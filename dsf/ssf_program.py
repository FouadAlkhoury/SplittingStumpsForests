# This script finds the splitting nodes, maps the input data to the new feature space induced by the splitting stumps, and trains a linear layer over it.
# To run this script, give a dataset, forest depth and size, filtering threshold values, and preferred decimal places 'lines: 27-32'
# The trained forests will be exported to forests folder.
# The selected splitting stumps will be exported to stumps folder.
# The results of linear layer training will be written in reports folder.
# datasets to choose from: 'adult','aloi', 'bank', 'credit', 'drybean', 'letter', 'magic', 'rice', 'room', 'shopping', 'spambase','satlog','waveform'
import math
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
import read_data
import ssf

def get_nodes_count(rf):
    nodes_count = 0
    for e in rf.estimators_:
        nodes_count += e.tree_.node_count
    return nodes_count

def writeToReport(path,content):
    f= open(path,"a")
    f.write(content+'\n')
    f.close()

datasets = ['adult','bank','credit','drybean','letter','magic','rice','room','shopping','spambase','satlog']
datasets = ['adult']
split_iterations = 1
forest_depth = 5
forest_size = 64
thresholds = [0.4]
decimal_places = [d for d in range(4,5)]

report_file = 'results_'+str(forest_size) +'_' + str(forest_depth) + '.csv'
report_file_lr = 'results_lr_.csv'
#writeToReport(report_file_lr,'Dataset, Accuracy')
writeToReport(report_file, 'Dataset, threshold, decimal places, avg. Test Accuracy, Std, avg. RF Nodes count, std. RF Nodes count, avg. SSF Nodes count, std. SSF Nodes count  ' + '\n')
for dataset in datasets:
    print(dataset)

    random_seed = random.randint(30, 50)
    decimal_place = decimal_places[0]
    X, y = read_data.readData(dataset)
    X_train, X_test_, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random_seed)
    X_train = np.round(X_train, decimals=decimal_place)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train,y_train)
    print('Logistic Regression accuracy: ' + str(lr.score(X_test_,y_test)))
    writeToReport(report_file_lr,
                  dataset + ',' + str(lr.score(X_test_,y_test))+ '\n')


    accuracy_array = np.zeros((len(thresholds), len(decimal_places), split_iterations), dtype=np.float32)


    for t,threshold in enumerate(thresholds):
        print('Threshold: ' + str(threshold))
        for d,decimal_place in enumerate(decimal_places):
            print('Decimal Place: ' + str(decimal_place))
            X, y = read_data.readData(dataset)
            test_acc = []
            nodes_count = []
            stumps_count = []

            acc_list = []

            for i in range(split_iterations):

                    random_seed = random.randint(30,50)

                    X_train, X_test_, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random_seed)
                    X_train = np.round(X_train, decimals=decimal_place)

                    rf = RandomForestClassifier(n_estimators=forest_size, max_depth=forest_depth)

                    lr = LogisticRegression(max_iter=10000)
                    rf.fit(X_train,y_train)
                    acc_list.append(rf.score(X_test_, y_test))

                    print('RF accuracy: ' + str(rf.score(X_test_,y_test)))
                    print(get_nodes_count(rf))

                    ssf_object, forest, lr, stumps = ssf.SSF.from_data(X_train, y_train, rf, lr, threshold, decimal_place)
                    nodes_count.append(get_nodes_count(rf))
                    X_test = (math.pow(10,decimal_place) * X_test_).astype(int)
                    ssf_mlgen_acc = ssf_object.score(X_test, y_test)
                    print('SSF accuracy: ' + str(round(ssf_mlgen_acc['Accuracy'],4)))
                    test_acc.append(round(ssf_mlgen_acc['Accuracy'],4))
                    stumps_count.append(len(stumps))

            writeToReport(report_file, dataset + ',' + str(threshold) + ',' + str(decimal_place) + ','  + str(np.mean(test_acc)) + ',' + str(np.std(test_acc)) + ',' +
                          str(np.mean(nodes_count)) + ',' + str(np.std(nodes_count)) + ',' + str(np.mean(stumps_count)) + ',' + str(np.std(stumps_count))
                          + '\n')
