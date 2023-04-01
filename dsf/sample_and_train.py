import json
import random
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import ReadData as ReadData
import DecisionSnippetFeatures as DecisionSnippetFeatures
import datetime
from util import writeToReport
import numpy as np

dataPath = "../data/"
forestsPath = "../tmp/forests_64/"
snippetsPath = "../tmp/snippets_64_no_edges/"
samplesPath = "../tmp/samples_64_no_edges/"
resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports_64_no_edges/"
patternList = []




def sampleRandomForest(dataset, samples_count, patternsCount):
    with open(snippetsPath + dataset + '/RF_64_15_pruned_1.0.json') as json_file:
        data = json.load(json_file)
        for c in range(0, samples_count):

            patternList = []
            for i in range(0, patternsCount - 1):
                print(str(data[random.randint(0, len(data) - 1)]).replace("'", '"'))
                patternList.append(str(data[random.randint(0, len(data) - 1)]).replace("'", '"') + ',')
            patternList.append(str(data[random.randint(0, len(data) - 1)]).replace("'", '"') + ']')
            f = open(samplesPath + dataset + '/sample_'+ str(c + 1) + '.json', "w")
            f.write('[')
            for pattern in patternList:
                f.write(str(pattern) + '\n')

            f.close()

    json_file.close()


#dataset = sys.argv[1]
dataset = 'credit'
samples_count = 10
#patternsCount = int(sys.argv[2])
patternsCount = 194
#max_depth = int(sys.argv[2])
sampleRandomForest(dataset, samples_count, patternsCount)


X_train, Y_train = ReadData.readData(dataset, 'train', dataPath)
X_test, Y_test = ReadData.readData(dataset, 'test', dataPath)
X = X_train

scoring_function = 'accuracy'

learners = {'LR': LogisticRegression}

learners_parameters = {'LR': {'max_iter': 10000}}

verbose = True

report_model_dir = reportsPath+'/'+dataset
report_file = report_model_dir + '/samples.csv'
#report_thresholds = report_model_dir + '/report_thresholds.csv'
#report_time_file = report_model_dir + '/report_time.txt'


print('\n\nStart Training\n')

results_list = list()

# save results list
if not os.path.exists(os.path.join(resultsPath, dataset)):
    os.makedirs(os.path.join(resultsPath, dataset))


def train_model_on_top(model, fts_onehot, Y_train, scoring_function, model_name, descriptor, scaling=False):
    if scaling:
        model = Pipeline([('scaler', StandardScaler()), (model_name, model)])

    dsf_score = 0
    dsf_std = 0
    #writeToReport(report_file, str(model_name) + '\t' + str(descriptor) + '\t' + str(dsf_score) + ' +- ' + str(dsf_std))

    model.fit(fts_onehot, Y_train)
    train_acc = model.score(fts_onehot, Y_train)
    return train_acc, model


def dsf_transform_edges(snippets_file, X, is_train):
    global pruning_time
    global patterns_count
    global patterns_ratio
    global filtered_patterns_indexes

    with open(snippets_file, 'r') as f_decision_snippets:
        # load decision snippets and create decision snippet features
        frequentpatterns = json.load(f_decision_snippets)
        patterns_count = len(frequentpatterns)

        dsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(
            map(lambda x: x['pattern'], frequentpatterns))
        fts = dsf.fit_transform(X)
        # print(len(fts[0]))

        # transform to onehot encodings for subsequent processing by linear methods
        categories = dsf.get_categories()
        fts_onehot_sparse = OneHotEncoder(handle_unknown='ignore',
                                          categories=categories).fit_transform(fts)
        fts_onehot = fts_onehot_sparse.toarray()

        return fts_onehot

writeToReport(report_file, 'Sample, Test Accuracy')
test_acc_list = []
# train several models on the various decision snippet features
# for graph_file in filter(lambda x: x.endswith('.json'), sorted(os.listdir(os.path.join(snippetsPath, dataset)))):
for i in range(1, samples_count+1):


                graph_file = 'sample_'+ str(i) + '.json'
                snippets_file = os.path.join(samplesPath, dataset, graph_file)

                fts_onehot = dsf_transform_edges(snippets_file, X_train, True)
                filtered_patterns_indexes = []

                # train models
                for model_type, model_class in learners.items():
                    start_training = datetime.datetime.now()

                    train_acc, learner_model = train_model_on_top(model_class(**learners_parameters[model_type]),
                                                                  fts_onehot, Y_train, scoring_function, model_type,
                                                                  graph_file)
                    # results_list.append((xval_score, model_type, graph_file, learner_model, xval_results))

                    end_training = datetime.datetime.now()
                    training_time = (end_training - start_training)
                    print('training result = ' + str(train_acc))
                    print('Training Time for ' + graph_file + ' : ' + str(training_time))

                    #writeToReport(report_file,'Training Time for ' + graph_file + ' , ' + model_type + ' : ' + str(training_time))

                    fts_onehot_test = dsf_transform_edges(os.path.join(samplesPath, dataset, graph_file), X_test,
                                                          False)

                    test_acc = learner_model.score(fts_onehot_test, Y_test)
                    test_acc_list.append(test_acc)
                    #Y_pred = learner_model.predict(fts_onehot_test)
                    # Y_test = np.argmax(Y_test, axis=0)
                    #f1_macro = f1_score(Y_pred, Y_test, average='macro')
                    #f1_micro = f1_score(Y_pred, Y_test, average='micro')
                    # roc_auc = roc_auc_score(Y_test,Y_pred)
                    #roc_auc = 0
                    # test_acc = accuracy_score(Y_test, pred_test)
                    print('test accuracy = ' + str(test_acc))


                    writeToReport(report_file, str(i) + ',' + str(np.round(test_acc, 3)))

                    # writeToReport(report_thresholds,str(size)+','+str(depth)+','+str(frequency)+','+str(threshold)+','+str(learner_model)+','+str(np.round(train_acc,3))+','+str(np.round(test_acc,3))+',' +str(np.round((test_acc - acc_ref),3))+','+str(np.round(f1_macro,3))+',' +str(np.round((f1_macro - f1_macro_ref),3))+','+str(np.round(f1_micro,3))+','+str(np.round(roc_auc,3))+',' +str(np.round((roc_auc - roc_auc_ref),3))+','+str(patterns_count)+','+str(np.round(patterns_ratio,3))+','+str(training_time)+','+str(pruning_time)+',\n')

                    print(str(learner_model) + ' ' + str(model_type) + ' ' + str(np.round(test_acc, 3)) + ' ' + str(
                        fts_onehot.shape[1]))

                    # cleanup
                    xval_score, learner_model, xval_results = None, None, None

                    # writeToReport(report_file, str(best_result[2]) + ',' + str(model_type) + ',' + str(np.round(best_result[0],3)) + ',' + str(np.round(test_acc,3)) + ',' + str(fts_onehot.shape[1]))
writeToReport(report_file, '\n' + ' ,mean, std')
print(filtered_patterns_indexes)
writeToReport(report_file, ',' +str(round(np.mean(test_acc_list),3)) + ',' + str(round(np.std(test_acc_list),3)))
#print(np.mean(test_acc_list))
#print(np.std(test_acc_list))
#end_training_total = datetime.datetime.now()
#training_total_time = (end_training_total - start_training_total)

#print('Total Training Time: ' + str(training_total_time))
#writeToReport(report_time_file, 'Total Training Time: ' + str(training_total_time) + '\n')

# %% Find, for each learner, the best decision snippet features on training data (the proposed model) and evaluate its performance on test data

#writeToReport(report_file, '\n Best \n')
#writeToReport(report_file, 'DSF, Model, Accuracy, Deviation,')

