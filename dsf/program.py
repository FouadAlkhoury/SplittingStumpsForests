# This script finds the splitting nodes, maps the input data to the new feature space induced by the splitting stumps, and trains a linear layer over it.
# To run this script, give a dataset, forest depth and size, and filtering threshold values 'lines: 58-61'
# The trained forests will be exported to forests folder.
# The selected splitting stumps will be exported to stumps folder.
# The results of linear layer training will be written in reports folder.
# datasets to choose from: 'adult','aloi', 'bank', 'credit', 'drybean', 'letter', 'magic', 'rice', 'room', 'shopping', 'spambase','satlog','waveform'
# imports
import sys
sys.path.append('../arch-forest/code/')
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import ReadData as ReadData
from fitModels import fitModels
import DecisionSnippetFeatures as DecisionSnippetFeatures
import datetime
import pruning
import Forest
from util import writeToReport
import numpy as np

class SSF:
    def __init__(self, stumpsPath, dataset, graph_file):
        self.model = LogisticRegression()
        self.snippets_file = os.path.join(stumpsPath, dataset, graph_file)

    def train(self, X, y):
        XX = ssf_transform_data(self.snippets_file, X, True)
        self.model.fit(XX,y)

    def predict(self, X):
        XX = ssf_transform_data(self.snippets_file, X, True)
        return self.model.predict(XX)

    def score(self,X,y):
        XX = ssf_transform_data(self.snippets_file, X, True)
        return accuracy_score(y,self.model.predict(XX))


dataPath = "../data/"
forestsPath = "../tmp/forests/"
stumpsPath = "../tmp/stumps/"
samplesPath = "../tmp/samples/"
resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports/"

def list_to_str(list):
    str_list = ''
    for l in list:
        str_list += str(l) + ','
    return str_list

datasets = ['adult']  # datasets that can be used: adult, bank, credit, drybean, letter, magic, rice, room, shopping, spambase, and satlog.
forest_depths = [5]
forest_sizes = [16]
thresholds = [0.25]
edge_thresholds = [1.0 - x  for x in thresholds]
forest_types = ['RF']
maxPatternSize = 1
minThreshold = 1
maxThreshold = 1
scoring_function = 'accuracy'
# learners on top of the Splitting Stumps
learners = {'LR': LogisticRegression}
learners_parameters = {'LR': {'max_iter': 10000}}
verbose = True
fitting_models_time = datetime.timedelta()
pruning_time = datetime.timedelta()

def traverse(tree, threshold):
    if ("probLeft" in tree and "probRight" in tree):

        if (tree["probLeft"] >= threshold or tree["probRight"] >= threshold):
            feature = tree["feature"]
            split = tree["split"]
            patterns.add((feature, split))

        traverse(tree["leftChild"], threshold)
        traverse(tree["rightChild"], threshold)

def ssf_transform_data(stumps_file, X, is_train):
    global pruning_time
    global patterns_count
    global patterns_ratio
    global filtered_patterns_indexes

    with open(stumps_file, 'r') as stumps:
        # load test nodes and create decision stumps
        frequentpatterns = json.load(stumps)
        patterns_count = len(frequentpatterns)

        ssf = DecisionSnippetFeatures.FrequentSubtreeFeatures(
            map(lambda x: x['pattern'], frequentpatterns))
        fts = ssf.fit_transform(X)

        # transform to onehot encodings for subsequent processing by linear methods
        categories = ssf.get_categories()
        fts_onehot_sparse = OneHotEncoder(handle_unknown='ignore',
                                          categories=categories).fit_transform(fts)
        fts_onehot = fts_onehot_sparse.toarray()

        return fts_onehot


def compute_nodes_count(stumps_file):

    with open(stumps_file, 'r') as f:
        text = f.read()
        nodes_count = text.count('"id"')
    f.close()

    return nodes_count

def train_model_on_top(model, fts_onehot, Y_train, scoring_function, model_name, descriptor, scaling=False):
        if scaling:
            model = Pipeline([('scaler', StandardScaler()), (model_name, model)])
        model.fit(fts_onehot, Y_train)
        train_acc = model.score(fts_onehot, Y_train)
        return train_acc, model

for dataset in datasets:

    print('dataset:' + dataset)
    split_iterations = 2
    accuracy_array = np.zeros((len(forest_sizes), len(forest_depths), split_iterations),dtype=np.float32)
    X, Y = ReadData.readData(dataset, 'train', dataPath)

    for i in range(split_iterations):

        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

        report_model_dir = reportsPath + '/' + dataset
        report_file = report_model_dir + '/report.txt'
        report_thresholds = report_model_dir + '/report_thresholds.csv'
        report_time_file = report_model_dir + '/report_time.txt'

        if not os.path.exists(report_model_dir):
            os.makedirs(report_model_dir)

        # %% create forest data, evaluate and report accuracy on test data
        start_fitting_models = datetime.datetime.now()
        accuracy_array = fitModels(roundSplit=True, iterations= split_iterations, dataset=dataset,
                  XTrain=X_train, YTrain=Y_train,
                  XTest=X_test, YTest=Y_test,
                  createTest=False, model_dir=os.path.join(forestsPath, dataset), types=forest_types,
                  forest_depths=forest_depths, forest_sizes=forest_sizes)
        end_fitting_models = datetime.datetime.now()
        fitting_models_time = (end_fitting_models - start_fitting_models)

        report_file = reportsPath + '/' + dataset + '/report_rf_test.csv'
        writeToReport(report_file, 'Forest size, Forest depth, avg. Test Accuracy, Std, ' + '\n')
        for i, size in enumerate(forest_sizes):
            for j, depth in enumerate(forest_depths):
                writeToReport(report_file, str(size) + ', ' + str(depth) + ', ' + str(np.mean(accuracy_array[i][j])) + ',' + str(np.std(accuracy_array[i][j])) + ', \n')

        writeToReport(report_time_file, 'Fitting Models Time \t ' + str(fitting_models_time) + '\n')
        writeToReport(report_thresholds,
                      'Forest Size, Forest Depth, Patterns threshold, Edges Threshold, Learner, Train Accuracy, Test Accuracy, Test Accuracy (gain), F1_macro, F1_macro (gain), F1_micro, Patterns Count, Patterns Ratio, Nodes Count, Nodes Ratio, Training Time, Pruning Time' + '\n')





        patterns_count = 0
        patterns_ratio = 0
        start_training_total = datetime.datetime.now()

        counter = 0
        pruning_time = datetime.timedelta()
        report_pruning_dir = reportsPath + '/' + dataset
        report_pruning_file = report_pruning_dir + '/report_pruning_time.txt'
        score = 0
        scoreStr = ''
        writeToReport(report_pruning_file, 'Forest Size, Forest Depth, Pruning threshold, Pruning Time' + '\n')

        for graph_file in filter(lambda x: x.endswith('.json'), sorted(os.listdir(os.path.join(forestsPath, dataset)))):
                forests_file = os.path.join(forestsPath, dataset, graph_file)
                with open(forests_file, 'r') as f_decision_forests:
                    trees = json.load(f_decision_forests)
                    for th in edge_thresholds:
                        features = []
                        splits = []
                        patterns = set()
                        counter = 0
                        pruned_file = graph_file[:-5] + '_pruned_' + str(th) + '.json'
                        snippets_file = os.path.join(stumpsPath, dataset, pruned_file)
                        with open(snippets_file, 'w') as f_out:
                            f_out.write("[")
                            for tree in trees:
                                traverse(tree, th)
                            for p in patterns:
                                f_out.write(
                                    "{\"patternid\":" + str(counter) + ",\"pattern\":{\"id\":" + str(
                                        0) + ",\"feature\":" + str(
                                        p[0]) + ",\"split\":" + str(p[1]) + "}}")
                                f_out.write(",\n")
                                counter += 1
                            f_out.seek(0, 2)
                            f_out.seek(f_out.tell() - 2, 0)
                            f_out.truncate()
                            f_out.write("]")
        acc_ref = 0.0
        f1_macro_ref = 0.0
        patterns_count_ref = 0
        nodes_count_ref = 0

        print('\n\nStart Training\n')

        results_list = list()

        # save results list
        if not os.path.exists(os.path.join(resultsPath, dataset)):
                    os.makedirs(os.path.join(resultsPath, dataset))

        acc_ref = 1.0
        f1_macro_ref = 1.0
        patterns_count_ref = 1.0
        nodes_count_ref = 1.0
        train_acc = 0.0

        for size in forest_sizes:
            for depth in forest_depths:
                    for frequency in range(minThreshold, maxThreshold + 1):
                        for threshold in edge_thresholds:

                            graph_file = 'RF_' + str(size) + '_' + str(depth) + '_pruned_' + str(threshold) + '.json'
                            model_ssf = SSF(stumpsPath, dataset, graph_file)
                            nodes_count = compute_nodes_count(snippets_file)
                            if (threshold == 1.0):
                                patterns_count_ref = patterns_count
                                nodes_count_ref = nodes_count

                            patterns_ratio = patterns_count / patterns_count_ref
                            nodes_ratio = nodes_count / nodes_count_ref
                            filtered_patterns_indexes = []

                            # train models
                            for model_type, model_class in learners.items():
                                start_training = datetime.datetime.now()
                                model_ssf.train(X_train,Y_train)
                                train_acc = model_ssf.score(X_train,Y_train)
                                end_training = datetime.datetime.now()
                                training_time = (end_training - start_training)
                                print('training result = ' + str(train_acc))
                                print('Training Time for ' + graph_file + ' : ' + str(training_time))
                                writeToReport(report_time_file,
                                              'Training Time for ' + graph_file + ' , ' + model_type + ' : ' + str(
                                                  training_time))

                                start_testing = datetime.datetime.now()
                                Y_pred = model_ssf.predict(X_test)
                                test_acc = model_ssf.score(X_test, Y_test)

                                end_testing = datetime.datetime.now()
                                testing_time = (end_testing - start_testing)
                                print('Testing Time for ' + graph_file + ' : ' + str(testing_time))
                                writeToReport(report_time_file,
                                              'Testing Time for ' + graph_file + ' , ' + ' : ' + str(
                                                  testing_time))
                                f1_macro = f1_score(Y_pred, Y_test, average='macro')
                                f1_micro = f1_score(Y_pred, Y_test, average='micro')


                                if (threshold == 1):
                                    acc_ref = test_acc
                                    f1_macro_ref = f1_macro

                                print('test accuracy = ' + str(test_acc))
                                print('f1 score macro = ' + str(f1_macro))

                                writeToReport(report_thresholds,
                                              str(size) + ',' + str(depth) + ',' + str(frequency) + ',' + str(
                                                  threshold) + ',' + str(model_ssf) + ',' + str(
                                                  np.round(train_acc, 3)) + ',' + str(np.round(test_acc, 3)) + ',' + str(
                                                  np.round((test_acc - acc_ref), 3)) + ',' + str(
                                                  np.round(f1_macro, 3)) + ',' + str(
                                                  np.round((f1_macro - f1_macro_ref), 3)) + ',' + str(
                                                  np.round(f1_micro, 3)) + ',' + str(
                                                  patterns_count) + ',' + str(np.round(patterns_ratio, 3)) + ',' + str(
                                                  nodes_count) + ',' + str(np.round(nodes_ratio, 3)) + ',' + str(
                                                  training_time) + ',' + str(pruning_time) + ',' + str(
                                                  testing_time) + ',\n')

                                # cleanup
                                xval_score, learner_model, xval_results = None, None, None

            print(filtered_patterns_indexes)

            end_training_total = datetime.datetime.now()
            training_total_time = (end_training_total - start_training_total)
            print('Total Training Time: ' + str(training_total_time))
            writeToReport(report_time_file, 'Total Training Time: ' + str(training_total_time) + '\n')





