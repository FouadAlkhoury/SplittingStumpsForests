# The script samples randomly a specific number of nodes from the trained random forest and train them as splitting stumps (lines: 44-46)
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
forestsPath = "../tmp/forests/"
snippetsPath = "../tmp/stumps/"
samplesPath = "../tmp/samples/"
resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports/"
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


dataset = 'credit'
samples_count = 10
patternsCount = 194
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

    model.fit(fts_onehot, Y_train)
    train_acc = model.score(fts_onehot, Y_train)
    return train_acc, model

def dsf_transform_edges(snippets_file, X, is_train):
    global pruning_time
    global patterns_count
    global patterns_ratio
    global filtered_patterns_indexes

    with open(snippets_file, 'r') as f_decision_snippets:
        frequentpatterns = json.load(f_decision_snippets)
        patterns_count = len(frequentpatterns)

        dsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(
            map(lambda x: x['pattern'], frequentpatterns))
        fts = dsf.fit_transform(X)
        categories = dsf.get_categories()
        fts_onehot_sparse = OneHotEncoder(handle_unknown='ignore',
                                          categories=categories).fit_transform(fts)
        fts_onehot = fts_onehot_sparse.toarray()

        return fts_onehot

writeToReport(report_file, 'Sample, Test Accuracy')
test_acc_list = []

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

                    end_training = datetime.datetime.now()
                    training_time = (end_training - start_training)
                    print('training result = ' + str(train_acc))
                    print('Training Time for ' + graph_file + ' : ' + str(training_time))

                    fts_onehot_test = dsf_transform_edges(os.path.join(samplesPath, dataset, graph_file), X_test,
                                                          False)

                    test_acc = learner_model.score(fts_onehot_test, Y_test)
                    test_acc_list.append(test_acc)

                    print('test accuracy = ' + str(test_acc))
                    writeToReport(report_file, str(i) + ',' + str(np.round(test_acc, 3)))

                    print(str(learner_model) + ' ' + str(model_type) + ' ' + str(np.round(test_acc, 3)) + ' ' + str(
                        fts_onehot.shape[1]))

                    # cleanup
                    xval_score, learner_model, xval_results = None, None, None

writeToReport(report_file, '\n' + ' ,mean, std')
print(filtered_patterns_indexes)
writeToReport(report_file, ',' +str(round(np.mean(test_acc_list),3)) + ',' + str(round(np.std(test_acc_list),3)))
