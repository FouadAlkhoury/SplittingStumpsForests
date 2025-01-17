# The script trains a random forest model.
import os
import json
from sklearn.utils.estimator_checks import check_estimator
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

import ReadData as ReadData
import cString2json as cString2json
import json2graphNoLeafEdgesWithSplitValues as json2graphNoLeafEdgesWithSplitValues
from fitModels import fitModels
import DecisionSnippetFeatures as DecisionSnippetFeatures
import Forest
import datetime
from util import writeToReport
import numpy as np

dataPath = "../data/"
forestsPath = "../tmp/forests/"
snippetsPath = "../tmp/stumps/"
samplesPath = "../tmp/samples/"
resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports/"

dataset = 'letter'
forest_types = ['RF']
forest_depths = [5,10,15]
forest_sizes = [16,32]
param_grid = {  'min_samples_leaf': [1,5,10,20],  'ccp_alpha': [0.005,0.01,0.015,0.02]}
edge_thresholds = [0.7]

maxPatternSize = 1
minThreshold = 1
maxThreshold = 1

scoring_function = 'accuracy'
# learners that are to be used on top of Decision Snippet Features
learners = {'LR': LogisticRegression}

learners_parameters = {'LR': {'max_iter': 10000}}

verbose = True

fitting_models_time = datetime.timedelta()
pruning_time = datetime.timedelta()

X_train, Y_train = ReadData.readData(dataset, 'train', dataPath)
X_test, Y_test = ReadData.readData(dataset, 'test', dataPath)
X = X_train

report_model_dir = reportsPath+'/'+dataset
report_file = report_model_dir + '/report.txt'
report_thresholds = report_model_dir + '/report_thresholds.csv'
report_time_file = report_model_dir + '/report_time.txt'


def testModel(roundSplit, dataset, XTrain, YTrain, XTest, YTest, model, name, model_dir, size, depth):
    report_file = reportsPath + '/' + dataset + '/report_rf_test.csv'

    print("Fitting", name)
    model.fit(XTrain, YTrain)

    print("Testing ", name)
    # start = timeit.default_timer()
    start_testing = datetime.datetime.now()
    YPredicted = model.predict(XTest)
    end_testing = datetime.datetime.now()
    testing_time = (end_testing - start_testing)
    print('Testing Time ' + str(testing_time))

    print("Saving model")
    if (issubclass(type(model), DecisionTreeClassifier)):
        mymodel = Tree.Tree()
    else:
        mymodel = Forest.Forest()

    mymodel.fromSKLearn(model, roundSplit)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(os.path.join(model_dir, name + ".json"), 'w') as outFile:
        outFile.write(mymodel.str())

    SKPred = model.predict(XTest)
    # MYPred = mymodel.predict_batch(XTest)
    accuracy = accuracy_score(YTest, SKPred)
    print("Accuracy:", accuracy)

    # auc = roc_auc_score(YTest, SKPred)
    auc = 0
    f1_macro = f1_score(YTest, SKPred, average='macro')
    f1_micro = f1_score(YTest, SKPred, average='micro')

    with open(report_file, 'a') as outFile:
        outFile.write(str(size) + ', ' + str(depth) + ', ' + str(accuracy) + ',' + str(f1_macro) + ',' + str(
            f1_micro) + ',' + str(auc) + ', \n')
    outFile.close()


    print("Saving model to PKL on disk")
    joblib.dump(model, os.path.join(model_dir, name + ".pkl"))

    print("*** Summary ***")
    print("#Examples\t #Features\t Accuracy\t Avg.Tree Height")
    print(str(accuracy) + "\t" + str(mymodel.getAvgDepth()))

    print()



if not os.path.exists(report_model_dir):
    os.makedirs(report_model_dir)

# %% create forest data, evaluate and report accuracy on test data
start_fitting_models = datetime.datetime.now()

print('\n\nHERE ARE THE ACCURACIES ON TEST DATA OF THE ORIGINAL RANDOM FORESTS\n(don\'t worry, test data is not used for training)\n')

for size in forest_sizes:
    for depth in forest_depths:

        gridSearch = GridSearchCV(RandomForestClassifier(n_estimators=size, n_jobs=8, max_depth=depth), param_grid, cv=5, n_jobs=1)
        gridSearch.fit(X_train, Y_train)
        print('Initial score: ', gridSearch.best_score_)
        print('Initial parameters: ', gridSearch.best_params_)

