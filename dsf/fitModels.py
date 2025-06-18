import sys
import csv
import numpy as np
import os.path
import json
import timeit
import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import joblib

sys.path.append('../arch-forest/code/')
import Forest
import Tree
import m2cgen as m2c

resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports/"

accuracy_list = []


def testModel(roundSplit, iterations, dataset, XTrain, YTrain, XTest, YTest, model, name, model_dir, size, depth):
    report_file = reportsPath + '/' + dataset + '/report_rf_test.csv'

    print("Fitting", name)
    model.fit(XTrain, YTrain)
    c_code_model = m2c.export_to_c(model)
    with open("rf_model_c.c", "w") as f:
        f.write(c_code_model)

    # cross_val_scores = cross_val_score(model, XTrain,YTrain)
    # print(cross_val_scores)
    print("Testing ", name)
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
    accuracy = accuracy_score(YTest, SKPred)
    print("Accuracy:", accuracy)
    accuracy_list.append(accuracy)

    auc = 0
    f1_macro = f1_score(YTest, SKPred, average='macro')
    f1_micro = f1_score(YTest, SKPred, average='micro')

    # report_file = reportsPath + '/' + dataset + '/report_rf_test.csv'
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

    return accuracy


def fitModels(roundSplit, iterations, dataset, XTrain, YTrain, XTest=None, YTest=None, createTest=False,
              model_dir='text',
              types=['RF', 'ET', 'DT'],
              forest_depths=[1, 2],
              forest_sizes=[20, 30]):
    accuracy_arr = np.zeros((len(forest_sizes), len(forest_depths), iterations), dtype=np.float32)
    print(accuracy_arr)

    param_grid = {'min_samples_split': [50, 100, 150], 'min_samples_leaf': [50, 100, 150],
                  'ccp_alpha': [0.01, 0.02, 0.03]}
    report_file = reportsPath + '/' + dataset + '/report_rf.csv'
    if XTest is None or YTest is None:
        XTrain, XTest, YTrain, YTest = train_test_split(XTrain, YTrain, test_size=0.25)
        createTest = True

    if createTest:
        with open("test.csv", 'w') as outFile:
            for x, y in zip(XTest, YTest):
                line = str(y)
                for xi in x:
                    line += "," + str(xi)

                outFile.write(line + "\n")

    if 'DT' in types:
        for depth in forest_depths:
            testModel(roundSplit, XTrain, YTrain, XTest, YTest,
                      RandomForestClassifier(n_estimators=1, n_jobs=8, max_depth=depth), f"DT_{depth}", model_dir)

    if 'ET' in types:
        for depth in forest_depths:
            testModel(roundSplit, XTrain, YTrain, XTest, YTest,
                      ExtraTreesClassifier(n_estimators=forest_size, n_jobs=8, max_depth=depth), f"ET_{depth}",
                      model_dir)

    if 'RF' in types:

        with open(report_file, 'a') as outFile:
            outFile.write('Forest Size, Forest depth, Accuracy, F1_macro, F1_micro, AUC ROC, \n')
        outFile.close()
        for i, size in enumerate(forest_sizes):
            for j, depth in enumerate(forest_depths):
                for k in range(iterations):
                    acc = testModel(roundSplit, iterations, dataset, XTrain, YTrain, XTest, YTest,
                                    RandomForestClassifier(n_estimators=size, n_jobs=8, bootstrap=True,
                                                           max_depth=depth), f"RF_{size}_{depth}", model_dir, size,
                                    depth)
                    accuracy_arr[i][j][k] += acc

        print(accuracy_list)
        print(accuracy_arr)
        return accuracy_arr