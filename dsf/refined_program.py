#import os
import datetime
import pandas as pd
import numpy as np
import sklearn
#from sklearn import cross_validation
#from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from src.refined_rf import RefinedRandomForest
from src.telstra_data import TelstraData, multiclass_log_loss
from matplotlib import pyplot as plt
import ReadData
from sklearn.metrics import accuracy_score

featureparams = {'location_min_count': 0,
 'n_common_events': 20,
 'n_common_log_features': 40,
 'n_common_resources': 5,
 'n_label_encoded_log_features': 4}
aggregateparams = {"loc_agg_prior_weight":3.0}


datasets = ['adult','aloi', 'bank', 'credit', 'drybean', 'letter', 'magic', 'rice', 'room', 'shopping', 'spambase', 'satlog','waveform']
datasets = ['aloi','waveform']
dataPath = "../data/"

for dataset in datasets:
    print(dataset)
    X_train, Y_train = ReadData.readData(dataset, 'train', dataPath)
    X_test, Y_test = ReadData.readData(dataset, 'test', dataPath)

    clf = RandomForestClassifier(n_estimators=64, max_depth=5,n_jobs=8)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    rf_accuracy = accuracy_score(Y_test, Y_pred)
    print(rf_accuracy)


    rrf = RefinedRandomForest(clf, C = 0.01, n_prunings = 0)
    rrf.fit(X_train, Y_train)
    Y_pred = rrf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(accuracy)



    rrf.n_prunings = 1
    accuracy_list = [rf_accuracy]
    nleaves = [sum(rrf.n_leaves_)]
    n_not_leaves = [sum(rrf.n_not_leaves_)]
    total_nodes = [sum(rrf.n_leaves_) + sum(rrf.n_not_leaves_)]

    accuracy_ref = 0.0
    compress_ref= 1.0
    iter_ref = 1
    for k in range(50):
        print('Iteration: ' + str(k))
        rrf.fit(X_train, Y_train)
        start_testing = datetime.datetime.now()
        Y_pred = rrf.predict(X_test)
        end_testing = datetime.datetime.now()
        testing_time = (end_testing - start_testing)
        print('Testing Time for ' + dataset + ' : ' + str(testing_time))


        accuracy = accuracy_score(Y_test, Y_pred)
        print(accuracy)

        nleaves.append(sum(rrf.n_leaves_))
        n_not_leaves.append(sum(rrf.n_not_leaves_))
        total_nodes.append(sum(rrf.n_leaves_) + sum(rrf.n_not_leaves_))

        if (accuracy_ref < accuracy):
            accuracy_ref = accuracy
            compress_ref = nleaves[0]/nleaves[-1]
            iter_ref = k
        accuracy_list.append(accuracy)



    print(accuracy_list)
    print(nleaves)
    compression_ratio = nleaves[0]/nleaves[-1]

    print('--- Result ---')
    print(iter_ref)
    print(accuracy_ref)
    print(compress_ref)
