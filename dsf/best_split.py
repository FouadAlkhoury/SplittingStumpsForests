# This is the (cleaned up) code accompanying the publication
#
# Pascal Welke, Fouad Alkhoury, Christian Bauckhage, Stefan Wrobel: Decision Snippet Features.
# International Conference on Pattern Recognition (ICPR) 2021.
#
# Code was written by Pascal Welke and Fouad Alkhoury and is based on
# code written by Sebastian Buschjaeger (TU Dortmund) that is used for 
# json-serialization of random forest models.


# %% imports

import os
import json
import subprocess
import pickle
import sys

from sklearn.utils.estimator_checks import check_estimator
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import ReadData as ReadData
import cString2json as cString2json
import json2graphNoLeafEdgesWithSplitValues as json2graphNoLeafEdgesWithSplitValues
from fitModels import fitModels
import DecisionSnippetFeatures as DecisionSnippetFeatures
import pruning
import Forest
import datetime
from util import writeToReport
import numpy as np
# %% Parameters. 

dataPath = "../data/"
forestsPath = "../tmp/forests/"
snippetsPath = "../tmp/snippets/"
resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports/"

# current valid options are ['sensorless', 'satlog', 'mnist', 'magic', 'spambase', 'letter', 'bank', 'adult', 'drinking']

dataset = sys.argv[1]
print(dataset)
#dataSet = 'satlog'
# dataSet = 'adult'
# dataSet = 'drinking'

# possible forest_types ['RF', 'DT', 'ET']
forest_types = ['RF']
forest_depths = [5]
sigma_values = [0.3]
#forest_depths = [5, 10, 15, 20]
forest_size = 25

maxPatternSize = 4
minThreshold = int(sys.argv[2])
maxThreshold = int(sys.argv[3])

scoring_function = 'accuracy'

# learners that are to be used on top of Decision Snippet Features
#learners = {'NB': MultinomialNB,
#            'SVM': LinearSVC, 
#            'LR': LogisticRegression}

learners = {'NB': MultinomialNB}

# specify parameters that are given at initialization
# learners_parameters = {'NB': {},
#                       'SVM': {'max_iter': 30000},
#                       'LR': {'max_iter': 5000}}

learners_parameters = {'NB': {}}


# for quick debugging, let the whole thing run once. Afterwards, you may deactivate individual steps
# each step stores its output for the subsequent step(s) to process
run_fit_models = True
run_mining = True
run_training = True
run_eval = True

verbose = True

fitting_models_time = datetime.timedelta()
pruning_time = datetime.timedelta()



# %% load data

X_train, Y_train = ReadData.readData(dataset, 'train', dataPath)
X_test, Y_test = ReadData.readData(dataset, 'test', dataPath)
X = X_train

print(X[][0])
