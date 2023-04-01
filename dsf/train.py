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
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
import statistics

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
snippetsPath = "../tmp/samples/"
resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports/"

dataset = sys.argv[1]
print(dataset)
scoring_function = 'accuracy'
# learners that are to be used on top of Decision Snippet Features
learners = {#'NB': MultinomialNB,
            #'SVM': LinearSVC,
            'LR': LogisticRegression}

#learners = {'NB': MultinomialNB}

# specify parameters that are given at initialization
learners_parameters = {#'NB': {},
                       #'SVM': {'max_iter': 10000},
                       'LR': {'max_iter': 10000}}

verbose = True
# %% load data
X_train, Y_train = ReadData.readData(dataset, 'train', dataPath)
X_test, Y_test = ReadData.readData(dataset, 'test', dataPath)
X = X_train

report_model_dir = reportsPath+'/'+dataset 
report_file = report_model_dir + '_new_splits/report2.csv'

writeToReport(report_file, 'Accuracy, Deviation,')

acc_ref = 0.0
f1_macro_ref = 0.0
roc_auc_ref = 0.0
patterns_count_ref = 0
test_accuracy_list = []

def dsf_transform(snippets_file, X, is_train):
    
    with open(snippets_file, 'r') as f_decision_snippets:

        frequentpatterns = json.load(f_decision_snippets)        
        dsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(
            map(lambda x: x['pattern'], frequentpatterns))
        fts = dsf.fit_transform(X)

        categories = dsf.get_categories()
        fts_onehot_sparse = OneHotEncoder(
            categories=categories).fit_transform(fts)
        fts_onehot = fts_onehot_sparse.toarray()
       
        return fts_onehot
        
    
results_list = list()

# save results list
if not os.path.exists(os.path.join(resultsPath, dataset)):
        os.makedirs(os.path.join(resultsPath, dataset))

def train_model_on_top(model, fts_onehot, Y_train, scoring_function, model_name, descriptor, scaling=False):

        if scaling:
            model = Pipeline([('scaler', StandardScaler()), (model_name, model)])

        dsf_score = 0
        dsf_std = 0
        writeToReport(report_file, str(model_name) + '\t' + str(descriptor) + '\t' + str(dsf_score)  + ' +- ' + str(dsf_std))
      
        model.fit(fts_onehot, Y_train)
        train_acc = model.score(fts_onehot,Y_train)
        return train_acc, model
    
start_training_total = datetime.datetime.now()   
for graph_file in filter(lambda x: x.endswith('.json'), sorted(os.listdir(os.path.join(snippetsPath, dataset)))):
                    print(graph_file)
     
                    fts_onehot = dsf_transform(os.path.join(snippetsPath, dataset, graph_file), X_train, True)
                  
                    # train models
                    for model_type, model_class in learners.items():
                        start_training = datetime.datetime.now()
                       
                        train_acc, learner_model = train_model_on_top(model_class(**learners_parameters[model_type]), fts_onehot, Y_train, scoring_function, model_type, graph_file)
                            
                        end_training = datetime.datetime.now()
                        training_time = (end_training - start_training)
                        print('training result = ' + str(train_acc))
                        print('Training Time for '+ graph_file +' : '+str(training_time))
                        
                       
                        writeToReport(report_file, 'Training Time for '+ graph_file+' , ' + model_type +' : ' +str(training_time)) 

                        fts_onehot_test = dsf_transform(os.path.join(snippetsPath, dataset, graph_file), X_test, False)
                       
                        test_acc = learner_model.score(fts_onehot_test,Y_test)
                        Y_pred = learner_model.predict(fts_onehot_test)
                        #Y_test = np.argmax(Y_test, axis=0)
                        f1_macro = f1_score(Y_pred,Y_test,average = 'macro')
                        f1_micro = f1_score(Y_pred,Y_test,average = 'micro')
                        roc_auc = roc_auc_score(Y_test,Y_pred)
                        test_accuracy_list.append(test_acc)
                        #test_accuracy_avg += test_acc
                        #test_acc = accuracy_score(Y_test, pred_test)
                        print('test accuracy = ' + str(test_acc))
                        print('f1 score macro = ' + str(f1_macro))
                        #print('f1 score micro = ' + str(f1_micro))
                        print('roc auc score = ' + str(roc_auc))
                        
                        writeToReport(report_file,str(learner_model)+','+str(np.round(train_acc,3))+','+str(np.round(test_acc,3))+',' +','+str(np.round(f1_macro,3))+','+str(np.round(f1_micro,3))+','+str(np.round(roc_auc,3))+','+str(training_time)+',\n')

                        print(str(learner_model) + ' ' +str(model_type) + ' ' + str(np.round(test_acc,3)) + ' ' +str(fts_onehot.shape[1]))                       
                        
                        # cleanup
                        xval_score, learner_model, xval_results = None, None, None
                
end_training_total = datetime.datetime.now()
training_total_time = (end_training_total - start_training_total)
print('Total Training Time: '+str(training_total_time))  
mean = statistics.mean(test_accuracy_list)
stdev = statistics.stdev(test_accuracy_list)
print('Average Test Accuracy: ' + str(mean))
print('Standard Deviation: ' + str(stdev))
#writeToReport(report_time_file, 'Total Training Time: '+str(training_total_time) + '\n')

