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
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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

dataset = sys.argv[1]
#max_size = int(sys.argv[2])
max_sizes = [2,6,12,24,32,48,64]
#max_nodes_count = int(max_size/0.025)
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
features_count = len(X_train[0])
print(features_count)


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
        
  
def train_model_on_top(model, fts_onehot, Y_train, scoring_function, model_name, descriptor, scaling=False):

        if scaling:
            model = Pipeline([('scaler', StandardScaler()), (model_name, model)])

        dsf_score = 0
        dsf_std = 0
        #writeToReport(report_file, str(model_name) + '\t' + str(descriptor) + '\t' + str(dsf_score)  + ' +- ' + str(dsf_std))
      
        model.fit(fts_onehot, Y_train)
        train_acc = model.score(fts_onehot,Y_train)
        return train_acc, model

    
for size in max_sizes:
    

    max_nodes_count = int(size/0.025)
    nodes_per_feature = int(max_nodes_count / features_count)
    print(nodes_per_feature)
    features=[]
    splits=[]
    
    for f in range(features_count):



        c = [i[f] for i in X_train]
        #print(c)
        c = np.sort(c)
        #print(c)
        mid = c[int(5*(len(c)-1)/10)]
        #print(mid)
        #splits.append(mid)
        for i in range(1,int(nodes_per_feature)+1):


            a = c[int(i*(len(c)-1)/(nodes_per_feature+1))]
            #b = c[int((nodes_per_feature - i)*(len(c)-1)/10)]
            #if (mid == a):
            #    mid += 0.5
            #    print('Feature: ' + str(f) +', split: '+str(mid))
            #elif (mid == b):
            #    mid -= 0.5
            #    print('Feature: ' + str(f) +', split: '+str(mid))
            #else:
            #    print('Feature: ' + str(f) +', split: '+str(mid))
            #    print('Feature: ' + str(f) +', split: '+str(a))
            #    print('Feature: ' + str(f) +', split: '+str(b))
            #print(a)
            #print(b)
            #splits.append(mid)
            features.append(f)
            splits.append(a)
            #splits.append(b)
    
    print(features)
    print(splits)


    snippets_file = os.path.join(snippetsPath, dataset+'_new_splits',str(size)+ '.json')    
    with open(snippets_file,'w') as f_out:

                        f_out.write("[")
                        for index in range(len(splits)):

                            #print(features[int(index/9)])
                            #print(splits[index])



                            f_out.write("{\"patternid\":" + str(index) + ",\"pattern\":{\"id\":" +str(0)+",\"feature\":"+str(features[index]) + ",\"split\":"+str(splits[index]) + "}}")
                            f_out.write(",\n")

                        #f_out.write("\n")    
                        f_out.seek(0,2)
                        f_out.seek(f_out.tell() - 2, 0)
                        f_out.truncate()    
                        f_out.write("]")    


    report_model_dir = reportsPath+'/'+dataset 
    report_file = report_model_dir + '_new_splits/report.csv'

    writeToReport(report_file, 'Model Size (KB), Nodes Count, Train Accuracy, Test Accuracy, R2, MAE, MSE, Training Time,')

    acc_ref = 0.0
    f1_macro_ref = 0.0
    roc_auc_ref = 0.0
    patterns_count_ref = 0



    results_list = list()

    # save results list
    if not os.path.exists(os.path.join(resultsPath, dataset)):
            os.makedirs(os.path.join(resultsPath, dataset))



    start_training_total = datetime.datetime.now()   
    #for graph_file in filter(lambda x: x.endswith('.json'), sorted(os.listdir(os.path.join(snippetsPath, dataset+'_new_splits')))):
                        #print(graph_file)
    graph_file = snippets_file

    fts_onehot = dsf_transform(graph_file, X_train, True)

    # train models
    for model_type, model_class in learners.items():
                            start_training = datetime.datetime.now()

                            train_acc, learner_model = train_model_on_top(model_class(**learners_parameters[model_type]), fts_onehot, Y_train, scoring_function, model_type, graph_file)

                            end_training = datetime.datetime.now()
                            training_time = (end_training - start_training)
                            print('training result = ' + str(train_acc))
                            print('Training Time for '+ graph_file +' : '+str(training_time))


                            #writeToReport(report_file, 'Training Time for '+ graph_file+' , ' + model_type +' : ' +str(training_time)) 

                            fts_onehot_test = dsf_transform(graph_file, X_test, False)

                            test_acc = learner_model.score(fts_onehot_test,Y_test)
                            Y_pred = learner_model.predict(fts_onehot_test)
                            #Y_test = np.argmax(Y_test, axis=0)
                            r2 = r2_score(Y_test,Y_pred)
                            mae = mean_absolute_error(Y_test,Y_pred)
                            mse = mean_squared_error(Y_test,Y_pred)

                            #test_acc = accuracy_score(Y_test, pred_test)
                            print('r2 = ' + str(r2))
                            print('mae = ' + str(mae))
                            print('mse = ' + str(mse))

                            writeToReport(report_file,str(size)+','+str(max_nodes_count) + ','+str(np.round(train_acc,3))+','+str(np.round(test_acc,3))+','+str(np.round(r2,3))+','+str(np.round(mae,3))+','+str(np.round(mse,3))+','+str(training_time)+',\n')

                            print(str(learner_model) + ' ' +str(model_type) + ' ' + str(np.round(test_acc,3)) + ' ' +str(fts_onehot.shape[1]))                       

                            # cleanup
                            xval_score, learner_model, xval_results = None, None, None
                
    end_training_total = datetime.datetime.now()
    training_total_time = (end_training_total - start_training_total)
    print('Total Training Time: '+str(training_total_time))  
    #writeToReport(report_time_file, 'Total Training Time: '+str(training_total_time) + '\n')

