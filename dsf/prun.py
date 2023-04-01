# %% imports

import os
import json
import subprocess
import pickle
import sys
import re

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
forestsPath = "../tmp/forests_64/"
snippetsPath = "../tmp/snippets_64/"
samplesPath = "../tmp/samples/"
resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports_64/"

#dataset = sys.argv[1]
dataset = 'satlog'
forest_types = ['RF']
forest_depths = [5,10,15]
forest_sizes = [64]
#forest_depths = [5]
#forest_sizes = [32]

maxPatternSize = 1
minThreshold = 1
maxThreshold = 1

#edge_thresholds = [0.85,0.8,0.775,0.75,0.725,0.7,0.675,0.65,0.625,0.6,0.575,0.55]
edge_thresholds = [1.0, 0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6]

#edge_thresholds = [0.98]

scoring_function = 'accuracy'
# learners that are to be used on top of Decision Snippet Features
learners = {'LR': LogisticRegression}
learners_parameters = {'LR': {'max_iter': 10000}}

#learners = {'SVM': LinearSVC}
#learners_parameters = {'SVM': {'max_iter': 10000}}

#learners = {'NB': MultinomialNB}
#learners_parameters = {'NB': {}}

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

if not os.path.exists(report_model_dir):
    os.makedirs(report_model_dir)

# %% create forest data, evaluate and report accuracy on test data
start_fitting_models = datetime.datetime.now()
'''
print('\n\nHERE ARE THE ACCURACIES ON TEST DATA OF THE ORIGINAL RANDOM FORESTS\n(don\'t worry, test data is not used for training)\n')

fitModels(roundSplit=True,  dataset = dataset,
              XTrain=X_train, YTrain=Y_train, 
              XTest=X_test, YTest=Y_test, 
              createTest=False, model_dir=os.path.join(forestsPath, dataset), types=forest_types, forest_depths = forest_depths, forest_sizes = forest_sizes)

end_fitting_models = datetime.datetime.now()
fitting_models_time = (end_fitting_models- start_fitting_models)

print('Fitting Models Time: '+str(fitting_models_time))  


        
writeToReport(report_time_file,'Fitting Models Time \t ' + str(fitting_models_time) + '\n')

writeToReport(report_thresholds,'Forest Size, Forest Depth, Patterns threshold, Edges Threshold, Learner, Train Accuracy, Test Accuracy, Test Accuracy (gain), F1_macro, F1_macro (gain), F1_micro, ROC_AUC, ROC_AUC (gain), Patterns Count, Patterns Ratio, Nodes Count, Nodes Ratio, Training Time, Pruning Time'+ '\n')
'''
start_pruning_total = datetime.datetime.now()
writeToReport(report_time_file,'Pruning Time \t ')
#writeToReport(report_file,'RF \t Sigma \t Pruning Time')

patterns_count = 0
patterns_ratio = 0

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
        #print(len(fts[0]))

        # transform to onehot encodings for subsequent processing by linear methods
        categories = dsf.get_categories()
        fts_onehot_sparse = OneHotEncoder(handle_unknown='ignore',
            categories=categories).fit_transform(fts)
        fts_onehot = fts_onehot_sparse.toarray()
       
        return fts_onehot
         
start_training_total = datetime.datetime.now()   

writeToReport(report_file, 'Depth, Frequency, Model, Accuracy, Deviation,')

### prun

#edge_thresholds = [0.7,0.6]
#edge_thresholds = [1.0,0.975,0.95,0.925,0.9,0.85,0.8,0.7,0.6]
counter = 0
pruning_time = datetime.timedelta()
report_pruning_dir = reportsPath+'/'+dataset 
report_pruning_file = report_pruning_dir + '/report_pruning_time.txt'
score = 0
scoreStr = ''

def exist_pattern(pattern, patternsSet):
    
    for p in patternsSet:
        if pattern in p:
            return True
        if p in pattern:
            return True
    return False
    
# Python program to perform iterative preorder traversal


writeToReport(report_pruning_file,'Forest Size, Forest Depth, Pruning threshold, Pruning Time'+ '\n')

def is_valid(tree, threshold, index):
    global subtreeStr
    if ("probLeft" in tree and "leftChild" in tree):

        if (tree["probLeft"] <= threshold and tree["probRight"] <= threshold):

            if("probLeft" in tree["leftChild"] and "leftChild" in tree):

                leftTree = tree["leftChild"]
                is_valid(leftTree, threshold, index)
            else:
                del tree["leftChild"]
            if ("probRight" in tree["rightChild"] and "rightChild" in tree):
                rightTree = tree["rightChild"]
                is_valid(rightTree, threshold, index)
            else:
                del tree["rightChild"]

        else:

            treeLeft = tree["leftChild"]
            treeRight = tree["rightChild"]
            subtrees.append(treeLeft)
            subtrees.append(treeRight)

            subtreeStr = subtreeStr.replace(str(tree),'')

    else:

        subtreeStr = subtreeStr.replace(str(tree), '')

    if (tree is not None and "prediction" in tree):

        subtreeStr = subtreeStr.replace(str(tree), '')



def remove_a_key(d, remove_key):
    if isinstance(d, dict):
        for key in list(d.keys()):


                if key == remove_key:
                    del d[key]
                else:
                    remove_a_key(d[key], remove_key)

def remove_a_child_key(d):
    if isinstance(d, dict):
        for key in list(d.keys()):
            if ((key =='leftChild' or key == 'rightChild') and (len(d[key]) == 0)):


                    del d[key]

            else:
                    remove_a_child_key(d[key])


for size in forest_sizes:
    for depth in forest_depths:
            
        graph_file = 'RF_'+str(size)+'_'+str(depth)+'.json'    
        forests_file = os.path.join(forestsPath, dataset, graph_file)
        #data = json.loads(forests_file)

        print(forests_file)
        with open(forests_file, 'r') as f_decision_forests:

            jsonforestStr = f_decision_forests.read()


            
            
            for th in edge_thresholds:
                print('th: '+str(th))


                patterns = set()
                # global subtrees
                global subtreeStr
                subtrees = []
                print('Threshold: ' + str(th))
                #jsonforest = json.loads(forest)

                jsonforest = json.loads(jsonforestStr)

                for jsontree in jsonforest:

                    #jsontree = jsontree["pattern"]
                    #print('Now')
                    #print(jsontree)
                    # remove_a_key(jsontree, 'id')
                    remove_a_key(jsontree, 'numSamples')
                    remove_a_key(jsontree, 'isCategorical')
                    remove_a_key(jsontree, 'prediction')
                    subtrees.append(jsontree)
                for index, subtree in enumerate(subtrees):
                    subtreeStr = str(subtree)
                    is_valid(subtree, th, index)
                    subtreeStr = re.sub(", 'leftChild': {'id': [0-9]+}", '', subtreeStr)
                    subtreeStr = re.sub(", 'leftChild': ,", ',', subtreeStr)
                    subtreeStr = re.sub(", 'rightChild': {'id': [0-9]+}", '', subtreeStr)
                    subtreeStr = re.sub(", 'rightChild': ,", ',', subtreeStr)
                    subtreeStr = re.sub(", 'rightChild': }", '}', subtreeStr)
                    subtreeStr = re.sub(", 'leftChild': }", '}', subtreeStr)
                    subtreeStr = re.sub("'split'", '"split"', subtreeStr)
                    subtreeStr = re.sub("'feature'", '"feature"', subtreeStr)

                    subtreeStr = re.sub("'probLeft':(.)+?,", '', subtreeStr)
                    subtreeStr = re.sub("'probRight':(.)+?,", '', subtreeStr)
                    # subtreeStr = re.sub("'probRight'", '"probRight"', subtreeStr)
                    subtreeStr = re.sub("'id'", '"id"', subtreeStr)
                    subtreeStr = re.sub("'leftChild'", '"leftChild"', subtreeStr)
                    subtreeStr = re.sub("'rightChild'", '"rightChild"', subtreeStr)

                    if (len(subtreeStr) > 2):
                        patterns.add(subtreeStr)
                    #print(subtreeStr)

                    subtreeStr = ''

                pruned_file = graph_file[:-5] + '_pruned_' +str(th)+ '.json'    
                snippets_file = os.path.join(snippetsPath, dataset, pruned_file)  
                
                with open(snippets_file,'w') as f_out:
                    f_out.write("[")

                    for p in patterns:
                        f_out.write('{"patternid": 0, "pattern": ')
                        #f_out.write('{')

                        f_out.write(p)
                        f_out.write('},\n')
                        print("tree")


                    connected_file = os.path.join("../tmp/snippets_64_no_edges/", dataset, pruned_file)
                    with open(connected_file,'r') as f_connected_in:
                        text = f_connected_in.read()
                        text = text[1:]
                        text += '}'
                    f_connected_in.close()
                    f_out.write(text)
                    if (len(patterns) > 0):

                        f_out.seek(0,2)
                        f_out.seek(f_out.tell() - 2, 0)
                        f_out.truncate()
                        f_out.write("]")


                    
                    end_pruning_time = datetime.datetime.now()
                    #pruning_time = (end_pruning_time - start_pruning_time)
                    #writeToReport(report_pruning_file, str(size)+ ', ' + str(depth) + ', ' + str(th) + ', ' + str(pruning_time) + '\n')

## nodes count

def compute_nodes_count(snippets_file):
        
        with open(snippets_file,'r') as f:
            text = f.read()
            nodes_count = text.count('"id"')
            #print(count)
        f.close()  
        
        return nodes_count

        
### training
acc_ref = 0.0
f1_macro_ref = 0.0
roc_auc_ref = 0.0
patterns_count_ref = 1
nodes_count_ref = 1


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
        writeToReport(report_file, str(model_name) + '\t' + str(descriptor) + '\t' + str(dsf_score)  + ' +- ' + str(dsf_std))
      
        model.fit(fts_onehot, Y_train)
        train_acc = model.score(fts_onehot,Y_train)
        return train_acc, model
    
# train several models on the various decision snippet features
#for graph_file in filter(lambda x: x.endswith('.json'), sorted(os.listdir(os.path.join(snippetsPath, dataset)))):
for size in forest_sizes:
       for depth in forest_depths:
            for frequency in range(minThreshold, maxThreshold+1):
                 for threshold in edge_thresholds: 
        
                    graph_file = 'RF_'+str(size)+'_'+str(depth)+'_pruned_'+str(threshold)+'.json'
                    snippets_file = os.path.join(snippetsPath, dataset, graph_file)  
                    nodes_count = compute_nodes_count(snippets_file)
                    fts_onehot = dsf_transform_edges(snippets_file, X_train, True)
                    if (threshold == 1.0):
                        patterns_count_ref = patterns_count
                        nodes_count_ref = nodes_count
                    patterns_ratio = patterns_count / patterns_count_ref
                    nodes_ratio = nodes_count / nodes_count_ref
                    filtered_patterns_indexes = []
        
                    # train models
                    for model_type, model_class in learners.items():
                        start_training = datetime.datetime.now()
                       
                        train_acc, learner_model = train_model_on_top(model_class(**learners_parameters[model_type]), fts_onehot, Y_train, scoring_function, model_type, graph_file)
                        #results_list.append((xval_score, model_type, graph_file, learner_model, xval_results))
                            
                        end_training = datetime.datetime.now()
                        training_time = (end_training - start_training)
                        print('training result = ' + str(train_acc))
                        print('Training Time for '+ graph_file +' : '+str(training_time))
                       
                        writeToReport(report_file, 'Training Time for '+ graph_file+' , ' + model_type +' : ' +str(training_time)) 

                       
                        fts_onehot_test = dsf_transform_edges(os.path.join(snippetsPath, dataset, graph_file), X_test, False)
                       
                        test_acc = learner_model.score(fts_onehot_test,Y_test)
                        Y_pred = learner_model.predict(fts_onehot_test)
                        #Y_test = np.argmax(Y_test, axis=0)
                        f1_macro = f1_score(Y_pred,Y_test,average = 'macro')
                        f1_micro = f1_score(Y_pred,Y_test,average = 'micro')
                        #roc_auc = roc_auc_score(Y_test,Y_pred)
                        roc_auc = 0
                        if (threshold == 1):
                            acc_ref = test_acc
                            f1_macro_ref = f1_macro
                            roc_auc_ref = roc_auc
                        #test_acc = accuracy_score(Y_test, pred_test)
                        print('test accuracy = ' + str(test_acc))
                        print('f1 score macro = ' + str(f1_macro))
                        #print('roc auc score = ' + str(roc_auc))
                        
                        writeToReport(report_thresholds,str(size)+','+str(depth)+','+str(frequency)+','+str(threshold)+','+str(learner_model)+','+str(np.round(train_acc,3))+','+str(np.round(test_acc,3))+',' +str(np.round((test_acc - acc_ref),3))+','+str(np.round(f1_macro,3))+',' +str(np.round((f1_macro - f1_macro_ref),3))+','+str(np.round(f1_micro,3))+','+str(np.round(roc_auc,3))+',' +str(np.round((roc_auc - roc_auc_ref),3))+','+str(patterns_count)+','+str(np.round(patterns_ratio,3))+','+str(nodes_count)+','+str(np.round(nodes_ratio,3))+','+str(training_time)+','+str(pruning_time)+',\n')
                        
#writeToReport(report_thresholds,str(size)+','+str(depth)+','+str(frequency)+','+str(threshold)+','+str(learner_model)+','+str(np.round(train_acc,3))+','+str(np.round(test_acc,3))+',' +str(np.round((test_acc - acc_ref),3))+','+str(np.round(f1_macro,3))+',' +str(np.round((f1_macro - f1_macro_ref),3))+','+str(np.round(f1_micro,3))+','+str(np.round(roc_auc,3))+',' +str(np.round((roc_auc - roc_auc_ref),3))+','+str(patterns_count)+','+str(np.round(patterns_ratio,3))+','+str(training_time)+','+str(pruning_time)+',\n')                        

                        print(str(learner_model) + ' ' +str(model_type) + ' ' + str(np.round(test_acc,3)) + ' ' +str(fts_onehot.shape[1]))                       
                        
                        # cleanup
                        xval_score, learner_model, xval_results = None, None, None
                
                        #writeToReport(report_file, str(best_result[2]) + ',' + str(model_type) + ',' + str(np.round(best_result[0],3)) + ',' + str(np.round(test_acc,3)) + ',' + str(fts_onehot.shape[1]))    
            
print(filtered_patterns_indexes)            

end_training_total = datetime.datetime.now()
training_total_time = (end_training_total - start_training_total)
print('Total Training Time: '+str(training_total_time))  
writeToReport(report_time_file, 'Total Training Time: '+str(training_total_time) + '\n')

# %% Find, for each learner, the best decision snippet features on training data (the proposed model) and evaluate its performance on test data

writeToReport(report_file,'\n Best \n')
writeToReport(report_file,'DSF, Model, Accuracy, Deviation,')

       
