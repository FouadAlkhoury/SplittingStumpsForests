# This script finds the splitting nodes, maps the input data to the new feature space induced by the splitting stumps, and trains a linear layer over it.
# To run this script, give a dataset, forest depth and size, and filtering threshold values 'lines: 31-34'
# The trained forests will be exported to forests folder.
# The selected splitting stumps will be exported to stumps folder.
# The results of linear layer training will be written in reports folder.

# imports
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import ReadData as ReadData
from fitModels import fitModels
import DecisionSnippetFeatures as DecisionSnippetFeatures
import datetime
from util import writeToReport
import numpy as np

dataPath = "../data/"
forestsPath = "../tmp/forests/"
stumpsPath = "../tmp/stumps/"
samplesPath = "../tmp/samples/"
resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports/"

dataset = 'adult' # datasets that can be used: adult, bank, credit, drybean, letter, magic, rice, room, shopping, spambase, and satlog.
forest_depths = [5,10,15]
forest_sizes = [16,32,64]
thresholds = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45]
edge_thresholds = [1.0 - x  for x in thresholds]
forest_types = ['RF']
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

if not os.path.exists(report_model_dir):
    os.makedirs(report_model_dir)

# %% create forest data, evaluate and report accuracy on test data
start_fitting_models = datetime.datetime.now()

fitModels(roundSplit=True,  dataset = dataset,
              XTrain=X_train, YTrain=Y_train, 
              XTest=X_test, YTest=Y_test, 
              createTest=False, model_dir=os.path.join(forestsPath, dataset), types=forest_types, forest_depths = forest_depths, forest_sizes = forest_sizes)

end_fitting_models = datetime.datetime.now()
fitting_models_time = (end_fitting_models- start_fitting_models)

print('Fitting Models Time: '+str(fitting_models_time))
        
writeToReport(report_time_file,'Fitting Models Time \t ' + str(fitting_models_time) + '\n')

writeToReport(report_thresholds,'Forest Size, Forest Depth, Patterns threshold, Edges Threshold, Learner, Train Accuracy, Test Accuracy, Test Accuracy (gain), F1_macro, F1_macro (gain), F1_micro, ROC_AUC, ROC_AUC (gain), Patterns Count, Patterns Ratio, Nodes Count, Nodes Ratio, Training Time, Pruning Time'+ '\n')

start_pruning_total = datetime.datetime.now()
writeToReport(report_time_file,'Pruning Time \t ')

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

        # transform to onehot encodings for subsequent processing by linear methods
        categories = dsf.get_categories()
        fts_onehot_sparse = OneHotEncoder(handle_unknown='ignore',
            categories=categories).fit_transform(fts)
        fts_onehot = fts_onehot_sparse.toarray()
       
        return fts_onehot
         
start_training_total = datetime.datetime.now()   

writeToReport(report_file, 'Depth, Frequency, Model, Accuracy, Deviation,')

counter = 0
pruning_time = datetime.timedelta()
report_pruning_dir = reportsPath+'/'+dataset 
report_pruning_file = report_pruning_dir + '/report_pruning_time.txt'
score = 0
scoreStr = ''

writeToReport(report_pruning_file,'Forest Size, Forest Depth, Pruning threshold, Pruning Time'+ '\n')

# select nodes that surpasses the threshold
def traverse(tree, threshold):
    if ("probLeft" in tree and "probRight" in tree):

        if (tree["probLeft"] <= threshold and tree["probRight"] <= threshold):
            feature = tree["feature"]
            split = tree["split"]
            patterns.add((feature, split))

        traverse(tree["leftChild"], threshold)
        traverse(tree["rightChild"], threshold)

# write the selected nodes to a new patterns file
for graph_file in filter(lambda x: x.endswith('.json'), sorted(os.listdir(os.path.join(forestsPath, dataset)))):
    forests_file = os.path.join(forestsPath, dataset, graph_file)
    print(forests_file)
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
                    print("tree")
                    traverse(tree, th)
                for p in patterns:
                    f_out.write(
                        "{\"patternid\":" + str(counter) + ",\"pattern\":{\"id\":" + str(0) + ",\"feature\":" + str(
                            p[0]) + ",\"split\":" + str(p[1]) + "}}")
                    f_out.write(",\n")
                    counter += 1
                f_out.seek(0, 2)
                f_out.seek(f_out.tell() - 2, 0)
                f_out.truncate()
                f_out.write("]")

def compute_nodes_count(snippets_file):
        
        with open(snippets_file,'r') as f:
            text = f.read()
            nodes_count = text.count('"id"')
        f.close()  
        
        return nodes_count
        
### training of the selected nodes
acc_ref = 0.0
f1_macro_ref = 0.0
roc_auc_ref = 0.0
patterns_count_ref = 0
nodes_count_ref = 0

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

acc_ref = 1.0
f1_macro_ref = 1.0
roc_auc_ref = 1.0
patterns_count_ref = 1.0
nodes_count_ref = 1.0
train_acc = 0.0

for size in forest_sizes:
       for depth in forest_depths:
            for frequency in range(minThreshold, maxThreshold+1):
                 for threshold in edge_thresholds: 
        
                    graph_file = 'RF_'+str(size)+'_'+str(depth)+'_pruned_'+str(threshold)+'.json'
                    snippets_file = os.path.join(stumpsPath, dataset, graph_file)
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
                        end_training = datetime.datetime.now()
                        training_time = (end_training - start_training)
                        print('training result = ' + str(train_acc))
                        print('Training Time for '+ graph_file +' : '+str(training_time))
                        writeToReport(report_file, 'Training Time for '+ graph_file+' , ' + model_type +' : ' +str(training_time))
                        fts_onehot_test = dsf_transform_edges(os.path.join(stumpsPath, dataset, graph_file), X_test, False)
                        start_testing = datetime.datetime.now()
                        Y_pred = learner_model.predict(fts_onehot_test)
                        end_testing = datetime.datetime.now()
                        test_acc = learner_model.score(fts_onehot_test,Y_test)
                        testing_time = (end_testing - start_testing)
                        print('Testing Time for ' + graph_file + ' : ' + str(testing_time))
                        f1_macro = f1_score(Y_pred,Y_test,average = 'macro')
                        f1_micro = f1_score(Y_pred,Y_test,average = 'micro')
                        roc_auc = 0
                        if (threshold == 1):
                            acc_ref = test_acc
                            f1_macro_ref = f1_macro
                            roc_auc_ref = roc_auc
                        print('test accuracy = ' + str(test_acc))
                        print('f1 score macro = ' + str(f1_macro))
                        print('roc auc score = ' + str(roc_auc))
                        writeToReport(report_thresholds,str(size)+','+str(depth)+','+str(frequency)+','+str(threshold)+','+str(learner_model)+','+str(np.round(train_acc,3))+','+str(np.round(test_acc,3))+',' +str(np.round((test_acc - acc_ref),3))+','+str(np.round(f1_macro,3))+',' +str(np.round((f1_macro - f1_macro_ref),3))+','+str(np.round(f1_micro,3))+','+str(np.round(roc_auc,3))+',' +str(np.round((roc_auc - roc_auc_ref),3))+','+str(patterns_count)+','+str(np.round(patterns_ratio,3))+','+str(nodes_count)+','+str(np.round(nodes_ratio,3))+','+str(training_time)+','+str(pruning_time)+','+str(testing_time)+',\n')

                        # cleanup
                        xval_score, learner_model, xval_results = None, None, None

print(filtered_patterns_indexes)            

end_training_total = datetime.datetime.now()
training_total_time = (end_training_total - start_training_total)
print('Total Training Time: '+str(training_total_time))  
writeToReport(report_time_file, 'Total Training Time: '+str(training_total_time) + '\n')

writeToReport(report_file,'\n Best \n')
writeToReport(report_file,'DSF, Model, Accuracy, Deviation,')


