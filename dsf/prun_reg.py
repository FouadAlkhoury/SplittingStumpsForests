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
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import ReadData as ReadData
import cString2json as cString2json
import json2graphNoLeafEdgesWithSplitValues as json2graphNoLeafEdgesWithSplitValues
from fitModels_reg import fitModels
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
dataset = 'temperature'
forest_types = ['RF']
forest_depths = [5, 10, 15]
forest_sizes = [64]
#forest_depths = [5]
#forest_sizes = [32]

maxPatternSize = 1
minThreshold = 1
maxThreshold = 1

edge_thresholds = [1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6]
#edge_thresholds = [0.98]

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

print('\n\nHERE ARE THE ACCURACIES ON TEST DATA OF THE ORIGINAL RANDOM FORESTS\n(don\'t worry, test data is not used for training)\n')

fitModels(roundSplit=True,  dataset = dataset,
              XTrain=X_train, YTrain=Y_train, 
              XTest=X_test, YTest=Y_test, 
              createTest=False, model_dir=os.path.join(forestsPath, dataset), types=forest_types, forest_depths = forest_depths, forest_sizes = forest_sizes)

end_fitting_models = datetime.datetime.now()
fitting_models_time = (end_fitting_models- start_fitting_models)

print('Fitting Models Time: '+str(fitting_models_time))  


        
writeToReport(report_time_file,'Fitting Models Time \t ' + str(fitting_models_time) + '\n')

(report_thresholds,'Forest Size, Forest Depth, Patterns threshold, Edges Threshold, Learner, Train R2, Test R2, Test R2 (gain), MAE, MAE (gain), MSE, MSE (gain), Patterns Count, Patterns Ratio, Nodes Count, Nodes Ratio, Training Time, Pruning Time'+ '\n')
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

'''
def traverse(tree, threshold, pattern, index, suffix):
    global score
    global scoreStr
     
    if ("probLeft" in tree and "probRight" in tree):
        
        traverse(tree["leftChild"], threshold, '', 0, '')
        traverse(tree["rightChild"], threshold, '', 0, '')
        
        if (tree["probLeft"] <= threshold and tree["probRight"] <= threshold ):
            #score = min(tree["probLeft"],tree["probRight"])
            feature = tree["feature"]
            split = tree["split"]
            if (index == 0): 
                pattern = "{\"patternid\":" + str(counter) + ",\"pattern\":{\"id\":" +str(0)+",\"feature\":"+str(feature) + ",\"split\":"+str(split)               
                #scoreStr = str(score)
            if (index == 1):
                pattern += ",\"leftChild\":{\"id\":"+str(0)+",\"feature\":" + str(feature) + ",\"split\":"+str(split)
                #scoreStr += str(score)                                   
            if (index == 2):
                pattern += ",\"rightChild\":{\"id\":"+str(0)+",\"feature\":" +str(feature) + ",\"split\":"+str(split)
                #scoreStr += str(score) 
                
            suffix += '}'
            traverse(tree["leftChild"], threshold, pattern, 1, suffix)  
            traverse(tree["rightChild"], threshold, pattern, 2, suffix)
                   
            
    else:
        pattern += suffix
        if (len(pattern) > 0):
            #patterns_scores.add(pattern+'#'+str(scoreStr)+'\n')
            pattern += "},\n"
            
            patterns.add(pattern)
            


def traversal(root, threshold):
        res = []
        if root:
            res.append(root.data)
            res = res + self.traversal(root.left)
            res = res + self.traversal(root.right)
        return res

    
    
'''
def exist_pattern(pattern, patternsSet):
    
    for p in patternsSet:
        if pattern in p:
            return True
        if p in pattern:
            return True
    return False
    
# Python program to perform iterative preorder traversal

# A binary tree node
class Node:

	# Constructor to create a new node
	def __init__(self, tree):
#		self.tree = self["id"]
		self.left = self["leftChild"]
		self.right = self["rightChild"]
		self.feature = self["feature"]
		self.split = self["split"]
		self.probLeft = self["probLeft"]        
		self.probRight = self["probRight"]                

# An iterative process to print preorder traversal of BT
def iterativePreorder(root):
	global threshold
	# Base CAse
	if root is None:
		return

	# create an empty stack and push root to it
	nodeStack = []
	nodeStack.append(root)

	# Pop all items one by one. Do following for every popped item
	# a) print it
	# b) push its right child
	# c) push its left child
	# Note that right child is pushed first so that left
	# is processed first */
	while(len(nodeStack) > 0):
		
		# Pop the top item from stack and print it
		node = nodeStack.pop()
		if (node.probLeft <= threshold and node.probRight <= threshold ):
			print (node.probLeft, end=" ")
		
		# Push right and left children of the popped node
		# to stack
		if node.right is not None:
			nodeStack.append(node.right)
		if node.left is not None:
			nodeStack.append(node.left)
	
# Driver program to test above function
#root = Node(10)
#root.left = Node(8)
#root.right = Node(2)
#root.left.left = Node(3)
#root.left.right = Node(5)
#root.right.left = Node(2)
#iterativePreorder(root)


def traverse(tree, threshold, index, pattern, suffix):
    global score
    global scoreStr
    #global counter
    #global pattern
    #global suffix
    
    
    #if (index == 0):
        #if ("feature" in tree and "split" in tree):
            
            #feature = tree["feature"]
            #split = tree["split"]
            #pattern = "{\"patternid\":" + str(counter) + ",\"pattern\":{\"id\":" +str(0)+",\"feature\":"+str(feature) + ",\"split\":"+str(split)
    
    if ("probLeft" in tree and "probRight" in tree):
        left = tree["leftChild"]
        right = tree["rightChild"]
        #traverse(tree["leftChild"], threshold, 0)
        # tree = tree_tmp["rightChild"]
        #traverse(tree["rightChild"], threshold, 0)
        
        if (tree["probLeft"] <= threshold and tree["probRight"] <= threshold ):
            #score = min(tree["probLeft"],tree["probRight"])
            feature = tree["feature"]
            split = tree["split"]
            #global pattern
            #global suffix
            if (index == 0): 
                pattern = "{\"patternid\":" + str(counter) + ",\"pattern\":{\"id\":" +str(0)+",\"feature\":"+str(feature) + ",\"split\":"+str(split)               
                #scoreStr = str(score)
            if (index == 1):
                
                pattern += ",\"leftChild\":{\"id\":"+str(0)+",\"feature\":" + str(feature) + ",\"split\":"+str(split)

                #counter+=1
                #scoreStr += str(score)                                   
            if (index == 2):
                pattern += ",\"rightChild\":{\"id\":"+str(0)+",\"feature\":" +str(feature) + ",\"split\":"+str(split)

                #counter-=1
                #scoreStr += str(score) 
                
            suffix += '}'


            #tree_tmp = tree
            #tree = tree["leftChild"]
            #print('')
            traverse(tree["leftChild"], threshold, 1,pattern, suffix)
            #pattern += '}'
            #suffix = ''
            #tree = tree_tmp["rightChild"]
            traverse(tree["rightChild"], threshold, 2,pattern, suffix)
            #cont = False
            #pattern += '}'
            #if (counter == 0):

            #pattern += suffix
            #pattern += "},\n"
            #patterns.add(pattern)
            #pattern = ''
            #suffix = ''
            #index = 0

            '''if ("probLeft" in tree and "probRight" in tree):
                # tree = tree["leftChild"]
                # tree_tmp = tree
                # tree = tree["leftChild"]
                traverse(tree["leftChild"], threshold, 0)
                # tree = tree_tmp["rightChild"]
                traverse(tree["rightChild"], threshold, 0)

            '''

        elif (len(pattern) > 0):
                pattern += suffix
                #patterns_scores.add(pattern+'#'+str(scoreStr)+'\n')
                pattern += "},\n"
                
                #if (not exist_pattern(pattern, patterns)):
            
                patterns.add(pattern)
                pattern = ''
                suffix = ''
                index = 0


        #else:
            #suffix += '}'
            #return
    '''

        
    
    elif (len(pattern) > 0):
                #patterns_scores.add(pattern+'#'+str(scoreStr)+'\n')
                pattern += suffix
                pattern += "},\n"

                patterns.add(pattern)
                pattern = ''
                suffix = ''
                index = 0


            '''
    # if (not exist_pattern(pattern, patterns)):

    #return pattern

def traverseiter(tree, threshold, index, pattern, suffix):
        global score
        global scoreStr
        # global counter
        # global pattern
        # global suffix

        # if (index == 0):
        # if ("feature" in tree and "split" in tree):

        # feature = tree["feature"]
        # split = tree["split"]
        # pattern = "{\"patternid\":" + str(counter) + ",\"pattern\":{\"id\":" +str(0)+",\"feature\":"+str(feature) + ",\"split\":"+str(split)

        while ("probLeft" in tree and "probRight" in tree):
            left = tree["leftChild"]
            right = tree["rightChild"]
            # traverse(tree["leftChild"], threshold, 0)
            # tree = tree_tmp["rightChild"]
            # traverse(tree["rightChild"], threshold, 0)

            if (tree["probLeft"] <= threshold and tree["probRight"] <= threshold):
                # score = min(tree["probLeft"],tree["probRight"])
                feature = tree["feature"]
                split = tree["split"]
                # global pattern
                # global suffix
                if (index == 0):
                    pattern = "{\"patternid\":" + str(counter) + ",\"pattern\":{\"id\":" + str(
                        0) + ",\"feature\":" + str(feature) + ",\"split\":" + str(split)
                    # scoreStr = str(score)
                if (index == 1):
                    pattern += ",\"leftChild\":{\"id\":" + str(0) + ",\"feature\":" + str(
                        feature) + ",\"split\":" + str(split)

                    # counter+=1
                    # scoreStr += str(score)
                if (index == 2):
                    pattern += ",\"rightChild\":{\"id\":" + str(0) + ",\"feature\":" + str(
                        feature) + ",\"split\":" + str(split)

                    # counter-=1
                    # scoreStr += str(score)

                suffix += '}'

                # tree_tmp = tree
                # tree = tree["leftChild"]
                # print('')
                traverse(tree["leftChild"], threshold, 1, pattern, suffix)
                # pattern += '}'
                # suffix = ''
                # tree = tree_tmp["rightChild"]
                traverse(tree["rightChild"], threshold, 2, pattern, suffix)
                # pattern += '}'
                # if (counter == 0):

                # pattern += suffix
                # pattern += "},\n"
                # patterns.add(pattern)
                # pattern = ''
                # suffix = ''
                # index = 0

                '''if ("probLeft" in tree and "probRight" in tree):
                    # tree = tree["leftChild"]
                    # tree_tmp = tree
                    # tree = tree["leftChild"]
                    traverse(tree["leftChild"], threshold, 0)
                    # tree = tree_tmp["rightChild"]
                    traverse(tree["rightChild"], threshold, 0)

                '''

                if (len(pattern) > 0):
                    pattern += suffix
                    # patterns_scores.add(pattern+'#'+str(scoreStr)+'\n')
                    pattern += "},\n"

                    # if (not exist_pattern(pattern, patterns)):

                    patterns.add(pattern)
                    pattern = ''
                    suffix = ''
                    index = 0


'''
def extract_subtrees(node):
    subtrees = []
    if ("probLeft" in node and "probRight" in node):
        if node['probLeft'] < 0.9:
            subtrees.append(node)
        if 'leftChild' in node:
            subtrees += extract_subtrees(node['leftChild'])
        if 'rightChild' in node:
            subtrees += extract_subtrees(node['rightChild'])
        return subtrees


def extract_subtrees(node):
    if "probLeft" in node and "probRight" in node and node["probLeft"] < 0.8 and node["probRight"] < 0.8:
        return [node]
    elif "leftChild" in node or "rightChild" in node:
        subtrees = []
        if "leftChild" in node:
            subtrees += extract_subtrees(node["leftChild"])
        if "rightChild" in node:
            subtrees += extract_subtrees(node["rightChild"])
        return subtrees
    return []



def extract_subtrees(node):
    subtrees = []
    if "leftChild" in node and "rightChild" in node:
        if node["probLeft"] < 0.7 and node["probRight"] < 0.7:
            left_subtrees = extract_subtrees(node["leftChild"])
            right_subtrees = extract_subtrees(node["rightChild"])
            subtrees = [{**node, "leftChild": t, "rightChild": {} } for t in left_subtrees] + [{**node, "leftChild": {}, "rightChild": t} for t in right_subtrees]
        else:
            if node["probLeft"] < 0.7:
                left_subtrees = extract_subtrees(node["leftChild"])
                subtrees = [{**node, "leftChild": t, "rightChild": {} } for t in left_subtrees]
            if node["probRight"] < 0.7:
                right_subtrees = extract_subtrees(node["rightChild"])
                subtrees = [{**node, "leftChild": {}, "rightChild": t} for t in right_subtrees]
    elif "prediction" in node:
        subtrees = [node]
    return subtrees
'''

def maximal_patterns(tree, threshold):
    if not tree:
        return []

    if tree["leftChild"] is None and tree["rightChild"] is None:
        return [{"patternid": 0, "pattern": tree}]

    left_patterns = maximal_patterns(tree["leftChild"], threshold)
    right_patterns = maximal_patterns(tree["rightChild"], threshold)

    patterns = []
    if tree["probLeft"] < threshold and tree["probRight"] < threshold:
        patterns.append({"patternid": 0, "pattern": tree})
    else:
        patterns.extend(left_patterns)
        patterns.extend(right_patterns)

    return patterns

#input = [{"probLeft":0.6533164012894402,"probRight":0.34668359871055976,"feature":63,"split":43,"leftChild": {"...}]}]
#threshold = 0.5
def get_pattern(probLeft, probRight, feature, split, leftChild, rightChild):
    if probLeft < 0.5 and probRight < 0.5:
        left_child_pattern = None
        right_child_pattern = None
        if leftChild:
            left_child_pattern = get_pattern(**leftChild)
        if rightChild:
            right_child_pattern = get_pattern(**rightChild)

        if left_child_pattern and right_child_pattern:
            return {
                'id': 0,
                'feature': feature,
                'split': split,
                'leftChild': left_child_pattern,
                'rightChild': right_child_pattern
            }
        elif left_child_pattern:
            return left_child_pattern
        elif right_child_pattern:
            return right_child_pattern
        else:
            return {
                'id': 0,
                'feature': feature,
                'split': split
            }
    return None

def extract_patterns(input_tree):
    root = input_tree
    pattern = get_pattern(**root)
    return [pattern] if pattern else []

writeToReport(report_pruning_file,'Forest Size, Forest Depth, Pruning threshold, Pruning Time'+ '\n')
#for graph_file in filter(lambda x: x.endswith('.json'), sorted(os.listdir(os.path.join(forestsPath, dataset)))):
def is_valid(tree, threshold, index):
    global subtreeStr
    #global subtrees
    if ("probLeft" in tree and "leftChild" in tree):
        #print(tree["probLeft"])

        if (tree["probLeft"] <= threshold and tree["probRight"] <= threshold):

            #del tree["numSamples"]
            #del tree["isCategorical"]
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

            #subtrees.remove(tree)
            #print('Removed')
            #print(tree)

            #tree["id"] = -1
            #del tree["numSamples"]
            #del tree["probLeft"]
            #del tree["probRight"]
            #del tree["isCategorical"]
            #del tree["feature"]
            #del tree["split"]
            treeLeft = tree["leftChild"]
            treeRight = tree["rightChild"]
            #print(treeLeft)
            #print(treeRight)
            subtrees.append(treeLeft)
            subtrees.append(treeRight)

            #index = subtrees.index(tree)
            #subtrees[index] = ""
            #print('removed index: '+str(index))
            #tree = "new"
            #print(subtrees[index])
            #if("numSamples" in tree):
                #del tree["numSamples"]
            subtreeStr = subtreeStr.replace(str(tree),'')
            #tree = ""

            #subtrees[index] = None
            #print(subtree)

    else:

        subtreeStr = subtreeStr.replace(str(tree), '')

    if (tree is not None and "prediction" in tree):

        subtreeStr = subtreeStr.replace(str(tree), '')
        #subtrees[index] = None
        #tree = ""



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
                    print('Now')
                    print(jsontree)
                    # remove_a_key(jsontree, 'id')
                    remove_a_key(jsontree, 'numSamples')
                    remove_a_key(jsontree, 'isCategorical')
                    remove_a_key(jsontree, 'prediction')
                    # remove_a_key(jsontree, 'probLeft')
                    # remove_a_key(jsontree, 'probRight')
                    # remove_a_child_key(jsontree)

                    # remove_a_child_key(jsontree, 'rightChild')
                    # remove_a_key(jsontree, 'rightChild')
                    # remove_a_key(jsontree, 'rightChild')
                    # remove a node if it has only one child (left: id)
                    subtrees.append(jsontree)
                for index, subtree in enumerate(subtrees):
                    # print(index)
                    # print(subtree)
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
                    # jsontree = json.loads(subtreeStr)
                    # remove_a_key(jsontree, 'leftChild')
                    # remove_a_key(jsontree, 'rightChild')
                    # subtreeStr = str(jsontree)
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
r2_ref = 0.0
mae_ref = 0.0
mse_ref = 0.0
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
    writeToReport(report_file, str(model_name) + '\t' + str(descriptor) + '\t' + str(dsf_score) + ' +- ' + str(dsf_std))

    model.fit(fts_onehot, Y_train)
    Y_pred = model.predict(fts_onehot)
    r2_train = r2_score(Y_train, Y_pred)
    # train_acc = model.score(fts_onehot,Y_train)
    return r2_train, model



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
                       
                        r2_train, learner_model = train_model_on_top(model_class(**learners_parameters[model_type]), fts_onehot, Y_train, scoring_function, model_type, graph_file)
                        #results_list.append((xval_score, model_type, graph_file, learner_model, xval_results))
                            
                        end_training = datetime.datetime.now()
                        training_time = (end_training - start_training)
                        print('training result = ' + str(r2_train))
                        print('Training Time for '+ graph_file +' : '+str(training_time))
                       
                        writeToReport(report_file, 'Training Time for '+ graph_file+' , ' + model_type +' : ' +str(training_time)) 

                       
                        fts_onehot_test = dsf_transform_edges(os.path.join(snippetsPath, dataset, graph_file), X_test, False)
                       
                        test_acc = learner_model.score(fts_onehot_test,Y_test)
                        Y_pred = learner_model.predict(fts_onehot_test)
                        #Y_test = np.argmax(Y_test, axis=0)
                        r2 = r2_score(Y_test, Y_pred)
                        mae = mean_absolute_error(Y_test, Y_pred)
                        mse = mean_squared_error(Y_test, Y_pred)

                        if (threshold == 1):
                            r2_ref = r2
                            mae_ref = mae
                            mse_ref = mse

                        # test_acc = accuracy_score(Y_test, pred_test)
                        print('r2 = ' + str(r2))
                        print('mae = ' + str(mae))
                        print('mse = ' + str(mse))

                        writeToReport(report_thresholds,str(size)+','+str(depth)+','+str(frequency)+','+str(threshold)+','+str(learner_model)+','+str(np.round(r2_train,3))+','+str(np.round(r2,3))+',' +str(np.round((r2 - r2_ref),3))+','+str(np.round(mae,3))+',' +str(np.round((mae - mae_ref),3))+','+str(np.round(mse,3))+',' +str(np.round((mse - mse_ref),3))+','+str(patterns_count)+','+str(np.round(patterns_ratio,3))+','+str(nodes_count)+','+str(np.round(nodes_ratio,3))+','+str(training_time)+','+str(pruning_time)+',\n')
                        
#writeToReport(report_thresholds,str(size)+','+str(depth)+','+str(frequency)+','+str(threshold)+','+str(learner_model)+','+str(np.round(train_acc,3))+','+str(np.round(test_acc,3))+',' +str(np.round((test_acc - acc_ref),3))+','+str(np.round(f1_macro,3))+',' +str(np.round((f1_macro - f1_macro_ref),3))+','+str(np.round(f1_micro,3))+','+str(np.round(roc_auc,3))+',' +str(np.round((roc_auc - roc_auc_ref),3))+','+str(patterns_count)+','+str(np.round(patterns_ratio,3))+','+str(training_time)+','+str(pruning_time)+',\n')                        

                        # cleanup
                        xval_score, learner_model, xval_results = None, None, None
                
                        #writeToReport(report_file, str(best_result[2]) + ',' + str(model_type) + '
                        #print(str(learner_model) + ' ' +str(model_type) + ' ' + str(np.round(test_acc,3)) + ' ' +str(fts_onehot.shape[1]) +', ' + str(np.round(best_result[0],3)) + ',' + str(np.round(test_acc,3)) + ',' + str(fts_onehot.shape[1]))
            
print(filtered_patterns_indexes)            

end_training_total = datetime.datetime.now()
training_total_time = (end_training_total - start_training_total)
print('Total Training Time: '+str(training_total_time))  
writeToReport(report_time_file, 'Total Training Time: '+str(training_total_time) + '\n')

# %% Find, for each learner, the best decision snippet features on training data (the proposed model) and evaluate its performance on test data

writeToReport(report_file,'\n Best \n')
writeToReport(report_file,'DSF, Model, Accuracy, Deviation,')

       
