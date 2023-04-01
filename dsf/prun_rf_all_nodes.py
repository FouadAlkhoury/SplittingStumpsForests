import numpy as np # helps with the math
import matplotlib
import matplotlib.pyplot as plt # to plot error during training
import ReadData as ReadData
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import datetime
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import cross_val_score
from datetime import datetime
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import DecisionSnippetFeatures as DecisionSnippetFeatures
import datetime
from util import writeToReport


dataPath = "../data/"
forestsPath = "../tmp/forests/"
snippetsPath = "../tmp/snippets_pruned_patterns/"
scoresPath = "../tmp/snippets_pruned_patterns_scores/"
resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports_pruned_patterns/"
support = 100
dataset = sys.argv[1]

forest_depths = [5,10]
forest_sizes = [16,32,64,128,256]

#forest_depths = [5,10]
#forest_sizes = [16,32]

#edge_thresholds = [0.7,0.6]
edge_thresholds = [1.0,0.975,0.95,0.925,0.9,0.85,0.8,0.7,0.6]
counter = 0
pruning_time = datetime.timedelta()
report_pruning_dir = reportsPath+'/'+dataset 
report_pruning_file = report_pruning_dir + '/report_pruning_time.txt'
score = 0
scoreStr = ''

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
            

writeToReport(report_pruning_file,'Forest Size, Forest Depth, Pruning threshold, Pruning Time'+ '\n')            
#for graph_file in filter(lambda x: x.endswith('.json'), sorted(os.listdir(os.path.join(forestsPath, dataset)))):
for size in forest_sizes:
    for depth in forest_depths:
            
        graph_file = 'RF_'+str(size)+'_'+str(depth)+'.json'    
        forests_file = os.path.join(forestsPath, dataset, graph_file)
        print(forests_file)
        with open(forests_file, 'r') as f_decision_forests:
        
            trees = json.load(f_decision_forests)
            
            
            
            for th in edge_thresholds:
                
                start_pruning_time = datetime.datetime.now()

                features = []
                splits = []
                patterns = set()
                patterns_scores = set()
                counter = 0
                score = 0
                scoreStr = ''

                pruned_file = graph_file[:-5] + '_pruned_' +str(th)+ '.json'    
                snippets_file = os.path.join(snippetsPath, dataset, pruned_file)  
                scores_file = os.path.join(scoresPath, dataset, pruned_file)
                with open(snippets_file,'w') as f_out:
                    f_out.write("[")
                    for tree in trees:
                        print("tree")
                        pattern = ''
                        suffix = ''

                        traverse(tree, th,pattern, 0, suffix)
                    for p in patterns:
                        f_out.write(p)
                        
                     
                        counter += 1
                     
                    f_out.seek(0,2)
                    f_out.seek(f_out.tell() - 2, 0)
                    f_out.truncate()    
                    f_out.write("]") 
                    
                    end_pruning_time = datetime.datetime.now()
                    pruning_time = (end_pruning_time - start_pruning_time)
                    writeToReport(report_pruning_file, str(size)+ ', ' + str(depth) + ', ' + str(th) + ', ' + str(pruning_time) + '\n')
                '''with open(scores_file,'w') as f_out:
                    f_out.write("[")
                    for tree in trees:
                        print("tree")
                        pattern = ''
                        suffix = ''

                        
                    for score in patterns_scores:
                        f_out.write(score)
                        
                     
                        counter += 1
                     
                    f_out.seek(0,2)
                    f_out.seek(f_out.tell() - 2, 0)
                    f_out.truncate()    
                    f_out.write("]")         
            '''

'''      
def traverse(tree, threshold, pattern, index):
     
    if ("probLeft" in tree and "probRight" in tree):
        
        traverse(tree["leftChild"], threshold, '', 0)
        traverse(tree["rightChild"], threshold, '', 0)
        
        if (tree["probLeft"] <= threshold and tree["probRight"] <= threshold ):            
            feature = tree["feature"]
            split = tree["split"]
            if (index == 0):
                
                pattern = "{\"patternid\":" + str(counter) + ",\"pattern\":{\"id\":" +str(0)+",\"feature\":"+str(feature) + ",\"split\":"+str(split) 
                
                if ("probLeft" in tree["leftChild"] and "probRight" in tree["leftChild"]):
                    
                    if (tree["leftChild"]["probLeft"] <= threshold and tree["leftChild"]["probRight"] <= threshold ):
                        traverse(tree["leftChild"], threshold, pattern, 1)
                        pattern += "}"
                
                if ("probLeft" in tree["rightChild"] and "probRight" in tree["rightChild"]):
                    if (tree["rightChild"]["probLeft"] <= threshold and tree["rightChild"]["probRight"] <= threshold ):
                        traverse(tree["rightChild"], threshold, pattern, 2)
                        pattern += "}"
                
            if (index == 1):
                pattern += ",\"leftChild\":{\"id\":"+str(0)+",\"feature\":" + str(feature) + ",\"split\":"+str(split)
                suffix = "}"
                
                if ("probLeft" in tree["leftChild"] and "probRight" in tree["leftChild"]):
                    if (tree["leftChild"]["probLeft"] <= threshold and tree["leftChild"]["probRight"] <= threshold ):
                        traverse(tree["leftChild"], threshold, pattern, 1)
                        pattern += "}"
                       
                if ("probLeft" in tree["rightChild"] and "probRight" in tree["rightChild"]):        
                    if (tree["rightChild"]["probLeft"] <= threshold and tree["rightChild"]["probRight"] <= threshold ):
                        traverse(tree["rightChild"], threshold, pattern, 2) 
                        pattern += "}"
                #pattern += "}"
                
                
            if (index == 2):
                pattern += ",\"rightChild\":{\"id\":"+str(0)+",\"feature\":" +str(feature) + ",\"split\":"+str(split)
                suffix = "}"
                
                if ("probLeft" in tree["leftChild"] and "probRight" in tree["leftChild"]):
                    if (tree["leftChild"]["probLeft"] <= threshold and tree["leftChild"]["probRight"] <= threshold ):
                        traverse(tree["leftChild"], threshold, pattern, 1)
                        pattern += "}"
                
                if ("probLeft" in tree["rightChild"] and "probRight" in tree["rightChild"]): 
                    if (tree["rightChild"]["probLeft"] <= threshold and tree["rightChild"]["probRight"] <= threshold ):
                        traverse(tree["rightChild"], threshold, pattern, 2)
                        pattern += "}"
                #pattern += "}"
            #patterns.add((feature,split,index)) 
            #leftIndex = index + '0'
            #rightIndex = index + '1'

            
        if (len(pattern) > 0):
            pattern += "}},\n"
            
            patterns.add(pattern)
            
        
        pattern += "}"
    
    
'''