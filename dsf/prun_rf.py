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

dataPath = "../data/"
forestsPath = "../tmp/forests/"
snippetsPath = "../tmp/snippets/"
resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports/"

dataset = sys.argv[1]

edge_thresholds = [1.0,0.975,0.95,0.925,0.9,0.85,0.8,0.7,0.6]

counter = 0
def traverse(tree, threshold):
     
    if ("probLeft" in tree and "probRight" in tree):
        
        if (tree["probLeft"] <= threshold and tree["probRight"] <= threshold ):
            
            feature = tree["feature"]
            split = tree["split"]
            patterns.add((feature,split)) 
            traverse(tree["leftChild"], threshold)
            traverse(tree["rightChild"], threshold)
        
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

                pruned_file = graph_file[:-5] + '_pruned_' +str(th)+ '.json'    
                snippets_file = os.path.join(snippetsPath, dataset, pruned_file)    
                with open(snippets_file,'w') as f_out:
                    f_out.write("[")
                    for tree in trees:
                        print("tree")

                        traverse(tree, th)
                    for p in patterns:
                        f_out.write("{\"patternid\":" + str(counter) + ",\"pattern\":{\"id\":" +str(0)+",\"feature\":"+str(p[0]) + ",\"split\":"+str(p[1]) + "}}")
                        f_out.write(",\n")
                        counter += 1
                    #f_out.write("\n")    
                    f_out.seek(0,2)
                    f_out.seek(f_out.tell() - 2, 0)
                    f_out.truncate()    
                    f_out.write("]")    
                        
            

      

    
    
