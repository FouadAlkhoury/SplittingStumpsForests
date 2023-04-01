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

dataPath = "../data/"
forestsPath = "../tmp/forests/"
snippetsPath = "../tmp/snippets/"
resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports/"

dataset = sys.argv[1]
patterns_count = int(sys.argv[2])
#dataset = 'magic'
#patterns_count = 136
#dataset = 'magic'
print(dataset)

forest_types = ['RF']
forest_depths = [5]
sigma_values = [0.3]
forest_size = 25

maxPatternSize = 4

scoring_function = 'accuracy'

# learners that are to be used on top of Decision Snippet Features
learners = {'NB': MultinomialNB,
            'SVM': LinearSVC}
#            'LR': LogisticRegression}

#learners = {'NB': MultinomialNB}

learners_parameters = {'NB': {},
                       'SVM': {'max_iter': 10000}}
#                       'LR': {'max_iter': 5000}}

#learners_parameters = {'NB': {}}

run_fit_models = False
run_mining = False
run_training = True
run_eval = False

verbose = True

#fitting_models_time = datetime.timedelta()

X_train, Y_train = ReadData.readData(dataset, 'train', dataPath)
X_test, Y_test = ReadData.readData(dataset, 'test', dataPath)
X_train_org = X_train

splits = []
for graph_file in filter(lambda x: x.endswith('.json'), sorted(os.listdir(os.path.join(snippetsPath, dataset)))):
        snippets_file = os.path.join(snippetsPath, dataset, graph_file)
        with open(snippets_file, 'r') as f_decision_snippets:
        
            frequentpatterns = json.load(f_decision_snippets)
            #print(len(frequentpatterns))
            #print(frequentpatterns[0])
            pattern = frequentpatterns[0]["pattern"]
            feature = pattern["feature"]
            split = pattern["split"]
            

for pattern in frequentpatterns:
    splits.append(pattern["pattern"]["split"])

for pattern in frequentpatterns:
    #pattern = frequentpatterns[0]["pattern"]
    feature = pattern["pattern"]["feature"]
    split = pattern["pattern"]["split"]
    print(feature)
    print(split)
    #splits.append(pattern["pattern"]["split"])
    
splits = np.array(splits).reshape(len(splits),1).astype(dtype = np.float64)
m = splits.mean()
std = splits.std()

def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
extended_x_train = []
extended_x_test = []
for x in X_train:
    for p in frequentpatterns:
        extended_x_train.append(x[p["pattern"]["feature"]])
        
for x in X_test:
    for p in frequentpatterns:
        extended_x_test.append(x[p["pattern"]["feature"]])        
'''
feature_class = []
counter_left = 0
counter_left_y1 = 0
counter_right = 0
counter_right_y1 = 0

for pattern in frequentpatterns:
    for i,x in enumerate(X_train_org):
        if (x[pattern["pattern"]["feature"]] < pattern["pattern"]["split"]):
            counter_left += 1
            if (Y_train[i] == 1):
                counter_left_y1 += 1
                
        else:
            counter_right += 1
            if (Y_train[i] == 1):
                counter_right_y1 += 1
        
    
    if (counter_left != 0):
        feature_class.append(0.5)
        #feature_class.append(counter_left_y1/counter_left)
    else:
        feature_class.append(0)
    #if (counter_right != 0):    
    #    feature_class.append(counter_right_y1/counter_right)
    #else:
    #    feature_class.append(0)    
            
    
    print(counter_left)
    print(counter_left_y1)
    print(counter_right)
    print(counter_right_y1)
    
    counter_left = 0  
    counter_left_y1 = 0        
    counter_right = 0
    counter_right_y1 = 0
            
    
    #pattern = frequentpatterns[0]["pattern"]
    feature = pattern["pattern"]["feature"]
    split = pattern["pattern"]["split"]
    #print(feature)
    #print(split)
    #splits.append(pattern["pattern"]["split"])
print(feature_class)    

     
feature_class = []
counter = 0
for pattern in frequentpatterns:
    for i,x in enumerate(X_train_org):
        if ((x[pattern["pattern"]["feature"]] < pattern["pattern"]["split"] and Y_train[i] == 1) or 
            x[pattern["pattern"]["feature"]] >= pattern["pattern"]["split"] and Y_train[i] == 0):
            counter += 1
        else:
            counter -= 1
    #print(counter)
    if (counter > 0):
        feature_class.append(0.5)
        feature_class.append(0.5)
    else:
        feature_class.append(-0.5)
        feature_class.append(-0.5)
    counter = 0      
        
    #pattern = frequentpatterns[0]["pattern"]
    feature = pattern["pattern"]["feature"]
    split = pattern["pattern"]["split"]
    #print(feature)
    #print(split)
    #splits.append(pattern["pattern"]["split"])
print(feature_class)    

       

feature_class = []
counter_left = 0
counter_right = 0
for pattern in frequentpatterns:
    for i,x in enumerate(X_train_org):
        if ((x[pattern["pattern"]["feature"]] < pattern["pattern"]["split"] and Y_train[i] == 1) or 
            x[pattern["pattern"]["feature"]] >= pattern["pattern"]["split"] and Y_train[i] == 0):
            counter_left += 1
        else:
            counter_right += 1        

    print(counter_left)
    if (counter_left > counter_right):
        feature_class.append(counter_left/len(X_train_org))
    else:
        feature_class.append(-counter_right/len(X_train_org))
    counter_left = 0      
    counter_right = 0
    
    #pattern = frequentpatterns[0]["pattern"]
    feature = pattern["pattern"]["feature"]
    split = pattern["pattern"]["split"]
    #print(feature)
    #print(split)
    #splits.append(pattern["pattern"]["split"])
print(feature_class)    
'''
feature_class = []
counter_left = 0
counter_left_y1 = 0
counter_right = 0
counter_right_y1 = 0

for pattern in frequentpatterns:
    for i,x in enumerate(X_train_org):
        if (x[pattern["pattern"]["feature"]] < pattern["pattern"]["split"]):
            counter_left += 1
            if (Y_train[i] == 1):
                counter_left_y1 += 1
                
        else:
            counter_right += 1
            if (Y_train[i] == 1):
                counter_right_y1 += 1
        
    
    if (counter_left != 0):
        #feature_class.append(0.5)
        feature_class.append(counter_left_y1/counter_left)
    else:
        feature_class.append(0)
    if (counter_right != 0): 
        #feature_class.append(-0.5)
        feature_class.append(-counter_right_y1/counter_right)
    else:
        feature_class.append(0)    
            
    #feature_class.append(0.5)
    #feature_class.append(-0.5)
    print(counter_left)
    print(counter_left_y1)
    print(counter_right)
    print(counter_right_y1)
    
    counter_left = 0  
    counter_left_y1 = 0        
    counter_right = 0
    counter_right_y1 = 0
            
    
    #pattern = frequentpatterns[0]["pattern"]
    feature = pattern["pattern"]["feature"]
    split = pattern["pattern"]["split"]
    #print(feature)
    #print(split)
    #splits.append(pattern["pattern"]["split"])
print(feature_class)    
       

feature_class = np.array(feature_class)
#feature_class = np.ones((1,patterns_count*2))
feature_class = torch.tensor(feature_class.reshape(patterns_count*2,1),dtype = torch.float)
print(feature_class)

extended_x_train = np.array(extended_x_train)
extended_x_train = extended_x_train.reshape(len(X_train),len(frequentpatterns))

extended_x_test = np.array(extended_x_test)
extended_x_test = extended_x_test.reshape(len(X_test),len(frequentpatterns))

inputs_train = extended_x_train
inputs_test = extended_x_test
weights = splits

outputs_train = Y_train
outputs_test = Y_test

new_x_train = sigmoid(np.subtract(inputs_train , np.transpose(splits)))
new_x_test = sigmoid(np.subtract(inputs_test , np.transpose(splits)))

outputs_train = outputs_train.reshape(len(outputs_train),1)
outputs_test = outputs_test.reshape(len(outputs_test),1)

#splits = np.ones((1,patterns_count))
#splits = np.random.normal(0,10,(1,patterns_count))
weights = torch.tensor(splits.reshape(1,patterns_count),dtype = torch.float)
print(weights)

class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        #self.inputSize = 136
        #self.outputSize = 1
        #self.hiddenSize = 136
        
        #self.W1 = Variable(torch.tensor(np.ones((patterns_count)),dtype = torch.float),requires_grad = True)
        
        self.bias = Variable(torch.tensor(weights),requires_grad = True)
        #self.X2 = Variable(torch.tensor(weights),requires_grad = True)
        #splits = np.ones((1,patterns_count))
        #self.W2 = Variable(torch.tensor(torch.randn(patterns_count,1)),requires_grad = True)
        self.W2 = Variable(torch.tensor(feature_class),requires_grad = True)
        
        
        
        #self.W2 = Variable(torch.tensor(np.ones((patterns_count*2,1)),dtype = torch.float),requires_grad = True)
        print(self.W2)
        
    def forward(self, X):
        
       
        X =  X.subtract(self.bias)
       
        #print('X-S: ' + str(X))
        X = self.sigmoid(X) # activation function
        
        
        #print('sigmoid x-s: ' + str(X))
        #self.X2 = 1 - X
        #print(type(X))
        X = torch.stack((X,1 - X), 2).reshape(len(X),patterns_count*2)
        #print('new X: ' + str(X))
        
        #X = X.reshape(14999,patterns_count * 2)
        #X = torch.tensor(X.reshape(len(X),patterns_count*2),dtype = torch.float,requires_grad = True)
        #print('new X: ' + str(X))
        
        
        #print('sigmoid(x-s): ' + str(X))
        X = torch.matmul(X, self.W2)
        #print(X)
        #print('WX: ' + str(X))
        #o = F.softmax(X,dim=-1)
        #if (X > 0):
        #    o = 1
        #else: 
        #    o = 0
        o = self.sigmoid(X) # final activation function
        #print(o)
        #o = m(X)
        #print('output: ' + str(X))
        return o
        
    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
    
    def sigmoidPrime(self, s):
        return s * (1 - s)
    
    
        
    def saveWeights(self, model):
        torch.save(model, "NN")
        
    def predict(self):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forward(xPredicted)))
        

def train(X, y):
        
        
        optimizer = torch.optim.SGD([{'params':[NN.bias,NN.W2], 'lr':0.01}])
        optimizer.zero_grad()
        loss = criterion(NN(X),y)
        train_loss = loss.item()
        #print('train loss: ' + str(train_loss))
        NN.bias.retain_grad()
        NN.W2.retain_grad()
        loss.backward()
        optimizer.step()
        return train_loss
        
def evaluation(X, y):
        
        #criterion = nn.MSELoss().cuda()
        loss = criterion(NN(X),y)
        valid_loss = loss.item()
        #log('valid_loss', valid_loss)
        #print('eval loss: ' + str(valid_loss))
        return valid_loss
        
X_train = torch.tensor((inputs_train), dtype=torch.float) # 3 X 2 tensor
Y_train = torch.tensor((outputs_train), dtype=torch.float) # 3 X 1 tensor

X_test = torch.tensor((inputs_test), dtype=torch.float) # 3 X 2 tensor
Y_test = torch.tensor((outputs_test), dtype=torch.float) # 3 X 1 tensor

print(X_train.size())
print(Y_train.size())

print(X_test.size())
print(Y_test.size())

ex_train = torch.tensor(extended_x_train)
ex_test = torch.tensor(extended_x_test)

m = X_train.mean(dim = 0)
std = X_train.std(dim = 0)
X_train = (ex_train - m)/std
X_train = torch.tensor((X_train), dtype=torch.float)

m = X_test.mean(dim = 0)
std = X_test.std(dim = 0)
X_test = (ex_test - m)/std
X_test = torch.tensor((X_test), dtype=torch.float)

#X,y = random_split()

weights = (weights - m)/std
print(weights)

NN = Neural_Network()
loss_list = [] 
#criterion = nn.MSELoss().cuda()
#class_count = [1,1,1,1,1,1,1,1]
#class_weights = 1./torch.tensor(class_count, dtype=torch.float)

criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss().cuda()

valid_loss = 0.0
min_valid_loss = np.inf
# separate the split from the other weight 
# literature (soft decision tree)
# classification loss
# accuracy NN vs original DT
# compare converging speed
now = datetime.now()
results_file = os.path.join(resultsPath, dataset, 'train_rf_splits_' + str(now) + '.csv')            

    
results_list = []    
max_acc = 0.0
patience = 4000
count_patience = 0

for i in range(30000): 
    if(i % 100 == 0):
        print ("#" + str(i) + " Loss: " + str(torch.mean((Y_train - NN(X_train))**2).detach().item()))  # mean sum squared loss
    loss_list.append('iteration : ' + str(i) + ': ' + str(torch.mean((Y_train - NN(X_train))**2).detach().item()))
    
    
    
    train_loss = train(X_train, Y_train)
    
    #NN.eval()
    valid_loss = evaluation(X_test,Y_test)
    
    Y_pred = (NN(X_test) > 0.5).float()
    #print(Y_test)
    #print(Y_pred)
    accuracy = (Y_test == Y_pred).sum().item()/len(X_test)
    results_list.append(str(i+1) + ',' + str(train_loss)+','+str(valid_loss)+','+str(accuracy))
    if (i % 100 == 0):
        print('Epoch: ' + str(i))
        print('train loss: ' + str(train_loss))
        print('eval loss: ' + str(valid_loss))
        print('Accuracy: ' + str(accuracy))
    
    #print(valid_loss)
    '''
    if (min_valid_loss > valid_loss):
        min_valid_loss = valid_loss
        print('decreased')
    else: 
        print(str(i) + ' No')
        break
    '''
    if (max_acc < accuracy):
        max_acc = accuracy
        count_patience = 0
        #print('increased')
    else: 
        if (count_patience < patience):
            count_patience += 1
            #print(str(i) + ' No')
        else:
            break
        
    
        
    #print('bias grad: ' + str(NN.bias.grad))
NN.saveWeights(NN)
#for l in loss_list:
#    print(l)

with open(results_file, 'w') as fout:
    fout.write('Epoch, Train Loss, Valid Loss, Accuracy \n')
    for e in results_list:
        fout.write(e + '\n')
    
    
#results_file = os.path.join(reportsPath, dataset, 'loss_' + graph_file + '.txt')            
#with open(results_file, 'w') as fout:
#    for l in loss_list:
#        fout.write(l + '\n')

new_weights = (NN.bias * std) + m
print(new_weights)
print(splits)



'''
fts_onehot_nb_cv_score = cross_val_score(NN, X_test, Y_test, cv=5, scoring='accuracy')
NN.fit(X_test, Y_test)
dsf_score = fts_onehot_nb_cv_score.mean()
dsf_std = fts_onehot_nb_cv_score.std()
print(str(dsf_score) + ' +- ' + str(dsf_std))

    

'''
#new_frequentpatterns = frequentpatterns

snippets_file = os.path.join(snippetsPath, dataset, 'trained_splits_'+ str(now) +'_'+ graph_file )            
with open(snippets_file, 'w') as f:
    f.write('[')

    for i,pattern in enumerate(frequentpatterns):
        #splits.append(pattern["pattern"]["split"])
        pattern["pattern"]["split"] = float("{:.2f}".format(new_weights[0][i].item()))
        print(pattern["pattern"]["split"])
        print(pattern)
        
    
    
        f.write(str(pattern).replace('\'','\"')+',\n')
        
    f.write(']')  
    