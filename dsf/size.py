
# set dataset name and run.
# The output results file and plot are exported to the folder SizeComparison/dataset/  
dataset = "satlog"


import csv
import matplotlib.pyplot as plt
import numpy as np

variant = "NoLeafEdges"
scoring_function = 'accuracy'
pattern_max_size = 6
filesPath = "../tmp/snippets"
filesPath_RF = "../tmp/forests"
sizePath = "../tmp/nodesCount"
nodes_count_list = []
for rf_depth in (5,10,15,20):
    for frequency in range(2,26):
      
     with open(filesPath+'/'+dataset+'/'+'RF_'+str(rf_depth)+'_t'+str(frequency)+'.json') as json_file:
         line = json_file.read()
         list = [i for i in range(len(line)) if line.startswith('"id"', i)]
         
         nodes_count_list.append('RF_'+str(rf_depth)+'_t'+str(frequency)+','+str(len(list))+',\n')
                                                
json_file.close()

f= open(sizePath+'/'+dataset+'/'+'nodesCount_'+dataset+'.csv',"w")
f.write('RF,nodes Count,\n')
for line in nodes_count_list:
        f.write(line)
f.close()


nodes_count_list = []
for rf_depth in (5,10,15,20):
    
      
     with open(filesPath_RF+'/'+dataset+'RF_'+str(rf_depth)+'.json') as json_file:
         line = json_file.read()
         list = [i for i in range(len(line)) if line.startswith('"id"', i)]
         
         nodes_count_list.append('RF_'+str(rf_depth)+','+str(len(list))+',\n')
                
        
          
            
json_file.close()

f= open(sizePath+'/'+dataSet+'/'+'nodesCount_'+dataset+'_RandomForestClassifier'+'.csv',"w")
f.write('RF,nodes Count,\n')
for line in nodes_count_list:
        f.write(line)
f.close()





accuracy_list = []
accuracy_list_rf = []
size_list = []
size_list_rf = []
rf_list = []


with open(sizePath+'/'+dataSet+'/'+'nodesCount_'+dataset+'.csv') as size_file:
        size_reader = csv.reader(size_file, delimiter='\n')
        line_count = 1
        for row in size_reader:
            if (line_count > 1 ):
                
                rowStr = str(row).split(',')                                       
                size_list.append(rowStr[1])
            line_count +=1    

size_file.close()

with open(sizePath+'/'+dataSet+'/'+'nodesCount_'+dataset+'_RandomForestClassifier'+'.csv') as size_file:
        size_reader = csv.reader(size_file, delimiter='\n')
        line_count = 1
        for row in size_reader:
            if (line_count > 1 ):
                
                rowStr = str(row).split(',')                                       
                size_list_rf.append(rowStr[1])
            line_count +=1    

size_file.close()

'''
for rf_depth in (5,10,15,20):
    
    with open(filesPath+'/'+dataset+'/Results_'+variant+'/leq'+str(pattern_max_size)+'/'+'RF_'+str(rf_depth)+'_'+scoring_function+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        line_count = 2
        for row in csv_reader:
            rowStr = str(row).split(',')
            if line_count < 26:
                accuracy_list.append(rowStr[1])
                rf_list.append('RF_'+str(rf_depth)+'_t'+str(line_count))
                if (line_count == 5):
                    accuracy_list_rf.append(rowStr[4])
                line_count+=1
    csv_file.close()
'''    
    
    
f= open(sizePath+'/'+dataSet+'/'+'nodesCount_all_'+dataset+'.csv',"w")
f.write('RF,nodes Count,\n')
for i in range(0,len(size_list)):
        f.write(rf_list[i]+','+size_list[i]+',\n')
for depth in (5,10,15,20):
        f.write('RF_'+str(depth)+','+size_list_rf[int(depth/5) -1]+',\n')
f.close()



variant = "NoLeafEdges"
scoring_function = 'accuracy'
pattern_max_size = 6
filesPath = "../frequentTreesInRandomForests/forests/rootedFrequentTrees"
resultsPath = "../frequentTreesInRandomForests/SizeComparison"
accuracy_list = []
size_list = []

accuracy_list_dsf = []
size_list_dsf = []

with open(resultsPath+'/'+'nodesCount_accuracy_'+dataset+'_'+variant+'_leq'+str(pattern_max_size)+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        lineCount =1    
            
        for row in csv_reader:
                if (lineCount > 1 ):
                    rowStr = str(row).split(',')
                    if (lineCount > 97):
                        size_list.append(rowStr[1])
                        accuracy_list.append(rowStr[2])
                    else:
                        size_list_dsf.append(rowStr[1])
                        accuracy_list_dsf.append(rowStr[2])
                        
            
                    
                lineCount+=1
                
            
                
csv_file.close()



size = np.array(size_list, dtype=np.float32)
accuracy = np.array(accuracy_list, dtype=np.float32)

size_dsf = np.array(size_list_dsf, dtype=np.float32)
accuracy_dsf = np.array(accuracy_list_dsf, dtype=np.float32)

best_dsf_rf5 = 0
best_dsf_rf10 = 0
best_dsf_rf15 = 0
best_dsf_rf20 = 0
best_dsf_rf5_index = 0
best_dsf_rf10_index = 0
best_dsf_rf15_index = 0
best_dsf_rf20_index = 0

for i in range(0,len(size_dsf)):
    if (i < 24):
        if (accuracy_dsf[i] > best_dsf_rf5):
            best_dsf_rf5 = accuracy_dsf[i]
            best_dsf_rf5_index = i
    if (i >= 24 and i < 48):
        if (accuracy_dsf[i] > best_dsf_rf10):
            best_dsf_rf10 = accuracy_dsf[i]
            best_dsf_rf10_index = i    
    if (i >= 48 and i < 72):
        if (accuracy_dsf[i] > best_dsf_rf15):
            best_dsf_rf15 = accuracy_dsf[i]
            best_dsf_rf15_index = i
    if (i >= 72 and i < 96):
        if (accuracy_dsf[i] > best_dsf_rf20):
            best_dsf_rf20 = accuracy_dsf[i]
            best_dsf_rf20_index = i        
    

accuracy_dsf_best = []
accuracy_dsf_best.append(best_dsf_rf5)
accuracy_dsf_best.append(best_dsf_rf10)
accuracy_dsf_best.append(best_dsf_rf15)
accuracy_dsf_best.append(best_dsf_rf20)
size_dsf_best = []
size_dsf_best.append(size_dsf[best_dsf_rf5_index])
size_dsf_best.append(size_dsf[best_dsf_rf10_index])
size_dsf_best.append(size_dsf[best_dsf_rf15_index])
size_dsf_best.append(size_dsf[best_dsf_rf20_index])

best_rf_index = np.argmax(accuracy)
best_dsf_index = np.argmax(accuracy_dsf_best)
best_rf = np.max(accuracy)
best_dsf = np.max(accuracy_dsf_best)

f= open(sizePath+'/'+dataset+'/'+'best_accuracy_'+dataset+'_'+'.csv',"w")
f.write('DSF,size,accuracy,\n')
f.write('RF_5_t'+str(best_dsf_rf5_index+2)+','+str(size_dsf_best[0])+','+str(accuracy_dsf_best[0])+',\n')
f.write('RF_10_t'+str(best_dsf_rf10_index+2-24)+','+str(size_dsf_best[1])+','+str(accuracy_dsf_best[1])+',\n')
f.write('RF_15_t'+str(best_dsf_rf15_index+2-48)+','+str(size_dsf_best[2])+','+str(accuracy_dsf_best[2])+',\n')
f.write('RF_20_t'+str(best_dsf_rf20_index+2-72)+','+str(size_dsf_best[3])+','+str(accuracy_dsf_best[3])+',\n')
f.write('\n')
f.write('Best DSF'+','+str(best_dsf)+',\n')
f.write('\n')
f.write('RF_5'+','+str(size[0])+','+str(accuracy[0])+',\n')
f.write('RF_10'+','+str(size[1])+','+str(accuracy[1])+',\n')
f.write('RF_15'+','+str(size[2])+','+str(accuracy[2])+',\n')
f.write('RF_20'+','+str(size[3])+','+str(accuracy[3])+',\n')
f.write('\n')
f.write('Best RF'+','+str(best_rf)+',\n')
f.close()

margin=0
if (dataset == 'adult'):
    margin = 0.005
if (dataset == 'spambase'):
    margin = 0.002
if (dataset == 'letter'):
    margin = 0.05
    
for i in range(0,len(size)):
    if (i == 2 or i==3):
         plt.scatter(size[i], accuracy[i], c='red')
         plt.text(size[i] - 2000, accuracy[i]-margin, (i+1)*5, fontsize=9)
    else:
         plt.scatter(size[i], accuracy[i], c='red')
         plt.text(size[i], accuracy[i]-margin, (i+1)*5, fontsize=9)
   
    plt.scatter(size_dsf_best[i], accuracy_dsf_best[i], c='blue')
    plt.text(size_dsf_best[i], accuracy_dsf_best[i]-margin, (i+1)*5, fontsize=9)
    
plt.style.use("seaborn")
plt.xlabel('Size (nodes count)')
plt.ylabel('Accuracy')

ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.60, box.height])
plt.legend(['Complete RF','DSF'],bbox_to_anchor=(1.0, 0.5), loc='upper left')
plt.text(size[3]*2,accuracy[3], 'max depth: {5,10,15,20}')
plt.text(size[3]*2,accuracy_dsf_best[3], 'Learning Algorithm: Naive Bayes')
plt.text(size[3]*2,accuracy_dsf_best[3]+margin*2, 'Dataset: '+dataset)
plt.xscale("log")
fig = plt.gcf()
fig.savefig(sizePath+'/'+dataset+'/'+'size_'+dataset+'_'+'.png', dpi=150)
fig.show()
