import sys
# set dataset name and run.
# The output results file and plot are exported to the folder SizeComparison/dataset/  
dataset = sys.argv[1]


import csv
#import matplotlib.pyplot as plt
import numpy as np

variant = "NoLeafEdges"
scoring_function = 'accuracy'
pattern_max_size = 4
filesPath = "../tmp/snippets"
samplesPath = "../tmp/samples"
filesPath_RF = "../tmp/forests"
sizePath = "../tmp/nodesCount"
nodes_count_list = []
rf_depth = 5
    
with open(filesPath_RF+'/'+dataset+'/RF_'+str(rf_depth)+'.json') as json_file:
         line = json_file.read()
         list = [i for i in range(len(line)) if line.startswith('"id"', i)]
         
         nodes_count_list.append('RF_'+str(rf_depth)+','+str(len(list))+',\n')
'''    
    for sigma in ('0_0','0_1','0_2','0_3'):
        
        with open(filesPath_RF+'/'+dataset+'/'+'RF_'+str(rf_depth)+'_pruned_with_sigma_'+sigma+'.json') as json_file:
         line = json_file.read()
         list = [i for i in range(len(line)) if line.startswith('"id"', i)]
         
         nodes_count_list.append('RF_'+str(rf_depth)+'_sigma_'+sigma+','+str(len(list))+',\n')   
'''    
     
frequency=1
      
with open(filesPath+'/'+dataset+'/'+'RF_'+str(rf_depth)+'_t'+str(frequency)+'.json') as json_file:
         line = json_file.read()
         list = [i for i in range(len(line)) if line.startswith('"id"', i)]
         
         nodes_count_list.append('RF_'+str(rf_depth)+'_t'+str(frequency)+','+str(len(list))+',\n')
            
for sample in range(1,11):
    
    with open(samplesPath+'/'+dataset+'/'+'RF_'+str(rf_depth)+'_t'+str(frequency)+'_'+str(sample)+'.json') as json_file:
         line = json_file.read()
         list = [i for i in range(len(line)) if line.startswith('"id"', i)]
         
         nodes_count_list.append('RF_'+str(rf_depth)+'_t'+str(frequency)+'_'+str(sample)+','+str(len(list))+',\n')            
'''       
     for sigma in ('0_0','0_1','0_2','0_3'):
        
           with open(filesPath+'/'+dataset+'/'+'RF_'+str(rf_depth)+'_pruned_with_sigma_'+sigma+'_t'+str(frequency)+'.json') as json_file:
                line = json_file.read()
                list = [i for i in range(len(line)) if line.startswith('"id"', i)]
         
                nodes_count_list.append('RF_'+str(rf_depth)+'_t'+str(frequency)+'_sigma_'+sigma+','+str(len(list))+',\n')  
        
                                                
json_file.close()
'''
f= open(sizePath+'/'+dataset+'/'+'nodesCount_'+dataset+'.csv',"w")
f.write('RF,nodes Count,\n')
for line in nodes_count_list:
        f.write(line)
f.close()


nodes_count_list = []
#for rf_depth in (5,10,15,20):
    
      
                
        
          
            
#json_file.close()

#f= open(sizePath+'/'+dataSet+'/'+'nodesCount_'+dataset+'_RandomForestClassifier'+'.csv',"w")
#f.write('RF,nodes Count,\n')
#for line in nodes_count_list:
#        f.write(line)
#f.close()





accuracy_list = []
accuracy_list_rf = []
size_list = []
size_list_rf = []

rf_list = []


'''with open(sizePath+'/'+dataSet+'/'+'nodesCount_'+dataset+'.csv') as size_file:
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
    
    
    
f= open(sizePath+'/'+dataSet+'/'+'nodesCount_all_'+dataset+'.csv',"w")
f.write('RF,nodes Count,\n')
for i in range(0,len(size_list)):
        f.write(rf_list[i]+','+size_list[i]+',\n')
for depth in (5,10,15,20):
        f.write('RF_'+str(depth)+','+size_list_rf[int(depth/5) -1]+',\n')
f.close()
'''