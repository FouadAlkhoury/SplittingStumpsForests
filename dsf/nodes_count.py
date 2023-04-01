import os
import numpy as np
import math
import sys

resultsPath = "../tmp/final_reports_64/"
forestsPath = "../tmp/forests_64_pruned/"
#dataset = sys.argv[1]
dataset ='adult'
#compression_achieved = float(sys.argv[2])
#print(compression_achieved)
results_file = os.path.join(resultsPath, dataset+'/', 'forest_size.csv')
node_size = 25
sizes = [8]
#sizes = [25,50,100]
depths = [5]
#pattern_thresholds = [1]
#edge_thresholds = [1.0,0.975,0.95,0.925,0.9,0.85,0.8,0.7,0.6]

with open(results_file,'a') as f_out:
            f_out.write('')
for s in sizes:
    for d in depths:
        forest_file = os.path.join(forestsPath, dataset+'/RF_'+str(s)+'_'+str(d)+'.json')
        with open(forest_file,'r') as f:
            text = f.read()
            count = text.count('"id"')
            print(count)
        f.close()  
        with open(results_file,'a') as f_out:
            f_out.write(str(s)+','+str(d)+','+str(count)+','+str(int((count * node_size)/1000))+'\n')
        
f_out.close()
    
        
        





















