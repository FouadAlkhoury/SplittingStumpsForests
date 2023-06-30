# The script counts the number of nodes of the model.
import os
import numpy as np
import math

import sys

resultsPath = "../tmp/final_reports/"
forestsPath = "../tmp/forests/"
dataset ='aloi'
results_file = os.path.join(resultsPath, dataset+'/', 'forest_size.csv')
node_size = 25
sizes = [8]
depths = [5]


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
    
        
        





















