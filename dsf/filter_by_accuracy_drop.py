import os
import numpy as np
import math
import sys

resultsPath = "../tmp/final_reports_patterns/"
dataset = sys.argv[1]
accuracy_drop = float(sys.argv[2])
print(accuracy_drop)
results_file = os.path.join(resultsPath, dataset+'/', 'report_thresholds.csv')

sizes = [16,32,64,128]
depths = [5]
pattern_thresholds = [1]
edge_thresholds = [1.0,0.975,0.95,0.925,0.9,0.85,0.8,0.7,0.6]

with open(results_file,'r') as f:
    lines = f.readlines()
f.close()  


stats_file = os.path.join(resultsPath, dataset+'/', 'report_thresholds_'+str(accuracy_drop)+'.csv')
with open(stats_file,'w') as f_out:
    f_out.write('Forest Size, Forest Depth, Patterns Threshold, Edges Threshold, Test Accuracy, Accuracy drop, F1 Macro drop, ROC AUC drop, Patterns Count,Compression, Training Time, Pruning Time,\n')
    index = 0
    counter = 0
    line_index = []

    print('Forest Size, Forest Depth, Patterns Threshold, Edges Threshold, Test Accuracy, Accuracy drop, Patterns Count,Compression, Training Time, Pruning Time,')
    for i_s,s in enumerate(sizes):
        for i_d,d in enumerate(depths):
            for i_p,p in enumerate(pattern_thresholds):

                for i_e,e in enumerate(edge_thresholds):
                    line_arr = lines[counter *(len(edge_thresholds)) * 3  + ((i_e + 1) *3)].split(',')
                    
                    if (i_e == (len(edge_thresholds) - 1) and abs(float(line_arr[7])) <=  accuracy_drop):
                        index = len(edge_thresholds) - 1
                        line_index = lines[counter *(len(edge_thresholds)) * 3  + ((i_e + 1) *3)].split(',')
                        print(edge_thresholds[index])
                        #line_arr = lines[line_index].split(',')
                        compression = np.round(1/float(line_index[14]),2)
                        print(str(s)+','+str(d)+','+str(p)+','+str(line_index[3])+','+str(line_index[6])+','+str(line_index[7])+','+
                             str(line_index[13])+','+str(compression)+','+str((line_index[15]))+','+str(line_index[16])+',')
                        f_out.write(str(s)+','+str(d)+','+str(p)+','+str(line_index[3])+','+str(line_index[6])+','+str(line_index[7])+','+str(line_index[9])+','+str(line_index[12])+','+str(line_index[13])+','+str(compression)+','+str((line_index[15]))+','+str(line_index[16])+',\n')

                        counter += 1


                    else:
                        
                        if (abs(float(line_arr[7])) <=  accuracy_drop):
                            index = i_e
                            line_index = lines[counter *(len(edge_thresholds)) * 3  + ((i_e + 1) *3)].split(',')

                        else:
                            print(edge_thresholds[index])
                            
                            compression = np.round(1/float(line_index[14]),2)
                            print(str(s)+','+str(d)+','+str(p)+','+str(line_index[3])+','+str(line_index[6])+','+str(line_index[7])+','+
                                 str(line_index[13])+','+str(compression)+','+str((line_index[15]))+','+str(line_index[16])+',')
                            f_out.write(str(s)+','+str(d)+','+str(p)+','+str(line_index[3])+','+str(line_index[6])+','+str(line_index[7])+','+str(line_index[9])+','+str(line_index[12])+','+
                                 str(line_index[13])+','+str(compression)+','+str((line_index[15]))+','+str(line_index[16])+',\n')

                            counter += 1
                            break





























