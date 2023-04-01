import os
import numpy as np
import math
import sys

resultsPath = "../tmp/reports_64_no_edges/"
#dataset = sys.argv[1]
#size_wanted = float(sys.argv[2])
dataset = 'letter'
size_wanted = 64
size_wanted = size_wanted * 40
#print(compression_achieved)
results_file = os.path.join(resultsPath, dataset + '/', 'report_thresholds.csv')


with open(results_file, 'r') as f:
    lines = f.readlines()
f.close()

test_acc_list = []
nodes_list = []

for line in lines:
    if (len(line) > 1 and 'Nodes' not in line):

        line_arr = line.split(',')
        print(line_arr)
        if (line_arr[15] != ''):

            if (int(line_arr[15]) <= size_wanted):

                test_acc_list.append((float(line_arr[6]),int(line_arr[15])))
        #nodes_list.append(line_arr[15])

test_acc_list.sort(key=lambda x: x[0], reverse= True)
print(test_acc_list)
print('Acc: ' + str(test_acc_list[0][0]))
print('nodes count: ' + str(test_acc_list[0][1]))
print('size ' + str((test_acc_list[0][1] * 25)/1024))


'''stats_file = os.path.join(resultsPath, dataset + '/',
                          'report_below_size_' + str(size_wanted) + '.csv')
with open(stats_file, 'w') as f_out:
    f_out.write(
        'Forest Size, Forest Depth, Patterns Threshold, Edges Threshold, Test Accuracy, Accuracy drop, F1 Macro drop, ROC AUC drop, Patterns Count,Compression, Training Time, Pruning Time,\n')
'''












