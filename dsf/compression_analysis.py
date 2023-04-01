import os
import numpy as np
import math
import sys

resultsPath = "../tmp/reports_64_no_edges/"
#dataset = sys.argv[1]
#size_wanted = float(sys.argv[2])
datasets = ['adult','bank','credit','drybean','magic','rice','room','satlog','shopping','spambase']
edge_thresholds = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6]
sizes = [16,32,64]
depths = [5,10,15]
rf_acc_16_5 = [0.791,0.898,0.809,0.879,0.846,0.934,0.995,0.851,0.902,0.911]
rf_acc_16_10 = [0.826,0.898,0.81,0.922,0.87,0.938,0.997,0.898,0.909,0.932]
rf_acc_16_15 = [0.827,0.897,0.81,0.928,0.876,0.933,0.998,0.907,0.905,0.941]
rf_acc_32_5 = [0.793,0.898,0.81,0.882,0.847,0.934,0.996,0.849,0.873,0.919]
rf_acc_32_10 = [0.826,0.901,0.812,0.926,0.869,0.939,0.998,0.902,0.912,0.931]
rf_acc_32_15 = [0.83,0.898,0.811,0.923,0.876,0.933,0.999,0.904,0.912,0.948]
rf_acc_64_5 = [0.792,0.898,0.809,0.885,0.843,0.934,0.996,0.858,0.896,0.92]
rf_acc_64_10 = [0.828,0.899,0.811,0.926,0.87,0.937,0.998,0.895,0.909,0.937]
rf_acc_64_15 = [0.83,0.898,0.81,0.93,0.877,0.93,0.998,0.906,0.91,0.945]

rf_size_16_5 = [880,936,980,940,936,916,842,936,930,830]
rf_size_16_10 = [9054,11728,12752,8012,9134,4252,1982,6332,8510,3700]
rf_size_16_15 = [29692,49892,43998,16710,25080,5688,2098,10472,22906,6502]
rf_size_32_5 = [1784,1908,1956,1848,1888,1770,1628,1932,1858,1546]
rf_size_32_10 = [18622,22544,25268,16246,17638,8648,3782,13160,18078,7788]
rf_size_32_15 = [59134,97552,86556,33178,51344,11382,4080,21240,45194,12774]
rf_size_64_5 = [3588,3820,3908,3702,3802,3538,3144,3798,3722,3150]
rf_size_64_10 = [35798,46006,50620,32082,35422,16996,7482,25990,35780,15396]
rf_size_64_15 = [116302,181172,175186,67182,101812,23300,8324,41976,87148,26746]


shape = (3,3,len(edge_thresholds))
arr = np.zeros(shape)
size = 64
depth = 15
print(arr)

test_acc_list = []
nodes_list = []

for d,dataset in enumerate(datasets):
    test_acc_list=[]

    results_file = os.path.join(resultsPath, dataset + '/', 'report_thresholds.csv')

    with open(results_file, 'r') as f:
        lines = f.readlines()
    f.close()


    for line in lines:
        if (len(line) > 1 and 'Nodes' not in line):

            line_arr = line.split(',')
            #print(line_arr)
            if (line_arr[15] != ''):

                if (float(line_arr[6]) >= rf_acc_64_15[d] -0.02 and int(line_arr[0]) == size and int(line_arr[1]) == depth and float(line_arr[3]) != 1.0):
                    test_acc_list.append((float(line_arr[6]),rf_size_64_15[d]/int(line_arr[15])))
                    #i = sizes.index(int(line_arr[0]))
                    #j = depths.index(int(line_arr[1]))
                    #k = edge_thresholds.index(float(line_arr[3]))

                    #arr[i][j][k] += 1
    print(dataset)
    test_acc_list.sort(key=lambda x: x[1], reverse=True)
    if (len(test_acc_list) > 0):

        print(test_acc_list[0])

    #nodes_list.append(line_arr[15])


#print(arr)


'''
test_acc_list.sort(key=lambda x: x[0], reverse= True)
print(test_acc_list)
print('Acc: ' + str(test_acc_list[0][0]))
print('nodes count: ' + str(test_acc_list[0][1]))
print('size ' + str((test_acc_list[0][1] * 25)/1024))

stats_file = os.path.join(resultsPath, dataset + '/',
                          'report_below_size_' + str(size_wanted) + '.csv')
with open(stats_file, 'w') as f_out:
    f_out.write(
        'Forest Size, Forest Depth, Patterns Threshold, Edges Threshold, Test Accuracy, Accuracy drop, F1 Macro drop, ROC AUC drop, Patterns Count,Compression, Training Time, Pruning Time,\n')
'''












