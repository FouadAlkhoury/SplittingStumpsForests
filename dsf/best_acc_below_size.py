# Given dataset and wanted space (lines: 5 and 6), the scripts outputs the best performing splitting stumps that fits in a specific space budget.
import os

resultsPath = "../tmp/reports/"
dataset = 'letter'
size_wanted = 64 # in KB
size_wanted = size_wanted * 40  # x1000 Bytes / 25 Bytes per node
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

test_acc_list.sort(key=lambda x: x[0], reverse= True)
print(test_acc_list)
print('Acc: ' + str(test_acc_list[0][0]))
print('nodes count: ' + str(test_acc_list[0][1]))
print('size ' + str((test_acc_list[0][1] * 25)/1024))









