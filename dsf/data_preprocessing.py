#train test split
import os

dataPath = "../data/"
dataset = 'wpbc'
data_file = os.path.join(dataPath, dataset + '/', dataset)
data_all = data_file + '_all.train'
data_train = data_file + '.train'
data_test = data_file + '.test'

file_write_train = open(data_train, 'w')
file_write_test = open(data_test, 'w')
file_read = open(data_all, 'r')
lines = file_read.readlines()
for i in range(len(lines)):
    if ('NaN' not in lines[i]):

        if (i % 5 != 0):
            file_write_train.write(lines[i])
        else:
            file_write_test.write(lines[i])

file_write_train.close()
file_read.close()