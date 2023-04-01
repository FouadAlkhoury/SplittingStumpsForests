p1: dataset e.g. magic
p2: data size e.g. 1000
p3: number of epochs e.g. 1000

train splits:

python3 dsf/train_splits_multilayer.py p1 p2 p3

the new trained patterns will be saved in snippets/dataset/
'remove the last comma in the generated file :)'


evaluate accuracy of trained splits:

python3 dsf/program_exp.py p1
