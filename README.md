# Splitting Stump Forests

Code and data accompanying the paper

Anonymous et al.: 
Splitting Stump Forests



## How to run the code

The file prgram.py contains the main experiment, it runs as follows: 

The script finds the splitting nodes, maps the input data to the new feature space induced by the splitting stumps, and trains a linear layer over it.
To run this script, give a dataset, forest depth and size, and filtering threshold values 'lines: 31-34'
The trained forests will be exported to forests folder.
The selected splitting stumps will be exported to stumps folder.
The results of linear layer training will be written in reports folder.
The scripts program.py uses classes and functions which exist in other files.

## Compression Analysis:
To compute the compression achieved when allowing a drop in accuracy, and to compute the accuracy achieved from a model which fits a space constrained device, and to compute the pareto frontier use the files:

best_acc_below_size.py
compression_analysis.py
nodes_count.py
pareto.py







