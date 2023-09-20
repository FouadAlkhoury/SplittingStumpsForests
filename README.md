# Splitting Stump Forests

Code and data accompanying the paper

Splitting Stump Forests: Tree Ensemble Compression for Edge Devices.
Authors: Fouad Alkhoury and Pascal Welke.

## Repository Structure

The folder 'dsf' contains the scripts used to run the experiments. 'data' folder contains the datasets used.
The results, reports, forests and stumps files are exported to 'tmp' folder. 

## How to run the code

The file dsf/program.py contains the main experiment, it finds the splitting nodes, maps the input data to the new feature space induced by the splitting stumps, and trains a linear layer over it.
The script runs as follows:
In lines: 31-34, give a dataset, forest depth and size, and filtering threshold values. 
The trained forests will be exported to tmp/forests folder.
The selected splitting stumps will be exported to tmp/stumps folder.
The results of linear layer training will be written in tmp/reports folder.
The scripts program.py uses classes and functions which exist in other files in the same folder.


## Example
Running the current code (dataset=adult, depth=5, size=64) gives the following results:
Random Forest accuracy: 0.795 
Splitting Stumps test accuracy: 0.837
f1 macro = 0.837
Testing time = 5 ms

## Compression Analysis:
To compute the compression achieved when allowing a drop in accuracy, and to compute the accuracy achieved from a model which fits a space constrained device, and to compute the pareto frontier use the files:

best_acc_below_size.py
compression_analysis.py
nodes_count.py
pareto.py







