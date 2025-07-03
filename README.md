# Splitting Stump Forests

Code and data accompanying the paper

Splitting Stump Forests: Tree Ensemble Compression for Edge Devices (Extended Version)
Authors: Fouad Alkhoury, Sebastian Buschjäger, and Pascal Welke.

## Repository Structure

The folder 'dsf' contains the scripts used to run the experiments. 'data' folder contains the datasets used.
The results, reports, forests and stumps files are exported to 'tmp' folder. 

## How to run the code

The file dsf/ssf_program.py contains the main experiment, it finds the splitting nodes, maps the input data to the new feature space induced by the splitting stumps, and trains a linear layer over it.
The script runs as follows:
In lines: 27-32, give a dataset, forest depth and size, filtering threshold, and decimal places values. 
The trained forests will be exported to tmp/forests folder.
The selected splitting stumps will be exported to tmp/stumps folder.
The results of linear layer training will be written in tmp/reports folder.
The scripts program.py uses classes and functions which exist in other files in the same folder.


## Example
Running the current code (dataset=adult, depth=5, size=64, threshold=0.4, decimal_places=4) gives the following results:
It shows first both baselines Random forest and Logistic regression:
Random Forest accuracy: 0.795 
Logistic Regression accuracy: 0.798 
and then the splitting stump forest accuracy:
Splitting Stumps test accuracy: 0.827
It shows also the count of nodes in the original random forest and the count of stumps selected by the splitting stump forest method.








