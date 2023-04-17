import sys
import csv
import numpy as np
import os.path
import json
import timeit

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import joblib

sys.path.append('../arch-forest/code/')
import Forest
import Tree

resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports/"

def testModel(roundSplit, dataset, XTrain, YTrain, XTest, YTest, model, name, model_dir, size, depth):

	report_file = reportsPath+'/'+dataset + '/report_rf.csv' 

	print("Fitting", name)
	model.fit(XTrain,YTrain)

	print("Testing ", name)
	start = timeit.default_timer()
	YPredicted = model.predict(XTest)
	end = timeit.default_timer()
	print("Total time: " + str(end - start) + " ms")

	print("Saving model")
	if (issubclass(type(model), DecisionTreeClassifier)):
		mymodel = Tree.Tree()
	else:
		mymodel = Forest.Forest()

	mymodel.fromSKLearn(model,roundSplit)

	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	with open(os.path.join(model_dir, name + ".json"), 'w') as outFile:
		outFile.write(mymodel.str())

	YPred = model.predict(XTest)
	r2 = r2_score(YTest,YPred)
	mae = mean_absolute_error(YTest,YPred)
	mse = mean_squared_error(YTest,YPred,squared=False)
	print("r2:", r2)

	with open(report_file, 'a') as outFile:
		outFile.write(str(size) + ', ' + str(depth) + ', ' + str(r2) + ',' + str(mae)+ ',' + str(mse)+ ', \n')
	outFile.close()

	print("Saving model to PKL on disk")
	joblib.dump(model, os.path.join(model_dir, name + ".pkl"))
	
	print("*** Summary ***")
	print("#Examples\t #Features\t R2\t Avg.Tree Height")
	print(str(r2) + "\t" + str(mymodel.getAvgDepth()))

	print()


def fitModels(roundSplit, dataset, XTrain, YTrain, XTest=None, YTest=None, createTest=False, model_dir='text', 
              types=['RF', 'ET', 'DT'], 
			  forest_depths = [1,2],
              forest_sizes = [20,30]):
	''' Fit a bunch of forest models to the given train data and write the resulting models to disc.
	Possible forest types are: 
	- DT (decision tree)
	- ET (extra trees)
	- RF (random forest)
	- AB (adaboost) '''
	report_file = reportsPath+'/'+dataset + '/report_rf.csv' 
	if XTest is None or YTest is None:
		XTrain,XTest,YTrain,YTest = train_test_split(XTrain, YTrain, test_size=0.25)
		createTest = True

	if createTest:
		with open("test.csv", 'w') as outFile:
			for x,y in zip(XTest, YTest):
				line = str(y)
				for xi in x:
					line += "," + str(xi)

				outFile.write(line + "\n")

	if 'DT' in types:
		for depth in forest_depths:
			testModel(roundSplit, XTrain, YTrain, XTest, YTest, RandomForestClassifier(n_estimators=1, n_jobs=8, max_depth=depth), f"DT_{depth}", model_dir)

	if 'ET' in types:
		for depth in forest_depths:
			testModel(roundSplit, XTrain, YTrain, XTest, YTest, ExtraTreesClassifier(n_estimators=forest_size, n_jobs=8, max_depth=depth), f"ET_{depth}", model_dir)

	if 'RF' in types:
		with open(report_file, 'w') as outFile:
			outFile.write('Forest Size, Forest depth, R2, MAE, RMSE, \n')
		outFile.close()            
		for size in forest_sizes:        
			for depth in forest_depths:
				testModel(roundSplit, dataset, XTrain, YTrain, XTest, YTest, RandomForestClassifier(n_estimators=size, n_jobs=8, max_depth=depth), f"RF_{size}_{depth}", model_dir, size, depth)

