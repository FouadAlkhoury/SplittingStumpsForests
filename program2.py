# This is the (cleaned up) code accompanying the publication
#
# Pascal Welke, Fouad Alkhoury, Christian Bauckhage, Stefan Wrobel: Decision Snippet Features.
# International Conference on Pattern Recognition (ICPR) 2021.
#
# Code was written by Pascal Welke and Fouad Alkhoury and is based on
# code written by Sebastian Buschjaeger (TU Dortmund) that is used for 
# json-serialization of random forest models.


# %% imports

import os
import json
import subprocess
import pickle

from sklearn.utils.estimator_checks import check_estimator
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import ReadData as ReadData
import cString2json as cString2json
import json2graphNoLeafEdgesWithSplitValues as json2graphNoLeafEdgesWithSplitValues
from fitModels import fitModels
import DecisionSnippetFeatures as DecisionSnippetFeatures
import pruning
import Forest
import datetime
from util import writeToReport
import numpy as np
# %% Parameters. 

dataPath = "../data/"
forestsPath = "../tmp/forests/"
snippetsPath = "../tmp/snippets/"
resultsPath = "../tmp/results/"
reportsPath = "../tmp/reports/"

# current valid options are ['sensorless', 'satlog', 'mnist', 'magic', 'spambase', 'letter', 'bank', 'adult', 'drinking']
dataSet = 'satlog'
# dataSet = 'adult'
# dataSet = 'drinking'

# possible forest_types ['RF', 'DT', 'ET']
forest_types = ['RF']
forest_depths = [5]
sigma_values = [0.0,0.1,0.2,0.3]
#forest_depths = [5, 10, 15, 20]
forest_size = 25

maxPatternSize = 3
minThreshold = 2
maxThreshold = 25

scoring_function = 'accuracy'

# learners that are to be used on top of Decision Snippet Features
learners = {'DSF_NB': MultinomialNB,
            'DSF_SVM': LinearSVC, 
            'DSF_LR': LogisticRegression}

# specify parameters that are given at initialization
learners_parameters = {'DSF_NB': {},
                       'DSF_SVM': {'max_iter': 10000},
                       'DSF_LR': {'max_iter': 1000}}


# for quick debugging, let the whole thing run once. Afterwards, you may deactivate individual steps
# each step stores its output for the subsequent step(s) to process
run_fit_models = True
run_mining = True
run_training = True
run_eval = True

verbose = True

fitting_models_time = datetime.timedelta()
pruning_time = datetime.timedelta()



# %% load data

X_train, Y_train = ReadData.readData(dataSet, 'train', dataPath)
X_test, Y_test = ReadData.readData(dataSet, 'test', dataPath)
X = X_train

# %% create forest data, evaluate and report accuracy on test data
start_fitting_models = datetime.datetime.now()

if run_fit_models:
    print('\n\nHERE ARE THE ACCURACIES ON TEST DATA OF THE ORIGINAL RANDOM FORESTS\n(don\'t worry, test data is not used for training)\n')

    fitModels(roundSplit=True, 
              XTrain=X_train, YTrain=Y_train, 
              XTest=X_test, YTest=Y_test, 
              createTest=False, model_dir=os.path.join(forestsPath, dataSet), types=forest_types, forest_depths = forest_depths)

end_fitting_models = datetime.datetime.now()
fitting_models_time = (end_fitting_models- start_fitting_models)

print('Fitting Models Time: '+str(fitting_models_time))    
# Pruning
report_model_dir = reportsPath+'/'+dataSet 
report_file = report_model_dir + '/report.txt'
if not os.path.exists(report_model_dir):
    os.makedirs(report_model_dir)
        
writeToReport(report_file,'Fitting Models Time \t ' + str(fitting_models_time) + '\n')

start_pruning_total = datetime.datetime.now()
writeToReport(report_file,'Pruning Time \t ')
writeToReport(report_file,'RF \t Sigma \t Pruning Time')

'''
for depth in forest_depths:
    
    input_file = forestsPath+dataSet+'/RF_'+ str(depth) +'.json'
    #feature_vectors_file = X
    
    
    f = Forest.Forest()
    f.fromJSON(input_file)
    t = f.trees[0]
    
    feature_vectors = X

    for sigma in sigma_values:
        
        start_pruning = datetime.datetime.now()
        
        pruning.Min_DBN(feature_vectors, f, sigma)
        pruning.post_processing(f)
        #print(f.pstr())
    
        #save the pruned forest in a json file
        sigma_as_string = '_'.join(str(sigma).split('.'))
        output_file = forestsPath + dataSet + '/RF_'+ str(depth) +'_pruned_with_sigma_' + sigma_as_string + '.json'
    
        with open(output_file, 'w') as outfile:
            outfile.write(f.str())
            
        end_pruning = datetime.datetime.now()
        pruning_time = (end_pruning - start_pruning)
        print('Pruning Time for RF_'+str(depth)+' and sigma='+str(sigma)+' : '+str(pruning_time)) 
        writeToReport(report_file, str(depth)+ '\t ' +str(sigma)+ '\t \t' + str(pruning_time))
    
end_pruning_total = datetime.datetime.now()
pruning_total_time = (end_pruning_total - start_pruning_total)

print('Total Pruning Time: '+str(pruning_total_time))  
writeToReport(report_file, 'Total Pruning Time: '+str(pruning_total_time) + '\n')
# %% compute decision snippets
'''

start_mining_total = datetime.datetime.now()

writeToReport(report_file,'Mining Time \t ')

if run_mining:
    print('\n\nFEEL FREE TO IGNORE THIS OUTPUT\n')

    def pattern_frequency_filter(f_patterns, frequency, f_filtered_patterns):
        pattern_count = 0
        for line in f_patterns:
            tokens = line.split('\t')
            if int(tokens[0]) >= frequency:
                f_filtered_patterns.write(line)
                pattern_count += 1
        return pattern_count

    start_json_to_graph = datetime.datetime.now()
    # translate json to graph files
    for json_file in filter(lambda x: x.endswith('.json'), os.listdir(os.path.join(forestsPath, dataSet))):
        #print(json_file[:-4])
        graph_file = json_file[:-4] + 'graph'
        with open(os.path.join(forestsPath, dataSet, json_file), 'r') as f_in:
            with open(os.path.join(forestsPath, dataSet, graph_file), 'w') as f_out:
                json2graphNoLeafEdgesWithSplitValues.main(f_in, f_out)

    end_json_to_graph = datetime.datetime.now()
    json_to_graph_time = (end_json_to_graph - start_json_to_graph)
    print('Translating Time :' + str(json_to_graph_time)) 
    writeToReport(report_file, 'Translating Time :' + str(json_to_graph_time))    

    # run frequent pattern mining
    if not os.path.exists(os.path.join(snippetsPath, dataSet)):
        os.makedirs(os.path.join(snippetsPath, dataSet))

    for graph_file in filter(lambda x: x.endswith('.graph'), os.listdir(os.path.join(forestsPath, dataSet))):
        
        # pattern mining for smallest minThreshold
        #print(f"mining {minThreshold}-frequent patterns for {graph_file}")
        
        start_mining = datetime.datetime.now()
        print("mining "+ str(minThreshold) + " frequent patterns for " + str(graph_file))
        pattern_file=os.path.join(snippetsPath, dataSet, graph_file[:-6] + f'_t{minThreshold}.patterns')
        feature_file=os.path.join(snippetsPath, dataSet, graph_file[:-6] + f'_t{minThreshold}.features')
        log_file=os.path.join(snippetsPath, dataSet, graph_file[:-6] + f'_t{minThreshold}.logs')

        args = ['../lwgr', '-erootedTrees', '-mbfs', f'-t{minThreshold}', f'-p{maxPatternSize}', 
                f'-o{pattern_file}', os.path.join(forestsPath, dataSet, graph_file)]
        with open(feature_file, 'w') as f_out:
            with open(log_file, 'w') as f_err:        
                subprocess.run(args, stdout=f_out, stderr=f_err)

        # filtering of patterns for larger thresholds
        print(f"filtering more frequent patterns for {graph_file}")
        for threshold in range(maxThreshold, minThreshold, -1):
            filtered_pattern_file=os.path.join(snippetsPath, dataSet, graph_file[:-6] + f'_t{threshold}.patterns')
            pattern_count = -1
            with open(pattern_file, 'r') as f_patterns:
                with open(filtered_pattern_file, 'w') as f_filtered_patterns:
                    pattern_count = pattern_frequency_filter(f_patterns, threshold, f_filtered_patterns)
            
            # if there are no frequent patterns for given threshold, remove the file.
            if pattern_count == 0:
                os.remove(filtered_pattern_file)
            else:
                if verbose:
                    print(f'threshold {threshold}: {pattern_count} frequent patterns')


        # transform canonical string format to json
        for threshold in range(maxThreshold, minThreshold-1, -1):
            filtered_pattern_file = os.path.join(snippetsPath, dataSet, graph_file[:-6] + f'_t{threshold}.patterns')
            filtered_json_file = os.path.join(snippetsPath, dataSet, graph_file[:-6] + f'_t{threshold}.json')
            try:
                with open(filtered_pattern_file, 'r') as f_filtered_patterns:
                    with open(filtered_json_file, 'w') as f_filtered_json:
                        json_data = cString2json.parseCStringFileUpToSizePatterns(f_filtered_patterns, patternSize=maxPatternSize)
                        f_filtered_json.write(json_data)
            except EnvironmentError:
                # this might happen if a certain threshold resulted in no frequent patterns and is OK
                pass
            
            
        end_mining = datetime.datetime.now()
        mining_time = (end_mining - start_mining)
        print('Mining Time for '+ graph_file +' : '+str(mining_time)) 
        writeToReport(report_file, 'Mining Time for '+ graph_file +' : '+str(mining_time))    

end_mining_total = datetime.datetime.now()
mining_total_time = (end_mining_total - start_mining_total)
writeToReport(report_file, 'Total Mining Time: '+str(mining_total_time) + '\n')

# %% Training of classifiers. For later selection of best candidate learners, run xval on train to estimate generalization

writeToReport(report_file,'Training Time \t ')

def dsf_transform(snippets_file, X):
    with open(snippets_file, 'r') as f_decision_snippets:

        # load decision snippets and create decision snippet features
        frequentpatterns = json.load(f_decision_snippets)
        dsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(
            map(lambda x: x['pattern'], frequentpatterns))
        fts = dsf.fit_transform(X)

        # transform to onehot encodings for subsequent processing by linear methods
        categories = dsf.get_categories()
        fts_onehot_sparse = OneHotEncoder(
            categories=categories).fit_transform(fts)
        fts_onehot = fts_onehot_sparse.toarray()

        return fts_onehot

start_training_total = datetime.datetime.now()   

if run_training:
    print('\n\nFEEL FREE TO IGNORE THIS OUTPUT\n')

    results_list = list()

    # save results list
    if not os.path.exists(os.path.join(resultsPath, dataSet)):
        os.makedirs(os.path.join(resultsPath, dataSet))

    def train_model_on_top(model, fts_onehot, Y_train, scoring_function, model_name, descriptor, scaling=False):

        if scaling:
            model = Pipeline([('scaler', StandardScaler()), (model_name, model)])

        fts_onehot_nb_cv_score = cross_val_score(model, fts_onehot, Y_train, cv=5, scoring=scoring_function)

        dsf_score = fts_onehot_nb_cv_score.mean()
        dsf_std = fts_onehot_nb_cv_score.std()
        print(f'{model_name} {descriptor} {dsf_score} +- {dsf_std}')
        writeToReport(report_file, str(model_name) + '\t' + str(descriptor) + '\t' + str(dsf_score)  + ' +- ' + str(dsf_std))
        model.fit(fts_onehot, Y_train)
        return dsf_score, model, fts_onehot_nb_cv_score

    # train several models on the various decision snippet features
    # store all xval results on traning data in a list
    for graph_file in filter(lambda x: x.endswith('.json'), sorted(os.listdir(os.path.join(snippetsPath, dataSet)))):
        
        start_training = datetime.datetime.now()
        # get Decision Snippet Features
        fts_onehot = dsf_transform(os.path.join(snippetsPath, dataSet, graph_file), X_train)
       
        # train models
        for model_type, model_class in learners.items():
            xval_score, learner_model, xval_results = train_model_on_top(model_class(**learners_parameters[model_type]), fts_onehot, Y_train, scoring_function, model_type, graph_file)
            results_list.append((xval_score, model_type, graph_file, learner_model, xval_results))
            # cleanup
            xval_score, learner_model, xval_results = None, None, None

        # dump after each decision snippet
        with open(os.path.join(resultsPath, dataSet, "training_xval.pkl"), 'wb') as f_pickle:
            pickle.dump(results_list, f_pickle)
            
            
        end_training = datetime.datetime.now()
        training_time = (end_training - start_training)
        print('Training Time for '+ graph_file +' : '+str(training_time)) 
        writeToReport(report_file, 'Training Time for '+ graph_file +' : '+str(training_time))    
    

end_training_total = datetime.datetime.now()
training_total_time = (end_training_total - start_training_total)
print('Total Training Time: '+str(training_total_time))  
writeToReport(report_file, 'Total Training Time: '+str(training_total_time) + '\n')

# %% Find, for each learner, the best decision snippet features on training data (the proposed model) and evaluate its performance on test data

if run_eval:
    print('\n\nHERE ARE THE ACCURACIES ON TEST DATA OF THE BEST DECISION SNIPPET FEATURES\n(for each model and each RF depth)\n')
    
    with open(os.path.join(resultsPath, dataSet, "training_xval.pkl"), 'rb') as f_pickle:
        results_list = pickle.load(f_pickle)

        unique_forests = map(lambda x: x[:-5] + '_', filter(lambda x: x.endswith('.json'), os.listdir(os.path.join(forestsPath, dataSet))))

        print(f'best_snippets model_type train_xval_acc test_acc n_features')
        for forest in unique_forests:
            for model_type in learners:
                # print('processing', model_type, forest)
                best_score = 0.
                scores = list()
                labels = list()
                for result in filter(lambda x: x[2].startswith(forest), filter(lambda x: x[1] == model_type, results_list)):
                    # store run with max score
                    if result[0] > best_score:
                        best_result = result
                        best_score = result[0]
                    scores.append(result[0])
                    labels.append(result[2])

                # evaluate on test
                graph_file = best_result[2]
                fts_onehot = dsf_transform(os.path.join(snippetsPath, dataSet, graph_file), X_test)
                pred_test = best_result[3].predict(fts_onehot)
                test_acc = accuracy_score(Y_test, pred_test)

                print(f'{best_result[2]} {model_type} {best_result[0]:.3f} {test_acc:.3f} {fts_onehot.shape[1]}')
                writeToReport(report_file, str(best_result[2]) + '\t' + str(model_type) + '\t' + str(np.round(best_result[0],3)) + '\t' + str(np.round(test_acc,3)) + '\t' + str(fts_onehot.shape[1]))

