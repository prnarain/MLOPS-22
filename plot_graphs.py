# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics

from matplotlib import test
from sklearn import datasets, svm, metrics
from sklearn import tree
from matplotlib import pyplot as plt

## function to be loaded from util 
from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    dt_param_tuning,
    data_viz,
    pred_image_viz,
)
from joblib import dump, load

# train dev and test fraction 
train_frac= 0.8
dev_frac = (1-train_frac)/2
test_frac = (1-train_frac)/2 

assert train_frac + dev_frac + test_frac == 1.0

# set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.5, 1, 5, 10]

h_param_comb = [{"gamma": g, "C": c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list) * len(c_list)
metric_list = []
metric_best5_list_svm = [] 
metric_best5_list_dt = [] 

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

metric = metrics.accuracy_score

train_frac_list = [0.6, 0.66, 0.72, 0.76, 0.8]

metric_list.append("++++++++++++++++  SVM accuracy  ++++++++++++++++")

for train_frac in train_frac_list:

    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac)

    clf = svm.SVC()
   
    best_model, best_metric, best_h_params, metric_all_list = h_param_tuning(
        h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric)
    
    metric_list.append([best_h_params, 
                        "train %", train_frac*100, 
                        "dev %", round((1-train_frac)*100/2,2), 
                        "test %", round((1-train_frac)*100/2,2),
                        "best metric",best_metric
                        ])
    metric_best5_list_svm.append(best_metric)

    ## predicted vs expected 
    predicted = best_model.predict(x_test)
    pred_image_viz(x_test, predicted)


print("\n\n\n *************     end of svm classifier    ****************\n\n")

# for DT 
metric_list.append(" ")
metric_list.append("+++++++++++++  Decision Tree accuracy +++++++++++++++")

max_depth_list = [20,50,100,200,300]
max_leaf_nodes_list = [100,200,300,400,500]

dt_param_comb = [{"max_depth": d, "max_leaf_nodes": n} for d in max_depth_list for n in max_leaf_nodes_list]

for train_frac in train_frac_list:
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac)

    clft = tree.DecisionTreeClassifier()

    best_model, best_metric, best_h_params, metric_all_list_dt = dt_param_tuning(
        dt_param_comb, clft, x_train, y_train, x_dev, y_dev, metric)

    metric_list.append([best_h_params, 
                        "train %", train_frac*100, 
                        "dev %", round((1-train_frac)*100/2,2), 
                        "test %", round((1-train_frac)*100/2,2),
                        "best metric",best_metric
                        ])
    metric_best5_list_dt.append(best_metric)
    
    ## predicted vs expected 
    predicted = best_model.predict(x_test)
    pred_image_viz(x_test, predicted)


print("\n\n************    end of decision tree  classifier    ***********\n\n")

###################### for comparing with results  #########################
from sklearn.datasets import load_digits
digits = load_digits()

fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

plt.show()
###############################################

#print average and mean of both classifier 

import statistics
print("\n+++++++++++ Mean and Standard Deviation (all values) of both classifier  +++++++++\n")
print ("\nMean of SVM", statistics.mean(metric_all_list))
print ("\nSD of SVM", statistics.pstdev(metric_all_list))

print ("\nMean of total Decision Tree", statistics.mean(metric_all_list_dt))
print ("\nSD of total Decision Tree", statistics.pstdev(metric_all_list_dt))


print("\n\n+++++++++++ Mean and Standard Deviation (5 values) of both classifier  +++++++++\n")
print ("\nMean of best SVM", statistics.mean(metric_best5_list_svm))
print ("\nSD of best SVM", statistics.pstdev(metric_best5_list_svm))

print ("\nMean of best Decision Tree", statistics.mean(metric_best5_list_dt))
print ("\nSD of best Decision Tree", statistics.pstdev(metric_best5_list_dt))

########################################