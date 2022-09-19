# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


#PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Standard scientific Python imports
from ctypes import alignment
from tkinter import CENTER
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# 1. set the ranges of hyper parameters 
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10] 

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()

#PART: sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
n_samples = len(digits.images)

'''
[[1,2,3],
 [4,5,6],
 [7,8,9]]
'''


data = digits.images.reshape((n_samples, -1))

#[1,2,3,4,5,6,7,8,9]

#other type of preprocessing for plot_digits 

    # image : 8 X 8 : RESIZE 16 X 16 , 32 X 32 , 
    # normalise : mean normalise  =  [X - mean(X)]
    #             min max normalisation = 
    # smoothing the image : blur on the image 
    # flattening 

#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model


dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)

best_train_acc = -1.0
best_train_model = None
best_train_h_params = None

best_val_acc = -1.0
best_val_model = None
best_val_h_params = None

best_test_acc = -1.0
best_test_model = None
best_test_h_params = None

accuracy_list = []
accuracy_list.append(" ")
accuracy_list.append("--------------------------------------------------"
                    "---------------------------------------------------")

accuracy_list.append([" Gamma :  g, '    C': c}", "            "
                    "Train Accuracy",       "  "
                    "Validation Accuracy",  "  "
                    "Test Accuracy"])

accuracy_list.append("--------------------------------------------------"
                    "---------------------------------------------------")
accuracy_list.append(" ")

# 2. For every combination-of-hyper-parameter values
for cur_h_params in h_param_comb:

    #PART: Define the model
    # Create a classifier: a support vector classifier
    clf = svm.SVC()

    #PART: setting up hyperparameter
    hyper_params = cur_h_params
    clf.set_params(**hyper_params)

##-----  Train accuracy 

    #PART: Test model
    # 2.a train the model 
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    
    # print(cur_h_params)
    #PART: get dev set predictions
    predicted_train = clf.predict(X_train)

    # 2.b compute the accuracy on the validation set
    cur_train_acc = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)

    # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
    if cur_train_acc > best_train_acc:
        best_train_acc = cur_train_acc
        best_model_train = clf
        best_train_h_params = cur_h_params
        #print("Found new best acc with :"+str(best_train_h_params))
        #print("val accuracy:" + str(best_train_acc))

##----- Validation accuracy 
  
    #PART: Train model
    # 2.a train the model 
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    
    # print(cur_h_params)
    #PART: get dev set predictions
    predicted_dev = clf.predict(X_dev)

    # 2.b compute the accuracy on the validation set
    cur_val_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)

    # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
    if cur_val_acc > best_val_acc:
        best_val_acc = cur_val_acc
        best_model_val = clf
        best_val_h_params = cur_h_params
        #print("Found new best acc with :"+str(best_val_h_params))
        #print("val accuracy:" + str(best_val_acc))
        
##----- Test accuracy 

    #PART: Train model
    # 2.a train the model 
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    
    # print(cur_h_params)
    #PART: get dev set predictions
    predicted_test = clf.predict(X_test)

    # 2.b compute the accuracy on the validation set
    cur_test_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)

    # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
    if cur_test_acc > best_test_acc:
        best_test_acc = cur_test_acc
        best_model_test = clf
        best_test_h_params = cur_h_params
        #print("Found new best acc with :"+str(best_test_h_params))
        #print("val accuracy:" + str(best_test_acc))
    
    #best_train_acc_list = "{:^15}".format(best_train_acc)
    #best_val_acc_list =   "{:^15}".format(best_val_acc) 
    #best_test_acc_list =  "{:^15}".format(best_test_acc)
    best_train_acc_list = "{:^{width}.{precis}f}".format(best_train_acc, precis=4, width=15)
    best_val_acc_list = "{:^{width}.{precis}f}".format(best_val_acc, precis=4, width=15)
    best_test_acc_list = "{:^{width}.{precis}f}".format(best_test_acc, precis=4, width=15)

    right_aligned_dict = {str(key).rjust(6): str(value).rjust(4) for key, value in cur_h_params.items()}

    accuracy_list.append([right_aligned_dict, 
                         best_train_acc_list, 
                         best_val_acc_list ,
                         best_test_acc_list])

#print(len(accuracy_list))
for i in range (0, len(accuracy_list)):
    print (accuracy_list[i])

#PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model_test.predict(X_test)

#PART: Sanity check of predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# 4. report the test set accurancy with that best model.
#PART: Compute evaluation metrics

## Train
predicted = best_model_train.predict(X_train)
print("\n----------------------------  Train  ------------------------------")
print(
    f"\nClassification report for classifier {clf}:\n\n"
    f"{metrics.classification_report(y_train, predicted)}\n")
print("Best Train hyperparameters were:", best_train_h_params)

## Validation
predicted = best_model_val.predict(X_dev)
print("\n-------------------------  Validation  -------------------------- ")
print(
    f"\nClassification report for classifier {clf}:\n\n"
    f"{metrics.classification_report(y_dev, predicted)}\n")
print("Best Validation hyperparameters were:", best_val_h_params)

## Test
predicted = best_model_test.predict(X_test)
print("\n ---------------------------  Test  ---------------------------- ")
print(
    f"\nClassification report for classifier {clf}:\n\n"
    f"{metrics.classification_report(y_test, predicted)}\n")
print("Best Test hyperparameters were:", best_test_h_params)
print("\n\n")

