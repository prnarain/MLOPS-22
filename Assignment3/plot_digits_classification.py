"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
from tkinter import Scale
from turtle import shape
from unicodedata import digit
import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.




###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.
#######################################

def train_test():

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

    ## ----------- Train ------------------------------

    for cur_h_params in h_param_comb:

        # Create a classifier: a support vector classifier
        clf = svm.SVC()

        #PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted_train = clf.predict(X_train)
        cur_train_acc = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)

        if cur_train_acc > best_train_acc:
            best_train_acc = cur_train_acc
            best_model_train = clf
            best_train_h_params = cur_h_params


    ## -----------  Validation  -------------------------

    # Create a classifier: a support vector classifier
        clf = svm.SVC()

        #PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        # Learn the digits on the train subset
        clf.fit(X_dev, y_dev)

        # Predict the value of the digit on the test subset
        predicted_dev = clf.predict(X_dev)
        cur_val_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)


        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc
            best_model_val = clf
            best_val_h_params = cur_h_params


    ## ------------  Test  -------------------------

    # Create a classifier: a support vector classifier
        clf = svm.SVC()

        #PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted_test = clf.predict(X_test)
        cur_test_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)


        if cur_test_acc > best_test_acc:
            best_test_acc = cur_test_acc
            best_model_test = clf
            best_test_h_params = cur_h_params

        best_train_acc_list = "{:^{width}.{precis}f}".format(best_train_acc, precis=4, width=15)
        best_val_acc_list = "{:^{width}.{precis}f}".format(best_val_acc, precis=4, width=15)
        best_test_acc_list = "{:^{width}.{precis}f}".format(best_test_acc, precis=4, width=15)
        
        accuracy_list.append([cur_h_params, "        ",
                            best_train_acc_list, 
                            best_val_acc_list ,
                            best_test_acc_list])

    #print(len(accuracy_list))
    for i in range (0, len(accuracy_list)):
        print (accuracy_list[i])

    ###############################################################################
    # Below we visualize the first 4 test samples and show their predicted
    # digit value in the title.
    '''
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted_test):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    '''
    ###############################################################################
    # 4. report the test set accurancy with that best model.
    #PART: Compute evaluation metrics

    ## ------------  Train  -----------------------

    predicted = best_model_train.predict(X_train)
    print(
        f"\nTrain Classification report for classifier {best_model_train}:\n\n"
        f"{metrics.classification_report(y_train, predicted)}\n\n")
    print("Best Train hyperparameters were:", best_train_h_params)
    '''
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_train, predicted)
    disp.figure_.suptitle("Confusion Matrix Train")
    print(f"\nConfusion matrix (train):\n{disp.confusion_matrix}")
    plt.show()
    '''

    ## -----------  Validation ---------------------

    predicted = best_model_val.predict(X_dev)
    print(
        f"\nValidation Classification report for classifier {best_model_val}:\n\n"
        f"{metrics.classification_report(y_dev, predicted)}\n\n")
    print("Best Validation hyperparameters were:", best_val_h_params)
    '''
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_dev, predicted)
    disp.figure_.suptitle("Confusion Matrix Validation")
    print(f"\nConfusion matrix (validation):\n{disp.confusion_matrix}")
    plt.show()
    '''

    ## -------------   Test --------------------------

    predicted = best_model_test.predict(X_test)
    print(
        f"\nTest Classification report for classifier {best_model_test}:\n\n"
        f"{metrics.classification_report(y_test, predicted)}\n\n")
    print("Best Test hyperparameters were:", best_test_h_params)
    '''
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix Test")
    print(f"\nConfusion matrix (test ):\n{disp.confusion_matrix}")
    plt.show()
    '''


# 1. set the ranges of hyper parameters 

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10] 

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

### Define train / test / validation split 

dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)
'''
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
'''
#----------------------------------------------- 8 X 8 ----------------------------------
## use the same X_data

print ("\n++++++++++++++++++++++++++++  Digit resolution 8 X 8 ++++++++++++++++++++++++++++n")
print("\nResolution of original digits dataset (x,y) :", digits.images.shape[1:])
print("\n Dataset shape",(digits.images).shape)
train_test()

#----------------------------------------------- 16 X 16 ----------------------------------
## change the X_data

X_data = []
print ("\n++++++++++++++++++++++++++++  Digit resolution 16 X 16 ++++++++++++++++++++++++++++n")
for i in range (0,len(digits.images)):
    digits16 = rescale(digits.images[i],2)
    X_data.append (digits16)

print("\nResolution of digits dataset changed to (x,y) :", np.array(X_data).shape[1:])
print("\n Dataset shape ",(np.array(X_data).shape))
train_test()

#------------------------------------------------ 32 X 32 -------------------------------
## change the X_data

X_data = []
print ("\n++++++++++++++++++++++++++++  Digit resolution 32 X 32 ++++++++++++++++++++++++++++\n")
for i in range (0,len(digits.images)):
    digits16 = rescale(digits.images[i],4)
    X_data.append (digits16)

print("\nResolution of digits dataset changed to (x,y) :", np.array(X_data).shape[1:])
print("\n Dataset shape ",(np.array(X_data).shape))
train_test()

#---------------------------  X  ----  X  ----  X  ----  X  ----  X  ----  X  ----