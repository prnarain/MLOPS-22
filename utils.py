
## for importing files 

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## for preprocessing digits 

def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label


## for data visualisation 

def data_viz(dataset):
    # PART: sanity check visualization of the data
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, dataset.images, dataset.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    plt.show()

# for checking the models 
def pred_image_viz(x_test, predictions):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x_test, predictions):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    plt.show()

# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model

def train_dev_test_split(data, label, train_frac, dev_frac):

    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test

## for hyper parameter tuning for svm and classifer training for all splits  
def h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric):
    best_metric = -1.0
    best_model = None
    best_h_params = None
    metric_all_list = []
    # For every combination-of-hyper-parameter values
    for cur_h_params in h_param_comb:

        # setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        # train the model
        clf.fit(x_train, y_train)

        # print(cur_h_params)
        # get dev set predictions
        predicted_dev = clf.predict(x_dev)

        # compute the accuracy on the validation set
        cur_metric = metric(y_pred=predicted_dev, y_true=y_dev)

        # identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_model = clf
            best_h_params = cur_h_params
            print("-----------------------------------------------------")
            print("Found new best metric with :" + str(cur_h_params))
            print("New best val metric:" + str(cur_metric))
        
        metric_all_list.append(cur_metric)
    return best_model, best_metric, best_h_params, metric_all_list
    

## for Decision tree 
## for hyper parameter tuning for DT  and classifer training for all splits  
def dt_param_tuning(dt_param_comb, clf, x_train, y_train, x_dev, y_dev, metric):
    best_metric = -1.0
    best_model = None
    best_h_params = None
    metric_all_list_dt = []

    #For every combination-of-hyper-parameter values
    for cur_h_params in dt_param_comb:

        # setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        # Train model
        clf.fit(x_train, y_train)

        # print(cur_h_params)
        # get dev set predictions
        predicted_dev = clf.predict(x_dev)

        # compute the accuracy on the validation set
        cur_metric = metric(y_pred=predicted_dev, y_true=y_dev)

        # identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_model = clf
            best_h_params = cur_h_params
            print("--------------------------------------------------------------")
            print("Found new best metric with :" + str(cur_h_params))
            print("New best val metric:" + str(cur_metric))
        
        metric_all_list_dt.append(cur_metric)

    return best_model, best_metric, best_h_params, metric_all_list_dt

##################################################