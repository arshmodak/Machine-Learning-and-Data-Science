# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:28:20 2020

@author: Arsh Modak
"""


#%%

import numpy as np
import scipy.io
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


#%%

def loadData(data_path):
    """
    

    Parameters
    ----------
    data_path : string
        filepath to data.

    Returns
    -------
    X_train : array of float64
        x train set.
    X_test : array of float64
        x test set.
    y_train : array of uint8
        y train set.
    y_test : array of uint8
        y test set.

    """
    data_mat = scipy.io.loadmat(data_path)
    X_train = data_mat["X_trn"]
    X_test = data_mat["X_tst"]
    Y_train = data_mat["Y_trn"]
    Y_test = data_mat["Y_tst"]
    
    return X_train, X_test, Y_train.flatten(), Y_test.flatten()


#%%

def plotDecisionBoundary(X, y, clf, legend, kernel, set_):
    """
    

    Parameters
    ----------
    X : array of float64
        train or test set.
    y : array of uint8
        actual y values (y train or y test).
    clf : model
        class of the model.
    legend : int
        no of labels.
    name_to_print : string
        name of the model (Logistic Regression or Naive Bayes).
    method : string
        type of method (scratch or sklearn).
    set_ : string
        type of set (train or test).

    Returns
    -------
    None.

    """
    plot_decision_regions(X = X, y = y, clf = clf, legend = legend, colors = "#b82121,#0915ed")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Decision Boundary for {} SVM on the {} set.".format(kernel, set_.capitalize()))
    plt.show()
    
    return

#%%

# Function to print Evaluation Metrics:
def printMetrics(y, y_pred, kernel, set_):
    """


    Parameters
    ----------
    y : array of uint8
        actual y values (y train or y test).
    y_pred : array of uint8
        predicted y values.
    name_to_print : string
        name of the model (Logistic Regression or Naive Bayes).
    method : string
        type of method (scratch or sklearn).
    set_ : string
        type of set (train or test).

    Returns
    -------
    None.


    """
    
    print("\nDecision Boundary for {} SVM on the {} set.".format(kernel.upper(), set_.capitalize()))
    cfm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix: ")
    print("\n", cfm)
    print("\n\nClassification Report:" )
    print(classification_report(y, y_pred))
    acc = accuracy_score(y, y_pred)
    print("\nAccuracy of the Model: ", round(acc, 3))
    print("Classification Error of the Model: ", round(1-acc, 3))
    
    return

#%%

def SVM(X_train, X_test, y_train, kernel, set_, C = 1, gamma = "auto", legend = 2):
    
    classifier = SVC(kernel = kernel , C = C, gamma = gamma, probability = False)
    classifier.fit(X_train, y_train)
    
    if set_ == "train":
        y_pred = classifier.predict(X_train)
        printMetrics(y_train, y_pred, kernel, set_)
        plotDecisionBoundary(X_train, y_train, classifier, legend, kernel, set_)
    elif set_ == "test":
        y_pred = classifier.predict(X_test)
        printMetrics(y_test, y_pred, kernel, set_)
        plotDecisionBoundary(X_test, y_test, classifier, legend, kernel, set_)
    else:
        print("Invalid set_!")

    
    return classifier, y_pred

#%%

if __name__ == "__main__":
    data_path = "datasets/svm_dataset.mat"
    X_train, X_test, y_train, y_test = loadData(data_path)
    kernel = "linear"      # poly or rbf
    set_ = "test"          # or train
    C = 10
    gamma = "auto"
    legend = 2
    classifier, y_pred = SVM(X_train, X_test, y_train, kernel, set_, C, gamma, legend)
    print("Coefficients: {}, Intercept: {}".format(classifier.coef_, classifier.intercept_))

#%%
