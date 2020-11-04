# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:30:57 2020

@author: Arsh Modak
"""


#%%

# Importing Necessary Packages:
    
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from mlxtend.plotting import plot_decision_regions

#%%

# Importing Data:
def loadData(dataPath):
    """
    

    Parameters
    ----------
    dataPath : string
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
    data_mat = scipy.io.loadmat(dataPath)
    X_train = data_mat["X_trn"]
    X_test = data_mat["X_tst"]
    Y_train = data_mat["Y_trn"]
    Y_test = data_mat["Y_tst"]
    
    return X_train, X_test, Y_train.flatten(), Y_test.flatten()


#%%

# LOGISTIC REGRESSION WITH GRADIENT DESCENT:
#==============================================================================

class myLogisticRegression():
    def __init__(self, l_rate, max_it, name_to_print, method, set_):
        """
        

        Parameters
        ----------
        l_rate : float
            learning rate.
        max_it : int
            maximum iterations.
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
        self.l_rate = l_rate
        self.max_it = max_it
        self.name_to_print = name_to_print
        self.method = method
        self.set_ = set_
        self.w = None
        self.b = None
        self.eps = 1e-05

    # Function to Predict y:
    def predict(self, X):
        """
        

        Parameters
        ----------
        X : array of float64
            train or test set.

        Returns
        -------
        y_pred : array of uint8
            predicted y values.

        """
        sigmoid = lambda w, b, X : 1.0/(1.0 + np.exp(-(np.dot(X, self.w) + self.b)))
        y_pred_prob = sigmoid(self.w, self.b, X)
        y_pred = np.where(y_pred_prob > 0.50, 1, 0)
        
        return y_pred
    
    # Function to print Evaluation Metrics:
    def printMetrics(self, y, y_pred, name_to_print, method, set_):
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
        print("\nPrinting Metrics for {} from {} on the {} set.".format(name_to_print.capitalize(), method.capitalize(), set_.capitalize()))
        cfm = confusion_matrix(y, y_pred)
        print("\nConfusion Matrix: ")
        print("\n", cfm)
        print("\n\nClassification Report:" )
        print(classification_report(y, y_pred))
        acc = accuracy_score(y, y_pred)
        print("\nAccuracy of the Model: ", round(acc, 3))
        print("Classification Error of the Model: ", round(1-acc, 3))
        
        return
    
    # Function to run Logistic Regression using Gradient Descent:
    def logisticRegression(self, X, y):
        """
        

        Parameters
        ----------
        X : array of float64
            train or test set.
        y : array of uint8
            actual y values (y train or y test).

        Returns
        -------
        w : array of size x-features
            optimal weights for logistic regression.
        b : float
            bias.

        """
        # Sigmoid Function:
        sigmoid = lambda w, b, X : 1/(1 + np.exp(-(np.dot(X, w) + b)))
        x_rows, x_cols = X.shape[0], X.shape[1]
        self.w = np.zeros(x_cols) # [w0, w1]
        self.b = 0   # bias
        
        for it in range(self.max_it):
            h_x = sigmoid(self.w, self.b, X)
            temp_w = (1.0/x_rows)*(np.dot(X.T, (y.T - h_x).reshape(x_rows)))
            temp_b = (1.0/x_rows)*(np.sum((y.T - h_x).reshape(x_rows)))
            
            self.w = self.w + self.l_rate*temp_w
            self.b = self.b + self.l_rate*temp_b
            
            if abs((self.w - temp_w)).all() <= self.eps and abs((self.b - temp_b)).all() <= self.eps:
                break
        
        return self.w, self.b

#%%

# LOGISTIC REGRESSION AND NAIVE BAYES USING SKLEARN:
#==============================================================================

def runMLModels(modelLabel, X, y):
    """
    

    Parameters
    ----------
    modelLabel : string
        kind of model (logistic or NB).
    X : array of float64
        train or test set.
    y : array of uint8
        actual y values (y train or y test).

    Returns
    -------
    classifier : model
        class of the model.

    """
    if modelLabel == "logistic":
        classifier = LogisticRegression(random_state = 0)
    elif modelLabel == "NB":
        classifier = GaussianNB()
    else:
        print("Invalid Parameters")
    classifier.fit(X, y)
    
    return classifier

#%%

# Predicting using Scikit - Learn
def sklearnPredict(X, classifier):
    """
    

    Parameters
    ----------
    X : array of float64
        train or test set.
    classifier : model
        class of the model.

    Returns
    -------
        y_pred : array of uint8
            predicted y values..

    """
    y_pred = classifier.predict(X)
    return y_pred
    
    
#%%

# Function to print Evaluation Metrics:
def printMetrics(y, y_pred, name_to_print, method, set_):
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
    
    print("\nPrinting Metrics for {} from {} on the {} set.".format(name_to_print.capitalize(), method.capitalize(), set_.capitalize()))
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


def plotDecisionBoundary(X, y, clf, legend, name_to_print, method, set_):
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
    plt.title("Decision Boundary for {} from {} on the {} set.".format(name_to_print.capitalize(), method.capitalize(), set_.capitalize()))
    plt.show()
    
    return

#%%

def main(X_train, X_test, y_train, y_test, modelLabel, method, l_rate, max_it, name_to_print, set_):
    """
    

    Parameters
    ----------
    X_train : array of float64
        x train set.
    X_test : array of float64
        x test set.
    y_train : array of uint8
        y train set.
    y_test : array of uint8
        y test set.
    modelLabel : string
        kind of model (logistic or NB).
    method : string
        type of method (scratch or sklearn).
    l_rate : float
        learning rate.
    max_it : int
        maximum iterations.
    name_to_print : string
        name of the model (Logistic Regression or Naive Bayes).
    set_ : string
        type of set (train or test).

    Returns
    -------
    None.
    

    """
    if method == "sklearn":
        classifier = runMLModels(modelLabel, X_train, y_train)
        if set_ == "train":
            y_pred = sklearnPredict(X_train, classifier)
            printMetrics(y_train, y_pred, name_to_print, method, set_)
            plotDecisionBoundary(X_train, y_train, classifier, 2, name_to_print, method, set_)
        elif set_ == "test":
            y_pred = sklearnPredict(X_test, classifier)
            printMetrics(y_test, y_pred, name_to_print, method, set_)
            plotDecisionBoundary(X_test, y_test, classifier, 2, name_to_print, method, set_)
        else:
            print("Invalid set_ parameter")
    elif method == "scratch":
        classifier  = myLogisticRegression(l_rate, max_it, name_to_print, method, set_)
        w, b = classifier.logisticRegression(X_train, y_train)
        if set_ == "train":
            y_pred = classifier.predict(X_train)
            classifier.printMetrics(y_train, y_pred,name_to_print, method, set_)
            plotDecisionBoundary(X_train, y_train, classifier, 2, name_to_print, method, set_)
        elif set_ == "test":
            y_pred = classifier.predict(X_test)
            classifier.printMetrics(y_test, y_pred,name_to_print, method, set_)
            plotDecisionBoundary(X_test, y_test, classifier, 2, name_to_print, method, set_)
        else:
            print("Invalid set_ parameter")
    else:
        print("Invalid method parameter")
    
    return
     
#%%

# Driver Code:
if __name__ == "__main__":
    dataPath = r"give data path"
    X_train, X_test, y_train, y_test = loadData(dataPath)
    
    modelLabel = "logistic"                         # or "NB" for Naive Bayes
    method = "scratch"                              # or "sklearn" (only sklearn for NB)
    l_rate = 0.01                                   # learning rate
    max_it = 1000                                   # maximum iterations
    set_ = "test"                                   # or "train"
    name_to_print = "Logistic Regression"           # change to "Naive Bayes" when using it.
    
    main(X_train, X_test, y_train, y_test, modelLabel, method, l_rate, max_it, name_to_print, set_)
    
    
    



