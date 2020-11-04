# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:35:26 2020

@author: Arsh Modak
"""


#%%

# Importing Neccessary Libraries:
    
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from scipy import linalg 

# %matplotlib auto

#%%

# 3D plot of convex function:
#==============================================================================

# f(x1, x2) = x1 ** 2 + (x2 -2) ** 2

def concavefunc_3dplot():
    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig)
    x1 = np.arange(-10, 10, 0.35)
    x2 = np.arange(-10, 10, 0.35)
    x1, x2 = np.meshgrid(x1, x2)
    z  = (x1 ** 2 + (x2 -2) ** 2)
    ax.plot_surface(x1, x2, z, rstride=1, cstride=1, cmap=plt.cm.hot)
    ax.contourf(x1, x2, z, zdir='z', offset=-2, cmap=plt.cm.hot)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Z")
    ax.set_title("Solution 7.1")
    plt.show()
    
    return

# concavefunc_3dplot()


#%%

# GRADIENT DESCENT
#==============================================================================

# f(x1, x2) = x1 ** 2 + (x2 -2) ** 2
# df/dx1 = 2x1
# df/dx2 = 2(x2 -2)

# df/dx = [2x1, 2(x2 -1)] (a vector)

def gradientDescent(init_input, rate = float, max_it = 10000):
    """
    

    Parameters
    ----------
    init_input : list
        Input: (x1, x2).
    rate : float, optional
        DESCRIPTION. The default is float. Learning Rate
    max_it : int, optional
        DESCRIPTION. The default is 10000. Maximum Iterations.

    Returns
    -------
    points : 
        Numpy array which consist of sequence of points of Gradient Descent.

    """

    eps = 1e-3
    i = 0
    convergence = False
    terminator = False
    points = list()
    gradient = lambda x1, x2 : np.array([2*x1, 2*(x2-2)]).T
    curr = init_input
    print("Iteration: {}, Value: {}".format(0, curr))
    while terminator == False or convergence == False:
        prev = curr
        # print("Prev: {}".format(prev))
        curr = curr - rate * gradient(curr[0], curr[1])
        # print("Curr: {}".format(curr))
        term_var = abs(curr-prev).T
        # print("Termination Var.: {}".format(term_var))
        i += 1
        print("Iteration: {}, Value: {}".format(i, curr))
        points.append(curr)
        if i > max_it:
            terminator = True
            print("Max Iterations Reached!")
            break
        # if term_var[0] < eps and term_var[1] < eps:
        #     convergence = True
        #     print("Converged!")
        #     break
        if np.linalg.norm(gradient(curr[0], curr[1])) < eps:
            convergence = True
            print("Converged!")
            break
        if (term_var[0] == math.inf or term_var[1] == math.inf) or (term_var[0] == -math.inf or term_var[1] == -math.inf):
            print("Value reached Infinity")
            break
        
    print("\n\n")
    print("Total Number of Iterations: {}".format(i))
    print("Minimum: {}".format(curr))
    print("\n\n")
    
    return np.array(points)



def plotGD(points, title = str()):
    """

    Parameters
    ----------
    points : numpy ndarray
        Numpy array which consist of sequence of points of Gradient Descent
    title : string, optional
        Title of the graph. The default is str().

    Returns
    -------
    None.

    """
    x1 = points[:,0]
    x2 = points[:, 1]
    z_func = lambda x1, x2 : (x1**2 + (x2-2)**2)
    z = z_func(x1, x2)
    fig = plt.figure()
    ax = fig.gca(projection = "3d")
    ax.plot(x1, x2, "ro", label = "x1,x2")
    ax.plot(x1, x2, z, lw = 2.0, label = "f(x1, x2) = ((x1)^2 + (x2-2)^2)")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc = "best")
    
    return


def runGradientDescent(user_in = False):
    """
    

    Parameters
    ----------
    user_in : bool, optional
        Pass True to take user inputs. The default is False.

    Returns
    -------
    None.

    """
    if user_in == True:
        init_input = input("Enter Initial Points (x1, x2): ").split(",")
        init_input = [float(i) for i in init_input] 
        rate = float(input("Enter Learning Rate: "))
        max_it = int(input("Enter Maximum Number of Iterations: "))
        points = gradientDescent(init_input, rate, max_it)
        # df = pd.DataFrame(points, columns = ("x1", "x2"))
        # plt.plot("x1", "x2", "bo", data = df)
        plotGD(points, "Solution 7: Custom Inputs")
    else:
        points = gradientDescent([1.0,1.0], 0.01)
        plotGD(points, "GD with lr = 0.01")
        points = gradientDescent([1.0,1.0], 0.1)
        plotGD(points, "GD with lr = 0.1")
        points = gradientDescent([1.0,1.0], 5.0)
        plotGD(points, "DG with lr = 5.0")
    
    return

#%%

# CLOSED FORM LINEAR REGRESSION
#==============================================================================
# y = theta1*x + theta0
# closed form => (X_tilda.T*X)^-1(X_tilda.T*Y) # here * is dot product


def closedFormLR(train_set):
    """
    

    Parameters
    ----------
    train_set : numpy.ndarray
        A 2D array that consists of x-values(feature), y-values(response)
        

    Returns
    -------
    solution : numpy.ndarray
        Optimal solutions for a closed form linear regression problem
        solution = (X_tilda.T*X)^-1(X_tilda.T*Y)

    """
    
    X = train_set[:, 0]
    Y = train_set[:, 1]
    X_tilda = sm.add_constant(X)[:, [1, 0]] # adding a constant to match dimentions as well as calculate intercept
    closed_form = lambda X, Y : np.dot(linalg.inv(np.dot(X_tilda.T, X_tilda)), np.dot(X_tilda.T, Y))
    solution = closed_form(X_tilda, Y)
    print("Optimum Values: {}. {}".format(solution[0], solution[1]))
    
    return solution

def plot2DLR(solution, X, Y, title = str()):
    """
    

    Parameters
    ----------
    solution : numpy.ndarray
        Optimal solutions for a closed form linear regression problem
        solution = (X_tilda.T*X)^-1(X_tilda.T*Y)
        See function closedFormLR
    X : numpy ndarray
        x-values(feature)
    Y : numpy ndarray
        y-values(response)
    title: string, optional
        Title of the graph. The default is str()

    Returns
    -------
    None.

    """
    funct = solution[0]*X + solution[1]
    plt.plot(X, Y, "ro")
    plt.plot(X, funct)
    plt.xlabel("x-values (input)")
    plt.ylabel("y-values (response)")
    plt.title(title + " Regression (Line of Best Fit)")
    plt.show()

    
    return

        
#%%


# DRIVER CODE
if __name__ == "__main__":
    
    concavefunc_3dplot()

    runGradientDescent(False) # Pass True for custom (user) inputs
    

    train_set1 = np.array([[0.10, 0.65], [0.50, 0.10], [0.90, 0.35], [-0.20, 0.17], [-0.5, 0.42], [1.50, 2.62]])
    solution = closedFormLR(train_set1)
    plot2DLR(solution, train_set1[:, 0], train_set1[:, 1], "With All Given Points")
    

    train_set2 = train_set1[:-1]
    solution = closedFormLR(train_set2)
    plot2DLR(solution, train_set2[:, 0], train_set2[:, 1], "Without Last Point")


# ********************************************* END **************************************************
    
    

    
    
        
