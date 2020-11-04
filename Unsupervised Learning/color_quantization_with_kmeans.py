# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:57:32 2020

@author: Arsh Modak
"""


#%%

# Necessary Packages:
    
import numpy as np
import cv2.cv2 as cv2
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster import hierarchy
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import timeit
import time


#%%

# SOLUTION 2
#==============================================================================

def plotOG(X):
    plt.scatter(X[:, 0], X[:, 1], cmap = "rainbow")
    plt.xlabel("X-Values")
    plt.ylabel("Y-Values")
    plt.title("Original Points")
    plt.savefig("""give path""")
    plt.close()
    return


#%%
    
def plotKMeans(X, labels, centroids, iteration):
    plt.scatter(X[:, 0], X[:, 1],
            c = labels, cmap = "rainbow")
    plt.scatter(centroids[:, 0], centroids[:, 1], color = "black")
    plt.xlabel("X-Values")
    plt.ylabel("Y-Values")
    plt.title("Iteration : {}".format(iteration))
    plt.savefig("give path".format(iteration))
    plt.close()
    return

#%%

def plotFinalClusters(X, labels):
    plt.scatter(X[:, 0], X[:, 1],
        c = labels, cmap = "rainbow")
    plt.xlabel("X-Values")
    plt.ylabel("Y-Values")
    plt.title("Final Clustered Points")
    plt.savefig("give path")
    plt.close()
    return
    
    
#%%

def runQ2Kmeans(X, it, initial_centroids):
    for i in range(1, it + 1):
        # print(i)
        kmeans = KMeans(n_clusters = 3, random_state = 0, init = initial_centroids, max_iter = 1)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        # print(centroids)
        labels = kmeans.labels_
        plotKMeans(X, labels, centroids, i)
        initial_centroids = kmeans.cluster_centers_   
    plotFinalClusters(X, labels)
    return

#%%

def runHAC(X, hac_method = "single", hac_metric = "euclidean"):
    Z = hierarchy.linkage(X, method  = hac_method, metric = hac_metric)
    hierarchy.dendrogram(Z)
    plt.title("Dendrogram for {} Hierarchical Agglomorative Clustering".format(hac_method))
    plt.xlabel("Points")
    plt.ylabel("Closeness")
    plt.savefig(r"give path".format(hac_method))
    plt.close()
    return Z

#%%

# SOLUTION 3
#==============================================================================

# To Read Image into Python:
def readImage(imagePath):    
    return cv2.imread(imagePath)

#%%

# To Display the Image:
def showImage(image, title = str):
    return cv2.imshow(title, image)

#%%

def saveImage(imagePath, image):
    image = (image * 255).astype("uint8")
    cv2.imwrite(imagePath, image)
    return
    

#%%

# To Resize Image, Transform 3D Image Array to 2D Array
def processImage(image, only_resize = False):
    
    resImage = cv2.resize(image, None, fx = 0.427, fy = 0.4)
    img = np.array(resImage, dtype=np.float32)/255
    width, height, depth = original_shape = tuple(img.shape)
    img_arr = np.reshape(img, (width*height, depth)) 
    
    # OR
    # img_arr = np.float32(image).reshape(-1, 3) 
    
    if only_resize == False:
        return resImage, width, height, img_arr
    elif only_resize == True:
        return resImage
    else:
        print("Invalid Parameters!")
        

#%%

# COLOR QUANTIZATION USING KMEANS


def runKMEANS(image_array, km_method = "normal", k = 3, training_sampleSize = 2000):
    
    print("Running {} K-Means Algorithm for Color Quantization".format(km_method))
    # Creating a Data Sample for Training:
    dataSample = shuffle(image_array, random_state = 0)[:training_sampleSize]
    
    if km_method == "normal":
        # Initializing Model
        kmeans = KMeans(n_clusters = k, random_state = 0)
    elif km_method == "minibatch":
        kmeans = MiniBatchKMeans(n_clusters = k, random_state = 0)
    else:
        print("Invalid 'km_method' parameter!")
    
    print()
    # Fitting the data to the model:
    # %timeit kmeans.fit(dataSample)
    start = time.time()
    kmeans.fit(dataSample)
    # Predicting the labels (clusters)
    print("Time Taken to Train: {} ms".format(round((time.time()-start)*1000), 2))
    predict_time = time.time()
    labels = kmeans.predict(image_array)
    print("Time taken to Predict: {} ms".format(round((time.time()-predict_time)*1000), 2))
    
    
    return dataSample, kmeans, labels

#%%

# Function to Recreate the Image:
def recreateImage(cluster_centers, labels, width, height):
    
    depth = cluster_centers.shape[1]
    image = np.zeros((width, height, depth))
    
    labelIndex = 0
    for w in range(width):
        for h in range(height):
            image[w][h] = cluster_centers[labels[labelIndex]]
            labelIndex += 1
            
    return image

#%%

def elbowMethod(K, data):
    
    distortions = list()
    for k in K:
        kmeans = KMeans(n_clusters = k, random_state = 0)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
        
    plt.plot(K, distortions, "bo-")
    plt.xlabel("k")
    plt.ylabel("Distorition")
    plt.title("Elbow Method for Optiman value of k")
    plt.xticks(np.arange(min(K), max(K)+1, 1.0))
    plt.annotate("Elbow", 
                 xy = (3, distortions[2]), 
                 xytext = (4, 500),
                 fontsize = 14,
                 arrowprops = dict(facecolor = "black", shrink = 0.1))
    plt.show()
    plt.savefig(r"give path")
    plt.close()
    
    return distortions

#%%

def sol3(imagePath, method, k, K):
    # Reading Original Image
    og_image = readImage(imagePath)
    # Processed Image
    resizedImage, width, height, img_arr = processImage(og_image)
    # To show resized original image
    showImage(resizedImage, "Original Image Resized")
    # Run K-Means and get labels:
    dataSample, kmeans, labels = runKMEANS(img_arr, method, k, training_sampleSize)
    # Recreating Image:
    kmeansImage = recreateImage(kmeans.cluster_centers_, labels, width, height)
    showImage(kmeansImage, "Color Quantized Image with K-means (method = {}, k = {})".format(method, k))
    # Running the Elbow Method: (Optimal k = 4)
    distortions = elbowMethod(K, dataSample)
    
    return distortions, kmeansImage

#%%

def sol2(X, it, initial_centroids, hac_method, hac_metric, algorithm):
    
    if algorithm == "kmeans":
        # K-Means Clustering:
        runQ2Kmeans(X, it, initial_centroids)
    elif algorithm == "hac":
    # Hierarchical Agglomerative Clustering:
        runHAC(X, hac_method = hac_method, hac_metric = hac_metric)
    else:
        print("Invalid Algorithm!")
    
    return
      
#%%

if __name__ == "__main__":

    # For Q2:
    hac_method = "single" # can be average or complete
    hac_metric = "euclidean" # can be any other distance of your choice
    algorithm = "hac"    # (hac or kmeans)
    it = 4 # No. of iterations for q2 - kmeans
    
    Data:
    X = np.array([[2, 10], [2, 5], [8, 4], [5, 8],
                  [7, 5], [6, 4], [1, 2], [4, 9]])
    
    Initial Centroids:
    initial_centroids = np.array([[2, 10],
                                  [5, 8],
                                  [1, 2]])
    
    # To plot original points
    plotOG(X)
    # To plot original points with initial centroids
    plotKMeans(X, [1]*8, initial_centroids, 0)
    # hac_methods can be changed to "complete" or "average"
    # hac_metric can be changed to any other distance metric for HAC
    sol2(X, it, initial_centroids, hac_method, hac_metric, algorithm)
    
    # For Q3
    k = 3 # no. of clusters for k-means
    km_method = "minibatch" # type of kmeans (normal or minibatch)
    K = range(1, 11) # Range of k for elbow method
    imagePath = r"give path"
    quantizedImagePath = r"give path".format(k)
    training_sampleSize = 10000
    distortions, kmeansImage = sol3(imagePath, km_method, k, K)
    saveImage(quantizedImagePath, kmeansImage)
    
    

    

#%%





