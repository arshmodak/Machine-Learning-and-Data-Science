# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:09:59 2020

@author: Arsh Modak
"""


#%%

import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest, chi2

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)

#%%

def readData(path):
    return pd.read_csv(path)

housingDF = readData(r"E:\ARSH\NEU\Fall 2020\DS 5230 (USML)\Homeworks\HW1\housing.csv")
# housingDF.head()


#%%

# SOLUTION 2

# Function To Display Summary Statistics

def summaryStats(data, include_param = None):
    return data.describe(include = include_param) 

# pass True instead of None to show summary stats of categorical variables 
# as well as numerical variables.
# for numerical variables only:
print(summaryStats(housingDF, "all")) 


#%%

# SOLUTION 3
# There are multiple ways to compute and show Correlation within the dataset.
# %matplotlib inline

# Computing a Correlation Matrix:
    
def corrMatrix(data):
    corrdf = data.corr(method = "pearson")
    # Sorting the Correlation Coefficients w.r.t "median_house_value"
    sorted_corrdf = corrdf.loc["median_house_value"].sort_values(ascending = False)
    
    return corrdf, sorted_corrdf[1:]

corrdf, sorted_corrdf = corrMatrix(housingDF)

#%%

# Heat Map to show Correlation between Variables:
# heatmap = sns.heatmap(corrdf, linewidths = 2.0, annot = True)

def heatMap(data):
    
    sns.heatmap(data, linewidths = 2.0, annot = True,)
    plt.title("Heat Map to depict Correlation between all Features\n")
    plt.xticks(rotation = 20, horizontalalignment = "right")
    # heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 45, horizontalalignment = "right")
    plt.show()
    
    return

heatMap(corrdf)

#%%

# Barplot of Correlation Coefficients of variables w.r.t median house value
    
def barPlot(data):
    df = data.reset_index(level=0)
    df = df.rename(columns = {"index": "Features", 
                              "median_house_value": "Correlation Coefficient"})
    
    sns.barplot(x = "Correlation Coefficient", y = "Features", data = df)
    plt.title("Correlation of Features w.r.t Median House Value\n")
    plt.xticks(rotation = 45, horizontalalignment = "right")
    
    plt.show()
    
    return

barPlot(sorted_corrdf)

#%%

# SOLUTION 4

# Plot1: Distribution Plots

def distPlots(data):
    data = data.drop(["latitude", "longitude"], axis = 1)
    fig, axes = plt.subplots(3, 3)
    for i, col in enumerate(data.columns):
        sns.distplot(data[col], ax = axes[i//3, i%3], hist = 50)
    return

distPlots(housingDF.iloc[:, :-1])

# OR
# housingDF.hist(bins = 50, figsize = (15, 15), grid = False)

#%%

# Plot2: Box Plots

def boxPlots(data):
    data = data.drop(["latitude", "longitude"], axis = 1)
    fig, axes = plt.subplots(3, 3)
    for i, col in enumerate(data.columns):
        sns.boxplot(data[col], ax = axes[i//3, i%3], linewidth=2.0)
    
    return

boxPlots(housingDF.iloc[:, :-1])
    
# OR
# sns.boxplot(data = housingDF.iloc[:, :-1], orient = "h")
    
#%%

# Plot 3 & 4 Barplot and Pie Chart

def op_plots(data, plotType):
    ocean_proximity = data["ocean_proximity"].value_counts()
    labels = ocean_proximity.index
    values = ocean_proximity.values
    if plotType == "bar":
        sns.barplot(x = labels, y = values, palette="Set2")
        plt.xlabel("Ocean Proximity")
        plt.ylabel("Count")
        plt.title("Frequency of Ocean Proximity")
    elif plotType == "pie":
         plt.pie(values, explode = (0, 0, 0, 0, 0.2), 
                 shadow = True, startangle = 90, autopct='%1.1f%%', 
                 labels = labels)
         # plt.legend(loc = "best")
    else:
        print("Invalid Plot Type")  
    
    return plt.show() 

op_plots(housingDF, "pie") # For Barplot
# op_plots(housingDF, "pie") # For Piechart



#%%

# Plot 5 Scatter Plot of population and median_income using latitude and longitude:

def scatterPlot(data):
    data.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.3, 
        s = data['population']/50, c=housingDF['median_house_value'],
        cmap=plt.get_cmap('terrain'), colorbar=True)
    
    plt.title("Population(Size) and Median House Value(Color) in California")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    
    return

scatterPlot(housingDF)

#%%

# Solution 5 : In PDF

#%%

# SOLUTION 6

# Creating a copy of the data for preprocessing"
preprocDF = housingDF.copy()

def missingvaluesChecker(data):
    mvCount = data.isnull().sum()
    return mvCount

mvCount = missingvaluesChecker(preprocDF)
# print(mvCount)

# Handling Missing Values by replacing by:
    # 1. Median for numerical values.
    # 2. Mode for categorical values.
def handleMissingValues(data, typeofImputer, strategy_for_num = str(), col_list = list()):
    for col in col_list:
        if typeofImputer == "simple":
            if data[col].dtype == float or data[col].dtype == int:
                imputeVal = SimpleImputer(strategy = strategy_for_num)
                data[col] = imputeVal.fit_transform(data[[col]]).ravel()
            elif data[col].dtype == object:
                imputeVal = SimpleImputer(strategy = "most_frequent")
                data[col] = imputeVal.fit_transform(data[[col]]).ravel()
            else: 
                print("Invalid Data Type")
        elif typeofImputer == "filler":
            if data[col].dtype == float or data[col].dtype == int:
                data[col].fillna(data[col].median(), inplace = True)
            elif data[col].dtype == object:
                data[col].fillna(data[col].mode()[0], inplace = True)
            else: 
                print("Invalid Data Type")
        else:
            print("Invalid Imputer Type")
            
    return data

preprocDF = handleMissingValues(preprocDF, "simple", "median", col_list = ["total_bedrooms"])

# To confirm if missing values are imputed.
mvCount = missingvaluesChecker(preprocDF)
# print(mvCount)


#%%

# To encode (convert to numerical) categorical values:
    
def dataEncoder(data, encoder_type, columns = list()):
    
    if encoder_type == "Label":
        encoder = preprocessing.LabelEncoder()
        for col in columns:
            data[col] = encoder.fit_transform(data[col])
    elif encoder_type == "OneHot":
        encoder = preprocessing.OneHotEncoder()
        for col in columns:
            encoded = pd.DataFrame(encoder.fit_transform(data[[col]]).toarray())
            encoded.columns = encoder.get_feature_names([col])
            data = data.iloc[:, :-1].join(encoded)
    elif encoder_type == "Ordinal":
        encoder = preprocessing.OrdinalEncoder()     
        for col in columns:
            data[col] = encoder.fit_transform(data[[col]]).ravel()
    else:
        print("Invalid Encoder Type")

    
    return data

preprocDF = dataEncoder(preprocDF, "Label", ["ocean_proximity"])

#%%

# Data Transformation
def transformData(data, t_type = "log", col_list = list()):
    for col in col_list:
        if t_type == "log":
            data[col] = np.log1p(data[col])
        elif t_type == "sqrt":
            data[col] = np.sqrt(data[col])
        elif t_type == "square":
            data[col] = np.square(data[col])
        else:
            print("Invalid Tranformation Type")
        
    return data

preprocDF = transformData(preprocDF, t_type = "log", col_list = ["population", "total_rooms", "total_bedrooms", "median_house_value", "households", "median_income"])


#%%

# Handling Outliers
def removeOutliers(data, col_list = list()):
    for col in col_list:
        Q1 = data[col].quantile(0.20) 
        Q3 = data[col].quantile(0.80) 
        IQR = Q3-Q1 
        low  = Q1-(1.5*IQR) 
        high = Q3+(1.5*IQR)
        
        data_include = data.loc[(data[col] >= low) & (data[col] <= high)] 
        data_exclude = data.loc[(data[col] < low) | (data[col] > high)]
        
        imputeVal = data_include[col].median()
        data_exclude[[col]] = imputeVal
        data = pd.concat([data_include, data_exclude]) 
        
    return data

preprocDF = removeOutliers(preprocDF, col_list = ["total_rooms", "total_bedrooms", "population", "households", "median_income", "median_house_value"])


#%%

# Scaling Data
def dataScaler(data, scaler_type, columns = list()):
    if scaler_type == "Robust":
        scaler = preprocessing.RobustScaler()
    elif scaler_type == "MinMax":
        scaler = preprocessing.MinMaxScaler()
    elif scaler_type == "Standard":
        scaler = preprocessing.StandardScaler()
    else:
        print("Invalid Scaler Type!")
        
    data[columns] = scaler.fit_transform(data[columns])
    
    return data

preprocDF = dataScaler(preprocDF, "MinMax", preprocDF.iloc[:, :-1].columns)


#%%

# Feature Scaling

X = preprocDF.loc[:, preprocDF.columns != "median_house_value"]
Y = preprocDF[["median_house_value"]].astype(int)



skb = SelectKBest(score_func=chi2, k = 9)  # using chisquared test, trial and error by changing value of K
X1 = skb.fit(X, Y)

scores = [(item, score) for item, score in zip(X.columns, skb.scores_)]

sorted(scores, key = lambda x : -x[1])[:6]

#%%

# Data Sample:
sample = preprocDF.sample(frac = 1)

#%%


# SOLUTION 7

mod_housingDF = preprocDF.copy()
mod_housingDF["room_per_house_value"] = mod_housingDF["total_rooms"]/mod_housingDF["median_house_value"]
mod_housingDF["income_per_house_value"] = housingDF["median_income"]/housingDF["median_house_value"]
mod_housingDF["population_per_income"] = housingDF["population"]/housingDF["median_income"]
mod_housingDF["household_per_house_val"] = housingDF["households"]/housingDF["median_house_value"]



new = mod_housingDF[["median_house_value","population_per_income", "room_per_house_value", "income_per_house_value", "household_per_house_val"]]   
new_corr = new.corr(method = "pearson")

heatMap(new_corr)



#%%

