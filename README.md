# Machine-Learning-and-Data-Science
This repository consists of hard-coded machine learning algorithms (supervised learning) and a comparison to the algorithms provided by the Scikit-Learn Package in Python.
It also contains unsupervised learning algorithms for clustering and color quantization (using sklearn and scipy).

It consists of the following (as of 4/11/2020):

  1. Gradient Descent 
  2. Linear Regression using Closed Form
  3. Logistic Regression (using Gradient Descent) and its comparision with Logistic Regression using sklearn.
  4. Naive Bayes (sklearn)
  5. K-Means Clustering for Color Quantization (using sklearn) 
  6. Hierarchical Clustering [Single Linkage, Complete Linkage and Average Linkage] (using Scipy)
  7. Data Cleaning and Preprocessing and Analysis using the following techniques:
    a. Summary Statistics
    b. Treating Missing Values using SimpleImputer and/or fillna.
    c. Data Encoding using Label Encoder / Ordinal Encoder / One-Hot Encoder.
    d. Data Transformations using log, sqrt etc.
    e. Outlier Detection and Imputation using Boxplots (to detect) and Formulaic (Quartiles, IQR etc) for Imputation
    f. Data Scaling using MinMax Scaler / Robust Scaler / Standard Scaler
    g. Visualizations using Boxplots, Scatterplots, Distplots, Pie Charts and HeatMaps (using matplotlib and seaborn)
    h. Feature Selection using Select-k-best
    i. New Feature Extraction using Pandas and Analysis of Data.
    
The dataset I used for the task 7 is the california housing dataset, however, these functions can be used for any dataset for preprocessing.
