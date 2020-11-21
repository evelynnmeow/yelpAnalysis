#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:12:59 2019

@author: JM
"""

import csv
import json
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import pprint
import time
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import metrics
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA 
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
import statsmodels.formula.api as smapi
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
import warnings
import re
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc





# this function conducts 5 classification models 
# it takes the yelp dataset as input
# and prints result for both training and test sets
def classification_models(yelp):
    # select attributes to use in the models
    yelp_model_df = yelp[['Rating', 'Price Level', 'Log Review', 'City','Variety', 'Num Transaction Type',
                          'Num A&T TM', 'Num Music TM', 'Num Sports TM', 'Num Misc TM',
                          'Num Music EB', 'Num Perform EB', 'Num Sports EB']]
    # cast rating and city to category varaible for future use
    yelp_model_df['Rating'] = yelp_model_df['Rating'].astype('category').cat.codes
    yelp_model_df['City'] = yelp_model_df['City'].astype('category').cat.codes
    
    # change values to array
    value_arr = yelp_model_df.values
    # get predictors
    origin_X = value_arr[:,1:13]
    # normalize X
    X = preprocessing.normalize(origin_X)
    Y = value_arr[:, 0].astype('float')
    
    # use 20% of the data as test dataset
    test_size = 0.20
    # set seed
    seed = 5
    # split training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state=seed)
    
    # use 10 fold cross validation
    num_folds = 10
    # set random seed
    seed = 7
    # define scoring method
    scoring = 'accuracy'
    
    # define a list of model
    models = []
    # append models into the list
    # gaussian assumes normality
    models.append(('Naive Bayes', GaussianNB()))
    models.append(('SVM with linear Kernel', SVC(kernel = 'linear')))
    models.append(('Decision Tree', DecisionTreeClassifier()))
    # number of trees for each forest is 500 and the random state is fixed at seed 5
    models.append(('Random Forest', RandomForestClassifier(n_estimators = 500, 
                  max_depth = 2,random_state = 5)))
    models.append(('KNN', KNeighborsClassifier(n_neighbors = 100)))
    
    print('****************** Results for training set ****************************')
    # initialize result and name
    results = []
    names = []
    # get cross validation error for all models on training set
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        cv_results = cross_val_score(model, X_train , Y_train, 
                                     cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        # print the result out
        print('Results for training sets')
        print(msg)
        
    
    # for test sets
    print('****************** Results for test set ****************************')
    for name, model in models:
        # fit the model using training data
        model.fit(X_train,Y_train)
        # get predictions on test set
        predictions = model.predict(X_test)
        # print out result
        print(name)
        print('The accuracy score for ' + name + ' is: ')
        print(accuracy_score(Y_test, predictions))
        # print out confusion matrix
        print('The confusion matrix for ' + name)
        print(confusion_matrix(Y_test, predictions))
        # print(classification_report(Y_test, predictions))

# define main function
def main():
    # read in the yelp data with new features
    yelp = pd.read_csv('yelp_tm_eb.csv')
    # do log transformation for the Review Count variable since Gaussian Naive Bayes
    # assumes normality of predictors.
    yelp['Log Review'] = np.log(yelp['Review Count'])
    # call function to get training and test results
    classification_models(yelp)

# call main function
if __name__ == '__main__':
    main()
    
    
    



