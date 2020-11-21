#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:15:31 2019

@author: JM
"""

from itertools import cycle
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
import pylab as pl
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
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp

# online sources used for this part
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# https://stackoverflow.com/questions/37017400/sklearn-metrics-roc-curve-for-multiclass-classification


# this function generates ROC curve of each model for each classes
# it takes the yelp dataset as input
# and plots ROC curve for different models
def ROC_curves(yelp):
    yelp_model_df = yelp[['Rating', 'Price Level', 'Log Review', 'City','Variety', 'Num Transaction Type',
                          'Num A&T TM', 'Num Music TM', 'Num Sports TM', 'Num Misc TM',
                          'Num Music EB', 'Num Perform EB', 'Num Sports EB']]
        
    # cast the rating attribute to categgory
    yelp_model_df['Rating'] = yelp_model_df['Rating'].astype('category').cat.codes
    yelp_model_df['City'] = yelp_model_df['City'].astype('category').cat.codes
        
     # change values to array
    value_arr = yelp_model_df.values
    # get predictors
    origin_X = value_arr[:,1:12]
    # normalize predictors
    X = preprocessing.normalize(origin_X)
    # binarize the response variable to generate ROC curve for each classes
    Y = label_binarize(value_arr[:, 0], classes = [0, 1, 2, 3, 4, 5, 6])
    
    # split training and test sets 
    # use 20% of the data as test dataset
    test_size = 0.20
    seed = 5
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state=seed)
    # define number of classes
    n_classes = len(yelp_model_df['Rating'].unique()) 
    
    # roc curve for random forest
    classifier = RandomForestClassifier(n_estimators = 500, 
                                        max_depth = 2,random_state = 5)
    print('************ ROC Curve for Random Forest*******************')
    y_score = classifier.fit(X_train, Y_train).predict(X_test)
    # initialize fpr, tpr, and roc_auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # for loop to run through each classes
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # plot ROC curve 
    lw = 2
    colors = cycle(['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink' ])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw = lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1],'k--',lw = lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for random forest')
    plt.legend(loc="lower right")
    plt.savefig('ROC curve for Random Forests.png')
    plt.show()
    
    
    # roc curve for decision tree
    classifier = DecisionTreeClassifier()
    print('************ ROC Curve for Decision Tree*******************')
    y_score = classifier.fit(X_train, Y_train).predict(X_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # plot ROC curves for each classes
    lw = 2
    colors = cycle(['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink' ])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw = lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw = lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for Decision Tree')
    plt.legend(loc="lower right")
    plt.savefig('ROC curve for Decision Tree.png')
    plt.show()
    
    
    
     
    # roc curve for svm
    classifier = OneVsRestClassifier(SVC(kernel='linear', 
                     random_state=5))
    print('************ ROC Curve for SVM*******************')
    y_score = classifier.fit(X_train, Y_train).decision_function(X_test)
    
    # initialize fpr, tpr, roc_auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # plot the ROC curve for each classes
    lw = 2
    colors = cycle(['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink' ])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw = lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw = lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for SVM with linear kernel')
    plt.legend(loc="lower right")
    plt.savefig('ROC curve for SVM with linear kernel.png')
    plt.show()
    
    
    
    
    
    # roc curve for knn
    classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 100))
    print('************ ROC Curve for KNN*******************')
    y_score = classifier.fit(X_train, Y_train).predict_proba(X_test)
    
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # plot ROC curve for each class
    lw = 2
    colors = cycle(['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink' ])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color = color, lw = lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw = lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for KNN (N = 100)')
    plt.legend(loc="lower right")
    plt.savefig('ROC curve for KNN.png')
    plt.show()
    
    
    
    
    # roc curve for knn
    classifier = OneVsRestClassifier(GaussianNB())
    print('************ ROC Curve for Naive Bayes*******************')
    y_score = classifier.fit(X_train, Y_train).predict_proba(X_test)
    
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # plot ROC curve for each class
    lw = 2
    colors = cycle(['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink' ])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw = lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw = lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for Naive Bayes (Gaussian)')
    plt.legend(loc="lower right")
    plt.savefig('ROC curve for Naive Bayes.png')
    plt.show()
    
# define main   
def main():
    yelp = pd.read_csv('yelp_tm_eb.csv')
    # do log transformation on the review count 
    yelp['Log Review'] = np.log(yelp['Review Count'])
    ROC_curves(yelp)

# call main
if __name__ == '__main__':
    main()
    
