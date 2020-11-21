#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:32:54 2019

@author: JM

"""

# Hypothesis testing 
# anova test
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

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split




# this function conducts anova test for the yelp dataset
# it takes the yelp dataset as input
# and prints out the anova test result for each attribute
def anova_test_for_yelp(yelp):
    # print out anova test for Review Count for each city
    anova_review_count_by_cities = stats.f_oneway(list(yelp['Review Count'][yelp['City'] == 'Boston']), 
          list(yelp['Review Count'][yelp['City'] == 'Chicago']),
          list(yelp['Review Count'][yelp['City'] == 'Detroit']),
          list(yelp['Review Count'][yelp['City'] == 'Denver']),
          list(yelp['Review Count'][yelp['City'] == 'Las Vegas']),
          list(yelp['Review Count'][yelp['City'] == 'Los Angeles']),
          list(yelp['Review Count'][yelp['City'] == 'New York']),
          list(yelp['Review Count'][yelp['City'] == 'Philadelphia']),
          list(yelp['Review Count'][yelp['City'] == 'Pittsburgh']),
          list(yelp['Review Count'][yelp['City'] == 'San Francisco']))
    print('*********************************')
    print('Anova Result of Review Count for Each City')
    print(anova_review_count_by_cities)
    
    # print out anova test for Price Level (remember to change back to price level)
    anova_price_level_by_cities = stats.f_oneway(list(yelp['Price Level'][yelp['City'] == 'Boston']), 
          list(yelp['Price Level'][yelp['City'] == 'Chicago']),
          list(yelp['Price Level'][yelp['City'] == 'Detroit']),
          list(yelp['Price Level'][yelp['City'] == 'Denver']),
          list(yelp['Price Level'][yelp['City'] == 'Las Vegas']),
          list(yelp['Price Level'][yelp['City'] == 'Los Angeles']),
          list(yelp['Price Level'][yelp['City'] == 'New York']),
          list(yelp['Price Level'][yelp['City'] == 'Philadelphia']),
          list(yelp['Price Level'][yelp['City'] == 'Pittsburgh']),
          list(yelp['Price Level'][yelp['City'] == 'San Francisco']))
    
    print('*********************************')
    print('Anova Result of Price Level for Each City')
    print(anova_price_level_by_cities)
    
    # print out anova test for rating for each city
    anova_rating_each_cities = stats.f_oneway(list(yelp['Rating'][yelp['City'] == 'Boston']), 
          list(yelp['Rating'][yelp['City'] == 'Chicago']),
          list(yelp['Rating'][yelp['City'] == 'Detroit']),
          list(yelp['Rating'][yelp['City'] == 'Denver']),
          list(yelp['Rating'][yelp['City'] == 'Las Vegas']),
          list(yelp['Rating'][yelp['City'] == 'Los Angeles']),
          list(yelp['Rating'][yelp['City'] == 'New York']),
          list(yelp['Rating'][yelp['City'] == 'Philadelphia']),
          list(yelp['Rating'][yelp['City'] == 'Pittsburgh']),
          list(yelp['Rating'][yelp['City'] == 'San Francisco']))
    print('*********************************')
    print('Anova Result of Rating for Each City')
    print(anova_rating_each_cities)

# this function select attributes for the linear regression model and get the
# dataframe ready for linear regression model
# it takes the yelp dataset as input
# and returns the dataframe ready to run the linear regression
def yelp_model_dataframe(yelp):
    # predict rating in the yelp dataset by price level, review count, city, variety, number of transaction type
    yelp_model_df = yelp[['Rating', 'Price Level', 'Review Count', 'City', 'Variety', 'Num Transaction Type', 
                          'Num A&T TM', 'Num Music TM', 'Num Sports TM', 'Num Misc TM',
                          'Num Music EB', 'Num Perform EB', 'Num Sports EB']]
    ## change column names of the dataframe to apply models
    yelp_model_df.columns = ['Rating', 'PriceLevel', 'ReviewCount', 'City', 'Variety', 'NumTransactionType',
                             'NumATM', 'NumMusicTM', 'NumSportsTM', 'NumMiscTM',
                             'NumMusicEB', 'NumPerformEB', 'NumSportsEB']
    
    return(yelp_model_df)

# this function runs linear regression on selected attributes of yelp data
# it takes the selected attributes of yelp
# and prints out the summary table of linear regression model
def linear_regression_for_yelp(yelp_model_df):
    # run the ordinary least square regression
    yelp_lm = smapi.ols('ReviewCount ~ PriceLevel + Rating + City + Variety + NumTransactionType + NumATM + NumMusicTM + NumSportsTM + NumMiscTM + NumMusicEB + NumPerformEB + NumSportsEB' , data = yelp_model_df).fit() 
    # print out the linear regression summary
    print(yelp_lm.summary())
    
# define main 
def main():
    # read in the data
    yelp = pd.read_csv('yelp_tm_eb.csv')
    # do log transformation on the review count 
    yelp['Log Review'] = np.log(yelp['Review Count'])
    # call function for anova test
    anova_test_for_yelp(yelp)
    # modify the dataframe
    select_feature_dataframe = yelp_model_dataframe(yelp)
    # run linear regression
    linear_regression_for_yelp(select_feature_dataframe)
    
# call main
if __name__ == '__main__':
    main()
    
    




     
               








