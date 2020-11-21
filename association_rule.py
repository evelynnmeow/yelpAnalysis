#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:40:32 2019

@author: JM
"""






import pandas as pd

import numpy as np
import pprint
import time
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
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
from apyori import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import re

# online sources used for this part
# http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/


# read in the yelp dataset
yelp_data = pd.read_csv('yelp_tm_eb.csv')


# this function modift certain columns to run association rule
# it takes the yelp dataset as input
# and returns the dataframe after modification that is ready for association 
# rule mining
def modify_columns(yelp):
    
    ### Convert the Category column to lists of categories for each restaurant (instead of string)
    # use this list to store the flattened categories in each row
    flatten_categories = []
    # store the Category column separately for easy iteration
    cat = yelp['Category']
    for c in cat:
        match = re.findall(r'\'(.+?)\'', c) # use regular expression to match everything inside '', i.e. a category, in a non greedy way
        flatten_categories.append(match)
    yelp['Category'] = flatten_categories
    
    
    ### Create a new column containing the count of categories each restaurant has
    # yelp['Variety'] = [len(c) for c in yelp['Category']]
    
    
    ### Convert the Price Level column to numbers instead of $ signs
    #yelp['Price Level'] = [len(p) for p in yelp['Price Level']]
    
    
    ### Convert the Transaction Type column to lists for each restaurant (instead of string)
    # similar to what we did for Category at the beginning
    flatten_transtype = []
    transtype = yelp["Transaction Type"]
    for t in transtype:
        # if there is no additional transaction type, we append an empty list
        if t == '[]': 
            flatten_transtype.append([])
        else:
            match = re.findall(r'\'(.+?)\'', t) # use regular expression to match everything inside '', i.e. a transaction type, in a non greedy way
            flatten_transtype.append(match)
    yelp["Transaction Type"] = flatten_transtype
    
    
    ### Create a new column containing the count of transaction types
    # similar to Variety above
    ## for each restaurants, they all have dine in as underlying feature
    ## so aside from additional transaction types like delivery or take out, we add 1 for each restaurant
    yelp["Num Transaction Type"] = [len(t)+1 for t in yelp["Transaction Type"]]
    return (yelp)

# this function conducts association rule mining for the yelp dataset
# it takes a ticketmaster dataframe as input
# and prints out the most frequent itemsets with more than one items 
# and its support
def yelp_association_rule(yelp):
    # make the attribute 'Category' as a list
    category_list = yelp['Category'].values.tolist()
    # give the transactionbencoder a name for easy access
    TransEncode = TransactionEncoder()
    # apply the encoder to the category list
    TE_arr =  TransEncode.fit(category_list).transform(category_list)
    # change the list to a transaction dataframe
    category_df = pd.DataFrame(TE_arr, columns = TransEncode.columns_)
    # define a list of supports
    sup_list = [0.01, 0.02,0.03]
    # run through each support in the support list
    for sup in sup_list:
        # apply apriori method
        freq_sets = apriori(category_df, min_support = sup, use_colnames = True)
        freq_sets['Length'] = freq_sets['itemsets'].apply(lambda x: len(x))
        # get the most frequent itemset with length greater than 1
        if sup == max(sup_list):
            most_freq_sets = freq_sets[(freq_sets['Length'] >= 2)]
            # print out results
            print('********************')
            print('Most Frequent Set for Yelp Category' )
            print(most_freq_sets['itemsets'])
            print('Support of most frequent set is ')
            print(most_freq_sets['support'])


# this function conducts the first association rule mining 
# for the eventbrite dataset
# it takes a eventbrite dataframe with some selected attributes as input
# and prints out the most frequent itemsets with more than one items 
# and its support
def eb_set1_association(eb_subset):

    # convert the dataframe to a list
    eb_list = eb_subset.astype(str).values.tolist()
    
    # encode the list to true or false transaction types
    # give a name to the encoder for easy access
    TransEncode = TransactionEncoder()
    # encode the list
    TE_arr =  TransEncode.fit(eb_list).transform(eb_list)
    # change the list to a dataframe
    eb_df = pd.DataFrame(TE_arr, columns = TransEncode.columns_)
    
    
    # define a list of min support
    sup_list = [0.03, 0.05, 0.08]
    # apply apriori
    for sup in sup_list:
        freq_sets = apriori(eb_df, min_support = sup, use_colnames = True)
        freq_sets['Length'] = freq_sets['itemsets'].apply(lambda x: len(x))
        # get the most frequent itemset
        if sup == max(sup_list):
            most_freq_sets = freq_sets[(freq_sets['Length'] >= 2) &
                                      (freq_sets['support'] >= 0.09)]
            # print out the result
            print('********************')
            print('Most Frequent Set for Eventbrite Category' )
            print(most_freq_sets['itemsets'])
            print('Support of most frequent set is ')
            print(most_freq_sets['support'])
    

# this function conducts the second association rule mining 
# for the eventbrite dataset
# it takes a eventbrite dataframe with some selected attributes as input
# and prints out the most frequent itemsets with more than one items 
# and its support
def eb_set2_association(eb_df1):    
    
    # define a list of min support
    sup_list = [0.03, 0.05, 0.08]
    # apply apriori
    for sup in sup_list:
        freq_sets = apriori(eb_df1, min_support = sup, use_colnames = True)
        freq_sets['Length'] = freq_sets['itemsets'].apply(lambda x: len(x))
        # get the most frequent pattern
        if sup == max(sup_list):
            most_freq_sets = freq_sets[(freq_sets['Length'] >= 2) &
                                      (freq_sets['support'] >= 0.1)]
            print('********************')
            print('Most Frequent Set for Eventbrite Availability' )
            print(most_freq_sets['itemsets'])
            print('Support of most frequent set is ')
            print(most_freq_sets['support'])
                  

# this function conducts association rule mining for the ticketmaster dataset
# it takes a ticketmaster dataframe with some selected attributes as input
# and prints out the most frequent itemsets with more than one items 
# and its support
def tm_association(tm_subset):                           
    # get a list of items
    tm_list = tm_subset.astype(str).values.tolist()
    # define the encoder
    TransEncode = TransactionEncoder()
    # transform the list to a transaction list
    TE_arr1 =  TransEncode.fit(tm_list).transform(tm_list)
    # change the list to a dataframe
    tm_df = pd.DataFrame(TE_arr1, columns = TransEncode.columns_)
    # define a list of min support
    sup_list = [0.03, 0.05, 0.08]
    # apply apriori
    for sup in sup_list:
        freq_sets = apriori(tm_df, min_support = sup, use_colnames = True)
        freq_sets['Length'] = freq_sets['itemsets'].apply(lambda x: len(x))
        # get the most frequent pattern
        if sup == max(sup_list):
            most_freq_sets = freq_sets[(freq_sets['Length'] >= 2) &
                                      (freq_sets['support'] >= 0.25)]
            print('********************')
            print('Most Frequent Sets for TicketMaster')
            print(most_freq_sets)



# define main
def main():
    # modify columns for applying association rule mining
    yelp = modify_columns(yelp_data)
    # read in the data for eventbrite
    eventbrite = pd.read_csv('clean_eventbrite.csv')
    # choose subset
    eb_subset1 = eventbrite[['Format', 'Category', 'Subcategory', 'Venue Name', 'City']]
    eb_subset2 = eventbrite[['has Available Tickets', 'is Online', 'is Free']]
    # read in the data
    ticketmaster = pd.read_csv('clean_tm.csv')
    # select subset of ticketmaster for further use
    tm_subset1 = ticketmaster[['Segment', 'Genres', 'subGenres', 
                           'Venue Name', 'Venue City Name']]
    # run association rule mining for the yelp dataset
    yelp_association_rule(yelp)
    # run the first association rule mining for the eventbrite dataset
    eb_set1_association(eb_subset1)
    #run the second association rule mining for the eventbrite dataset
    eb_set2_association(eb_subset2)
    # run association rule mining for the ticketnaster dataset
    tm_association(tm_subset1)

# call main
if __name__ == '__main__':
    main()


























