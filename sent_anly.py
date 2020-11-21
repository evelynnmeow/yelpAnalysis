#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 07:56:51 2019

@author: JM
"""

# sentimental analysis yelp reviews

# import useful libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import nltk
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# this function preprocesses the textual comment
def preprocess_comment(comments):
    # remove puntuations
    re_punc =[char for char in comments if char not in string.punctuation]

    # join chars back
    re_punc = ''.join(re_punc)
    # remove stop words
    for word in re_punc.split():
        if word.lower() not in stopwords.words('english'):
            return word



# the function takes in the dataframe of yelp dataset
# do the sentimental analysis
# and print out result
def sent_anly_NB(yelp_tm_eb):
    
    # for the capacity of python and reduce the run time of the program
    # we will choose the first column of reviews in the dataframe
    yelp = yelp_tm_eb[['ID', 'Rating', 'Price Level', 'Comment 1']]
    # round down the rating for consistency
    yelp['Round Rating'] = yelp['Rating'].astype(int)
                
    # we are going to predict rating (class) of a store based on the 
    # sentimental of the testual comments
    X = yelp['Comment 1']
    Y = yelp['Round Rating']
    # convert word to vector
    vocab_X = CountVectorizer(analyzer= preprocess_comment).fit(X)
    X = vocab_X.transform(X)
    
    # split test and training set
    # we will use 20% of the data as test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1005)
    # using Naive Bayes Classifier
    NB = MultinomialNB()
    # fit the model on training set
    NB.fit(X_train,Y_train)
    # predict on test set
    preds_NB = NB.predict(X_test)
    # print out confusion matrix and classification report
    print('**********Confusion Matrix of Classification**********')
    print(confusion_matrix(Y_test, preds_NB))
    print('\n')
    print('**********Classification Report of Classification*********')
    print(classification_report(Y_test, preds_NB))
    
    
    

        
        
    

# define main
def main():
    # read in the data
    yelp_tm_eb = pd.read_csv('yelp_tm_eb.csv')
    sent_anly_NB(yelp_tm_eb)

# call main
if __name__ == '__main__':
    main()











