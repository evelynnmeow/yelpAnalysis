#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:12:13 2019

@author: JM
"""

import requests
from bs4 import BeautifulSoup
import csv
import json
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import pprint
import time
import warnings
warnings.filterwarnings("ignore")

def combine_city(y1, y2, y3, y4, y5, y6, y7, y8, y9, y10):
    # read in data files
    yelp_boston = pd.read_csv(y1)
    yelp_chicago = pd.read_csv(y2)
    yelp_denver = pd.read_csv(y3)
    yelp_detroit = pd.read_csv(y4)
    yelp_lasvegas = pd.read_csv(y5)
    yelp_losangeles = pd.read_csv(y6)
    yelp_newyork = pd.read_csv(y7)
    yelp_philadelphia = pd.read_csv(y8)
    yelp_pittsburgh = pd.read_csv(y9)
    yelp_sanfrancisco = pd.read_csv(y10)
    
    # concat dataframes together
    df = pd.concat([yelp_chicago, yelp_denver, yelp_detroit, yelp_lasvegas,yelp_losangeles,
                    yelp_newyork, yelp_boston, yelp_philadelphia, yelp_pittsburgh, yelp_sanfrancisco], axis = 0)

    return(df)


def check_quality_yelp(df):
    
    print('The quality score of a single attribute is the sum of number of rows with '+
          'and invalid values over the number of total data points.')
    print('The overall quality score is the sum of quality scores of all attributes '+
          'over the total number of attributes.')
    print('If there is no missing values or invalid values, the quality score will be 1.')
    
    
    # get the total number of data points
    num_cols = len(df.index)
    print('Total Number of rows is ')
    print(num_cols)
    df.drop_duplicates()
    print(len(df.index))
    # check 'name' column
    # checking missing value
    na_name = df[df['Name'] == 'na']
    number_missing_name = len(na_name.index)
    print('Number of rows with missing values in Name:')
    print(number_missing_name)
    # thus there is no missing value in this column
    
    # check ID column
    # check na
    na_id = df[df['ID'] == 'na']
    number_missing_id = len(na_id.index)
    print('Number of rows with missing values in ID:')
    print(number_missing_id)
    # check invalid ID value
    incorrect_id_len = df['ID'].apply(len)
    id_len_df = incorrect_id_len.to_frame()
    print('Length IDs are')
    print(id_len_df['ID'].unique())
    # thus there is no missing or invalid value
    
    # check rating col
    # check missing values
    na_rating = df[df['Rating'] == 'na']
    number_missing_rating = len(na_rating.index)
    print('Number of rows with missing values in Rating:')
    print(number_missing_rating)
    # check invalid
    # rating should be greater than 0
    negative_rating = df[df['Rating'] < 0]
    number_negative_rating = len(negative_rating.index)
    print('Number of rows with negative values in Rating:')
    print(number_negative_rating)
    # rating should be no more than 5
    invalid_rating = df[df['Rating'] > 5]
    number_invalid_rating = len(invalid_rating.index)
    print('Number of rows with Rating greater than 5:')
    print(number_invalid_rating)
    # thus there is no missing value or invalid value
    
    # check price level
    # check missing values
    whether_na_price = df['Price Level'].isna()
    whether_na_price_df = whether_na_price.to_frame()
    na_price = whether_na_price[whether_na_price_df['Price Level'] == True]
    number_missing_price = len(na_price)
    print('Number of rows with missing values in Price Level:')
    print(number_missing_price)
    # define quality score for price level
    score_price = (num_cols - number_missing_price) / num_cols
    print('Quality score of Price Level:')
    print(score_price)
    
    # check is closed
    # check missing value
    na_closed = df[df['Is Closed'].isna()]
    number_missing_closed = len(na_closed.index)
    print('Number of rows with missing values in Is Closed:')
    print(number_missing_closed)
    # check invalid
    invalid_close = df['Is Closed'].dtypes.name == 'bool'
    print('Are all values in Is Closed column boolean?') 
    print(invalid_close)
    
    # check category
    na_category = df[df['Category'].isna()]
    number_missing_category = len(na_category.index)
    print('Number of rows with missing values in Category:')
    print(number_missing_category)
    print('Since there is no specific criteria for what kind of category is valid, '+
          'we consider all categories to be valid.')
    
    """
    Since there is no specific criteria for what kind of category is valid,
    we consider all categories to be valid.
    
    """
    
    # check review count
    # check missing value
    na_review_count = df[df['Review Count'].isna()]
    number_missing_review_count = len(na_review_count.index)
    print('Number of rows with missing values in Review Count:')
    print(number_missing_review_count)
    # check invalid
    invalid_count_review = df[df['Review Count'] < 0]
    print('Number of rows with negative values in Review Count:')
    number_invalid_review_count = len(invalid_count_review.index)
    print(number_invalid_review_count)
    
    # check latitude
    # check missing values
    na_lat = df[df['Latitude'].isna()]
    number_missing_lat = len(na_lat.index)
    print('Number of rows with missing values in Latitude:')
    print(number_missing_lat)
    # check valid latitude
    max_lat = df['Latitude'].max()
    print('Maximum latitude')
    print(max_lat)
    min_lat = df['Latitude'].min()
    print('Minimum latitude')
    print(min_lat)
    # thus there is no invalid latitude value
    
    # check longitude
    # check missing value
    na_long = df[df['Longitude'].isna()]
    number_missing_long = len(na_long.index)
    print('Number of rows with missing values in Longitude:')
    print(number_missing_long)
    # check valid latitude
    max_long = df['Longitude'].max()
    print('Maximum longitude')
    print(max_long)
    min_long = df['Longitude'].min()
    print('Minimum longitude')
    print(min_long)
    # thus there is no invalid longitude value
    
    # check address 
    na_address = df[df['Address'].isna()]
    number_missing_address = len(na_address.index)
    print('Number of rows with missing values in Address:')
    print(number_missing_address)
    print('Since there is no specific criteria for what kind of address is valid, '+
          'we consider all categories to be valid.')
    
    """
    
    Since there is no specific criteria for what kind of category is valid,
    we consider all categories to be valid.
    
    """
    
    # check city
    # check missing value
    na_city = df[df['City'].isna()]
    number_missing_city = len(na_city.index)
    print('Number of rows with missing values in City:')
    print(number_missing_city)
    # check invalid
    print('Unique values of cities is')
    print(df['City'].unique())
    # define a desire city list
    city_list = ['New York', 'Chicago', 'Los Angeles', 'Pittsburgh', 'Boston', 
                 'Detroit', 'Philadelphia', 'Denver', 'San Francisco', 'Las Vegas']
    # check valid city 
    valid_city = df[df['City'].isin(city_list)]
    # number of valid city
    num_valid_city = len(valid_city.index)
    print('Number of rows with invalid values in City:')
    print(num_cols - num_valid_city)
    # define score for city
    score_city = (num_valid_city) / num_cols
    print('Quality score of City:')
    print(score_city)
    
    
    # check state
    # check missing value
    na_state = df[df['State'].isna()]
    number_missing_state = len(na_state.index)
    print('Number of rows with missing values in State:')
    print(number_missing_state)
    # check invalid
    uniq_state_list = df['State'].unique()
    print('Unique State list')
    print(uniq_state_list)
    # define a desired state list
    state_list = ['NY', 'IL' ,'CA' ,'PA' ,'MA', 'MI' ,'CO' ,'NV']
    # check valid state 
    valid_state = df[df['State'].isin(state_list)]
    # number of valid state
    num_valid_state = len(valid_state.index)
    print('Number of rows with invalid values in State:')
    print(num_cols - num_valid_state)
    # define score for state
    score_state = (num_valid_state) / num_cols
    print('Quality score of State:')
    print(score_state)
    
    # check zip code
    na_zip = df[df['Zipcode'].isna()]
    number_missing_zip = len(na_zip.index)
    print('Number of rows with missing values in Zip Code:')
    print(number_missing_zip)
    # check the length of zip code
    len_of_valid_zip = len(df['Zipcode'].astype(str).map(len) == 5)
    print('Whether zip code values are valid')
    print(len_of_valid_zip == num_cols)
    
    # check country code
    # check missing value
    na_country = df[df['Country'].isna()]
    number_missing_country = len(na_country.index)
    print('Number of rows with missing values in Country:')
    print(number_missing_country)
    uniq_country_code = df['Country'].unique()
    print('Unique country list')
    print(uniq_country_code)
    # define desired country list
    country_list = ['US']
    # check valid country 
    valid_country = df[df['Country'].isin(country_list)]
    # number of valid country
    num_valid_country = len(valid_country.index)
    print('Number of rows with invalid values in Country:')
    print(num_cols - num_valid_country)
    # define score for country
    score_country = (num_valid_country) / num_cols
    print('Quality score of Country:')
    print(score_country)
    
    # check transaction type
    na_transaction = df[df['Transaction Type'].isna()]
    number_missing_transaction = len(na_transaction.index)
    print('Number of rows with missing values in Transaction Type:')
    print(number_missing_transaction)
    print('Since the transaction type is any additional features of the business, '+
          'empty list in this case makes sense.'+
          'Thus, we will not consider this as data problems.')
    
    # check comment 1
    # check missing value
    na_comment1 = df[df['Comment 1'].isna()]
    number_missing_comment1 = len(na_comment1.index)
    print('Number of rows with missing values in Comment 1:')
    print(number_missing_comment1)
    
    # check comment 2
    # check missing value
    na_comment2 = df[df['Comment 2'].isna()]
    number_missing_comment2 = len(na_comment2.index)
    print('Number of rows with missing values in Comment 2:')
    print(number_missing_comment2)
    
    # check comment 3
    # check missing value
    na_comment3 = df[df['Comment 3'].isna()]
    number_missing_comment3 = len(na_comment3.index)
    print('Number of rows with missing values in Comment 3:')
    print(number_missing_comment3)
    
    # define total quality score
    total_score = ((score_price + score_city + 
                score_state + score_country + (num_cols - 4) * 1) / num_cols )
    print('The total quality score is')
    print(total_score)



def clean_yelp(df):
    
                 
    # data cleaning
    # drop the missing values in the dataset
    df = df.dropna()
    
    
    # define a desired city list
    # define a desired city list
    city_list = ['New York', 'Chicago', 'Los Angeles', 'Pittsburgh', 'Boston', 
                 'Detroit', 'Philadelphia', 'Denver', 'San Francisco', 'Las Vegas']
    # drop any data that does not have correct value in city
    valid_city = df[df['City'].isin(city_list)]
    
    # define a desired state list
    # define a desired state list
    state_list = ['NY', 'IL' ,'CA' ,'PA' ,'MA', 'MI' ,'CO' ,'NV']
    # check valid state 
    valid_state = valid_city[valid_city['State'].isin(state_list)]
    
    # define a desired country list
    country_list = ['US']
    # check valid country 
    valid_country = valid_state[valid_state['Country'].isin(country_list)]
    return(valid_country)













    

def main():
    raw_data = combine_city('Yelp_Boston.csv', 'Yelp_Chicago.csv',
                            'Yelp_Denver.csv', 'Yelp_Detroit.csv',
                            'Yelp_Las Vegas.csv', 'Yelp_Los Angeles.csv',
                            'Yelp_New York.csv', 'Yelp_Philadelphia.csv',
                            'Yelp_Pittsburgh.csv', 'Yelp_San Francisco.csv')
    print('\n' + '========Quality Report before cleaning========' )
    check_quality_yelp(raw_data)
    print('\n' + '========End of Quality Report before cleaning========')
    clean_data = clean_yelp(raw_data)
    clean_data.to_csv('clean_yelp.csv', index = False)
    print('\n' + '========Quality Report after cleaning========' )
    score_after = check_quality_yelp(clean_data)
    print('\n' + '========End of Quality Report after cleaning========')
    print('=========The data is clean========')

if __name__ == '__main__':
    main()
