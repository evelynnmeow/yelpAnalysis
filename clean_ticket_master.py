#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:32:23 2019

@author: JM
"""

"""
This file cleans the raw data of the ten selected cities from ticket master.

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


def check_quality_tm(tm):
    
    # read in the data
    df = pd.read_csv(tm, index_col = 0)
    # remove the unnamed col name
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print('The quality score of a single attribute is the sum of number of rows with '+
          'and invalid values over the number of total data points.')
    print('The overall quality score is the sum of quality scores of all attributes '+
          'over the total number of attributes.')
    print('If there is no missing values or invalid values, the quality score will be 1.')
    # check cleanliness of the data
    # check columns that contain missing values
    cols_with_na = df.columns[df.isna().any()].tolist()
    print('Columns with missing values is ')
    print(cols_with_na)
    
    # get the total number of data points
    num_cols = len(df.index)
    print('Total Number of rows is ')
    print(num_cols)
    df.drop_duplicates()
    print(len(df.index))
    # add fractions to the dataframe
    # na_counts_df['NaN Fraction'] = na_counts_df.iloc[0:] / num_cols
    # print(na_counts_df)
    """
    there are certain venus that do not have info, note, parking info and
    accessibility, thus we do not remove nas in those columns
    we will check nas in min price and max price later.
    
    """
    
    # check incorrect values
    # for the ID column, the length of each id should be 13, 14, 15, or 16, 17, 18 characters, 
    # with both letters and numbers
    incorrect_id_len = df['ID'].apply(len)
    id_len_df = incorrect_id_len.to_frame()
    print('Unique values of IDs are')
    print(id_len_df.ID.unique())
    # Thus there is no incorrect values in this column
    
    # check the incorrect value of min price and max price
    # first drop nas in the columns
    
    # check min price nas
    na_min_price = df[df['Min Price'] == 'na']
    # check max nas
    na_max_price = df[df['Max Price'] == 'na']
    # check whether the length of the two dataframe are the same
    print('Whether min price missing and max price missing happen simultaneously' )
    print(len(na_min_price) == len(na_max_price))
    # thus the missing value of min price and max price will be simutaneous. 
    
    # get number of rows that do not have nas in min price
    min_price_na_rows = len(na_min_price.index)
    # create the score of min price
    min_price_score = (num_cols - min_price_na_rows) / num_cols
    print('The quality score of min price is')
    print(min_price_score)
    # get number of rows have nas in max price
    max_price_na_rows = len(na_min_price.index)
    # create the score of min price
    max_price_score = (num_cols - max_price_na_rows) / num_cols
    print('The quality score of max price is')
    print(max_price_score)
    
    # check invalid max price
    # we will remove rows with na values
    drop_na_price = df[df['Max Price'] != 'na']
    drop_na_price['Max Price'] = drop_na_price['Max Price'].astype(float)
    invalid_max_price = drop_na_price[drop_na_price['Max Price'] < 0]
    num_invalid_max_price = len(invalid_max_price.index)
    print('Number of invalid max price is')
    print(num_invalid_max_price)
    # so there is no invalid max price
    
    # check invalid min price
    # we will remove rows with na values
    drop_na_price = df[df['Min Price'] != 'na']
    drop_na_price['Min Price'] = drop_na_price['Min Price'].astype(float)
    invalid_min_price = drop_na_price[drop_na_price['Min Price'] < 0]
    num_invalid_min_price = len(invalid_min_price.index)
    print('Number of invalid min price is')
    print(num_invalid_min_price)
    # so there is no invalid min price
    
    
    # check the length of venue id
    # The desired length should be 10, 11, 12
    incorrect_venue_id_len = df['Venue ID'].apply(len)
    unique_venue_id = incorrect_venue_id_len.to_frame()
    # get unique value of the length
    print('Unique of length of Venue ID')
    print(unique_venue_id['Venue ID'].unique())
    # Thus there is no incorrect value
    
    # check zip code
    print(type(df['Venue Zip Code'][1]))
    # so the type of a single zip code cell is int
    # check the length of zip code
    len_of_valid_zip = len(df['Venue Zip Code'].astype(str).map(len) == 5)
    print('Whether zip code values are valid')
    print(len_of_valid_zip == num_cols)
    # so zip codes are valid
    
    # check venue city name
    uniq_city_list = df['Venue City Name'].unique()
    print('Unique city name list')
    print(uniq_city_list)
    # so there is no invalid city name
    
    # check venue state name
    uniq_state_list = df['Venue State Name'].unique()
    print('Unique State name list')
    print(uniq_state_list)
    # so there is no invalid state name
    
    # check venue state code
    uniq_state_code_list = df['Venue State Code'].unique()
    print('Unique State code list')
    print(uniq_state_code_list)
    # so there is no invalid state name
    
    # check venue country code
    uniq_country_code = df['Venue Country Code'].unique()
    print('Unique country code list')
    print(uniq_country_code)
    # so there is no invalid country code
    
    # check valid longitude
    # max latitude
    max_long = df['Venue Longitude'].max()
    print('Maximum longitude')
    print(max_long)
    # there is invalid longitude
    num_max_long = df[df['Venue Longitude'] > 0]
    num_invalid_max_long = len(num_max_long.index)
    # min latitude
    min_long = df['Venue Longitude'].min()
    print('Minimum longitude')
    print(min_long)
    # there is no invalid value in min longitude
    # longitude score
    long_score = (num_cols - num_invalid_max_long) / num_cols
    print('Longitude quality score')
    print(long_score)
    
    # check valid latitude
    max_lat = df['Venue Latitude'].max()
    print('Maximum latitude')
    print(max_lat)
    min_lat = df['Venue Latitude'].min()
    print('Minimum latitude')
    print(min_lat)
    # thus there is no invalid latitude value
    
    # total data quality score
    # total number of columns in the data set
    total_col = len(df.columns)
    print('Total number of columns')
    print(total_col)
    
    """
    The column-wise score for columns with no nas and no invalid values will be 1
    
    """
    # get total score
    total_score = (max_price_score + min_price_score + long_score + 17) / total_col
    print('Total quality score')
    print(total_score)
    return(total_score)


def clean_tm(tm):
    # read in the data
    df = pd.read_csv(tm, index_col = 0)
    # remove the unnamed col name
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    
    # first we remove rows that have missing values in min and max price
    """
    since the missing of min and max price value is simutaneous, 
    we only consider the missing of min value in this case.
    
    """
    df = df[df['Min Price'] != 'na']
    
    
    # replace the bad value of longitude with good value
    print('We observed that the incorrect value is barely missing a negative sign')
    print('Thus, we will fix it by adding the negative sign')
    #for i, r in df.iterrows():
        #if df['Venue Longitude'][i] > 0:
            #df['Venue Longitude'][i] = df['Venue Longitude'][i] * (-1)
    for i in df['Venue Longitude']:
        if i > 0:
            i = -1 * i
    print(df['Venue Longitude'])
    # replace nas in info col with 'Not Available now'
    df['Info'].fillna('Not Available Now', inplace = True)
    print('Replace nas in info col with \'Not Available now\'')
    
    # replace nans in note col with 'No Note Available'
    df['Note'].fillna('No Note Available', inplace = True)
    print('Replace nans in note col with \'No Note Available\'')
   
    # replace nans in parking info col with 'No Parking Info Available'
    df['Parking Info'].fillna('No Parking Info Available', inplace = True)
    print('Replace nans in parking info col with \'No Parking Info Available\'')
    # replace nans in accessibility col with 'No Accessibility Info Available'
    df['Accessibility'].fillna('No Accessibility Info Available', inplace = True)
    print('Replace nans in accessibility col with \'No Accessibility Info Available\'')
    return(df)
        





    
def main():
    
    print('\n' + '========Quality Report before cleaning========' )
    score = check_quality_tm('Ticketmaster_total.csv')
    print('\n' + '========End of Quality Report before cleaning========')
    print('\n' + '========Begin data cleaning========')
    clean_data = clean_tm('Ticketmaster_total.csv')
    print('========End of data cleaning=========')
    print('========Write the clean data to csv========')
    clean_data.to_csv('clean_tm.csv', index = False)
    print('\n' + '========Quality Report after cleaning========' )
    score_after = check_quality_tm('clean_tm.csv')
    print('\n' + '========End of Quality Report after cleaning========')
    print('=========The data is clean========')


if __name__ == '__main__':
    main()
    
    
    
    
    
