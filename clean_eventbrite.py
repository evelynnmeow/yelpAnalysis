#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:55:25 2019

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

def combine_data(y1, y2, y3, y4, y5, y6, y7, y8, y9, y10):
    event_boston = pd.read_csv(y1)
    event_chicago = pd.read_csv(y2)
    event_denver = pd.read_csv(y3)
    event_detroit = pd.read_csv(y4)
    event_lasvegas = pd.read_csv(y5)
    event_losangeles = pd.read_csv(y6)
    event_newyork = pd.read_csv(y7)
    event_philadelphia = pd.read_csv(y8)
    event_pittsburgh = pd.read_csv(y9)
    event_sanfrancisco = pd.read_csv(y10)
    
    # concat dataframes together
    df = pd.concat([event_chicago, event_denver, event_detroit, event_lasvegas,event_losangeles,
                    event_newyork, event_boston, event_philadelphia, event_pittsburgh, event_sanfrancisco], axis = 0)
    # drop duplicate
    df.drop_duplicates()
    return(df)
    

def check_quality_eventbrite(df):
    # check columns that contain missing values
    cols_with_na = df.columns[df.isna().any()].tolist()
    print('Columns with missing values is ')
    print(cols_with_na)
    # get the total number of data points
    num_cols = len(df.index)
    print('Total Number of columns is ')
    print(num_cols)
    
    # check event name column
    # checking missing value
    na_name = df[df['Event Name'].isna()]
    number_missing_name = len(na_name.index)
    print('Number of rows with missing values in Event Name:')
    print(number_missing_name)
    # thus there is no missing value in this column
    print('Since there is no standard way to input the event name, '+
          'we will not consider the column having any invalid values.')
    
    # check event id column
    # check missing values
    na_id = df[df['Event ID'].isna()]
    number_missing_id = len(na_id.index)
    print('Number of rows with missing values in Event ID:')
    print(number_missing_id)
    print('Values of Event ID are integers, which may not be displayed correctly '+
          'in python console.')
    score_id = (num_cols - number_missing_id) / num_cols
    print('The Quality Score of ID is:')
    print(score_id)
    """
    Values of Event ID are integers, which may not be displayed correctly in python console.
    
    """
    
    # check description
    # check missing values
    na_description = df[df['Description'].isna()]
    number_missing_description = len(na_description.index)
    print('Number of rows with missing values in Description:')
    print(number_missing_description)
    score_description = (num_cols - number_missing_description) / num_cols
    print('As we skim through the column, we find some values have excessive white space before letters.')
    print('Since extra white space at the front is not necessarily bad values, '+
          'we will not count it as bad values but will remove it when we clean the data.')
    print('The Quality Score of Description is:')
    print(score_description)
    """
    Since extra white space at the front is not essentially bad values, 
    we will not count it as bad values but will remove it when we clean the data.
    """
    
    # remove excessive white spaces
    # df['Description'] = df['Description'].str.lstrip()
    
    # check format
    # check missing values
    na_format = df[df['Format'].isna()]
    number_missing_format = len(na_format.index)
    print('Number of rows with missing values in Format:')
    print(number_missing_format)
    print('Since there is no standard way of input format, we will consider all values to be valid.')
    score_format = (num_cols - number_missing_format) / num_cols
    print('The Quality Score of Format is:')
    print(score_format)
    
    # check subcategory
    # check missing values
    na_subcategory = df[df['Subcategory'].isna()]
    number_missing_subcategory = len(na_subcategory.index)
    print('Number of rows with missing values in Subcategory:')
    print(number_missing_subcategory)
    print('Since there is no standard way of input format, we will consider all values to be valid.')
    score_subcategory = (num_cols - number_missing_subcategory) / num_cols
    print('The Quality Score of Subcategory is:')
    print(score_subcategory)
    
    # check available tickets
    # check missing values
    na_available = df[df['has Available Tickets'].isna()]
    number_missing_available = len(na_available.index)
    print('Number of rows with missing values in has Available Tickets:')
    print(number_missing_available)
    # check invalid data
    uniq_available = df['has Available Tickets'].unique()
    print(uniq_available)
    # check bad values in the column
    invalid_available1 = df[df['has Available Tickets'] == 'True']
    invalid_available2 = df[df['has Available Tickets'] == 'False']
    invalid_available3 = df[df['has Available Tickets'] == 'CO']
    invalid_available = len(invalid_available1.index) + len(invalid_available2.index) + len(invalid_available3.index)
    print('Number of invalid value of available tickets')
    print(invalid_available)
    # get the score of this attribute
    score_available = (num_cols - invalid_available - number_missing_available) / num_cols
    print('The Quality Score of Available tickets is:')
    print(score_available)
    
    # check min price
    # check missing values
    na_min_price = df[df['Min Price'] == 'na']
    number_missing_min_price = len(na_min_price.index)
    print('Number of rows with missing values in Min Price:')
    print(number_missing_min_price)
    # check invalid data
    # cast data type after dropping na values
    drop_na_price = df[df['Min Price'] != 'na']
    drop_na_price['Min Price'] = drop_na_price['Min Price'].astype(float)
    valid_min_price = drop_na_price[drop_na_price['Min Price'] >= 0]
    num_valid_min_price = len(valid_min_price.index)
    print('The number of valid min price is:')
    print(num_valid_min_price)
    score_min_price = (num_valid_min_price) / num_cols
    print('The quality score of min price is')
    print(score_min_price)
    
    # check max price
    # check missing values
    na_max_price = df[df['Max Price'] == 'na']
    number_missing_max_price = len(na_max_price.index)
    print('Number of rows with missing values in Max Price:')
    print(number_missing_max_price)
    # check invalid data
    # cast data type after dropping na values
    drop_na_price = df[df['Max Price'] != 'na']
    print('Unique values of max price')
    print(drop_na_price['Max Price'].unique())
    # drop invalid values
    drop_invalid_price1 = drop_na_price[drop_na_price['Max Price'] != 'US']
    drop_invalid_price = drop_invalid_price1[drop_invalid_price1['Max Price'] != 'Not Available']
    # check valid values
    drop_invalid_price['Max Price'] = drop_invalid_price['Max Price'].astype(float)
    valid_max_price = drop_invalid_price[drop_invalid_price['Max Price'] >= 0]
    num_valid_max_price = len(valid_max_price.index)
    print('The number of valid max price is:')
    print(num_valid_max_price)
    score_max_price = (num_valid_max_price) / num_cols
    print('The quality score of max price is')
    print(score_max_price)
    
    # check start time
    # check missing values
    na_start_time = df[df['Start Time (in utc)'].isna()]
    number_missing_start_time = len(na_start_time.index)
    print('Number of rows with missing values in start time:')
    print(number_missing_start_time)
    score_start_time = (num_cols - number_missing_start_time) / num_cols
    print('The Quality Score of start time is:')
    print(score_start_time)
    
    # check published time
    # check missing values
    na_publish_time = df[df['Published Time'].isna()]
    number_missing_publish_time = len(na_publish_time.index)
    print('Number of rows with missing values in publish time:')
    print(number_missing_publish_time)
    score_publish_time = (num_cols - number_missing_publish_time) / num_cols
    print('The Quality Score of publish time is:')
    print(score_publish_time)
    
    # check status
    # check missing values
    na_publish_time = df[df['Status'].isna()]
    number_missing_status = len(na_publish_time.index)
    print('Number of rows with missing values in publish time:')
    print(number_missing_status)
    score_status = (num_cols - number_missing_status) / num_cols
    print('The Quality Score of status is:')
    print(score_status)
    
    
    # check is online
    # check missing values
    na_online = df[df['is Online'].isna()]
    number_missing_online = len(na_online.index)
    print('Number of rows with missing values in is online:')
    print(number_missing_online)
    uniq_online = df['is Online'].unique()
    print(uniq_online)
    print('No bad values for this variable')
    score_online = (num_cols - number_missing_online) / num_cols
    print('The Quality Score of online is:')
    print(score_online)
    
    
    # check is free
    # check missing values
    na_free = df[df['is Free'].isna()]
    number_missing_free = len(na_free.index)
    print('Number of rows with missing values in is free:')
    print(number_missing_free)
    uniq_online = df['is Free'].unique()
    print(uniq_online)
    print('No bad values for this variable')
    score_free = (num_cols - number_missing_free) / num_cols
    print('The Quality Score of online is:')
    print(score_free)
    
    
    # check summary
    # check missing values
    na_summary = df[df['Summary'].isna()]
    number_missing_summary = len(na_summary.index)
    print('Number of rows with missing values in summary:')
    print(number_missing_summary)
    score_summary = (num_cols - number_missing_summary) / num_cols
    print('The Quality Score summary is:')
    print(score_summary)
    
    # check Category ID
    # check missing values
    na_category = df[df['Category ID'].isna()]
    number_missing_category = len(na_category.index)
    print('Number of rows with missing values in Category ID:')
    print(number_missing_category)
    score_category = (num_cols - number_missing_category) / num_cols
    print('The Quality Score of Category ID is:')
    print(score_category)
    
    # check Venue Name
    # check missing values
    na_venue_name = df[df['Venue Name'].isna()]
    number_missing_venue_name = len(na_venue_name.index)
    print('Number of rows with missing values in Venue Name:')
    print(number_missing_venue_name)
    score_venue_name = (num_cols - number_missing_venue_name) / num_cols
    print('The Quality Score of Venue Name is:')
    print(score_venue_name)
    
    # check Venue ID
    # check missing values
    na_venue_id = df[df['Venue ID'].isna()]
    number_missing_venue_id = len(na_venue_id.index)
    print('Number of rows with missing values in Venue ID:')
    print(number_missing_venue_id)
    score_venue_id = (num_cols - number_missing_venue_id) / num_cols
    print('The Quality Score of Venue ID is:')
    print(score_venue_id)
    
    
    # check address
    # check missing values
    na_address = df[df['Address'].isna()]
    number_missing_address = len(na_address.index)
    print('Number of rows with missing values in Address:')
    print(number_missing_address)
    score_address = (num_cols - number_missing_address) / num_cols
    print('The Quality Score of Address is:')
    print(score_address)
    
    # check city
    # check missing value
    na_city = df[df['City'].isna()]
    number_missing_city = len(na_city.index)
    print('Number of rows with missing values in City:')
    print(number_missing_city)
    score_city = (num_cols - number_missing_city) / num_cols
    print('The Quality Score of City is:')
    print(score_city)
    
    
    # check state
    # check missing value
    na_state = df[df['State'].isna()]
    number_missing_state = len(na_state.index)
    print('Number of rows with missing values in State:')
    print(number_missing_state)
    score_state = (num_cols - number_missing_state) / num_cols
    print('The Quality Score of State is:')
    print(score_state)
    
    # check country
    # check missing value
    na_country = df[df['Country'].isna()]
    number_missing_country = len(na_country.index)
    print('Number of rows with missing values in Country:')
    print(number_missing_country)
    # valid country
    valid_country = df[df['Country']== 'US']
    num_valid_country = len(valid_country.index)
    print('Number of valid country')
    print(num_valid_country)
    score_country = (num_valid_country) / num_cols
    print('The Quality Score of Country is:')
    print(score_country)
    
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
    print('The quality score of latitude is') 
    score_lat = (num_cols - number_missing_lat) / (num_cols)
    print(score_lat)
    
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
    print('The quality score of longitude is') 
    score_long = (num_cols - number_missing_long) / (num_cols)
    print(score_long)
    
    print('The total quality score of the data is')
    total_cols = len(df.columns)
    total_score = (1 + score_id + score_description + score_format + score_subcategory + score_available + score_min_price + score_max_price + score_start_time + score_publish_time + score_status + score_online + score_free + score_summary + score_category + score_venue_name + score_venue_id + score_address + score_city + score_state + score_country + score_lat + score_long) / total_cols
    print(total_score)

def clean_eventbrite(df):
    # clean
    # drop all missing values in the dataset
    df = df.dropna()
    
    # remove excessive white spaces
    df['Description'] = df['Description'].str.lstrip()    
    
    # replace bad values in the available ticket column
    df = df[df['has Available Tickets'] != 'CO']
    
    # drop bad values in max price
    df = df[df['Max Price'] != 'US']
    df = df[df['Max Price'] != 'Not Available']

    # return the cleaned data
    return(df)
    
def main():
    raw_data = combine_data('Eventbrite_Boston.csv', 'Eventbrite_Chicago.csv',
                            'Eventbrite_Denver.csv', 'Eventbrite_Detroit.csv',
                            'Eventbrite_Las Vegas.csv', 'Eventbrite_Los Angeles.csv',
                            'Eventbrite_New York.csv', 'Eventbrite_Philadelphia.csv',
                            'Eventbrite_Pittsburgh.csv', 'Eventbrite_San Francisco.csv')
    print('\n' + '========Quality Report before cleaning========' )
    check_quality_eventbrite(raw_data)
    print('\n' + '========End of Quality Report before cleaning========')
    clean_data = clean_eventbrite(raw_data)
    clean_data.to_csv('clean_eventbrite.csv', index = False)
    print('\n' + '========Quality Report after cleaning========' )
    score_after = check_quality_eventbrite(clean_data)
    print('\n' + '========End of Quality Report after cleaning========')
    print('=========The data is clean========')
    
    

if __name__ == '__main__':
    main()
    
    
    
