#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:46:04 2019

@author: JM
"""

# jm3292
# Visualization
# reference link for this part
# https://plot.ly/python/v3/frequency-counts/
# https://plot.ly/python/axes/

# import useful libraries
import plotly
import chart_studio
import chart_studio.plotly as py
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np





# the function takes in the yelp dataset
# creates scatterplot of Number of Music Venues vs Number of Sports Venues
# on TicketMaster
def num_music_sports_scat(yelp):
    # first plot
    # creates a trace for the first plot
    # it will be a scatterplot
    trace0 = go.Scatter(
    	x = yelp['Num Music TM'],
    	y = yelp['Num Sports TM'],
    	mode = 'markers'
    )
    
    # Assign it to an iterable object named data0
    data0 = [trace0]
    
    # Add axes and title
    layout0 = go.Layout(
    	title = "Scatterplot of Number of Music Venues vs Number of Sports Venues on TicketMaster",
    	xaxis=dict(
    		title = 'Number of Music Venues on TicketMaster'
    	),
    	yaxis=dict(
    		title = 'Number of Sports Venues on TicketMaster'
    	)
    )
        
    # Setup figure
    figure0 = go.Figure(data = data0, layout = layout0)
    
    # Display the scatterplot
    py.plot(figure0, auto_open = True)


# second plot
# the side by side boxplot of review count in each city
def review_city_box(yelp):
    # second plot
    # it will be a boxplot
    
    figure1 = px.box(yelp, x = "City", y = "Review Count")
    figure1.update_layout(title_text = "Boxplot of Review Count in Each City")
    # Display the plot
    py.plot(figure1, auto_open = True)
    
# the third plot
# side by side boxplot of types of goods and services in each city
def variety_city_box(yelp):
    # third plot
    # it will be a boxplot
    
    figure2 = px.box(yelp, x = "City", y = "Variety")
    figure2.update_layout(title_text = "Boxplot of Types of Goods and Services in Each City")
    # Display the plot
    py.plot(figure2, auto_open = True)

# the fourth plot
# histogram of review count
def review_count_hist(yelp):
    # the fourth plot
    # create a trace for it
    trace3 = go.Histogram(x = yelp['Review Count'], 
                          xbins = dict(start = np.min(yelp['Review Count']), 
                                       size = 100, end = np.max(yelp['Review Count'])))
     # Assign it to an iterable object named data0
    data3 = [trace3]
    
    # Add axes and title
    layout3 = go.Layout(
    	title = "Histogram of Review Count",
        xaxis=dict(
    		title = 'Review Count'
    	),
    	yaxis=dict(
    		title = 'Count'
    	)
    )
    
     # Setup figure
    figure3 = go.Figure(data = data3, layout = layout3)
    
    # Display the plot
    py.plot(figure3, auto_open = True)



# plot the boxplot of transaction type in different cities
def trans_type_city_box(yelp):
    # the fifth plot
    # it will be a boxplot
    
    figure3 = px.box(yelp, x = "City", y = "Num Transaction Type")
    # add title and axies labels
    figure3.update_layout(title_text = "Boxplot of Number of Transaction Types in Each City")
    figure3.update_xaxes(title_text = 'City')
    figure3.update_yaxes(title_text = 'Number of Transaction Types')
                          
    # Display the plot
    py.plot(figure3, auto_open = True)

# scatter plot of number of music events on two websites
def num_music_facet(yelp):
    # the sixth plot
    # faceted scatter plot
    figure5 = px.scatter(yelp, x = "Num Music TM", 
                         y = "Num Music EB", 
                         facet_col = "City",
                         facet_col_wrap = 3)
    # add title and axis label
    figure5.update_layout(title_text = 'Number of Music Venues on Two different Websites')
    figure5.update_yaxes(autorange="reversed")

    # Display the plot
    py.plot(figure5, auto_open = True)

# Histogram of price level in each city
def price_city_hist(yelp):
    # the seventh plot
    figure6 = px.histogram(yelp, x = "Price Level", 
                         facet_col = "City",
                         facet_col_wrap = 3)
    # add title and axis label
    figure6.update_layout(title_text = 'Distribution of Price Level in Each City')
    

    # Display the plot
    py.plot(figure6, auto_open = True)

# histogram of rating per city
def rating_city_hist(yelp):
    # the eighth plot
    # faceted scatterplot
    figure6 = px.histogram(yelp, x = "Rating", 
                         facet_col = "City",
                         facet_col_wrap = 3)
    # add title and axis label
    figure6.update_layout(title_text = 'Distribution of Rating in Each City')
    

    # Display the plot
    py.plot(figure6, auto_open = True)

# histogram of overall price level
def price_overall_hist(yelp):
    # plot a histogram of the rating
    figure8 = px.histogram(yelp, x = "Price Level")
    # add title and axis label
    figure8.update_layout(title_text = 'Histogram of Price Level')
    # Display the plot
    py.plot(figure8, auto_open = True)

# histogram of overall review count    
def review_overall_hist(yelp):
    # plot a histogram of the review count
    figure9 = px.histogram(yelp, x = "Review Count")
    # add title and axis label
    figure9.update_layout(title_text = 'Histogram of Review Count')
    # Display the plot
    py.plot(figure9, auto_open = True)

# histogram of overall rating
def rating_overall_hist(yelp):
    # plot a histogram of the review count
    figure10 = px.histogram(yelp, x = "Rating")
    # add title and axis label
    figure10.update_layout(title_text = 'Histogram of Rating')
    # Display the plot
    py.plot(figure10, auto_open = True)

# define main
def main():
   # set credentials
   chart_studio.tools.set_credentials_file(username='miathenerd', 
                                      api_key='4lUP77Y8FPwjFu7Zrtx1')
    
   # read in the data
   yelp = pd.read_csv('yelp_tm_eb.csv') 
   
   # the first plot
   num_music_sports_scat(yelp)
   
   # the second plot
   review_city_box(yelp)
   
   # the third plot
   variety_city_box(yelp)
   
   # the fourth plot
   review_count_hist(yelp)
   
   # the fifth plot
   trans_type_city_box(yelp)
   
   # the sixth plot
   num_music_facet(yelp)
   
   # the seventh plot
   price_city_hist(yelp)
   
   # the eighth plot
   rating_city_hist(yelp)
   
   # the histogram of rating
   price_overall_hist(yelp)
   
   # the histogram of review count
   review_overall_hist(yelp)
   
   # the histogram of rating
   rating_overall_hist(yelp)
   
   
   

# call main
if __name__ == '__main__':
    main()



