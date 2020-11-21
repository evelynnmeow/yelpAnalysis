#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:28:51 2019

@author: JM
"""

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
from sklearn import metrics
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN 
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA

# online sources used in this part
# https://scikit-learn.org/stable/modules/clustering.html




# this function plots histogram for attributes in yelp and ticketmaster
# parameter: the yelp and ticketmaster dataset
# it plots 5 histograms for 5 different attributes
def hist_plot(yelp, ticketmaster):
    # plot three attributes
    # price level of yelp
    yelp['Price Level'].hist(range=[0, 5],align = 'left')
    plt.xlabel('Price Level')
    plt.ylabel('Count')
    plt.title('Histogram of Price Level in Yelp Dataset')
    plt.savefig('Histogram of Price Level (Yelp).png')
    plt.show()
    plt.clf()
    
    # rating in the yelp dataset
    yelp['Rating'].hist(range=[0, 6],align = 'left')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Histogram of Rating in Yelp Dataset')
    # save image
    plt.savefig('Histogram of Rating (Yelp).png')
    plt.show()
    plt.clf()
    
    # review count in the yelp dataset
    yelp['Review Count'].hist(range=[0, 10000],align = 'left')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Count')
    plt.title('Histogram of Review Count in Yelp Dataset')
    # save image
    plt.savefig('Histogram of Review Count (Yelp).png')
    plt.show()
    plt.clf()
    
    
    # min price for ticket master
    ticketmaster['Min Price'].hist(range=[0, 300],align = 'left')
    plt.xlabel('Minimum Price')
    plt.ylabel('Count')
    plt.title('Histogram of Minimum Price in TicketMaster Dataset')
    # save image
    plt.savefig('Histogram of Minimum Price (TicketMaster).png')
    plt.show()
    plt.clf()
    
    # max price for ticket master
    ticketmaster['Max Price'].hist(range=[0, 7500],align = 'left')
    plt.xlabel('Maximum Price')
    plt.ylabel('Count')
    plt.title('Histogram of Maximum Price in TicketMaster Dataset')
    # save image
    plt.savefig('Histogram of Maximum Price (TicketMaster).png')
    plt.show()
    plt.clf()

# this function calculated pair-wise correlation between rating, review count, 
# and price level in the yelp dataset
# parameter: the yelp dataset
# the prints three tables for three types of correlation methods, one scatter matrix,
# and three scatterplots
def corr_scatter(yelp):
    # Correlation between attributes
    # choose rating, price, and review count from yelp
    # get value for yelp
    yelp_select = yelp[['Rating', 'Review Count', 'Price Level']]
    # using pearson method
    print('Correlation of Pearson Method')
    print(yelp_select.corr(method = 'pearson'))
    # using kandall method
    print('Correlation of Kendall Method')
    print(yelp_select.corr(method = 'kendall'))
    # using spearman
    print('Correlation of Spearman Method')
    print(yelp_select.corr(method = 'spearman'))
    
    # scatter plot matrix
    scatter_matrix(yelp_select, alpha = 0.2, figsize=(20, 20))
    plt.savefig('Scatter Matrix Plot (Yelp).png')
    plt.show()
    
    # scatter plot for rating vs review count
    plt.scatter(yelp_select['Rating'], yelp_select['Review Count'],alpha = 0.5)
    plt.xlabel('Rating')
    plt.ylabel('Review Count')
    plt.title('Scatter Plot of Rating vs. Review Count')
    # save image
    plt.savefig('Scatter Plot of Rating vs. Review Count (Yelp).png')
    plt.show()
    plt.clf()
    
    
    # scatter plot for rating vs price level
    plt.scatter(yelp_select['Rating'], yelp_select['Price Level'],alpha = 0.5)
    plt.xlabel('Rating')
    plt.ylabel('Price Level')
    plt.title('Scatter Plot of Rating vs. Price Level')
    # save image
    plt.savefig('Scatter Plot of Rating vs. Price Level (Yelp).png')
    plt.show()
    plt.clf()
    
    
    # scatter plot for price level vs review count
    plt.scatter(yelp_select['Price Level'], yelp_select['Review Count'],alpha = 0.5)
    plt.xlabel('Price Level')
    plt.ylabel('Review Count')
    plt.title('Scatter Plot of Price Level vs. Review Count')
    # save image
    plt.savefig('Scatter Plot of Price Level vs. Review Count (Yelp).png')
    plt.show()
    plt.clf()


# this function normalizes variables for clustering analysis
# parameter: the yelp dataset with selectede attributes
# it returns a dataframe after normalization

def data_normal(df): 
    # normalize data
    x = df.values
    # using min_max_scaler
    min_max_scaler = preprocessing.MinMaxScaler()
    # normalize values
    x_scaled = min_max_scaler.fit_transform(x)
    # change back to dataframe
    df_normal = pd.DataFrame(x_scaled)
    # return the dataframe
    return(df_normal)



# this function performs partitional clustering (kmeans)
# it takes the normalized dataframe and a list of k (number of clusters) as parameters
# it prints the best performance of kmeans method and the corresponding
# silhouette score, Calinski Harabaz score, and k value
def kmeans_cluster(yelp_normal, k):
    # initialize max silhouette score for further use
    max_si_score = 0
    # initialize max calinski harabaz score
    max_ch_score = 0
    # partitional clustering method (k means)
    
    print('============Partitional Clustering============')
    # for loop to loop through different k values
    for i in k:
        kmeans = KMeans(n_clusters=i)
        cluster_labels = kmeans.fit_predict(yelp_normal)
        # get silhouette and calinski harabz score
        silhouette_avg = silhouette_score(yelp_normal, cluster_labels)
       
        # using calinski harabaz
        cal_har = metrics.calinski_harabaz_score(yelp_normal, cluster_labels)  
        
        # find the k value that maximizes both scores
        if silhouette_avg > max_si_score:
            max_si_score = silhouette_avg
        if cal_har > max_ch_score:
            max_ch_score = cal_har
    # print out results
    print('For Partitional Clustering (Kmeans)')
    print('The best performance (Silhouette score) is when k is ', i)
    print('The best silhouette score is ',max_si_score)
    print('The best performance (Calinski Harabaz score) is when k is ', i)
    print('The best Calinski Harabaz score is ',max_ch_score)





# this function performs hierarchical clustering with ward linkage
# it takes the normalized dataframe and a list of k (number of clusters) as parameters
# it prints the best performance of kmeans method and the corresponding
# silhouette score, Calinski Harabaz score, and k value
def ward_cluster(yelp_normal, k):
    print('===============Hierarchical Clustering=============')
    # initialize max silhouette score for further use
    max_si_score = 0
    # initialize max calinski harabaz score
    max_ch_score = 0
    # hiarchical clusterring method
    for i in k:
        ward_linkage = AgglomerativeClustering(n_clusters = i, affinity = 'euclidean', linkage = 'ward')
        cluster_labels = ward_linkage.fit_predict(yelp_normal)
        # silhouette score
        silhouette_avg = silhouette_score(yelp_normal, cluster_labels)
        
        # using calinski harabaz score
        cal_har = metrics.calinski_harabaz_score(yelp_normal, cluster_labels)  
        # get k that maximizes both scores
        if silhouette_avg > max_si_score:
            max_si_score = silhouette_avg
        if cal_har > max_ch_score:
            max_ch_score = cal_har
    print('For Hierarchical Clustering with ward linkage')
    print('The best performance (Silhouette score) is when k is ', i)
    print('The best silhouette score is: ',max_si_score)
    print('The best performance (Calinski Harabaz score) is when k is ', i)
    print('The best Calinski Harabaz score is ',max_ch_score)

# this function performs DBSCAN
# it takes the normalized dataframe, a list minPts, and a list of eps
# it prints the best performance of kmeans method and the corresponding
# silhouette score, Calinski Harabaz score, minPts, and eps
def dbscan(yelp_normal, min_sam, eps):
    print('============DBSCAN============')
    # initialize max silhouette score for further use
    max_si_score = 0
    # initialize max calinski harabaz score
    max_ch_score = 0
    # select min sample size 
    min_sam = [3, 5, 7, 10,12,15]
    # select eps
    eps = [0.5, 1, 3, 5, 8, 10]
    # run DBSCAN over min_sam and ep
    for i in eps:
        for j in min_sam:
            dbscan = DBSCAN(eps = 0.05, min_samples = i)
            cluster_labels = dbscan.fit_predict(yelp_normal)
            # silhouette score
            silhouette_avg = silhouette_score(yelp_normal, cluster_labels)
            # cal_har_score
            cal_har = metrics.calinski_harabaz_score(yelp_normal, cluster_labels)  
            if silhouette_avg > max_si_score:
                max_si_score = silhouette_avg
            if cal_har > max_ch_score:
                max_ch_score = cal_har
    # print result
    print('For DBSCAN')
    print('The best performance (Silhouette score) is when min sample is ', j)
    print('The best performance (Silhouette score) is when eps is ', i)
    print('The best silhouette score is: ',max_si_score)
    print('The best performance (Calinski Harabaz score) is when mean sample is ', j)
    print('The best performance (Silhouette score) is when eps is', i)
    print('The best Calinski Harabaz score is ',max_ch_score)


# this function performs 2-dimensional PCA
# it takes the normalized dataset as input
# it plots the 2-dimensional PCA project
def pca2d(yelp_normal):
    # For easier visualization, we will use 2D PCA
    pca2D = decomposition.PCA(2)
    pca2D = pca2D.fit(yelp_normal)
    # transform to 2d PCA
    plot_columns = pca2D.transform(yelp_normal)
        
    # This shows how good the PCA performs on this dataset
    print('\n' + 'The performance of the PCA is ')
    print(pca2D.explained_variance_)
        
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1])
    plt.title("2-dimensional scatter plot using PCA")
    # save image
    plt.savefig('2D_PCA_Yelp.png')
    plt.show()
    plt.clf()

# define main 
def main():
    # read in the data
    yelp = pd.read_csv('yelp_tm_eb.csv')
    ticketmaster = pd.read_csv('clean_tm.csv')
    print('********************** Histograms and Correlations*******************')
    hist_plot(yelp, ticketmaster)
    corr_scatter(yelp)
    
    # select attributes for the model
    yelp_select = yelp[['Rating', 'Is Closed', 'Review Count', 'Price Level',
                        'Num Transaction Type', 'Variety']]
    # three cluster anlysis
    # a list of number of clusters k
    k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]
    # select minPts 
    min_sam = [3, 5, 7, 10,12,15]
    # select eps
    eps = [0.5, 1, 3, 5, 8, 10]
    # normalize data
    yelp_normalize = data_normal(yelp_select)
    print('*********************** Clustering Analysis************************')
    # kmean clustering
    kmeans_cluster(yelp_normalize, k)
    # hierarchical clustering
    ward_cluster(yelp_normalize, k)
    # DBSCAN
    dbscan(yelp_normalize, min_sam, eps)
    # 2D PCA projection
    pca2d(yelp_normalize)

# call main
if __name__ == '__main__':
    main()
    























