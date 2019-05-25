#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage #for dendrogram specifically
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'sample_data'
data = pd.read_csv(file_name + '.csv', header=0)

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining all the data X
X = data[['X1', 'X2']]


#Define number of clusters
clusters = 8


#Define which KMeans algorithm to use and fit it
Y_Kmeans = KMeans(n_clusters = clusters)
Y_Kmeans.fit(X)
Y_Kmeans_labels = Y_Kmeans.labels_
Y_Kmeans_silhouette = metrics.silhouette_score(X, Y_Kmeans_labels, metric='sqeuclidean')
print("Silhouette for Kmeans: {0}".format(Y_Kmeans_silhouette))
print("Results for Kmeans: {0}".format(Y_Kmeans_labels))


#Define which hierarchical clustering algorithm to use and fit it
linkage_types = ['ward', 'average', 'complete']
Y_hierarchy = AgglomerativeClustering(linkage=linkage_types[0], n_clusters=clusters)
Y_hierarchy.fit(X)
Y_hierarchy_labels = Y_hierarchy.labels_
Y_hierarchy_silhouette = metrics.silhouette_score(X, Y_hierarchy_labels, metric='sqeuclidean')
print("Silhouette for Hierarchical Clustering: {0}".format(Y_hierarchy_silhouette))
print("Hierarchical Clustering: {0}".format(Y_hierarchy_labels))


#Define figure
colormap = np.array(['magenta', 'black', 'blue', 'red', 'orange', 'green', 'brown', 'yellow', 'white', 'cyan']) #Define colors to use in graph - could use c=Y but colors are too similar when only 2-3 clusters
fig = plt.figure() #Define an empty figure
fig.set_size_inches(8,4) #Define the size of the figure as 8 inches by 4 inches

#Plot KMeans results
fig1 = fig.add_subplot(1,2,1)
plt.title("KMeans")
plt.scatter(data.X1, data.X2, c=colormap[Y_Kmeans_labels])
plt.annotate("s = " + str(Y_Kmeans_silhouette.round(2)), xy=(1, 0), xycoords='axes fraction', horizontalalignment='right', verticalalignment='bottom')

#Plot Hierarchical clustering results
fig1 = fig.add_subplot(1,2,2)
plt.title("Hierarchical Clustering")
plt.scatter(data.X1, data.X2, c=colormap[Y_hierarchy_labels])
plt.annotate("s = " + str(Y_hierarchy_silhouette.round(2)), xy=(1, 0), xycoords='axes fraction', horizontalalignment='right', verticalalignment='bottom')


#Show plots
fig.savefig(file_name + '_clustering.png', dpi=300)
plt.show()

#Can also plot individual silhouette coefficients: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py


#Using Scipy to draw dendrograms - for more info, see: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
linkage_types = ['ward', 'average', 'complete']
Z = linkage(X, linkage_types[1])
dendro = plt.figure()
dendro.set_size_inches(12,8)
dendrogram(Z, labels=data.index)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index from Dataframe')
plt.ylabel('Distance')
plt.savefig(file_name + '_dendro.png', dpi=300)
plt.show()