#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Installing neupy can be difficult in anaconda. In the command, try:
    'conda install neupy==0.6.5', if this does not work, try:
    'pip install neupy==0.6.5', if this still does not work, try:
    '/Users/NAME/anaconda3/bin/pip install neupy==0.6.5', switching your path name to your anaconda bin folder.

Also note that we are installing neupy version 0.6.5 that is supposed to be more stable than the latest version.
"""

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from neupy.algorithms import RBFKMeans
from neupy.algorithms import GRNN


#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'course_data'
data = pd.read_csv(file_name + '.csv', header=0)

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining dataset and droppping values we won't use
dataset = data.drop(['Name', 'Letter', 'Percentile'], axis=1)

#Hacking a normalization - Needed since we need to make sure we know the Y values of the prototypes selected
df = (dataset - dataset.min()) / (dataset.max() - dataset.min())
minmax = dataset.Grade.max() - dataset.Grade.min() #needed after
minval = dataset.Grade.min() #needed after

#Define X and Y
X = df.drop(['Grade'], axis=1)
Y = df.Grade

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# RBF and kmeans clustering by class 
#Number of prototypes
#prototypes = int(input("Number of seed points:"))
prototypes = 12

#Finding cluster centers
df_cluster = X_train
df_cluster['Grade']= Y_train #Reproduce original data but only with training values
rbfk_net = RBFKMeans(n_clusters=prototypes) #Chose number of clusters that you want
rbfk_net.train(df_cluster, epsilon=1e-5)
center = pd.DataFrame(rbfk_net.centers)

# Turn the centers into prototypes values needed
X_prototypes = center.iloc[:, 0:-1]
Y_prototypes = center.iloc[:, -1] #Y_prototypes is the last column of center since 'Grade' is the last feature added to center.

#Train GRNN
GRNNet = GRNN(std=0.1)
GRNNet.train(X_prototypes, Y_prototypes)

# Cross validataion
score = cross_val_score(GRNNet, X_train, Y_train, scoring='r2', cv=5)
print("")
print("Cross Validation: {0} (+/- {1})".format(score.mean().round(2), (score.std() * 2).round(2)))
print("")


#Prediction
Y_predict = GRNNet.predict(X)
print(Y.values * minmax + minval)
print((Y_predict * minmax + minval)[:,0].round(2))
print("")
print("Accuracy: {0}".format(metrics.r2_score(Y, Y_predict).round(2)))
print("")