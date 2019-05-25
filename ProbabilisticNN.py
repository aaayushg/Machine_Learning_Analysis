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
from neupy.algorithms import PNN 
from sklearn.naive_bayes import GaussianNB #Like Probabilistic Neural Network with one layer


#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'course_data'
data = pd.read_csv(file_name + '.csv', header=0)

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining dataset and droppping values we won't use, but keeping Y since we'll need it.
data_X = data.drop(['Name', 'Grade', 'Letter', 'Percentile'], axis=1)

#Hacking a normalization but keeping columns names since min_max_scaler does not return a dataframe
data_norm = (data_X - data_X.min()) / (data_X.max() - data_X.min())
#Adding a letter column to the normalized dataset
data_norm["Letter"] = data.Letter

#Using Built in train test split function in sklearn
data_train, data_test = train_test_split(data_norm, test_size=0.2)

#Defining testing sets
X_test = data_test.drop(['Letter'], axis=1)
Y_test = data_test.Letter


#Defining test sets and picking prototypes

#Splitting classes
ClassA_df = data_train[data_train['Letter'].isin(['A'])]
ClassA_data = ClassA_df.drop(['Letter'], axis=1)
#print (ClassA_data)
#print("")

ClassB_df = data_train[data_train['Letter'].isin(['B'])]
ClassB_data = ClassB_df.drop(['Letter'], axis=1)
#print (ClassB_data)
#print("")

ClassC_df = data_train[data_train['Letter'].isin(['C'])]
ClassC_data = ClassC_df.drop(['Letter'], axis=1)
#print (ClassC_data)
#print("")

ClassD_df = data_train[data_train['Letter'].isin(['D'])]
ClassD_data = ClassD_df.drop(['Letter'], axis=1)
#print (ClassD_data)
#print("")


# RBF and kmeans clustering by class 

#Number of prototypes
#prototypes = int(input("Number of prototypes:"))
prototypes = 4
eps = 1e-5

#Finding cluster centers
rbfk_netA = RBFKMeans(n_clusters=prototypes) #Chose number of clusters that you want
rbfk_netA.train(ClassA_data, epsilon=eps)
center_classA = pd.DataFrame(rbfk_netA.centers)

rbfk_netB = RBFKMeans(n_clusters=prototypes)
rbfk_netB.train(ClassB_data, epsilon=eps)
center_classB = pd.DataFrame(rbfk_netB.centers)

rbfk_netC = RBFKMeans(n_clusters=prototypes)
rbfk_netC.train(ClassC_data, epsilon=eps)
center_classC = pd.DataFrame(rbfk_netC.centers)

#Defining my Y_train
letter_classA = ['A']*prototypes
letter_classB = ['B']*prototypes
letter_classC = ['C']*prototypes

# Stack center of clusters as the training data
X_train = np.vstack((center_classA, center_classB, center_classC))
Y_train = np.hstack((letter_classA, letter_classB, letter_classC))

method = input('Gaussian Naive Bayes (G) or PNN (P):')

#Cross validation
cross_validation = 3

#Call Gaussian Naive Bayesian classifier as PNN
if method == 'P' or method == 'p':
    pnn = PNN(std=0.1)
    pnn.train(X_train, Y_train)
    
    # Cross validataion
    score = cross_val_score(pnn, X_train, Y_train, scoring='accuracy', cv=cross_validation)
    print("")
    print("Cross Validation: {0} (+/- {1})".format(abs(score.mean().round(2)), (score.std() * 2).round(2)))
    print("")
    
    #Prediction
    Y_predict = pnn.predict(X_test)
    print(Y_test.values)
    print(Y_predict)
    print("Accuracy: {0}".format(metrics.accuracy_score(Y_test, Y_predict).round(2)))
    
else:
    pnn = GaussianNB()
    pnn.fit(X_train, Y_train)
    print('Sigma: {0}'.format(pnn.sigma_)) #Standard deviation is estimated using maximum likelihood, so one sigma per variable and per class
    
    # Cross validataion
    score = cross_val_score(pnn, X_train, Y_train, scoring='accuracy', cv=cross_validation)
    print("")
    print("Cross Validation: {0} (+/- {1})".format(abs(score.mean().round(2)), (score.std() * 2).round(2)))
    print("")
    
    #Prediction
    Y_predict = pnn.predict(X_test)
    print(Y_test.values)
    print(Y_predict)
    print("Accuracy: {0}".format(metrics.accuracy_score(Y_test, Y_predict).round(2)))
    print("")
    