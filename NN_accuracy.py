#!/usr/bin/env python3


import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing #to normalize the values
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'forestfires'
data = pd.read_csv(file_name + '.csv', header=0)

analysis_type = input("Analysis Type 'R' or 'C': ")

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining X
X_raw = data[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']]

#Normalizing or not the data
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X_raw)
#X = X_raw
#print(X)

#Defining Y variables depending on whether we have a regression or classification problem
if analysis_type == 'R' or analysis_type == 'r':
    Y = data.area
else:
    Y = data.area

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)

acc=[]
neurons=[]

if analysis_type == 'R' or analysis_type == 'r':
    for i in range(1,201):
        #Fit the neural network for Regression purposes (i.e., you expect a continuous variable out)
        #Note that 'sgd' and 'adam' require a batch_size and the function is not as clear
        acti = ['logistic', 'tanh', 'relu', 'identity']
        algo = ['lbfgs', 'sgd', 'adam']
        learn = ['constant', 'invscaling', 'adaptive']
        neural = MLPRegressor(activation=acti[0], solver=algo[0], learning_rate = learn[0], hidden_layer_sizes=(i,)) 
    
        #Cross validation
        neural_scores = cross_val_score(neural, X_train, Y_train, cv=5)
        #print("Cross Validation Accuracy: {0} (+/- {1})".format(neural_scores.mean().round(2), (neural_scores.std() * 2).round(2)))
        #print("")
        
        #Fitting final neural network
        neural.fit(X_train, Y_train)
        neural_score = neural.score(X_test, Y_test)
        acc.append(neural_score.round(4))
        neurons.append(i)
    print(acc)
    print(neurons)
    plt.plot(neurons,acc)
    plt.title("Regression")