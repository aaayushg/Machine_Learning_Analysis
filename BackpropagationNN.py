#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Libraries needed to run the tool
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
file_name = 'course_data'
data = pd.read_csv(file_name + '.csv', header=0)

analysis_type = input("Analysis Type 'R' or 'C': ")

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining X
X_raw = data[['AvgHW', 'AvgQuiz', 'AvgLab','MT1', 'MT2', 'Final', 'Participation']]

#Normalizing or not the data
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X_raw)
#X = X_raw
#print(X)

#Defining Y variables depending on whether we have a regression or classification problem
if analysis_type == 'R' or analysis_type == 'r':
    Y = data.Grade
else:
    Y = data.Letter

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)


if analysis_type == 'R' or analysis_type == 'r':
    #Fit the neural network for Regression purposes (i.e., you expect a continuous variable out)
    #Note that 'sgd' and 'adam' require a batch_size and the function is not as clear
    acti = ['logistic', 'tanh', 'relu', 'identity']
    algo = ['lbfgs', 'sgd', 'adam']
    learn = ['constant', 'invscaling', 'adaptive']
    neural = MLPRegressor(activation=acti[0], solver=algo[0], learning_rate = learn[0], hidden_layer_sizes=(7,)) 
    
    #Cross validation
    neural_scores = cross_val_score(neural, X_train, Y_train, cv=5)
    print("Cross Validation Accuracy: {0} (+/- {1})".format(neural_scores.mean().round(2), (neural_scores.std() * 2).round(2)))
    print("")
        
    #Fitting final neural network
    neural.fit(X_train, Y_train)
    neural_score = neural.score(X_test, Y_test)
    print("Shape of neural network: {0}".format([coef.shape for coef in neural.coefs_]))
    print("Coefs: ")
    print("")
    print(neural.coefs_[0].round(2))
    print("")
    print(neural.coefs_[1].round(2))
    print("")
    print("Intercepts: {0}".format(neural.intercepts_))
    print("")
    print("Loss: {0}".format(neural.loss_))
    print("")
    print("Iteration: {0}".format(neural.n_iter_))
    print("")
    print("Layers: {0}".format(neural.n_layers_))
    print("")
    print("Outputs: {0}".format(neural.n_outputs_))
    print("")
    print("Output Activation: {0}".format(neural.out_activation_)) #identity because we are looking for a value
    print("")

    #Assess the fitted Neural Network
    print("Y test and predicted")
    print(Y_test.values)
    print(neural.predict(X_test).round(1))
    print("")
    print("Accuracy as Pearson's R2: {0}".format(neural_score.round(4)))
    print("")

else:
    #Fit the neural network for Classification purposes (i.e., you don't expect a continuous variable out).
    #Note that 'sgd' and 'adam' require a batch_size and the function is not as clear
    acti = ['logistic', 'tanh', 'relu', 'identity']
    algo = ['lbfgs', 'sgd', 'adam']
    learn = ['constant', 'invscaling', 'adaptive']
    neural = MLPClassifier(activation=acti[2], solver=algo[0], learning_rate = learn[0], hidden_layer_sizes=(7,)) 
    
    #Cross validation
    neural_scores = cross_val_score(neural, X_train, Y_train, cv=5)
    print("Cross Validation Accuracy: {0} (+/- {1})".format(neural_scores.mean().round(2), (neural_scores.std() * 2).round(2)))
    print("")
        
    #Fitting final neural network    
    neural.fit(X_train, Y_train)
    neural_score = neural.score(X_test, Y_test)
    print("Classes: {0}".format(neural.classes_))
    print("")
    print("Shape of neural network: {0}".format([coef.shape for coef in neural.coefs_]))
    print("")
    print("Coefs: ")
    print(neural.coefs_[0].round(2))
    print("")
    print(neural.coefs_[1].round(2))
    print("")
    print("Intercepts: {0}".format(neural.intercepts_))
    print("")
    print("Loss: {0}".format(neural.loss_))
    print("")
    print("Iteration: {0}".format(neural.n_iter_))
    print("")
    print("Layers: {0}".format(neural.n_layers_))
    print("")
    print("Outputs: {0}".format(neural.n_outputs_))
    print("")
    print("Output Activation: {0}".format(neural.out_activation_)) #softmax to get a probability between 0 and 1
    print("")

    #Assess the fitted Neural Network
    print("Y test and predicted")
    print(Y_test.values)
    print(neural.predict(X_test))
    print("")
    print("Mean Accuracy: {0}".format(neural_score.round(4)))
    print("")