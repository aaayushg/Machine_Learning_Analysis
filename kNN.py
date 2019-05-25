#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn import neighbors # kNN
from sklearn import metrics #Accuracy metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score #K-fold cross validation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'HW3_data'
data = pd.read_csv(file_name + '.csv', header=0)

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining X1, X2, and Y
X = np.column_stack((data.X1, data.X2)) #Normally use X = data[['X1', 'X2']], but here we want a matrix to determine x_min and x_max
Y = data.Y

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

#Setting up k-Nearest Neighbor and fitting the model with training data
k = 1
knn = neighbors.KNeighborsClassifier(n_neighbors=k)

#Cross validation
knn_scores = cross_val_score(knn, X_train, Y_train, scoring='accuracy', cv=3)
print("kNN accuracies: {0}".format(knn_scores)) #Prints individual scores
print("kNN overall accuracy: {0} (+/- {1})".format(knn_scores.mean().round(2), (knn_scores.std() * 2).round(2))) #Prints average score and standard deviation
print('')

#Fit final kNN algorithm that uses all training data
knn.fit(X_train, Y_train)

#Run the model on the test (remaining) data and show accuracy
Y_pred = knn.predict(X_test)
print(Y_pred)
print(Y_test.values)
score = metrics.accuracy_score(Y_pred, Y_test) 
print('Accuracy score: {0}'.format(score))


#Adapted from http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/tutorial/plot_knn_iris.html
# Plot the decision boundary. For that, we will asign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5 #Defines min and max on the x-axis
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5 #Defines min and max on the y-axis
h = (x_max - x_min)/200 # step size in the mesh to plot entire areas
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #Defines meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]) #Uses the calibrated model knn and running on "fake" data in meshgrid

# Put the result into a color plot
Z = Z.reshape(xx.shape) #Reshape for matplotlib
plt.figure(1) #create one figure
plt.set_cmap(plt.cm.Paired) #Picks color for 
plt.pcolormesh(xx, yy, Z) #Plot for the data

# Plot also the training points
colormap = np.array(['white', 'black']) #BGive two colors based on values of 0 and 1 from HW6_Data
plt.scatter(X[:,0], X[:,1],c=colormap[Y]) #Plot the data as a scatter plot, note that the color changes with Y.

plt.xlabel("X1") #Adding axis labels
plt.ylabel("X2")
plt.xlim(xx.min(), xx.max()) #Setting limits of axes
plt.ylim(yy.min(), yy.max())
plt.savefig(file_name + '_plot.png') #Saving the plot

plt.show() #Showing the plot