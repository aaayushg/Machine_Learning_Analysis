#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score #New element this week!K-cross validation
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

#Defining X1, X2, and Y
#X = data[["X1", "X2"]].values #Different way to get the values from X1 and X2
X = np.column_stack((data.X1, data.X2))
Y = data.Y

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#Setting up SVM and fitting the model with training data
ker = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] #we won't use precomputed
vector = svm.SVC(kernel=ker[0], C=1, gamma='scale', degree=2) #degree is only relevant for the 'poly' kernel, and gamma is relevant for the 'rbf', 'poly', and 'sigmoid' kernels

#Cross Validation (CV) process
scores = cross_val_score(vector, X_train, Y_train, cv=5)
print(scores)
print("Accuracy: {0} (+/- {1})".format(scores.mean().round(2), (scores.std() * 2).round(2)))
print("")

#Fit final SVC
print("Now fit on entire training set")
print("")
vector.fit(X_train, Y_train)

print("Indices of support vectors:")
print(vector.support_)
print("")
print("Support vectors:")
print(vector.support_vectors_)
print("")
print("Intercept:")
print(vector.intercept_)
print("")

if vector.kernel == 'linear':
    c = vector.coef_
    print("Coefficients:")
    print(c)
    print("This means the linear equation is: y = -" + str(c[0][0].round(2)) + "/" + str(c[0][1].round(2)) + "*x + " + str(vector.intercept_[0].round(2)) + "/" + str(c[0][1].round(2)))
    print("")


#Run the model on the test (remaining) data and show accuracy
Y_predict = vector.predict(X_test)
print(Y_predict)
print(Y_test.values)
print(metrics.accuracy_score(Y_predict, Y_test))


#Adapted from http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/tutorial/plot_knn_iris.html
# Plot the decision boundary. For that, we will asign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5 #Defines min and max on the x-axis
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5 #Defines min and max on the y-axis
h = (x_max - x_min)/300 # step size in the mesh to plot entire areas
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #Defines meshgrid
Z = vector.predict(np.c_[xx.ravel(), yy.ravel()]) #Uses the calibrated model knn and running on "fake" data in meshgrid

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
plt.xticks(()) #Removing tick marks
plt.yticks(())
plt.savefig(file_name + '_plot.png') #Saving the plot

plt.show() #Showing the plot