#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#Libraries needed to run the tool
import numpy as np
np.set_printoptions(suppress=True, precision=5, linewidth=150) #to control what is printed: 'suppress=True' prevents exponential prints of numbers, 'precision=5' allows a max of 5 decimals, 'linewidth'=150 allows 150 characters to be shown in one line (thus not cutting matrices)
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder #To switch categorical letters to numbers
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'course_data'
data = pd.read_csv(file_name + '.csv', header=0, index_col=0)

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")


#Defining X with column_stack since we will use numpy to get the covariance matrix.
X1 = data["AvgHW"]
X2 = data.AvgQuiz
X3 = data.AvgLab
X4 = data.MT1
X5 = data.MT2
X6 = data.Final
X7 = data.Participation
X = np.column_stack((X1, X2, X3, X4, X5, X6, X7))
#X = np.column_stack((X1, X2)) #Use only two variables to illustrate how transformation is done with two variables (with more the distances get distorted in a graph)


#Calculate and show covariance matrix
print("Covariance matrix")
print(np.cov(X, rowvar=0).round(2)) #rowvar=0 means that each column is a variable. Anything else suggest each row is a variable.
print('')
print("Here 1") #print to know where you are or to check if a bug exists
a = np.linalg.eigvals(np.cov(X, rowvar=0))
print(a/a.sum()) #To show that percentage variance explained by components is the eigenvalues
print('')
print("Here 2")


#Calculate and show correlation coefficients between datasets
print("Correlation Coefficients")
print(np.corrcoef(X, rowvar=0).round(2))
print("")


#Define the PCA algorithm
ncompo = int(input("Number of components to study:"))
print("")
pca = PCA(n_components=ncompo)

#Find the PCA
pcafit = pca.fit(X) #Use all data points since we are trying to figure out which variables are relevant

print("Mean")
print(pcafit.mean_)
print("")
print("Principal Components Results")
print(pcafit.components_)
print("")
print("Percentage variance explained by components")
print(pcafit.explained_variance_ratio_)
print("")

print(X)
X_new = pca.transform(X)
print(X_new)

'''
#Plot percentage variance explained by components 
perc = pcafit.explained_variance_ratio_
perc_x = range(1, len(perc)+1)
plt.plot(perc_x, perc)
plt.xlabel('Components')
plt.ylabel('Percentage of Variance Explained')
plt.savefig(file_name + '_pervar', dpi=300)
plt.show()
'''

'''
#Before and After
#Use AvgHw and AvgQuiz so that X_new[:,0] and X_new[:,1] are always AvgHW and AvgQuiz
before_after = input("Before / After Plot (Y):")
if before_after == 'Y' or before_after == 'y':
    le = LabelEncoder() #used to turn categorical letters to numbers: 0, 1, 2, 3
    le.fit(data.Letter)
    number = le.transform(data.Letter)
    colormap = np.array(['blue', 'green', 'orange', 'red'])
    
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(121)
    ax.scatter(X1, X2, c=colormap[number])
    ax.set_xlabel('AvgHW')
    ax.set_ylabel('AvgQuiz')
    
    
    ax = fig.add_subplot(122)
    ax.scatter(X_new[:,0], X_new[:,1], c=colormap[number])
    ax.set_xlabel('AvgHW')
    ax.set_ylabel('AvgQuiz')
    
    plt.savefig(file_name + '_before_after', dpi=300)
    plt.show()
'''

'''
#Fun 3D Plot
plot_3D = input("3D Plot (Y):")
if plot_3D == 'Y' or plot_3D == 'y':
    le = LabelEncoder() #used to turn categorical letters to numbers: 0, 1, 2, 3
    le.fit(data.Letter)
    number = le.transform(data.Letter)
    colormap = np.array(['blue', 'green', 'orange', 'red'])
    
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X4, X5, X6, c=colormap[number])
    ax.set_xlabel('MT1')
    ax.set_ylabel('MT2')
    ax.set_zlabel('Final')
    
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(X_new[:,4], X_new[:,5], X_new[:,6], c=colormap[number])
    ax.set_xlabel('MT1')
    ax.set_ylabel('MT2')
    ax.set_zlabel('Final')
    
    plt.savefig(file_name + '_3D', dpi=300)
    plt.show()
'''