#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')


#Ask for file name
#file_name = input("Name of file:")
file_name = 'course_data'
file_header = input("File has labels and header (Y):")


#Create a pandas dataframe from the csv file.      
if file_header == 'Y' or file_header == 'y':
    data = pd.read_csv(file_name + '.csv', header=0, index_col=0) #Remove index_col = 0 if rows do not have headers
else:
    data = pd.read_csv(file_name + '.csv', header=None)


#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns.values)))
print('')


#Plot a pairplot
pairplot = input("Plot a pairplot (Y):")
if pairplot == 'Y' or pairplot == 'y':
    sns_pairplot = sns.pairplot(data, diag_kind="kde")
    sns_pairplot.savefig(file_name + "_pairplot.png")
    plt.show()


#Plot a jointplot
jointplot = input("Plot a jointplot (Y):")
if jointplot == 'Y' or jointplot == 'y':
    print(data.columns.values)
    x_var = input("X variable:")
    y_var = input("Y variale:")
    sns_jointplot = sns.jointplot(data[x_var], data[y_var], kind="kde")
    sns_jointplot.savefig(file_name + "_jointplot.png")
    plt.show()


#Perform regression analysis
while True:    
    regression = input("Perform a linear regression (Y):")
    if regression == 'Y' or regression == 'y':
        print('') #Add onle line for space
        print(data.columns.values)
        #Set the x and y variables.
        x_var = input("X variable:")
        y_var = input("Y variable:")
        print('')
        #sklearn prefers 2d arrays, so we simply change the format of the data using the "reshape" function.
        x = data[x_var].values.reshape(-1, 1)
        y = data[y_var].values.reshape(-1, 1)
        
        
        lr = linear_model.LinearRegression() #Call regression model
        lr.fit(x, y) #Fit regression
        print("{0} * x + {1}".format(lr.coef_[0][0].round(2), lr.intercept_[0].round(2))) #Show equations calculated
        
        y_pred = lr.predict(x) #Calculate points predicted to measure accuracy
        R2 = r2_score(y, y_pred) #Calculate R2 score
        print("MSE: {0} and R2: {1}".format(np.mean((y_pred - y)**2).round(2), R2))
        
    else:
        break

#Goodbye message
print('')
print("Good Bye")
