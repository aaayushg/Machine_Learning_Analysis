#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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

#Defining X and Y
X = data[['AvgHW', 'AvgQuiz', 'AvgLab','MT1', 'MT2', 'Final', 'Participation']]
Y = data.Letter

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)


#Fit the Decision tree
crit_choice = ['gini', 'entropy']
crit = crit_choice[0]
cross_val_number = 5
decitree = tree.DecisionTreeClassifier(criterion=crit, max_depth=10)
randfor = RandomForestClassifier(n_estimators = 5, criterion=crit)
gbm = GradientBoostingClassifier(n_estimators=100, max_depth=10)


#Cross Validation (CV) process
decitree_scores = cross_val_score(decitree, X_train, Y_train, cv=cross_val_number)
randfor_scores = cross_val_score(randfor, X_train, Y_train, cv=cross_val_number)
gbm_scores = cross_val_score(gbm, X_train, Y_train, cv=cross_val_number)
print("")
print("Decision Tree Accuracy: {0} (+/- {1})".format(decitree_scores.mean().round(2), (decitree_scores.std() * 2).round(2)))
print("Random Forest Accuracy: {0} (+/- {1})".format(randfor_scores.mean().round(2), (randfor_scores.std() * 2).round(2)))
print("Gradient Boosting Accuracy: {0} (+/- {1})".format(gbm_scores.mean().round(2), (gbm_scores.std() * 2).round(2)))
print("")

#Training final algorithms
decitree.fit(X_train, Y_train) #Decision Tree fitting
randfor.fit(X_train, Y_train) #Random Forest fitting
gbm.fit(X_train, Y_train) #Gradient Boosting fitting


#Final Predictions

print("Y test values: {0}".format(Y_test.values))
print("")

#Decision treee
decitree_predict = decitree.predict(X_test)
decitree_score = metrics.accuracy_score(decitree_predict, Y_test)
print("Decision Tree: {0}".format(decitree_predict))
print("Decision tree score: {0}".format(decitree_score))
print("")

#Random Forest
randfor_predict = randfor.predict(X_test)
randfor_score = metrics.accuracy_score(randfor_predict, Y_test)
print("Random Forest: {0}".format(randfor_predict))
print("Random Forest score: {0}".format(randfor_score))
print("")

#Gradient Boosting
gbm_predict = gbm.predict(X_test)
gbm_score = metrics.accuracy_score(gbm_predict, Y_test)
print("Gradient Boosting: {0}".format(gbm_predict))
print("Gradient Boosting score: {0}".format(gbm_score))
print("")


#Variable Importance Plots
nb_var = np.arange(len(X.columns))

#Decision Tree
plt.barh(nb_var, decitree.feature_importances_)
plt.yticks(nb_var, X.columns)
plt.title('Decision Tree')
plt.savefig(file_name + '_VI_decitree.png') #Saving the plot
plt.show()
print("")

#Random Forest
plt.barh(nb_var, randfor.feature_importances_)
plt.yticks(nb_var, X.columns)
plt.title('Random Forest')
plt.savefig(file_name + '_VI_randfor.png') #Saving the plot
plt.show()

#Gradient Boosting
plt.barh(nb_var, gbm.feature_importances_)
plt.yticks(nb_var, X.columns)
plt.title('Gradient Boosting')
plt.savefig(file_name + '_VI_gbm.png') #Saving the plot
plt.show()


#Export tree properties in graph format
#to see the graph need to use 'dot -Tpng HW7_data_decitree.dot -o HW7_data_decitree.png' in command prompt, if needed, install Graphviz from http://www.graphviz.org/
#alternatively, copy and paste the text from the .dot file to http://www.webgraphviz.com/
features = ['AvgHW', 'AvgQuiz', 'AvgLab','MT1', 'MT2', 'Final', 'Participation']
classes = ['A', 'B', 'C', 'D']
tree.export_graphviz(decitree, out_file = file_name + '_tree_decitree.dot', feature_names=features, class_names=classes)

'''
i_tree = 0
for tree_in_forest in randfor.estimators_:
    with open(file_name + '_tree_randfor_' + str(i_tree) + '.dot', 'w') as my_file:
        my_file = tree.export_graphviz(tree_in_forest, out_file = my_file, feature_names=features, class_names=classes)
    i_tree += 1
'''