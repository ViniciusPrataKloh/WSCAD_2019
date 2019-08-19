#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:50:11 2019

@author: vinicius
"""

# Importing the libraries
import io
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import  metrics
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.feature_selection import chi2

# Pre-processing the data
df = pd.read_csv('Data2/output_energy_smcis.csv')
del df['Unnamed: 0']
del df['seconds']
del df['Thread']
del df['cpu-clock']
del df['page-faults']
del df['minor-faults']
del df['Unnamed: 0']
df = pd.read_csv('NORM_output_energy_smcis.csv')

# Feature Scaling with StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled = sc.fit_transform(df)
column_names = list(df.columns)
df = pd.DataFrame(scaled, columns=column_names)

# Feature Scaling with MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
scaled = min_max_scaler.fit_transform(df)
column_names = list(df.columns)
df = pd.DataFrame(scaled, columns=column_names)

# Defining X and y
X = df.drop(['energy_joules'], axis=1)
y = df['energy_joules']

# Correlation Analysis
corr = df.corr()
fig, ax = plt.subplots(figsize=(17, 17))
colormap = sns.diverging_palette(260, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
timestr = time.strftime("%Y%m%d_%Hh%Mm%Ss")
plt.savefig('heatmap_'+ timestr +'.png', bbox_inches='tight')
plt.show()

# Select K-best 
selector = SelectKBest(mutual_info_regression, k=2).fit(X, y)
selector.fit(X, y)
X_new = selector.transform(X)
columns = list(X.columns)
features = list(X.columns[selector.get_support(indices=True)])

# Mutual Information Regression (Estimate mutual information for a continuous target variable.)
mi = mutual_info_regression(X, y)
mi = pd.Series(mi)
mi.sort_values(ascending = False)
mi.plot.bar(figsize=(10,4))

# Create the new dataframe
df2 = pd.DataFrame(X_new, columns=features)
X = df2
y = df['seconds']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)
X_train_PCA = X_train
X_test_PCA = X_test

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(criterion="mae", max_depth=8)
scores = cross_val_score(dtr, X_test_PCA, y_test, cv=10)
print(scores)
dtr.fit(X_train_PCA, y_train)
y_pred = dtr.predict(X_test_PCA)

# Error
from sklearn.metrics import r2_score
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Mean Absolute Percent Error:', str(np.mean(np.abs((y_test - y_pred) / y_test)) * 100) + '%')
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
score = r2_score(y_test, y_pred) #* 100
print("R2 Score: ", score)

# Table
df_runtime = pd.DataFrame(data={'actual_runtime': y_test, 'predicted_runtime': y_pred, 'runtime_difference': (y_test - y_pred)})
df_runtime['error_percentage'] = np.abs((y_test - y_pred) / y_test) * 100
df_runtime

# Save the model to disk
from sklearn import model_selection
from sklearn.externals import joblib
filename = 'dtr_model.sav'
joblib.dump(dtr, filename)

# Create the new dataframe
X_train_PCA = pd.DataFrame(X_train_PCA)

# Printing the Decision Tree
from sklearn.tree import export_graphviz

# Export as dot file
export_graphviz(dtr, out_file = 'trees/Energy/Selection_2/tree_8_MMS.dot', 
                feature_names = X_train_PCA.columns,
                class_names = 'seconds',
                rounded = True, proportion = False, 
                precision = 5, filled = True)

# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'trees/Energy/Selection_2/tree_8_MMS.dot', '-o', 'trees/Energy/Selection_2/tree_8_MMS.png'])

# Display in python
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree_depth_5_KBest_5.png'))
plt.axis('off');
plt.show();