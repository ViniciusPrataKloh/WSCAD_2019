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
df = pd.read_csv('Data/dataset.csv')
del df['Class']
del df['matrix_workload']
del df['Thread']
del df['energy_joules']

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
X = df.drop(['seconds'], axis=1)
y = df['seconds']

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
selector = SelectKBest(mutual_info_regression, k=10).fit(X, y)
selector.fit(X, y)
X = selector.transform(X)
columns = list(X.columns)
features = list(X.columns[selector.get_support(indices=True)])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(criterion="mae", max_depth=3)
scores = cross_val_score(dtr, X_test, y_test, cv=10)
print(scores)
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)

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
X_train = pd.DataFrame(X_train)

# Printing the Decision Tree
from sklearn.tree import export_graphviz

# Export as dot file
export_graphviz(dtr, out_file = 'trees/FS1/tree_3.dot', 
                feature_names = X_train.columns,
                class_names = 'seconds',
                rounded = True, proportion = False, 
                precision = 5, filled = True)

# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'trees/FS1/tree_3.dot', '-o', 'trees/FS1/tree_3.png'])

