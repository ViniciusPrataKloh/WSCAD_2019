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

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Data/dataset.csv')
y = df['Class']
X = df.drop(['Class'], axis=1)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Feature Scaling with MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
scaled = min_max_scaler.fit_transform(X)
column_names = list(X.columns)
X = pd.DataFrame(scaled, columns=column_names)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train_PCA = X_train
X_test_PCA = X_test

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_PCA, y_train)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train_PCA, y_train)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', splitter = 'best')
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(X_train_PCA, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_PCA)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Printing the Decision Tree
from sklearn.tree import export_graphviz

# Export as dot file
export_graphviz(classifier, out_file = 'trees/tree_cl.dot', 
                feature_names = X_train_PCA.columns,
                class_names = 'seconds',
                rounded = True, proportion = False, 
                precision = 5, filled = True)

# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'trees/tree_cl.dot', '-o', 'trees/tree_cl.png'])

# Table
df_runtime = pd.DataFrame(data={'actual': y_test, 'predicted': y_pred})
df_runtime

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train_PCA, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test_PCA, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
