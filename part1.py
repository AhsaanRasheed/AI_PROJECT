# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing the libraries

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

#importing the data Set
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, 1 :9].values
y = dataset.iloc[:, 9].values



"""----------------------------------------------------
#Taking care of missing data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3]  = imputer.transform(x[:, 1:3])

#Encoding catergorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y = LabelEncoder().fit_transform(y)

# onehotencoder = OneHotEncoder(categorical_features = [0])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct.fit_transform(x)
#------------------------------------------------------"""


#Encoding catergorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y = LabelEncoder().fit_transform(y)


#onehotencoder = OneHotEncoder(categorical_features = [0])
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.compose import ColumnTransformer
#ct = ColumnTransformer([("Name", OneHotEncoder(), [0])], remainder = 'passthrough')
#x = ct.fit_transform(x)
#print(y)

#splitting the dataSet into training set and test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Fitting classifier to the Training set

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2 )
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





