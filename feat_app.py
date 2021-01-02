# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:28:59 2020

@author: Mariano
"""

#-------------------------------------
# Import libraries
#-------------------------------------

import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

#---------------------------------
# Load Data
#---------------------------------

# Load cleaned data
filename = 'C:/Users/Mariano/app_feats/data_clean.csv'
df = read_csv(filename)

filename = 'C:/Users/Mariano/app_feats/data_clean.csv'
df_orig = read_csv(filename)

#---------------------------------
# Prepare data 
#---------------------------------

# Drop unnecesary columns
df = df.drop(columns=['Unnamed: 0', 'Key','Attrition_Flag'])
y = df['Y']

# Get the list of categorical descriptive features
categorical_cols = df.columns[df.dtypes==object].tolist()

# Encode categorical variables as one hot encoding
df = pd.get_dummies(df)

# Encode target 
target = y.values

target = np.where(target<0.5, 0, target)
target = np.where(target>0.5, 1, target)

target = target.astype(int)

np.unique(target, return_counts=True)

# Encode target using labelencoder from scikitlearn 
le = preprocessing.LabelEncoder()
le_fit = le.fit(target)
target_encoded_le = le_fit.transform(target)

# Get features from dataframe
df_feats = df.drop(columns=['Y'])

# Split train and test dataset 
X_train, X_test, y_train, y_test = train_test_split(df_feats, target_encoded_le, test_size=0.2)


#-----------------------------------------------
# Decision tree feature selection 
#-----------------------------------------------

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

trans = SelectFromModel(clf)
X_trans = trans.fit_transform(X_train, y_train)
test_trans = trans.fit_transform(X_test, y_test)

print("We started with {0} features but retained only {1} of them!".
      format(X_train.shape[1], X_trans.shape[1]))

columns_retained_FromMode = df.iloc[:, 1:].columns[trans.get_support()].values

print("Columns selected are: {0}".
      format(columns_retained_FromMode))


#-----------------------------------------------
# Model before and after feature engineering 
#-----------------------------------------------

# Model before feature engineering
rf = RandomForestClassifier(max_depth=2, random_state=0)

rf.fit(X_train, y_train)

prediction = rf.predict(X_test)

accuracy_norm_model = metrics.accuracy_score(y_test, prediction)

# Model after feature engineering
rf_trans = RandomForestClassifier(max_depth=2, random_state=0)

rf_trans.fit(X_trans, y_train)

prediction_trans = rf_trans.predict(test_trans)

accuracy_trans_model = metrics.accuracy_score(y_test, prediction_trans)

print("Increase of accuracy on a simple model from {0} to {1}:".
      format(accuracy_norm_model, accuracy_trans_model))

#-----------------------------------------------
# Save obtained feats for further analysis
#-----------------------------------------------

feats = columns_retained_FromMode.tolist()

data_feats = pd.DataFrame(data={"feats": feats})

data_feats.to_csv('C:/Users/Mariano/app_feats/feats.csv')



