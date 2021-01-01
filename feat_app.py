# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:28:59 2020

@author: Mariano
"""

#-------------------------------------
# Import libraries
#-------------------------------------

import csv
import xlrd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv
from numpy import loadtxt

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



#---------------------------------
# Data Analysis
#---------------------------------

# Load cleaned data
filename = 'C:/Users/Mariano/app_feats/data_clean.csv'
df = read_csv(filename)

# drop unnecesary columns
df = df.drop(columns=['Unnamed: 0', 'Key'])
y = df['Y']

# get the list of categorical descriptive features
categorical_cols = df.columns[df.dtypes==object].tolist()

# encode categorical variables as one hot encoding
df = pd.get_dummies(df)

# encode target 
target = y.values

target = np.where(target<0.5, 0, target)
target = np.where(target>0.5, 1, target)

target = target.astype(int)

np.unique(target, return_counts=True)

# encode target using labelencoder from scikitlearn 
le = preprocessing.LabelEncoder()
le_fit = le.fit(target)
target_encoded_le = le_fit.transform(target)

# get features from dataframe
df_feats = df.drop(columns=['Y'])

# split train and test dataset 
X_train, X_test, y_train, y_test = train_test_split(df_feats, target_encoded_le, test_size=0.2)


k_limit = 5
select_feature = SelectKBest(chi2, k=k_limit).fit(X_train, y_train)

selected_features_df = pd.DataFrame({'Feature':list(X_train.columns),
                                     'Scores':select_feature.scores_})
df_score = selected_features_df.sort_values(by='Scores', ascending=False)

select_feature.scores_

df_feats_selected = df_score.iloc[0:k_limit]

features = df_feats_selected['Feature'].values

print('features selected:', features)

