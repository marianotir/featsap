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
from matplotlib import pyplot
from pandas import read_csv
from numpy import loadtxt

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer


#---------------------------------
# Data Analysis
#---------------------------------

# Load data
filename = 'D:/Users/Mariano/Documents/Downloads/archive/BankChurners.csv'
data_orig = read_csv(filename)


