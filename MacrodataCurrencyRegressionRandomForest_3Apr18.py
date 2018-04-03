#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:21:10 2018

@author: nitinsinghal
"""

# Macrodata Currency Regression Tests - Multiple, Polynomial. Random Forest tests.
# Please refer the supporting document MacroCcyRegressionRandomForestResultsSummary_3Apr18.docx for details

#Import libraries
import numpy as np
import pandas as pd

# Import the macro and ccy data
# Make sure you change the file depending on currency pair you are evaluating
# US data is available in the same file so you can easily change column range to 
# set the regression data

macroccydata = pd.read_csv('/Users/macroccyeurusdalldata_3Apr18.csv')
X = macroccydata.iloc[:, 2:25].values
y = macroccydata.iloc[:, 1].values

# 6 month lag, 12 month lag, etc between ccy price and macro data can be done 
# by manipulating the source file or the row range as required


# Splitting the dataset into the Training set and Test set
# You can change the test_size and other parameters to get best fit
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
multlin_reg = LinearRegression()
multlin_reg.fit(X_train, y_train)

# Predicting a new result with Multiple Linear Regression
y_pred = multlin_reg.predict(X_test)


# Fitting Polynomial Regression to the dataset - NOT USEFUL
#from sklearn.preprocessing import PolynomialFeatures

# Fitting Random Forest Regression to the dataset
# You can change the n_estimators and other parameters to get best fit
from sklearn.ensemble import RandomForestRegressor
randomforest_reg = RandomForestRegressor(n_estimators = 20, random_state = 0)
randomforest_reg.fit(X_train, y_train)

# Predicting a new result with Random Forest Regression
y_pred = randomforest_reg.predict(X_test)
