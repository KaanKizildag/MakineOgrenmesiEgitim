# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:08:47 2020

@author: Kaan
"""

import pandas as pd
import numpy as np

veriler = pd.read_csv("YeniVeriler.csv")

print(veriler)
veriler.drop(columns = veriler.columns[0:1], inplace=(True))


from sklearn.model_selection import train_test_split
y = veriler.iloc[:,3]
x = veriler.drop(columns = ['boy'])
boy = y
x_train, x_test, y_train, y_test = train_test_split(x ,y)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values = veriler, axis = 1)

x_l = veriler.iloc[:,[0,1,2,3,4,5]].values
x_l = np.array(x_l,dtype = float)

model = sm.OLS(boy,x_l).fit()

print(model.summary())

x_l = veriler.iloc[:,[0,1,2,3,5]].values
x_l = np.array(x_l,dtype = float)

model = sm.OLS(boy,x_l).fit()

print(model.summary())







