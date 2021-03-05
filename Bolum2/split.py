# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:16:18 2020

@author: Kaan
"""

import pandas as pd
import numpy as np

veriler = pd.read_csv("YeniVeriler.csv")

print(veriler)

from sklearn.model_selection import train_test_split
y = veriler.iloc[:,5].values
x = veriler.drop(columns = ["yas","cinsiyet"]) 
'''
cinsiyet verisi string ifade olduğu için dönüşümlerde sorun çıkardı encoding
işlemi yaparak bu sorunu çözebilirdik
'''
x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size = .33)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_test = sc.fit_transform(x_test)

x_train = sc.fit_transform(x_train)
