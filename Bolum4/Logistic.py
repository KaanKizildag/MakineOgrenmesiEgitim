# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 21:24:20 2020

@author: Kaan
"""

import pandas as pd
import numpy as np


veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.33)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



from sklearn.linear_model import LogisticRegression
reg_logis = LogisticRegression(random_state = 0)
reg_logis.fit(x_train,y_train)
y_pred = reg_logis.predict(x_test)



from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
conf_mat = confusion_matrix(y_test, y_pred,labels = ['k','e'])
cm_display = ConfusionMatrixDisplay(conf_mat,display_labels=['k','e']).plot()

print(conf_mat)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
conf_mat = confusion_matrix(y_test, y_pred,labels=['k','e'])
cm_display = ConfusionMatrixDisplay(conf_mat,display_labels=['k','e']).plot()
print(conf_mat)









