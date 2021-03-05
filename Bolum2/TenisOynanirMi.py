# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:10:40 2020

@author: Kaan
"""

# odev 1

import pandas as pd
import numpy as np

veriler = pd.read_csv('odev_tenis.csv')
print(veriler)
# verileri alabildik yaşasın
def metin_sayisal(veriler, index):
    from sklearn.preprocessing import OneHotEncoder
    outlook = veriler.iloc[:,index:index+1].values
    oneHE = OneHotEncoder()
    print(veriler.columns[index])
    encoded = oneHE.fit_transform(X = outlook).toarray()
   
    
    df = pd.DataFrame(data = encoded, columns= oneHE.get_feature_names())
    veriler.drop(columns = [veriler.columns[index]],inplace = True)
    veriler = pd.concat([df,veriler], axis = 1)
    return veriler

from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)
'''
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#outlook[:,0] = le.fit_transform(outlook[:,0])
# verileri label encoder kullanarak sayısal verilere dönüştürdük
'''
veriler2 = metin_sayisal(veriler2, 0)
veriler = metin_sayisal(veriler, 0)
veriler = metin_sayisal(veriler, 5)
veriler = metin_sayisal(veriler, 7)
veriler.drop(columns = ['x0_False','x0_no'],inplace = True)

'''
Trick !!
'''

#veriler.to_csv('tenis_duzenli.csv',index= False)

y = veriler['temperature']
x = veriler.drop(columns = ['temperature'])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x ,y)



from sklearn.linear_model import LinearRegression

regresor = LinearRegression()
regresor.fit(x_train, y_train)

y_pred = regresor.predict(x_test)



import statsmodels.api as sm

#X = np.append(arr = np.ones((22,1)).astype(int), values = veriler, axis = 1)

x_l = veriler.iloc[:,[0,1,2,3,4,6]].values
x_l = np.array(x_l,dtype = float)

model = sm.OLS(y,x_l).fit()

print(model.summary())


x_l = veriler.iloc[:,[2,3,4,6]].values
x_l = np.array(x_l,dtype = float)

model = sm.OLS(y,x_l).fit()
veriler.drop(columns = ['x0_yes','x0_True'],inplace = True)
print(model.summary())



y = veriler['temperature']
x = veriler.drop(columns = ['temperature'])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x ,y)



from sklearn.linear_model import LinearRegression

regresor = LinearRegression()
regresor.fit(x_train, y_train)

y_pred = regresor.predict(x_test)







