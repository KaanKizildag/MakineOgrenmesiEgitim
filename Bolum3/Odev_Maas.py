# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 21:36:33 2020

@author: Kaan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# validation = ((10,'CEO',10,10,100),(17,'Mudur',7,10,100))
# validation_df = pd.DataFrame(data = validation)
veriler = pd.read_csv('maaslar_yeni.csv')

def metin_sayisal(veriler, index):
    from sklearn.preprocessing import OneHotEncoder
    outlook = veriler.iloc[:,index:index+1].values
    oneHE = OneHotEncoder()
    encoded = oneHE.fit_transform(X = outlook).toarray()
    df = pd.DataFrame(data = encoded, columns= oneHE.get_feature_names())
    veriler.drop(columns = [veriler.columns[index]],inplace = True)
    veriler = pd.concat([df,veriler], axis = 1)
    return veriler

y = veriler.maas
x = veriler.drop(columns = ['maas','unvan','Calisan ID'])
X = x.values
Y = y.values.reshape(-1,1)

# import statsmodels.api as sm

# X_L = x.iloc[:,[0,1,3]]
# X_L = np.array(X_L,dtype = float)
# model = sm.OLS(Y,X_L).fit()

# print(model.summary())

# MLR
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

tahminMLR = lin_reg.predict(X)
## Görselleştirme
plt.scatter(range(len(Y)), Y,color = 'red')
plt.plot(range(len(Y)),tahminMLR,color = 'blue')
plt.ylabel('LinearReg')
plt.show()

# PR
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # 2. dereceden
poly_x = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(poly_x,Y)

tahminPR = lin_reg2.predict(poly_x)
## Görselleştirme
plt.scatter(range(len(Y)), Y,color = 'red')
plt.plot(range(len(Y)),tahminPR,'b')
plt.ylabel('Polynomial Reg')
plt.show()

# SVR

# svr kullanırken scaler kullanmak gerekir.
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
scalerY = StandardScaler()
olcekli_x = scalerX.fit_transform(X)
olcekli_y = scalerY.fit_transform(Y)

from sklearn.svm import SVR
reg_svr = SVR(kernel= 'rbf')
reg_svr.fit(olcekli_x, olcekli_y)

tahminSVR = reg_svr.predict(olcekli_x)
# görselleştirme
plt.scatter(range(len(Y)), olcekli_y,color = 'red')
plt.plot(range(len(Y)),tahminSVR,'b')
plt.ylabel('SVR')
plt.show()


# Decision Tree
from sklearn.tree import DecisionTreeRegressor
reg_dt = DecisionTreeRegressor(random_state = 0)
reg_dt.fit(X, Y)

tahmin_dt = reg_dt.predict(X)
## Görselleştirme
plt.scatter(range(len(Y)), Y,color = 'red')
plt.plot(range(len(Y)),tahmin_dt,'b')
plt.ylabel('DecisionTree')
plt.show()


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor(n_estimators = 10, random_state=0)
reg_rf.fit(X, Y)
tahminRf = reg_rf.predict(X)

## Görselleştirme
plt.scatter(range(len(Y)), Y,color = 'red')
plt.plot(range(len(Y)),tahminRf,'b')
plt.ylabel('RandomForest')
plt.show()


from sklearn.metrics import r2_score

print('r^2 score lin_reg: ' + str(r2_score(Y,tahminMLR)))
print('r^2 score PR: ' + str(r2_score(Y,tahminPR)))
print('r^2 score SVR: ' + str(r2_score(olcekli_y,tahminSVR)))
print('r^2 score DT: ' + str(r2_score(Y,tahmin_dt)))
print('r^2 score RF: ' + str(r2_score(Y,tahminRf)))





