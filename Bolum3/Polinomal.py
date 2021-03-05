# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:48:47 2020

@author: Kaan
"""

import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar.csv')
# veriler yüklendi

y = veriler['maas']
x = veriler.drop(columns =['maas','unvan'])

X = x.values
Y = y.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

plt.scatter(X, Y, marker = '.')
plt.plot(X,lin_reg.predict(X),'-r')
plt.show()


from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)


lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, Y)

plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(x_poly),'r')

plt.show()

# 4. derece doğrusal olmayan (non-linear) polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3, Y)

plt.scatter(X,Y)
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),'r')

plt.show()




#SVR

from sklearn.preprocessing import StandardScaler

print(X.shape)
print(Y.shape)

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli, color = 'r')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli))

plt.show()


from sklearn.tree import DecisionTreeRegressor
dectree_reg = DecisionTreeRegressor(random_state=0)

dectree_reg.fit(X,Y)

plt.scatter(X,Y,color = 'red')
plt.plot(X,dectree_reg.predict(X))
plt.title = 'decisionTree'
plt.show()

print(dectree_reg.predict([[6.6]]))
'''
from sklearn.ensemble import RandomForest
randfor_reg = RandomForest(estimators = 10, random_state = 0)
randfor_reg.fit(X,Y)

plt.scatter(X,Y)
plt.plot(X,randfor_reg.predict(X))
plt.show()
'''






