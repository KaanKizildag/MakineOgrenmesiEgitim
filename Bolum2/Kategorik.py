# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:13:59 2020

@author: Kaan
"""

# KategorikVeriler

import pandas as pd
import numpy as np

veriler = pd.read_csv("veriler.csv")

print(veriler)
'''
iloc[:,0] ==> dersem dizi olarak veriyor iloc[:,0:1] ==> dersem
2 boyutlu dizi olarak alır
'''
ulke = veriler.iloc[:,0:1].values

genel = veriler.iloc[:,1:-1].values

'''
verileri encoder kullanarak düzenliyorum
'''
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)


'''
Bu  şekilde ülke verileri 0,1,2 gibi isimlendi fakat aralarında sayısal bir
oran olmadığı için bu verileri one hot encoder kullanarak 100,010,001
gibi kodlamalıyız
'''

oneHotEncoder = preprocessing.OneHotEncoder()

ulke = oneHotEncoder.fit_transform(ulke).toarray()
print(ulke)


# cinsiyet değeri için
c = veriler.iloc[:,-1:].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(c[:,-1])
print(c)

oneHotEncoder = preprocessing.OneHotEncoder()

c = oneHotEncoder.fit_transform(c).toarray()
print(c)





# dataframe oluşturma

sonuc = pd.DataFrame(data = ulke, index = range(len(ulke)), columns=["fr","tr","us"])

sonuc2 = pd.DataFrame(data = genel,columns=["boy","kilo","yas"])

sonuc3 = pd.DataFrame(data = c[:,1], columns =['cinsiyet_k'])
print(sonuc3)
tekDataFrame = pd.concat([sonuc,sonuc2,sonuc3],axis = 1)

print(tekDataFrame)
# veriyi geçerli path e kaydetmek için:
# tekDataFrame.to_csv("YeniVeriler.csv") 










