# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:45:58 2020

@author: Kaan
"""

# Eksik veriler

import pandas as pd
import numpy as np

eksikVeriler = pd.read_csv("eksikveriler.csv")

print(eksikVeriler)
'''
eksik veriler tamamlanırken sci kit learn kütüphanesinin
Simple imputer classını kullanacağız. import işlemiyle buna başlayalım
'''
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

'''
yas değişkenine tüm satırların 1 den 4 e kadar olan kolonlarını çekiyoruz
yani 1,2,3 kolonları alıyor
'''

yas = eksikVeriler.iloc[:,1:4].values
print(yas)

'''
imputer nesnesinin fit metoduyla eksik olan verileri öğretip,
transform metoduyla bunu uyguluyoruz.
'''

imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
print(yas)