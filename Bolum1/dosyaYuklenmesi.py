# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:30:09 2020

@author: Kaan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

data = pd.read_csv("veriler.csv")

print(data)

boy = data[["boy"]]

print(boy)