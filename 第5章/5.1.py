# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:14:17 2020

@author: meizihang
"""

import toad  
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
data = pd.read_csv('germancredit.csv')  
data.replace({'good':0,'bad':1},inplace=True)  
Xtr,Xts,Ytr,Yts = train_test_split(data.drop('creditability',axis=1),
                                          data['creditability'],
                                          test_size=0.25,
                                          random_state=450)
data_tr = pd.concat([Xtr,Ytr], axis=1)  
data_tr['type'] = 'train'  
data_ts = pd.concat([Xts,Yts], axis=1)  
data_ts['type'] = 'test'  
# 探索性数据分析  
toad.detect(data_tr).head(10)
