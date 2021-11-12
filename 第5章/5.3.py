# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:24:38 2020

@author: meizihang
"""

import toad  
combiner = toad.transform.Combiner()  
combiner.fit(dev_slct2, dev_slct2['bad_ind'], method='chi',
               min_samples=0.2, exclude=ex_lis)

combiner = toad.transform.Combiner()    
combiner.fit(dev_slct2, dev_slct2['bad_ind'],
               method='chi', min_samples=0.2,
               exclude=ex_lis, empty_separate=True)

import numpy as np  
import matplotlib.pyplot as plt  
  
# 先在4个中心点附近产生一堆数据  
real_center = [(1, 1), (1, 2), (2, 2), (2, 1)]  
point_number = 50  
  
points_x = []  
points_y = []  
  
for center in real_center:  
    offset_x, offset_y = np.random.randn(point_number)*0.3,
                              np.random.randn(point_number)*0.25
    x_val, y_val = center[0] + offset_x,center[1] + offset_y
  
    points_x.append(x_val)  
    points_y.append(y_val)  
      
points_x = np.concatenate(points_x)  
points_y = np.concatenate(points_y)

for method in ['chi', 'dt', 'quantile', 'step', 'kmeans']:  
    c2 = toad.transform.Combiner()  
    c2.fit(data_tr2[['duration.in.month','creditability']],
             y='creditability', method=method, n_bins=5)
    bin_plot(c2.transform(data_tr2,labels=True),
                x='duration.in.month',target='creditability')


c2 = toad.transform.Combiner()  
c2.fit(data_tr2[['purpose','creditability']],
        y='creditability', method='chi')  
bin_plot(c2.transform(data_tr2[['purpose','creditability']],
           labels=True), x='purpose', target='creditability')
print(c2.export())

c2.set_rules({'purpose': [
    ['domestic appliances','retraining','car (used)'], 
    ['radio/television'], 
    ['furniture/equipment','repairs','business','car (new)'], 
    ['education','others']]})
bin_plot(c2.transform(data_tr2[['purpose','creditability']],
           labels=True), x='purpose', target='creditability')


T = toad.transform.WOETransformer()  
dev_slct2_woe = t.fit_transform(data_tr2,data_tr2 ['creditability'],
                                      exclude=ex_lis) 
off_woe = t.transform(off3[dev_slct3.columns])  




















