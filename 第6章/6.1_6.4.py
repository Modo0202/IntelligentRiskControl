# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:27:21 2020

@author: meizihang
"""

# 皮尔逊相关系数  
X.corr(Y1, method="pearson")  

# 斯皮尔曼相关系数  
d = (X.sort_values().index - Y.sort_values().index)**2  
dd = d.to_series().sum()  
P = 1 – n * dd / (n * (n ** 2 - 1))

#肯德尔相关系数  
x.corr(y, method="kendall")  

import toad  
  
""" 
特征筛选 
empty：缺失率上限 
iv：信息量 
corr：相关系数 
return_drop：返回删除特征 
exclude：不参与筛选的变量名 
"""  
dev_slct1, drop_lst = toad.selection.select(
                           dev, dev['bg_result_compensate'],
                           empty=0.7, iv=0.02, 
                           corr=0.7, return_drop=True, 
                           exclude=ex_lis)  
print("keep:", dev_slct1.shape[1],  
      "drop empty:", len(drop_lst['empty']),  
      "drop iv:", len(drop_lst['iv']),  
      "drop corr:", len(drop_lst['corr'])) 



""" 
逐步回归 
检验方法（criterion）：'aic' 和 'bic' ，此处使用'aic'
"""  
import toad  
dev_woe_psi_stp = toad.selection.stepwise(
                     dev_woe_psi2,  
                     dev_woe_psi2['bg_result_compensate'],
                     exclude=ex_lis,
                     direction='both',   
                     criterion='aic')

""" 
计算训练集与时间外验证样本的PSI
删除PSI大于0.02的特征 
"""  
import toad  
psi_df = toad.metrics.PSI(dev_slct4_woe, off_woe).sort_values(0)  
psi_df = psi_df.reset_index()  
psi_df = psi_df.rename(columns = {'index': 'feature', 0: 'psi'})  
  
psi02 = list(psi_df[psi_df.psi<0.02].feature)  
for i in ex_lis:  
    if i in psi02:  
        pass  
    else:  
       psi02.append(i)

from toad.plot import proportion_plot,badrate_plot,bin_plot  
badrate_plot(data,x='type',target='bad_ind',
             by='sj_plate_province_name_woe')


adj_bin = {'sj_plate_province_name_woe': [-0.105334, 0.292488,                         
             0.5653520000000001]}
combiner.set_rules(adj_bin)  
# 分箱  
data_c2 = combiner.transform(data, labels=True)  
  
# 分箱后再次观察  
badrate_plot(data_c2, x='type',target='bg_result_compensate',                        
                by='sj_plate_province_name_woe')  













