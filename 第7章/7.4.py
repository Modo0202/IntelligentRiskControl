# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:38:48 2020

@author: meizihang
"""

# 加载相关库  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import roc_curve  
import pandas as pd  
import numpy as np  
import random  
import math  
import warnings  
warnings.filterwarnings("ignore")  
# 读取数据  
data = pd.read_csv('Acard_reject.txt')  
data.sample(5)


# 有真实标签  
kgb = data[data['bad_ind']!=-1].copy()  
# 无标签拒绝样本  
reject = data[data['bad_ind']==-1].copy()  
# 所有样本  
agb = data.copy()   
# 指定变量名，使用LR模型进行拟合  
feature_lst = ['person_info', 'finance_info', 
                  'credit_info', 'act_info']  
x = kgb[feature_lst]  
y = kgb['bad_ind']  
lr_model = LogisticRegression(C=0.1)  
lr_model.fit(x, y)


# 负样本概率，越大越可能是负样本  
reject['y_pred'] = lr_model.predict_proba(reject[feature_lst])[:,1]  
# 0.8分位点  
thd = reject['y_pred'].quantile(0.8)  
# 阈值以上硬截断为负样本  
reject['bad_ind'] = reject['y_pred'].map(
                        lambda x :1 if x >= thd else -1)
# 只保留负样本  
labeled = reject[reject['bad_ind']==1]  
labeled = labeled.drop('y_pred', axis=1)  
# 合后并重新建模  
final = pd.concat([labeled, kgb], ignore_index=True)  
x = final[feature_lst]  
y = final['bad_ind']  
lr_hard = LogisticRegression(C=0.1)  
lr_hard.fit(x,y) 

# 复制样本  
reject1 = reject.copy()  
reject2 = reject.copy()  
# 按照正负样本概率加权  
reject1['weight'] = lr_model.predict_proba(reject1[feature_lst])[:,1]  
reject2['weight'] = lr_model.predict_proba(reject2[feature_lst])[:,0]  
# 合并  
labeled = pd.concat([reject1, reject2], ignore_index=True)  
# 为KGB样本设置权重  
kgb['weight'] = 1  
# 合后并重新建模  
final = pd.concat([labeled, kgb], ignore_index=True)  
x = final[feature_lst]  
y = final['bad_ind']  
lr_fuzz = LogisticRegression(C=0.1)  
lr_fuzz.fit(x, y, sample_weight=final['weight'])  


# 负样本概率，越大越可能是负样本  
agb['y_pred'] = lr_model.predict_proba(agb[feature_lst])[:,1]  
# 等频分箱  
agb['range'] = pd.qcut(agb['y_pred'], 10)  
# 分组计算权重  
final = pd.DataFrame()  
for i in list(set(agb['range'])):  
    tt = agb[agb['range']==i].copy()  
    good = sum(tt['bad_ind']==1)  
    bad = sum(tt['bad_ind']==0)  
    re = sum(tt['bad_ind']==-1)  
    # 权重计算  
    tt['weight'] = (good+bad+re)/(good+bad)  
    final = final.append(tt)  
# 带入权重，重新拟合模型  
x = final[feature_lst]  
y = final['bad_ind']  
lr_weighted = LogisticRegression(C=0.1)  
lr_weighted.fit(x, y, sample_weight=final['weight'])  

# 负样本概率，越大越可能是负样本  
kgb['y_pred'] = lr_model.predict_proba(kgb[feature_lst])[:,1]  
# 等频分箱  
kgb['range'] = pd.qcut(kgb['y_pred'], 10)  
# 在AGB有标记样本上计算等频分箱阈值和负样本占比  
pmax = kgb['y_pred'].max()  
cutpoints = list(set(kgb['y_pred'].quantile(
                    [0.1 * n for n in range(1, 10)]))) + [pmax + 1]
cutpoints.sort(reverse=False)  
dct = {}  
for i in range(len(cutpoints) - 1):  
    # 分箱  
    data = kgb.loc[np.logical_and(kgb['y_pred'] >= cutpoints[i],  
                                  kgb['y_pred'] < cutpoints[i + 1]), 
                                         ['bad_ind']] 
    good = sum(data['bad_ind']==0)  
    bad = sum(data['bad_ind']==1)  
    # 通过递增的步长，使得经验风险因子从2增长至4  
    step = (i + 1) * 0.2  
    dct[i] = bad / (bad + good) * 2 * step  
# 拒绝样本按照阈值进行划分  
reject['y_pred'] = lr_model.predict_proba(reject[feature_lst])[:,1]  
rejectNew = pd.DataFrame()  
for i in range(len(cutpoints) - 1):  
    # 分箱  
    data = reject.loc[np.logical_and(reject['y_pred']>=cutpoints[i],  
                      reject['y_pred']<cutpoints[i+1])]  
    data['badrate'] = dct[i]  
    rejectNew.append(data)  
    if rejectNew is None:  
        rejectNew= data  
    else:  
        rejectNew = rejectNew.append(data)  
# 定义随机打分函数  
def assign(x):  
    tt = random.uniform(0, 1)  
    if tt < x:  
        return 1  
    else:  
        return 0  
# 按照加权负样本占比随机赋值  
rejectNew['bad_ind'] = rejectNew['badrate'].map(lambda x:assign(x))  
# 合后并重新建模  
final = pd.concat([rejectNew,kgb], ignore_index=True)  
x = final[feature_lst]  
y = final['bad_ind']  
lr_Extra = LogisticRegression(C=0.1)  
lr_Extra.fit(x,y)


maxKS = 0  
n = 0  
x = kgb[feature_lst]  
y = kgb['bad_ind']  
lr_hard = LogisticRegression(C=0.1)  
lr_hard.fit(x,y)  
reject['y_pred'] = lr_hard.predict_proba(reject[feature_lst])[:,1]  
while True:  
    # 负样本概率，越大越可能是负样本  
    reject['y_pred'] = lr_hard.predict_proba(reject[feature_lst])[:,1]  
    # 0.8分位点  
    thd = reject['y_pred'].quantile(0.4)  
    # 阈值以上硬截断为负样本  
    reject['bad_ind'] = reject['y_pred'].map(
                             lambda x :1 if x >= thd else -1)
    # 只保留负样本  
    labeled = reject[reject['bad_ind']==1]  
    labeled = labeled.drop('y_pred', axis=1)  
    # 合后并重新建模  
    final = pd.concat([labeled,kgb], ignore_index=True)  
    x = final[feature_lst]  
    y = final['bad_ind']  
    lr_hard = LogisticRegression(C=0.1)  
    lr_hard.fit(x,y)  
    y_pred = lr_hard.predict_proba(kgb[feature_lst])[:,1]  
    fpr_lr_train,tpr_lr_train,_ = roc_curve(kgb['bad_ind'],y_pred)  
    ks = abs(fpr_lr_train - tpr_lr_train).max()  
    if maxKS <= ks:  
        maxKS = ks  
        n+=1  
        print('迭代第%s轮，ks值为%s' % (n,ks))  
    else:  
        break  











