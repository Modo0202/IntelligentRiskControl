# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:48:52 2020

@author: meizihang
"""

import pandas as pd  
from sklearn.metrics import roc_auc_score,roc_curve,auc  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression   
import numpy as np  
import math  
import xgboost as xgb  
import toad  
# 加载数据
data_all = pd.read_csv("scorecard.txt")  

# 指定不参与训练列名  
ex_lis = ['uid', 'samp_type', 'bad_ind']  
# 参与训练列名  
ft_lis = list(data_all.columns)  
for i in ex_lis:      
    ft_lis.remove(i) 

# 开发样本、验证样本与时间外样本  
dev = data_all[(data_all['samp_type'] == 'dev')]
val = data_all[(data_all['samp_type'] == 'val') ]  
off = data_all[(data_all['samp_type'] == 'off') ]  

toad.detector.detect(data_all)

dev_slct1, drop_lst= toad.selection.select(dev, dev['bad_ind'], 
                                                   empty=0.7, iv=0.03, 
                                                   corr=0.7, 
                                                   return_drop=True, 
                                                   exclude=ex_lis)  
print("keep:", dev_slct1.shape[1],  
      "drop empty:", len(drop_lst['empty']), 
      "drop iv:", len(drop_lst['iv']),  
      "drop corr:", len(drop_lst['corr'])) 


# 得到切分节点  
combiner = toad.transform.Combiner()  
combiner.fit(dev_slct1, dev_slct1['bad_ind'], method='chi',
                min_samples=0.05, exclude=ex_lis)  
# 导出箱的节点  
bins = combiner.export()  
print(bins)

# 根据节点实施分箱  
dev_slct2 = combiner.transform(dev_slct1)
val2 = combiner.transform(val[dev_slct1.columns])
off2 = combiner.transform(off[dev_slct1.columns])
# 分箱后通过画图观察  
from toad.plot import  bin_plot, badrate_plot  
bin_plot(dev_slct2, x='act_info', target='bad_ind')  
bin_plot(val2, x='act_info', target='bad_ind')  
bin_plot(off2, x='act_info', target='bad_ind')   

bins['act_info']

adj_bin = {'act_info': [0.16666666666666666,0.35897435897435903,]}  
combiner.set_rules(adj_bin)

dev_slct3 = combiner.transform(dev_slct1)
val3 = combiner.transform(val[dev_slct1.columns])
off3 = combiner.transform(off[dev_slct1.columns])

# 画出Bivar图
bin_plot(dev_slct3, x='act_info', target='bad_ind')  
bin_plot(val3, x='act_info', target='bad_ind')  
bin_plot(off3, x='act_info', target='bad_ind') 

data = pd.concat([dev_slct3,val3,off3], join='inner')   
badrate_plot(data, x='samp_type', target='bad_ind', by='act_info') 

t = toad.transform.WOETransformer()  
dev_slct3_woe = t.fit_transform(dev_slct3, dev_slct3['bad_ind'], 
                                      exclude=ex_lis) 
val_woe = t.transform(val3[dev_slct3.columns])  
off_woe = t.transform(off3[dev_slct3.columns])  
data = pd.concat([dev_slct3_woe, val_woe, off_woe])

psi_df = toad.metrics.PSI(dev_slct3_woe, val_woe).sort_values(0)  
psi_df = psi_df.reset_index()  
psi_df = psi_df.rename(columns = {'index': 'feature', 0: 'psi'})  
psi_013 = list(psi_df[psi_df.psi<0.13].feature)  
for i in ex_lis:  
    if i in psi_013:  
        pass  
    else:
        psi_013.append(i)   
data = data[psi15]    
dev_woe_psi = dev_slct2_woe[psi_013]  
val_woe_psi = val_woe[psi15]  
off_woe_psi = off_woe[psi15]  
print(data.shape)

dev_woe_psi2, drop_lst = toad.selection.select(dev_woe_psi,
                                               dev_woe_psi['bad_ind'],
                                               empty=0.6,   
                                               iv=0.001, 
                                               corr=0.5, 
                                               return_drop=True, 
                                               exclude=ex_lis)  
print("keep:", dev_woe_psi2.shape[1],  
      "drop empty:", len(drop_lst['empty']),  
      "drop iv:", len(drop_lst['iv']),  
      "drop corr:", len(drop_lst['corr'])) 

dev_woe_psi_stp = toad.selection.stepwise(dev_woe_psi2,  
                                                  dev_woe_psi2['bad_ind'],  
                                                  exclude=ex_lis,  
                                                  direction='both',   
                                                  criterion='aic',  
                                                  estimator='ols',
                                              intercept=False)  
val_woe_psi_stp = val_woe_psi[dev_woe_psi_stp.columns]  
off_woe_psi_stp = off_woe_psi[dev_woe_psi_stp.columns]  
data = pd.concat([dev_woe_psi_stp, val_woe_psi_stp, off_woe_psi_stp]) 
print(data.shape)

def lr_model(x, y, valx, valy, offx, offy, C):  
    model = LogisticRegression(C=C, class_weight='balanced')      
    model.fit(x,y)  
      
    y_pred = model.predict_proba(x)[:,1]  
    fpr_dev,tpr_dev,_ = roc_curve(y, y_pred)  
    train_ks = abs(fpr_dev - tpr_dev).max()  
    print('train_ks : ', train_ks)  

    y_pred = model.predict_proba(valx)[:,1]  
    fpr_val,tpr_val,_ = roc_curve(valy, y_pred)  
    val_ks = abs(fpr_val - tpr_val).max()  
    print('val_ks : ', val_ks)  
    
    y_pred = model.predict_proba(offx)[:,1]  
    fpr_off,tpr_off,_ = roc_curve(offy, y_pred)  
    off_ks = abs(fpr_off - tpr_off).max()  
    print('off_ks : ', off_ks)  
      
    from matplotlib import pyplot as plt  
    plt.plot(fpr_dev, tpr_dev, label='dev')  
    plt.plot(fpr_val, tpr_val, label='val')  
    plt.plot(fpr_off, tpr_off, label='off')  
    plt.plot([0,1], [0,1], 'k--')  
    plt.xlabel('False positive rate')  
    plt.ylabel('True positive rate')  
    plt.title('ROC Curve')  
    plt.legend(loc='best')  
    plt.show() 

def xgb_model(x, y, valx, valy, offx, offy):  
    model = xgb.XGBClassifier(learning_rate=0.05,  
                              n_estimators=400,  
                              max_depth=2,  
                              class_weight='balanced',  
                              min_child_weight=1,  
                              subsample=1,   
                              nthread=-1,  
                              scale_pos_weight=1,  
                              random_state=1,  
                              n_jobs=-1,  
                              reg_lambda=300)  
    model.fit(x, y)  
      
    y_pred = model.predict_proba(x)[:,1]  
    fpr_dev,tpr_dev,_ = roc_curve(y, y_pred)  
    train_ks = abs(fpr_dev - tpr_dev).max()  
    print('train_ks : ', train_ks)  

    y_pred = model.predict_proba(valx)[:,1]  
    fpr_val,tpr_val,_ = roc_curve(valy, y_pred)  
    val_ks = abs(fpr_val - tpr_val).max()  
    print('val_ks : ', val_ks)  

    y_pred = model.predict_proba(offx)[:,1]  
    fpr_off,tpr_off,_ = roc_curve(offy, y_pred)  
    off_ks = abs(fpr_off - tpr_off).max()  
    print('off_ks : ', off_ks)  

    from matplotlib import pyplot as plt  
    plt.plot(fpr_dev, tpr_dev, label='dev')
    plt.plot(fpr_val, tpr_val, label='val') 
    plt.plot(fpr_off, tpr_off, label='off')  
    plt.plot([0,1], [0,1], 'k--')  
    plt.xlabel('False positive rate')  
    plt.ylabel('True positive rate')  
    plt.title('ROC Curve')  
    plt.legend(loc='best')  
    plt.show()  

def bi_train(data, dep='bad_ind', exclude=None):  
    from sklearn.preprocessing import StandardScaler  
    std_scaler = StandardScaler()  
    # 变量名  
    lis = list(data.columns)  
    for i in exclude:  
        lis.remove(i)  
    data[lis] = std_scaler.fit_transform(data[lis])  
    devv = data[(data['samp_type']=='dev')] 
    vall = data[(data['samp_type']=='val')] 
    offf = data[(data['samp_type']=='off')]
    x, y = devv[lis], devv[dep]
    valx, valy = vall[lis], vall[dep]
    offx, offy = offf[lis], offf[dep]
    # 逻辑回归正向
    print("逻辑回归正向：")
    lr_model(x, y, valx, valy, offx, offy, 0.1)  
    # 逻辑回归反向 
    print("逻辑回归反向：")
    lr_model(offx, offy, valx, valy, x, y, 0.1) 
    # XGBoost正向
    print("XGBoost正向：")
    xgb_model(x, y, valx, valy, offx, offy) 
    # XGBoost反向
    print("XGBoost反向：")
    xgb_model(offx, offy, valx, valy, x, y)  

bi_train(data, dep='bad_ind', exclude=ex_lis)

dep = 'bad_ind'  
lis = list(data.columns)  
for i in ex_lis:  
    lis.remove(i)
devv = data[data['samp_type']=='dev']
vall = data[data['samp_type']=='val']
offf = data[data['samp_type']=='off' ]
x, y = devv[lis], devv[dep]  
valx, valy = vall[lis], vall[dep]
offx, offy = offf[lis], offf[dep]
lr = LogisticRegression()  
lr.fit(x, y) 

from toad.metrics import KS, F1, AUC 

prob_dev = lr.predict_proba(x)[:,1]  
print('训练集')  
print('F1:', F1(prob_dev,y))  
print('KS:', KS(prob_dev,y))  
print('AUC:', AUC(prob_dev,y)) 

prob_val = lr.predict_proba(valx)[:,1]  
print('跨时间')  
print('F1:', F1(prob_val,valy))  
print('KS:', KS(prob_val,valy))  
print('AUC:', AUC(prob_val,valy)) 

prob_off = lr.predict_proba(offx)[:,1]  
print('跨时间')  
print('F1:', F1(prob_off,offy))  
print('KS:', KS(prob_off,offy))  
print('AUC:', AUC(prob_off,offy))

print('模型PSI:',toad.metrics.PSI(prob_dev,prob_off))  
print('特征PSI:','\n',toad.metrics.PSI(x,offx).sort_values(0))  

toad.metrics.KS_bucket(prob_off,offy,
                       bucket=10,
                       method='quantile')  

from toad.scorecard import ScoreCard  
card = ScoreCard(combiner=combiner, 
                    transer=t, C=0.1, 
                    class_weight='balanced', 
                    base_score=600,
                    base_odds=35,
                    pdo=60,
                    rate=2)  
card.fit(x,y)  
final_card = card.export(to_frame=True)  
final_card


