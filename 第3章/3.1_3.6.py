# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 23:34:43 2020

@author: meizihang
"""

from sklearn.linear_model import LinearRegression,Lasso, Ridge  
#简单线性回归  
model = LinearRegression()  
model.fit(x, y)  
#Lasso回归  
model = Lasso()  
model.fit(x, y)  
#Ridge回归  
model = Ridge()  
model.fit(x, y)  


from sklearn.linear_model import LogisticRegression  
lr_model = LogisticRegression()  
lr_model.fit(x,y) 


from sklearn.preprocessing import StandardScaler  
std_scaler = StandardScaler()  
x_new = std_scaler.fit_transform(x)  


from sklearn.preprocessing import robust_scale  
x_new= robust_scale(x)  


from scipy.stats import boxcox  
x_new, _ = boxcox(x)


def precision(y_true, y_pred):  
    true_positive = sum(y and y_p for y, y_p in zip(y_true, y_pred)) 
    predicted_positive = sum(y_pred)  
    return true_positive / predicted_positive  
def recall(y_true, y_pred):  
    true_positive = sum(y and y_p for y, y_p in zip(y_true, y_pred)) 
    real_positive = sum(y_true)  
    return true_positive / real_positive  

def true_negative_rate(y_true, y_pred):  
    true_negative = sum(1 - (yi or yi_hat) 
                             for yi, yi_hat in zip(y_true, y_pred))  
    actual_negative = len(y_true) - sum(y_true)  
    return true_negative / actual_negative  
def roc(y, y_hat_prob):  
    thresholds = sorted(set(y_hat_prob), reverse=True)  
    ret = [[0, 0]]  
    for threshold in thresholds:  
        y_hat = [int(yi_hat_prob >= threshold) 
                    for yi_hat_prob in y_hat_prob]  
        ret.append([recall(y, y_hat), 
                       1 - true_negative_rate(y, y_hat)])  
    return ret  
y_true = [1, 1, 1, 0, 1]  
y_hat_prob = [0.9, 0.95, 0.8, 0.5, 0.65]  
roc_list = roc(y_true, y_hat_prob)


def get_auc(y, y_hat_prob):  
    roc_val = iter(roc(y, y_hat_prob))  
    tpr_pre, fpr_pre = next(roc_val)  
    auc = 0  
    for tpr, fpr in roc_val:  
        auc += (tpr + tpr_pre) * (fpr - fpr_pre) / 2  
        tpr_pre = tpr  
        fpr_pre = fpr  
    return auc  
y_true = [1, 0, 1, 0, 1]  
y_hat_prob = [0.9, 0.85, 0.8, 0.7, 0.6]  
auc_val = get_auc(y_true, y_hat_prob) 


from sklearn.metrics import roc_auc_score, roc_curve, auc  
y_pred = model.predict_proba(x)[:,1]  
fpr_lgb_train, tpr_lgb_train, _ = roc_curve(y, y_pred)  
train_ks = abs(fpr_lgb_train - tpr_lgb_train).max()  
print('train_ks : ', train_ks)  
  
y_pred = model.predict_proba(evl_x)[:,1]  
fpr_lgb, tpr_lgb, _ = roc_curve(evl_y, y_pred)  
evl_ks = abs(fpr_lgb - tpr_lgb).max()  
print('evl_ks : ', evl_ks)  
  
from matplotlib import pyplot as plt  
plt.plot(fpr_lgb_train, tpr_lgb_train, label='train')  
plt.plot(fpr_lgb, tpr_lgb, label='evl')  
plt.plot([0,1], [0,1], 'k--')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC Curve')  
plt.legend(loc='best')


#保存模型至路径path  
import pickle      
with open(path, 'wb') as f:  
    pickle.dump(model, f, protocol=2)  
f.close()  

#从路径path加载模型  
from sklearn.externals import joblib   
model = joblib.load(path) 


#保存模型  
from sklearn2pmml import sklearn2pmml, PMMLPipeline  
from sklearn_pandas import DataFrameMapper  
mapper = DataFrameMapper([([i], None) for i in feat_list])  
pipeline = PMMLPipeline([('mapper', mapper), ("classifier", model)])  
sklearn2pmml(pipeline, pmml= path)  
  
#加载模型  
from pypmml import Model    
model = Model.fromFile(path)




















