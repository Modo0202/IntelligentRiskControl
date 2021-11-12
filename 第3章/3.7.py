# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 23:40:20 2020

@author: meizihang
"""

import numpy as np   
import pickle  
import xgboost as xgb  
# 基本例子，从libsvm文件中读取数据，做二分类  
# 数据是libsvm的格式  
#1 3:1 10:1 11:1 21:1 30:1 34:1 36:1 40:1 41:1 53:1 58:1 
#0 3:1 10:1 20:1 21:1 23:1 34:1 36:1 39:1 41:1 53:1 56:1 
#0 1:1 10:1 19:1 21:1 24:1 34:1 36:1 39:1 42:1 53:1 56:1 
dtrain = xgb.DMatrix('./data/agaricus.txt.train')  
dtest = xgb.DMatrix('./data/agaricus.txt.test')  
#超参数设定  
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
# 设定watchlist用于查看模型状态  
watchlist = [(dtest, 'eval'), (dtrain, 'train')]  
num_round = 2  
bst = xgb.train(param, dtrain, num_round, watchlist)  
# 使用模型预测  
preds = bst.predict(dtest)  
# 判断准确率  
labels = dtest.get_label()  
print ('错误类为%f' % \  
       (sum(1 for i in range(len(preds)) if 
          int(preds[i]>0.5) != labels[i]) / float(len(preds))))  
# 模型存储  
bst.save_model('./model/0001.model')


import pandas as pd  
import numpy as np  
import pickle  
import xgboost as xgb  
from sklearn.model_selection import train_test_split  
# 基本例子，从csv文件中读取数据，做二分类  
# 用pandas读入数据  
data = pd.read_csv('./data/Pima-Indians-Diabetes.csv')  
# 做数据切分  
train, test = train_test_split(data)  
# 转换成DMatrix格式  
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 
                      'SkinThickness', 'Insulin', 'BMI', 
                      'DiabetesPedigreeFunction', 'Age']  
target_column = 'Outcome'  
xgtrain = xgb.DMatrix(train[feature_columns].values, 
                          train[target_column].values)  
xgtest = xgb.DMatrix(test[feature_columns].values, 
                         test[target_column].values)  
# 参数设定  
param = {'max_depth':5, 'eta':0.1, 'silent':1, 
          'subsample':0.7, 'colsample_bytree':0.7, 
          'objective':'binary:logistic'}  
# 设定watchlist用于查看模型状态  
watchlist = [(xgtest, 'eval'), (xgtrain, 'train')]  
num_round = 10  
bst = xgb.train(param, xgtrain, num_round, watchlist)  
# 使用模型预测  
preds = bst.predict(xgtest)  
# 判断准确率  
labels = xgtest.get_label()  
print('错误类为%f' % \  
      (sum(1 for i in range(len(preds)) 
        if int(preds[i]>0.5) != labels[i]) / float(len(preds))))  
# 模型存储  
bst.save_model('./model/0002.model')


import warnings  
warnings.filterwarnings("ignore")  
import numpy as np  
import pandas as pd  
import pickle  
import xgboost as xgb  
from sklearn.model_selection import train_test_split  
from sklearn.externals import joblib  
# 基本例子，从csv文件中读取数据，做二分类  
# 用pandas读入数据  
data = pd.read_csv('./data/Pima-Indians-Diabetes.csv')  
# 做数据切分  
train, test = train_test_split(data)  
# 取出特征X和目标y的部分  
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 
                      'SkinThickness', 'Insulin', 'BMI', 
                      'DiabetesPedigreeFunction', 'Age']  
target_column = 'Outcome'  
train_X = train[feature_columns].values  
train_y = train[target_column].values  
test_X = test[feature_columns].values  
test_y = test[target_column].values  
# 初始化模型  
xgb_classifier = xgb.XGBClassifier(n_estimators=20,  
                                          max_depth=4,   
                                      learning_rate=0.1,   
                                      subsample=0.7,   
                                      colsample_bytree=0.7)  
# 拟合模型  
xgb_classifier.fit(train_X, train_y)  
# 使用模型预测  
preds = xgb_classifier.predict(test_X)  
# 判断准确率  
print('错误类为%f' % ((preds!=test_y).sum()/float(test_y.shape[0])))  
# 模型存储  
joblib.dump(xgb_classifier, './model/0003.model')

xgb.cv(param, dtrain, num_round, nfold=5, metrics={'error'}, seed=0) 

# 计算正负样本比，调整样本权重  
def fpreproc(dtrain, dtest, param):  
    label = dtrain.get_label()  
    ratio = float(np.sum(label==0)) / np.sum(label==1)  
    param['scale_pos_weight'] = ratio  
    return (dtrain, dtest, param)  
# 先做预处理，计算样本权重，再做交叉验证  
xgb.cv(param, dtrain, num_round, nfold=5,  
       metrics={'auc'}, seed=0, fpreproc=fpreproc) 


# 自定义对数损失函数   
def loglikelood(preds, dtrain):    
    labels = dtrain.get_label()    
    preds = 1.0 / (1.0 + np.exp(-preds))    
    grad = preds - labels    
    hess = preds * (1.0-preds)    
    return grad, hess    
# 评价函数：前20%正样本占比最大化    
def binary_error(preds, train_data):    
    labels = train_data.get_label()    
    dct = pd.DataFrame({'pred':preds,'percent':preds,'labels':labels})    
    # 取百分位点对应的阈值    
    key = dct['percent'].quantile(0.2)    
    # 按照阈值处理成二分类任务    
    dct['percent']= dct['percent'].map(lambda x :1 if x <= key else 0)      
    # 计算评价函数，权重默认0.5，可以根据情况调整    
    result = np.mean(dct[dct.percent== 1]['labels'] == 1)*0.5 \  
             + np.mean((dct.labels - dct.pred)**2)*0.5    
    return 'error', result    
watchlist = [(dtest,'eval'), (dtrain,'train')]    
param = {'max_depth':3, 'eta':0.1, 'silent':1}    
num_round = 100    
# 自定义损失函数训练    
bst = xgb.train(param, dtrain, num_round, watchlist, 
                   loglikelood, binary_err)

import numpy as np  
import pandas as pd  
import pickle  
import xgboost as xgb  
from sklearn.model_selection import train_test_split  
# 基本例子，从csv文件中读取数据，做二分类  
# 用pandas读入数据  
data = pd.read_csv('./data/Pima-Indians-Diabetes.csv')  
# 做数据切分  
train, test = train_test_split(data)  
# 转换成Dmatrix格式  
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 
                      'SkinThickness', 'Insulin', 'BMI', 
                      'DiabetesPedigreeFunction', 'Age']  
target_column = 'Outcome'  
xgtrain = xgb.DMatrix(train[feature_columns].values, 
                          train[target_column].values)  
xgtest = xgb.DMatrix(test[feature_columns].values, 
                         test[target_column].values)  
# 参数设定  
param = {'max_depth':5, 'eta':0.1, 'silent':1, 
          'subsample':0.7, 'colsample_bytree':0.7, 
          'objective':'binary:logistic' }  
# 设定watchlist用于查看模型状态  
watchlist = [(xgtest,'eval'), (xgtrain,'train')]  
num_round = 10  
bst = xgb.train(param, xgtrain, num_round, watchlist)  
# 只用第1棵树预测  
ypred1 = bst.predict(xgtest, ntree_limit=1)  
# 用前9棵树预测  
ypred2 = bst.predict(xgtest, ntree_limit=9)  
label = xgtest.get_label()  
print ('用前1棵树预测的错误率为 %f' % \
        (np.sum((ypred1>0.5)!=label) / float(len(label))))  
print ('用前9棵树预测的错误率为 %f' % \
         (np.sum((ypred2>0.5)!=label) / float(len(label))))


import pickle  
import xgboost as xgb  
import numpy as np  
from sklearn.model_selection import KFold, train_test_split, GridSearchCV  
from sklearn.metrics import confusion_matrix, mean_squared_error  
from sklearn.datasets import load_iris, load_digits, load_boston  
rng = np.random.RandomState(31337)  
# 二分类：混淆矩阵  
print("数字0和1的二分类问题")  
digits = load_digits(2)  
y = digits['target']  
X = digits['data']  
kf = KFold(n_splits=2, shuffle=True, random_state=rng)  
print("在2折数据上的交叉验证")  
for train_index, test_index in kf.split(X):  
    xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])  
    actuals = y[test_index]  
    print("混淆矩阵:")  
    print(confusion_matrix(actuals, predictions))  
# 多分类：混淆矩阵  
print("\nIris: 多分类")  
iris = load_iris()  
y = iris['target']  
X = iris['data']  
kf = KFold(n_splits=2, shuffle=True, random_state=rng)  
print("在2折数据上的交叉验证")  
for train_index, test_index in kf.split(X):  
    xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])  
    actuals = y[test_index]  
    print("混淆矩阵:")  
    print(confusion_matrix(actuals, predictions))  
# 回归问题：MSE  
print("\n波士顿房价回归预测问题")  
boston = load_boston()  
y = boston['target']  
X = boston['data']  
kf = KFold(n_splits=2, shuffle=True, random_state=rng)  
print("在2折数据上的交叉验证")  
for train_index, test_index in kf.split(X):  
    xgb_model = xgb.XGBRegressor().fit(X[train_index],y[train_index]) 
    predictions = xgb_model.predict(X[test_index])  
    actuals = y[test_index]  
    print("MSE:",mean_squared_error(actuals, predictions))


# 第2种训练方法的调参方法：使用sklearn接口的regressor + GridSearchCV  
print("参数最优化：")  
y = boston['target']  
X = boston['data']  
xgb_model = xgb.XGBRegressor()  
param_dict = {'max_depth': [2,4,6],  
              'n_estimators': [50,100,200]}  
clf = GridSearchCV(xgb_model, param_dict, verbose=1)  
clf.fit(X,y)  
print(clf.best_score_)  
print(clf.best_params_)


# 第1/2种训练方法的调参方法：early stopping  
# 在训练集上学习模型，一棵棵地添加树，在验证集上看效果  
# 当验证集效果不再提升，停止树的添加与生长  
X = digits['data']  
y = digits['target']  
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)  
clf = xgb.XGBClassifier()  
clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc",eval_set=[(X_val, y_val)])



iris = load_iris()  
y = iris['target']  
X = iris['data']  
xgb_model = xgb.XGBClassifier().fit(X,y)  
print('特征排序：')  
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
feature_importances = xgb_model.feature_importances_  
indices = np.argsort(feature_importances)[::-1]  
for index in indices:  
    print("特征 %s 重要度为 %f" 
            %(feature_names[index], feature_importances[index]))  
%matplotlib inline  
import matplotlib.pyplot as plt  
plt.figure(figsize=(16,8))  
plt.title("feature importances")  
plt.bar(range(len(feature_importances)), 
         feature_importances[indices], color='b')  
plt.xticks(range(len(feature_importances)), 
             np.array(feature_names)[indices], color='b')


import os  
if __name__ == "__main__":  
    from multiprocessing import set_start_method  
    set_start_method("forkserver")  
    import numpy as np  
    from sklearn.model_selection import GridSearchCV  
    from sklearn.datasets import load_boston  
    import xgboost as xgb  
    rng = np.random.RandomState(31337)  
    print("Parallel Parameter optimization")  
    boston = load_boston()  
    os.environ["OMP_NUM_THREADS"] = "2" 
    y = boston['target']  
    X = boston['data']  
    xgb_model = xgb.XGBRegressor()  
    clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],  
                       'n_estimators': [50, 100, 200]}, 
                            verbose=1, n_jobs=2)  
    clf.fit(X, y)  
    print(clf.best_score_)  
    print(clf.best_params_)


import xgboost  
import shap  

# 训练一个XGBoost 模型  
X, y = shap.datasets.boston()  
model = xgboost.train({"learning_rate": 0.1, "silent": 1}, 
                        xgboost.DMatrix(X, label=y), 100)


# 对模型文件model进行解释  
explainer = shap.TreeExplainer(model)  
# 传入特征矩阵X，计算SHAP值  
shap_values = explainer.shap_values(X)    
print(shap_values.shape)


# 可视化第一个预测的解释  
shap.force_plot(explainer.expected_value, 
                  shap_values[0,:], 
                  X.iloc[0,:], matplotlib=True)


# 所有样本Shap图  
shap.force_plot(explainer.expected_value, shap_values, X)

# 计算所有特征的影响  
shap.summary_plot(shap_values, X)


# 特征重要性  
shap.summary_plot(shap_values, X, plot_type="bar")  


# 加载xlearn包      
import xlearn as xl      
# 调用FM模型      
fm_model = xl.create_fm()      
# 训练集      
fm_model.setTrain("train.txt")      
# 设置验证集      
fm_model.setValidate("test.txt")     
# 分类问题：acc(Accuracy);prec(precision);f1(f1 score);auc(AUC score)      
param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'auc'}
# 训练模型  
fm_model.fit(param, "model.out")     
# 限制FM模型的输出在[0,1]之间  
fm_model.setSigmoid()      
fm_model.predict("model.out", "output.txt")      
# 保存模型  
fm_model.setTXTModel("model.txt")












