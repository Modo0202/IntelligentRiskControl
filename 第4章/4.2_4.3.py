# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:07:05 2020

@author: meizihang
"""

import pandas as pd  
import numpy as np  
import os  
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
data = pd.read_excel('./data/_data_for_tree.xlsx')  
x = data.drop('bad_ind',axis=1).copy()  
y = data.bad_ind.copy()  
from sklearn import tree  
Dtree = tree.DecisionTreeRegressor(max_depth=2, min_samples_leaf=500,
                                          min_samples_split=5000)  
dtree = dtree.fit(x,y)  
import pydotplus   
from IPython.display import Image  
from sklearn.externals.six import StringIO  
with open("dt.dot", "w") as f:  
    tree.export_graphviz(dtree, out_file=f)  
dot_data = StringIO()  
tree.export_graphviz(dtree, out_file=dot_data,  
                     feature_names=x.columns,  
                     class_names=['bad_ind'],  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())   
Image(graph.create_png())

import toad  
import pandas as pd  
import numpy as np  
import pydotplus  
from sklearn.externals.six import StringIO  
import os  
from sklearn import tree   
import warnings  
warnings.filterwarnings("ignore")   
class auto_tree(object):  
     
    def __init__(self, datasets, ex_lis=[], dep='bad_ind',
                    min_samples=0.05, min_samples_leaf=200,
                    min_samples_split=20, max_depth=5, is_bin=True):
        """ 
        datasets: 数据集 dataframe格式 
        ex_lis：不参与建模的特征，如id，时间切片等 list格式 
        dep：样本标签的变量名，string类型 
        min_samples：分箱时最小箱的样本占总比 numeric格式 
        max_depth：决策树最大深度 numeric格式 
        min_samples_leaf：决策树子节点最小样本个数 numeric格式 
        min_samples_split：决策树划分前，父节点最小样本个数 numeric格式 
        is_bin：是否进行卡方分箱 bool格式（True/False） 
        """  
        self.datasets = datasets  
        self.ex_lis = ex_lis  
        self.dep = dep  
        self.max_depth = max_depth  
        self.min_samples = min_samples  
        self.min_samples_leaf = min_samples_leaf  
        self.min_samples_split = min_samples_split  
        self.is_bin = is_bin  
        self.bins = 0  
        self.result = {}  

    def fit_plot(self):  
        os.environ["PATH"] += os.pathsep + \
                                    'D:/Program Files/Graphviz2.38/bin'  
        dtree = tree.DecisionTreeRegressor(
                   max_depth=self.max_depth,random_state=0,  
                 min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split)
        del_lis = []  
        for i in self.ex_lis:  
            if i in list(self.datasets.columns):  
                del_lis.append(i)  
        x = self.datasets.drop(del_lis, axis=1)         
        y = self.datasets[self.dep]  
        if self.is_bin:  
            # 分箱  
            combiner = toad.transform.Combiner()  
            combiner.fit(x,y,method='chi',min_samples=self.min_samples)
            x_bin= combiner.transform(x)  
            self.bins = combiner.export()          
        else:  
            combiner = 0  
            self.bins = []  
            x_bin = x.copy()   
        dtree = dtree.fit(x_bin, y)   
        self.estimator = dtree  
        df_bin = x_bin.copy()  
        df_bin[self.dep] = y  
        dot_data = StringIO()  
        tree.export_graphviz(dtree, out_file=dot_data,  
                               feature_names=x_bin.columns,  
                               class_names=[self.dep],  
                               filled=True, rounded=True,  
                               special_characters=True)  
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())   
        self.result['combiner'] = combiner  
        self.result['bins'] = self.bins  
        self.result['df_bin'] = df_bin  
        self.result['graph'] = graph.create_png()  
        self.result['model'] = dtree  
        return self.result  
    # 查看分箱结果          
    def value(result, col):  
        print('变量：', col)  
        print('分箱结果', result['bins'][col])  
        print('分箱后取值', set(result['df_bin'][col]))  
          
    # 计算负样本占比  
    def badrate(data, dep):  
        bad = sum(data[dep] == 1)  
        good = sum(data[dep] == 0)  
        print('负样本：',bad,'正样本：',good,'负样本占比,bad/(bad+good))  
    # 数据读取  
    def read_data(path,tp):  
        if tp == 'excel':   
            return pd.read_excel(path)  
        elif tp == 'csv' or tp == 'txt':  
            return pd.read_excel(path)  
        elif tp == 'sas':  
            return pd.read_excel(path)  
        else:  
            return '未知类型文件'  
    # 展示图像  
    def Image(graph):  
        from IPython.display import Image  
        return Image(graph)

# 导入函数  
import sys  
import pandas as pd  
import numpy as np  
sys.path.append('C:\\Users\\')  
from auto_decision import auto_tree  
# 加载数据  
data = pd.read_excel('./data/oil_data_for_tree.xlsx')  
base = data.sort_values(['uid','create_dt'], ascending=False)  
base = base.drop_duplicates(['uid'], keep='first')  
rh_base = base.fillna(0)  
# 指定不作为策略的变量名  
ex_lis = ['uid', 'create_dt', 'oil_actv_dt', 'class_new', 'bad_ind']

# 调用决策树函数  
result = auto_tree(datasets=rh_base, ex_lis=ex_lis,  
                    dep='bad_ind', min_samples=0.02, 
                      max_depth=4, min_samples_leaf=50,
                      min_samples_split=50).fit_plot()  
# result为字典类型，包含4个键  
print(result.keys())

# 展示图像  
auto_tree.Image(result['graph'])  

#原始数据的负样本占比  
auto_tree.badrate(rh_base , 'bad_ind')  

# 传入数据和变量名，查看分箱情况  
# 数值型区间左闭右开，字符型展示为完整分箱内容  
auto_tree.value(result, 'pay_amount_total')  
auto_tree.value(result, 'discount_amount')  
auto_tree.value(result, 'call_source') 

# 去掉上述规则拒绝的样本  
df_bin = result['df_bin']  
rh_base2 = df_bin[(df_bin['pay_amount_total']<=0.5) & \ 
                  (df_bin['discount_amount']<=0.5) & \ 
                  (df_bin['call_source']<=1.5) == False]  
rh_base2.shape


# 计算负样本占比  
auto_tree.badrate(rh_base2, 'bad_ind')  

# 分箱后不再分箱了，指定is_bin=False  
result2 = auto_tree(datasets=rh_base2, ex_lis=ex_lis,  
                    dep='bad_ind', min_samples=0.02,
                        max_depth=4, min_samples_leaf=50,
                        min_samples_split=50,
                        is_bin=False).fit_plot()  
# 展示图像  
auto_tree.Image(result2['graph'])

# 没有分箱结果  
result2['bins']

# 查看分箱结果  
auto_tree.value(result, 'pay_amount_total')  
auto_tree.value(result, 'oil_code') 


rh_base3 = rh_base2[(rh_base2['pay_amount_total']<=0.5) & \  
                    (rh_base2['oil_code']>0.5) == False]  
rh_base3.shape

# 重新计算负样本占比  
auto_tree.badrate(rh_base3, 'bad_ind') 



import matplotlib.pyplot as plt  
from sklearn.mixture import GaussianMixture as GMM  
gmm = GMM(n_components=3, covariance_type='full').fit(x) # 指定聚类中心个数为3  
labels = gmm.predict(x)  
plt.scatter(x['coupon_amount_cnt'], x['amount_tot'], c=labels, s=5, cmap='viridis')




























