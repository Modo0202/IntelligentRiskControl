from sklearn.linear_model import LogisticRegression  
lr_model = LogisticRegression(class_weight={0:4, 1:1})  
lr_model.fit(x,y) 

from sklearn.linear_model import LogisticRegression  
lr_model = LogisticRegression(class_weight="balanced")  
lr_model.fit(x,y)  

#生成数据集  
from sklearn.datasets import make_classification  
from collections import Counter  
X, y = make_classification(n_samples=5000, n_features=2,
                                n_informative=2, n_redundant=0, 
                                n_repeated=0, n_classes=3, 
                                n_clusters_per_class=1, 
                                weights=[0.01, 0.05, 0.94], 
                                class_sep=0.8, random_state=0)
#随机过采样  
from imblearn.over_sampling import RandomOverSampler  
X_resampled, y_resampled = RandomOverSampler().fit_sample(X, y)  
#SMOTE过采样及其变体  
from imblearn.over_sampling import SMOTE  
X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(X, y)  
X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(X, y)  
X_resampled, y_resampled = SMOTE(kind='borderline2').fit_sample(X, y)  
#ADASYN过采样  
from imblearn.over_sampling import ADASYN  
X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_sample(X, y)  
#随机欠采样  
from imblearn.under_sampling import RandomUnderSampler  
X_resampled, y_resampled = RandomUnderSampler().fit_sample(X, y)  
#基于k-means聚类的欠采样  
from imblearn.under_sampling import ClusterCentroids  
X_resampled, y_resampled = ClusterCentroids().fit_sample(X, y)  
#基于最近邻算法的欠采样  
from imblearn.under_sampling import RepeatedEditedNearestNeighbours  
X_resampled, y_resampled = 
     RepeatedEditedNearestNeighbours().fit_sample(X, y)  
#在数据上运用一种分类器
#然后将概率低于阈值的样本剔除掉
#从而实现欠采样  
from sklearn.linear_model import LogisticRegression  
from imblearn.under_sampling import InstanceHardnessThreshold  
lr_underS = InstanceHardnessThreshold(
               random_state=0, estimator=LogisticRegression())
X_resampled, y_resampled = lr_underS.fit_sample(X, y) 