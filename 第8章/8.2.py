# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:40:49 2020

@author: meizihang
"""

from toad.scorecard import ScoreCard  
card = ScoreCard(combiner=combiner, transer=t,
                    class_weight='balanced', C=0.1,
                    base_score=600, base_odds=35,
                    pdo=60, rate=2)  
card.fit(x, y)  
final_card = card.export(to_frame=True)  
final_card.head(8)


#8.2.2
lr_model = LogisticRegression(C=0.1)  
# LR模型训练      
lr_model.fit(x1, y)    
pred1 = lr_model.predict_proba(x1)[:,1]   
# LR模型训练    
lr_model.fit(x2, y)    
pred2 = lr_model.predict_proba(x2)[:,1]  
# LR模型训练    
lr_model.fit(x3, y)    
pred3 = lr_model.predict_proba(x2)[:,1]  
#前三个模型输出作为特征  
x4 = pd.concat([pred1,pred2,pred3])  
lr_model.fit(x4, y)  
#获得权重  
weight = lr_model.coef_  
#获得截距项  
intercept = lr_model.intercept_


