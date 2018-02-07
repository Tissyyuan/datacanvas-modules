#coding=utf-8
#递归特征消除法(Recursive Feature Elimination): 一种特征选择方法，基于算法输出的变量系数或者特征重要性，逐步地删除重要性小的变量。
#输入：x, y
#参数：step, n_features
#输出：x_new, y_new
#-----------------------------------------------------------------
import pandas as pd
import pickle
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def main(params, inputs, outputs):
    
    ### 读入输入变量和目标变量 ###
    x = pd.read_pickle(inputs.x)
    y = pd.read_pickle(inputs.y)
    
    ### 读入参数 ###
    step = params.step
    n_features = params.n_features
    
    ### 定义RFE使用的算法 ###
    estimator = RandomForestClassifier(n_estimators=20, criterion='gini', class_weight='balanced', n_jobs=-1) 
    ### 使用RFE进行训练 ###
    rfe = RFE(estimator, step=step, n_features_to_select=n_features)  
    rfe.fit(x,y) 
    
    ### 训练准确率 ###
    score = rfe.score(x,y) 
    
    ### 生成新dataframe ###
    df_rfe = pd.DataFrame(index=x.columns, data=rfe.support_, columns=['support'])
    rfe_columns = list(df_rfe[df_rfe.support==True].index)    
    x_new = x[rfe_columns]
    y_new = y.copy()
    
    ### 输出 ###
    x_new.to_pickle(outputs.x_new)
    y_new.to_pickle(outputs.y_new)
    
  