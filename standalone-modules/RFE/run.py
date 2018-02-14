#coding=utf-8
#递归特征消除法(Recursive Feature Elimination): 一种特征选择方法，基于算法输出的变量系数或者特征重要性，逐步地删除重要性小的变量。
#输入：x, y
#参数：step
#输出：rfe_columns, meta_json, df_rfe
#-----------------------------------------------------------------
import pandas as pd
import pickle
import json
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def main(params, inputs, outputs):
    
    ### 读入输入变量和目标变量 ###
    x = pd.read_pickle(inputs.x)
    y = pd.read_pickle(inputs.y)
    
    ### 读入参数 ###
    step = params.step
    
    ### 测试 ###
    #step = 1 
    
    ### 定义RFE使用的算法 ###
    estimator = RandomForestClassifier(n_estimators=20, criterion='gini', class_weight='balanced', n_jobs=-1) 
    ### 使用RFE进行训练 ###
    rfe = RFE(estimator, step=step)  
    rfe.fit(x,y) 
    
    ### 训练准确率 ###
    score = rfe.score(x,y) 
    
    ### 生成新dataframe ###
    df_rfe = pd.DataFrame(data = rfe.support_, index = x.columns, columns = ['support'])
    rfe_columns = list(df_rfe[df_rfe.support==True].index)    
    
    ### 输出 ###
    pickle.dump(rfe_columns, open(outputs.rfe_columns, 'wb'))
    
    ### 输出2 ###
    meta = \
    {
        "Number of variables before selection": x.shape[1],
        "Number of variables after selection": len(rfe_columns),
        "Variables after selection": rfe_columns,
    }
    meta_json = json.dumps(meta, indent=4)
    print("[meta]\n%s"%meta_json)
    with open(outputs.meta_json, 'w') as out:
        out.write(meta_json)
    
    ### 输出3 ###
    df_rfe = str(df_rfe)
    with open(outputs.df_rfe, 'wb') as out:
        out.write(df_rfe)