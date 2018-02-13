#coding=utf-8
#将是字符的类别型变量映射为数值, 缺失值仍保持为np.nan。注意：类别型变量若已为数值则不做转换。
#输入：df 
#参数：cols
#输出：df_new
#---------------------------------------------------------------
import pandas as pd
import pickle
import numpy as np

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    df = pd.read_pickle(inputs.df)
    
    ### 读入需转换变量 ###
    cols = params.cols
    
    ### 测试 ###
    #cols = ['Geography', 'Gender']
    
    ### 类别字段映射 ###
    for col in cols:
        class_mapping= {label:idx for idx,label in enumerate(set(df[col]), 1)}  
        
        if '' in class_mapping.keys():
            del class_mapping['']
            
        if np.NaN in class_mapping.keys():
            del class_mapping[np.NaN]
            
        if None in class_mapping.keys():
            del class_mapping[None]
            
        if len(class_mapping) > 1:
            df[col] = df[col].map(class_mapping, na_action='ignore').astype('object') #类别字段映射
        
        else:
            df.drop(col, inplace=True, axis=1)      #删除唯一值数据列
     
    print(df.head(5)) 
    print(df.dtypes)
     
    ### 输出dataframe ###
    df_new = df.copy()
    pickle.dump(df_new, open(outputs.df_new, 'wb'))
    
    