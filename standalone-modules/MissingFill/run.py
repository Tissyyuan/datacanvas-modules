#coding=utf-8
#小比例缺失值用众数或中位数填充(例如，类别变量缺失小于10%时用众数填充，非类别变量缺失小于30%时用中位数填充)。
#输入：df 
#参数：percent_obj, percent_non_obj
#输出：df_new
#---------------------------------------------------------------
import pandas as pd
import numpy as np
import pickle

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    df = pd.read_pickle(inputs.df)
    df_new = df.copy()
    
    ### 读入参数 ##
    percent_obj = params.percent_obj #填充小于该比率的空值字段(类别类型)
    percent_non_obj = params.percent_non_obj #填充小于该比率的空值字段(非类别类型)
    
    ### 测试 ###
    #percent_obj = 10 
    #percent_non_obj = 30 
    
    ### 计算每列空值比率 ###
    df_null=pd.DataFrame(data=df.dtypes,columns=['col_type'],index=df.dtypes.index) #列类型
    df_null.loc[:,'null_percent']=(df.isnull().sum()[:] * 100/ df.shape[0])  #计算每列的空值比率（%）
    
    ### 填充小于一定比率的类型字段 ###
    df_null_object = df_null[df_null.col_type=='object'] #对象类型
    obj_col=list(df_null_object[(df_null_object.null_percent<percent_obj) & (df_null_object.null_percent>0)].index) #小于指定空值比率的变量
    if len(obj_col) >=1:
        df_new[obj_col] = df_new[obj_col].fillna(df_new[obj_col].mode().iloc[0]) #空值填充为众数

    ### 填充小于一定比率的非类型字段 ###
    from sklearn.preprocessing import Imputer
    df_null_non_object = df_null[df_null.col_type!='object'] #对象类型
    non_obj_col=list(df_null_non_object[(df_null_non_object.null_percent<percent_non_obj) & (df_null_non_object.null_percent>0)].index) #小于指定空值比率的变量
    if len(non_obj_col) >= 1:
        df_new[non_obj_col]=Imputer(strategy='median').fit_transform(df_new[non_obj_col]) #空值填充为中值
    
    ### 输出结果测试 ###
    print(df_new.head(10))
    
    ### 输出结果 ###
    df_new.to_pickle(outputs.df_new)