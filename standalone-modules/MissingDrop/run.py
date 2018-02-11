#coding=utf-8
#删除几乎拥有唯一值的字段(比如单个变量最大类别百分比大于95%)；删除缺失百分比大于一定比率的字段(比如类别变量大于30%，连续变量大于60%)。
#输入：df 
#参数：percent_obj, percent_non_obj, percent_unique
#输出：df_new
#---------------------------------------------------------------
import pandas as pd
import numpy as np
import pickle

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    df = pd.read_pickle(inputs.df)
    
    ### 读入参数 ##
    percent_obj = params.percent_obj #删除大于该比率的空值字段(类别类型)
    percent_non_obj = params.percent_non_obj #删除大于该比率的空值字段(非类别类型)
    percent_unique = params.percent_unique  #删除大于等于该比率的唯一值字段（不包括空值）
    
    ### 测试 ###
    #percent_obj = 0.30 
    #percent_non_obj = 0.60 
    #percent_unique = 0.95

    ### ''转换为None ###
    df_new = df.applymap(lambda x:None if x=='' else x) #将''转换为None
    df_new.dropna(axis=1,how='all',inplace=True) #删除全部为空值的列

    ### 删除几乎拥有唯一值字段 ###
    cols = df_new.columns #全部字段
    
    for col in cols:
        value_count = df_new[col].value_counts(normalize=True,ascending=False,dropna=True)
        df_value_count = pd.DataFrame(data=value_count).reset_index()
        max_percent = df_value_count.iloc[0,1]
        if max_percent >= percent_unique:
            df_new.drop(col,axis=1,inplace=True) #删除唯一值列

    ### 计算每列空值比率 ###
    df_null=pd.DataFrame(data=df_new.dtypes,columns=['col_type'],index=df_new.dtypes.index) #列类型
    df_null.loc[:,'null_percent']=(df_new.isnull().sum()[:] * 100/ df_new.shape[0])  #计算每列的空值比率（%）
    
    ### 删除大于一定比率的对象类型字段 ###
    df_null_object = df_null[df_null.col_type=='object'] #对象类型
    df_null_object['delete'] = np.where(df_null_object['null_percent']>percent_obj,1,0) #得到大于某一空值比率的字段名
    drop_obj_cols = list(df_null_object[df_null_object.delete==1].index) #达到删除条件的空值字段(对象类型)
    df_new = df_new.drop(labels=drop_obj_cols,axis=1) #删除大于该比率的空值字段(对象类型)

    ### 删除大于一定比率的非对象类型字段 ###
    df_null_no_object = df_null[df_null.col_type!='object'] #非对象类型
    df_null_no_object['delete'] = np.where(df_null_no_object['null_percent']>percent_non_obj,1,0) #得到大于某一空值比率的字段名
    drop_no_obj_cols = list(df_null_no_object[df_null_no_object.delete==1].index) #达到删除条件的空值字段(其它类型)
    df_new = df_new.drop(labels=drop_no_obj_cols,axis=1) #删除大于该比率的空值字段(其它类型)

    ### 输出结果测试 ###
    print(df_new.head(10))
    
    ### 输出结果 ###
    df_new.to_pickle(outputs.df_new)