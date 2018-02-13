#coding=utf-8
#转换变量类型
#输入：df 
#参数：col, type
#输出：df_new, type
#---------------------------------------------------------------
import pandas as pd
import pickle

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    df = pd.read_pickle(inputs.df)
    
    ### 读入参数 ###
    col = params.col
    type = params.type
    
    ### 测试 ###
    #col = 'Age'
    #type = 'object'
    
    ### 转换变量类型 ###
    df[[col]] = df[[col]].astype(type)
    df_new = df.copy()
    
    ### 检查变量类型 ###
    Typ = df.dtypes
    typ = str(Typ)
    
    ### 输出结果 ###
    pickle.dump(df_new, open(outputs.df_new, 'wb'))
    
    with open(outputs.type, "w+") as out:
        out.write(typ) 
