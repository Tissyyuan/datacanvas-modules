#coding=utf-8
#对连续型变量进行归一化处理，把数变为（0，1）之间的小数
#输入：df
#参数：
#输出：df_new
#---------------------------------------------------------------
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

def main(params, inputs, outputs):
    ### 读入数据 ###
    df = inputs.df
    df_new = outputs.df_new
    df = pd.read_pickle(df)
    
    ### 从变量中选择连续变量 ###
    df_x = df.iloc[:,:-1]
    df_y = df.iloc[:,-1]
    df_x = df_x.drop(df_x.select_dtypes(['object']),axis=1)
    
    ### 变量标准化 ###
    df_x = MinMaxScaler().fit_transform(df_x)
    
    ### 合并数据 ###
    df_combine = df_x.join(df_y)
    
    ### 输出数据集 ###
    df_combine.to_pickle(df_new)