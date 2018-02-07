#coding=utf-8
#对连续型变量进行标准化处理，使处理后变量符合高斯分布
#输入：df
#参数：
#输出：df_new
#---------------------------------------------------------------
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def main(params, inputs, outputs):
    ### 读入数据 ###
    df = inputs.df
    df_new = outputs.df_new
    df = pd.read_pickle(df)
    
    ### 从变量中选择连续变量 ###
    df_x = df.iloc[:,:-1]
    df_y = df.iloc[:,-1]
    df_standard = df_x.drop(df_x.select_dtypes(['object']),axis=1)
    df_label = df_x.select_dtypes(['object'])
    
    ### 变量标准化 ###
    df_standard = StandardScaler().fit_transform(df_standard)
    df_standard = pd.DataFrame(df_standrad, columns=df_x.drop(df_x.select_dtypes(['object']),axis=1).columns)
    
    ### 合并数据 ###
    df_combine = df_standard.join(df_label)
    
    ### 输出数据集 ###
    df_combine.to_pickle(df_new)