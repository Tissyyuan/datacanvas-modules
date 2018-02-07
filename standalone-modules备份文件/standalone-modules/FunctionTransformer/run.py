#coding=utf-8
#FunctionTransformer将x传递给用户自定义的函数，并返回此函数的结果。可用于Pipeline中。
#输入：x  数据必须全为数值型
#参数：
#输出：x_new 
#---------------------------------------------------------------
import pandas as pd 
from sklearn.preprocessing import FunctionTransformer

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    x = pd.read_pickle(inputs.x)
    
    ### 定义函数：去除第一列数据 ###
    def all_but_first_column(X):
        return X[:, 1:]
    
    ### 使用FunctionTransformer训练并转化 ###
    ft = FunctionTransformer(all_but_first_column)
    x_new = ft.fit_transform(x) 
    
    ### 结果输出 ###
    x_new.to_pickle(outputs.x_new)
    