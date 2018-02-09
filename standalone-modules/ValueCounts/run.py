#coding=utf-8
#统计单个变量每一类的数量
#输入：df 
#参数：col
#输出：count
#---------------------------------------------------------------
import pandas as pd
import pickle

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    df = pd.read_pickle(inputs.df)
    
    ### 读入参数 ###
    col = params.col
    
    ### 统计单个变量每一类的数量 ###
    count = str(df["L2"].value_counts())
	
	### 输出结果 ###
    with open(outputs.count, "w+") as out:
        out.write(count) 
	
	