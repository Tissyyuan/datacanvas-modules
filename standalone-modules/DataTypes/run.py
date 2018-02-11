#coding=utf-8
#探查数据类型
#输入：df 
#参数：None
#输出：dtypes
#---------------------------------------------------------------
import pandas as pd
import pickle

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    df = pd.read_pickle(inputs.df)
    
    ### 统计每个变量的数据类型 ###
    dtypes = str(df.dtypes)
	
	### 输出结果 ###
    with open(outputs.dtypes, "w+") as out:
        out.write(dtypes) 
