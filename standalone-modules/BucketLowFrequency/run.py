#coding=utf-8
#对类别变量进行处理：对单个变量中数量较少的类(百分比小于0.05)合并成一类，统一赋值为99，该步骤应在对变量进行编码之后进行。
#输入：df 
#参数：None
#输出：df_new
#---------------------------------------------------------------
import pandas as pd
import pickle

def main(params, inputs, outputs):
	
	### 读入数据 ###
	df = pd.read_pickle(inputs.df)
	
	### 对类别变量做归类转换 ###
	for var in df.select_dtypes(['object']).columns:
		for val in df[var].unique():
			percent = df[var].value_counts() / len(df)
			if percent[val] < 0.05:
				df.loc[df[var] == val, var] = 99 
	
	### 输出结果测试 ###
	print(df.head(10))
	
	### 输出结果 ###
	df_new = df.copy()
	pickle.dump(df_new, open(outputs.df_new, 'wb'))