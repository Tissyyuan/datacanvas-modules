#coding=utf-8
#对连续变量进行分箱。
#输入：df 
#参数：box_type, box_num
#输出：df_new
#---------------------------------------------------------------
import pandas as pd
import pickle
from sklearn.preprocessing import QuantileTransformer

def main(params, inputs, outputs):
	
	### 读入数据 ###
	df = pd.read_pickle(inputs.df)
	
	### 读入参数 ###
	box_type = params.box_type
	box_num = params.box_num
	
	### 测试 ###
	#box_type = 1
	#box_num = 5
	
	### 选择连续型变量 ###
	trans_cols = df.drop(df.select_dtypes(['object']).columns, axis=1).columns #选择连续变量

	### 对连续变量进行分箱 ###
	for col_box in trans_cols:
		if box_type==0:
			df[col_box] = pd.DataFrame(pd.cut(df[col_box], box_num, labels=False, include_lowest=True), columns=[col_box], index=df.index)  #按区间分段
		else:
			df[col_box] = pd.DataFrame(pd.qcut(df[col_box], box_num, labels=False, duplicates='drop'), columns=[col_box], index=df.index)  #按分位数分段
	
	### 测试结果 ###
	print(df.head(5))
	
	### 数据输出 ###
	df_new = df.copy()
	pickle.dump(df_new, open(outputs.df_new, 'wb'))
   