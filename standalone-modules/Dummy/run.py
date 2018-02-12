#coding=utf-8
#对类别型变量哑编码(无论是类别中的字符还是数值)，缺失值也做了转换。
#输入：df 
#参数：None
#输出：df_new
#---------------------------------------------------------------
import pandas as pd
import pickle

def main(params, inputs, outputs):
	
	### 读入数据 ###
	df = pd.read_pickle(inputs.df)
	
	### 对类别变量做哑编码转换 ###
	df_new = pd.get_dummies(df)
	
	### 输出结果测试 ###
	print(df_new.head(5))
	
	### 输出结果 ###
	df_new.to_pickle(outputs.df_new)
