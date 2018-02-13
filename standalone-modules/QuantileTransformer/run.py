#coding=utf-8
#对连续变量进行正态化处理。
#输入：df 
#参数：None
#输出：df_new
#---------------------------------------------------------------
import pandas as pd
import pickle
from sklearn.preprocessing import QuantileTransformer

def main(params, inputs, outputs):
	
	### 读入数据 ###
	df = pd.read_pickle(inputs.df)
	
	### 选择连续型变量 ###
	trans_cols = df.drop(df.select_dtypes(['object']).columns, axis=1).columns #选择连续变量
	
	### 参数设置 ###
	sample = df.shape[0]  #采样数量
	n_quantiles = 100 #离散化数目
	output_distribution = 'normal'
	
	### 正态化 ###
	qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution, subsample=sample) #正态化
	df[trans_cols] = pd.DataFrame(qt.fit_transform(df[trans_cols]), columns=[trans_cols], index=df.index) #
	
	### 输出结果测试 ###
	df_new = df.copy()
	print(df_new.head(5))
	
	### 输出结果 ###
	pickle.dump(df_new, open(outputs.df_new, 'wb'))