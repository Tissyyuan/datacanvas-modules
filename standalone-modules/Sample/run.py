#coding=utf-8
#对数据进行抽样
#输入：df 
#参数：frac
#输出：df_new 
#---------------------------------------------------------------
import pandas as pd
import pickle
from datetime import datetime

def main(params, inputs, outputs):
	
	### 输入数据 ###
	df = pd.read_pickle(inputs.df)
	
	### 输入参数 ###
	sample = params.frac

    ### 数据抽样 ###
	NaN = "#"
	if sample != NaN:
		frac = float(sample)
		df_new = df.sample(frac=frac)
	
	### 输出数据 ###
	df_new.to_pickle(outputs.df_new)
	