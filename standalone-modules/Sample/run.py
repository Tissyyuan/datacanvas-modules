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
	print "001", datetime.now()
	sample = params.frac
	dataframe = inputs.df
	print "002", datetime.now()
	NaN = "#"
	if sample != NaN:
		frac = float(sample)
		df_new = df.sample(frac=frac)
	print "003", datetime.now()
	df_new.to_pickle(outputs.df_new)