#coding=utf-8
#将自变量和目标变量分开
#输入：df 
#参数：target
#输出：X, y
#---------------------------------------------------------------
import pandas as pd
import pickle

def main(params, inputs, outputs):
	
	### 读入数据 ###
	##df = pd.read_pickle(inputs.df)
	
	### 读入参数 ###
	target = params.target
	print(target)
	
	### 测试 ###
	#df = pd.read_csv(inputs.df)
	
    ### 分开X和y ###
	#y = df[[target]
	exec(("y=df[[%s]]" % target))
	exec(("X=df.drop([%s], axis=1)" % target))
	#X = df.drop(target, axis=1)
	
	### 输出结果测试 ###
	print(X.head(5))
	print(y.head(5))
	
	### 输出结果 ###
	pickle.dump(X, open(outputs.X, 'wb'))
	pickle.dump(y, open(outputs.y, 'wb'))