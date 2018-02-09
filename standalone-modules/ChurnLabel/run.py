#coding=utf-8
#定义客户流失标签，我们分为三个期：观察期(3个月)，稳定期(2个月)，表现期(3个月)；
#先过滤掉稳定期季日均资产下降超过35%以上的客户，再根据表现期季日均同观察期季日均资产相比是否下降超过30%来判定改客户是否为流失客户。
#输入：df 
#参数：None
#输出：df_new, df_filter 
#--------------------------------------------------------------
import pandas as pd
import pickle

def main(params, inputs, outputs):
	
	### 读入数据 ###
	df = pd.read_pickle(inputs.df)
	df_label = df[['cust_no','day_id','avg_puo','avg_sp','avg_pe','avg_bf_0m']]
	
	### 聚焦大资产客户 ###
	df_lasset = df_label[df_label['avg_puo'] >= 50000]
	
	### 剔除缺失值 ###
	df_lasset = df_lasset.dropna()
	
	### 计算稳定期和表现期资产下降率 ###
	df_lasset['sp_asset_ratio'] = (df_lasset['avg_puo'] - df_lasset['avg_sp']) / df_lasset['avg_puo']
	df_lasset['pe_asset_ratio'] = (df_lasset['avg_puo'] - df_lasset['avg_pe']) / df_lasset['avg_puo']
	
	### 过滤掉稳定期季日均资产下降超过35%的客户 ###
	df_filter = df_lasset[(df_lasset['sp_asset_ratio'] < 0.35)]
	
	### 给客户打标签 ###
	df_filter['Churn'] = df_filter['pe_asset_ratio'].apply(lambda x: 1 if x>=0.3 else 0)
	
	### 在输入数据集中聚焦我们关注的客户，并且添加流失标签 ###
	df_new = df.loc[df_filter.index, :]
	df_new['Churn'] = df_filter['Churn']
	
	### 数据集输出 ###
	df_new.to_pickle(outputs.df_new)
	df_filter.to_pickle(outputs.df_filter) #聚焦资产的变化情况