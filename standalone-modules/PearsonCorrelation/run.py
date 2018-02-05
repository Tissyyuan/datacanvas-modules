#coding=utf-8
#用皮尔森相关性系数(Pearson Correlation Coefficient)计算特征变量间相关系数, 并筛选掉相关系数过高的变量。
#输入：x
#参数：corr_thel #特征变量相关性阈值
#输出：x_new
#-----------------------------------------------------------------
import pandas as pd

def main(params, inputs, outputs):
    
    ### 读入输入变量 ###
    x = pd.read_pickle(inputs.x)
    
    ### 读入参数 ###
    corr_thel = params.corr_thel
    
    ### 计算pearson相关系数 ###
    corr = x.corr(method='pearson') 
    
    ### 按输入阈值进行特征变量高相关性排除 ###
    var = corr.index
    
    for i in corr.index:
        for j in corr.columns:
            if (corr.loc[i,j]>corr_thel and i<j):
                var = var.drop(labels=i,errors='ignore')
                
    x_new = x[var]
    
    ### 输出筛选后的特征向量 ###
    x_new.to_pickle(outputs.x_new)