#coding=utf-8
#用皮尔森相关性系数(Pearson Correlation Coefficient)计算特征变量间相关系数, 并筛选掉相关系数过高的变量。
#输入：x
#参数：corr_thel #特征变量相关性阈值
#输出：x_new, heatmap
#-----------------------------------------------------------------
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def main(params, inputs, outputs):
    
    ### 读入输入变量 ###
    x = pd.read_pickle(inputs.x)
    
    ### 读入参数 ###
    #corr_thel = params.corr_thel
     
    ### 测试 ###
    corr_thel = 0.9
    
    ### 类别型转换为数值型 ###
    cols = x.select_dtypes(['object']).columns
    x[cols] = x[cols].astype('int')
    
    ### 计算pearson相关系数 ###
    corr = x.corr(method='pearson') 
    
    ### 按输入阈值进行特征变量高相关性排除 ###
    var = corr.index
    
    for i in corr.index:
        for j in corr.columns:
            if (corr.loc[i,j]>corr_thel and i<j):
                var = var.drop(labels=i,errors='ignore')
                
    x_new = x[var]
    
    ### 相关性热力图 ###
    corr1 = x_new.corr(method='pearson') 
    plt.figure(figsize = (10,10))
    sns.heatmap(corr1, square = True, linewidth = 0.3, annot = False, cmap = 'Reds')
    plt.yticks(rotation = 0)
    plt.savefig(outputs.heatmap, format = "png")
    
    ### 数值型转换回类别型 ###
    cols2 = list(var&cols)
    x_new[cols2] = x_new[cols2].astype('object')
    
    ### 测试输出 ###
    print(cols)
    print(x_new.shape)
    print(str(x_new.dtypes))
    
    ### 输出筛选后的特征向量 ###
    pickle.dump(x_new, open(outputs.x_new, 'wb'))
    