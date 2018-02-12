#coding=utf-8
#统计变量缺失百分比并以柱形图显示。
#输入：df 
#参数：None
#输出：df_null, percent_plot
#---------------------------------------------------------------
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    df = pd.read_csv(inputs.df)
    
    ### 计算每列空值比率 ###
    df_null=pd.DataFrame(data=df.dtypes,columns=['col_type'],index=df.dtypes.index) #列类型
    df_null.loc[:,'null_percent']=(df.isnull().sum()[:] * 100/ df.shape[0])  #计算每列的空值比率（%）
    
    ### 输出零值缺失百分比柱形图 ###
    df_null.null_percent.plot(kind='bar',figsize=(100,20),sort_columns=True)
    plt.savefig(outputs.percent_plot,format = "png")
    
    ### 输出零值缺失百分比 ###
    df_null = str(df_null)
    with open(outputs.df_null, 'w+') as output:
        output.write(df_null)
    