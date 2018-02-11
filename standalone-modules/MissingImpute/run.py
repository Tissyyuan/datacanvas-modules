#coding=utf-8
#用传播算法对缺失值进行填充
#输入：df 
#参数：lower_null_percent, upper_null_percent, lower_null_percent1, upper_null_percent1
#输出：df_new

#-----------------------------------------------------------------
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    df = pd.read_pickle(inputs.df)
    
    ### 读入参数 ###
    lower_null_percent = params.lower_null_percent
    upper_null_percent = params.upper_null_percent
    lower_null_percent1 = params.lower_null_percent1
    upper_null_percent1 = params.upper_null_percent1
    
    ### 测试 ###
    #lower_null_percent = 10 
    #upper_null_percent = 30
    #lower_null_percent1 = 30
    #upper_null_percent1 = 60
    
    ### 计算每列空值比率 ###
    df_null = pd.DataFrame(data=df.dtypes,columns=['col_type'],index=df.dtypes.index) #列类型
    df_null.loc[:,'null_percent']=(df.isnull().sum()[:] * 100/ df.shape[0])  #计算每列的空值比率（%）
    df_null_object = df_null[df_null.col_type=='object'] #对象类型
    df_null_non_object = df_null[df_null.col_type!='object'] #非对象类型

    ### 需要传播的字段 ###
    spread_cols = list(df_null_object[(df_null_object.null_percent>=lower_null_percent) \
                                      & (df_null_object.null_percent<=upper_null_percent)].index) #对象类型
                                      
    regress_cols = list(df_null_non_object[(df_null_non_object.null_percent>=lower_null_percent1) \
                                      & (df_null_non_object.null_percent<=upper_null_percent1)].index) #非对象类型
                                      
    ### 非空值字段 ###
    not_null_cols = list(df_null[df_null.null_percent==0].index) 

    ### 定义模型 ###
    spread_model = RandomForestClassifier(max_features='sqrt',n_jobs=-1) #分类模型  
    regress_model=RandomForestRegressor(max_features='sqrt',n_jobs=-1)   #回归模型
    
    ### 模型填充缺失值 ###
    for spread_col in spread_cols:
        not_null_index = df[df[spread_col].isnull().values==False].index #非空值索引
        df_spread_x = df.loc[not_null_index, not_null_cols] #训练的特征变量
        df_spread_y = df.loc[not_null_index, spread_col] #训练的填充变量
        spread_weight = compute_sample_weight('balanced', df.loc[not_null_index, spread_col]) #填充列权重表
        spread_model.fit(df_spread_x, df_spread_y, spread_weight) #训练
        
        null_index = df[df[spread_col].isnull().values==True].index #空值索引
        df_spread_x = df.loc[null_index, not_null_cols] #预测的特征变量
        df_spread_y = spread_model.predict(df_spread_x)          #预测的填充变量
        df.loc[null_index, spread_col] = df_spread_y   #填充空值

    for regress_col in regress_cols:
        not_null_index = df[df[regress_col].isnull().values==False].index #非空值索引
        df_spread_x = df.loc[not_null_index, not_null_cols] #训练的特征变量
        df_spread_y = df.loc[not_null_index, regress_col] #训练的填充变量
        spread_weight = compute_sample_weight('balanced', df.loc[not_null_index, regress_col]) #填充列权重表
        regress_model.fit(df_spread_x, df_spread_y, spread_weight) #训练
        
        null_index = df[df[regress_col].isnull().values==True].index #空值索引
        df_spread_x = df.loc[null_index, not_null_cols] #预测的特征变量
        df_spread_y = regress_model.predict(df_spread_x)          #预测的填充变量
        df.loc[null_index, regress_col] = df_spread_y   #填充空值
    
    ### 输出结果测试 ###
    df_new = df.copy()
    print(df_new.head(10))
    
    ### 输出结果 ###
    df_new.to_pickle(outputs.df_new)

