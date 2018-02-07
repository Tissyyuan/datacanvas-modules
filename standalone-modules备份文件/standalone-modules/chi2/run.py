#coding=utf-8
#卡方检验：特征选择方法，计算自变量与目标变量间的卡方统计量，保留值相对较大的变量。另外特征变量的值必须非负。
#输入：x, y
#参数：sample_rate, percent
#输出：x_new, y_new
#---------------------------------------------------------------
import pandas as pd 
from sklearn.feature_selection import SelectPercentile, chi2

def main(params, inputs, outputs):
    ### 读入训练数据 ###
    x = pd.read_pickle(inputs.x)
    y = pd.read_pickle(inputs.y)
    
    ### 读入参数 ###
    sample_rate = params.sample_rate
    percent = params.percent
    
    ### 对样本进行抽样 ###
    sample_index = x.sample(frac = sample_rate).index 
    x_sample = x.loc[sample_index, :]
    y_sample = y.loc[sample_index]
    
    ### 用卡方检验筛选变量 ###
    chi2_info = SelectPercentile(score_func = chi2, percentile = percent)
    chi2_info = chi2_info.fit(x_sample, y_sample)
    chi2_feature = chi2_info.get_support(indices=True)
    
    ### 生成输出 ###
    x_new = x.iloc[:, chi2_feature]
    y_new = y.copy()
    
    ### 输出结果 ###
    x_new.to_pickle(outputs.x_new)
    y_new.to_pickle(outputs.y_new)
    