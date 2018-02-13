#coding=utf-8
#卡方检验：特征选择方法，计算自变量与目标变量间的卡方统计量，保留值相对较大的变量。另外特征变量的值必须非负。
#输入：x, y
#参数：sample_rate, percent
#输出：x_new, y_new, stat
#---------------------------------------------------------------
import pandas as pd 
import numpy as np
from sklearn.feature_selection import SelectPercentile, chi2

def main(params, inputs, outputs):
    ### 读入训练数据 ###
    x = pd.read_pickle(inputs.x)
    y = pd.read_pickle(inputs.y)
    
    ### 读入参数 ###
    sample_rate = params.sample_rate
    percent = params.percent
    
    ### 测试 ###
    #x = pd.read_csv(inputs.x)
    #y = pd.read_csv(inputs.y)
    #sample_rate = 0.3
    #percent = 70
    
    ### 对样本进行抽样 ###
    sample_index = x.sample(frac = sample_rate).index 
    x_sample = x.loc[sample_index, :]
    y_sample = y.loc[sample_index]
    
    ### 用卡方检验筛选变量 ###
    chi2_info = SelectPercentile(score_func = chi2, percentile = percent)
    chi2_info = chi2_info.fit(x_sample, y_sample)
    chi2_feature = chi2_info.get_support(indices=True)
    chi2_score=chi2_info.scores_  #卡方检验特征得分
    chi2_pv = np.round(chi2_info.pvalues_,4)  #卡方检验p-value
    
    ### 生成输出 ###
    x_new = x.iloc[:, chi2_feature]
    y_new = y.copy()
    
    ### 变量卡方值统计表 ###
    df_chi2 = pd.DataFrame(data = chi2_info.get_support(indices=False), index = x.columns, columns=['support'])
    df_chi2['chi2_score'] = chi2_score
    df_chi2['p-values'] = chi2_pv
    stat = df_chi2.sort_values(['chi2_score'], ascending= False)
    stat = str(stat)
    
    ### 打印 ###
    print(x_new.head())
    print(y_new.head())
    
    ### 输出结果 ###
    pickle.dump(x_new, open(outputs.x_new, 'wb'))
    pickle.dump(y_new, open(outputs.y_new, 'wb'))
    with open(outputs.stat, "w+") as out:
        out.write(stat)
    