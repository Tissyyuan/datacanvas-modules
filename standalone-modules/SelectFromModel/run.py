#coding=utf-8
#特征选择中的一类方法（embedded嵌入类方法）。该方法是基于机器学习模型对特征进行打分的方法。
#输入：x, y
#参数：method
#输出：x_new, y_new
#---------------------------------------------------------------
import pandas as pd 
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

def main(params, inputs, outputs):
    ### 读入训练数据 ###
    x = pd.read_pickle(inputs.x)
    y = pd.read_pickle(inputs.y)
    
    ### 读入参数 ###
    method = params.method
    
    ### 基于L1范数的特征选择 ###
    if method == 'L1':
        lsvc = LinearSVC(C=0.01, penalty='L1', dual=False).fit(x,y)
        model = SelectFromModel(lsvc, prefit=True)
        x_new = model.transform(x)
        y_new = y.copy()
        
    ### 基于树的特征选择 ###
    if method == 'Tree':
        clf = ExtraTreesClassifier()
        clf = clf.fit(x,y)
        model = SelectFromModel(clf, prefit=True)
        x_new = model.transform(x)
        y_new = y.copy()
    
    ### 输出结果 ###
    with open(outputs.x_new, "wb") as out:
        pickle.dump(x_new, out)
    with open(outpus.y_new, "wb") as out:
        pickle.dump(y_new, out)
    