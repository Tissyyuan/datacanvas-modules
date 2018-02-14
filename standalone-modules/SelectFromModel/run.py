#coding=utf-8
#特征选择中的一类方法（embedded嵌入类方法）。该方法是基于机器学习模型对特征进行打分的方法。
#输入：x, y
#参数：None
#输出：select_cols, meta_json, df_support
#---------------------------------------------------------------
import pandas as pd 
import pickle
import json
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

def main(params, inputs, outputs):
    
    ### 读入训练数据 ###
    x = pd.read_pickle(inputs.x)
    y = pd.read_pickle(inputs.y)
        
    ### 基于树的特征选择 ###
    clf = ExtraTreesClassifier(n_estimators = 100, criterion = 'entropy')
    clf = clf.fit(x,y)
    model = SelectFromModel(clf, prefit=True, threshold = 'median')
    df_support = pd.DataFrame(data = model.get_support(indices = False), index = x.columns, columns = ['support'])
    select_cols = list(df_support[df_support['support'] == True].index)
    df_support = str(df_support)
  
    ### 输出1 ###
    pickle.dump(select_cols, open(outputs.select_cols, 'wb')) 
        
    ### 输出2 ###
    meta = \
    {
        "Number of variables before selection": x.shape[1],
        "Number of variables after selection": len(select_cols),
        "Variables after selection": select_cols,
    }
    meta_json = json.dumps(meta, indent=4)
    print("[meta]\n%s"%meta_json)
    with open(outputs.meta_json, 'w') as out:
        out.write(meta_json)
        
    ### 输出3 ###
    with open(outputs.df_support, 'wb') as out:
        out.write(df_support)