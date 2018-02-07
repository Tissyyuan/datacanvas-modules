#coding=utf-8
#建立xgboost模型并使用训练集训练
#输入：x_train,y_train（dataframe)
#参数：
#输出：xgboost_classifier
#---------------------------------------------------------------
import pandas as pd 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def main(params, inputs, outputs):
    ### 读入训练数据 ###
    x_train = pd.read_pickle(inputs.x_train)
    y_train = pd.read_pickle(inputs.y_train)
    
    ### 训练模型 ###
    xgboost_classifier = XGBClassifier()
    xgboost_classifier.fit(X_train, y_train)
    
    ### 预测结果 ###
    pred = xgboost_classifier.predict(x_train)
    pred = [round(value) for value in pred] #预测结果是概率，所以要binary
    
    ## 模型评估 ###
    accuracy = accuracy_score(y_train, pred)
    
    ### 模型输出 ###
    with open(outputs.xgboost_classifier, "wb") as out:
        pickle.dump(xgboost_classifier, out) 