#coding=utf-8
#描述：用训练好的模型生成预测结果
#输入：model, xtest, ytest 
#参数：None
#输出: meta_json, pred, prob
#---------------------------------------------------------------
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    model = pd.read_pickle(inputs.model)
    xtest = pd.read_pickle(inputs.xtest)
    ytest = pd.read_pickle(inputs.ytest)

    ### 预测结果 ###
    pred = model.predict(xtest)
    prob = model.predict_proba(xtest)
    prob = prob[:,1]
    print(pred[:5])
    print(prob[:5])
    
    ### 模型准确率评估 ###
    accuracy = accuracy_score(ytest, pred)
    auc = roc_auc_score(ytest, prob)
    
    ### 输出1 ###
    meta = \
    {
        "Accuracy of the model is: ": accuracy,
        "AUC score of the model is: ": auc 
    }
    meta_json = json.dumps(meta, indent=4)
    print("[meta]\n%s"%meta_json)
    with open(outputs.meta_json, 'w') as out:
        out.write(meta_json)
        
    ### 输出2 ###
    pred = str(pred)
    prob = str(prob)
    
    with open(outputs.pred, 'wb') as out:
        out.write(pred)
        
    with open(outputs.prob, 'wb') as out:
        out.write(prob)
