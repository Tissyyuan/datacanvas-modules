#coding=utf-8
#描述：堆栈模型：分为两层，第一层是几个模型的集合，第二层是单独的一个模型，用第一层几个模型的输出作为第二层的输入来训练元模型。
#输入：xtrain, xtest, ytrian, ytest 
#参数：None
#输出：model, meta_json
#---------------------------------------------------------------
import pandas as pd
import pickle
import json
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from mlens.ensemble import SuperLearner

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    xtrain = pd.read_pickle(inputs.xtrain)
    xtest = pd.read_pickle(inputs.xtest)
    ytrain = pd.read_pickle(inputs.ytrain)
    ytest = pd.read_pickle(inputs.ytest)
    
    ### 基础模型 ###
    def get_models():
    ## Generate a library of base learners. ##
        nb = GaussianNB()
        svc = SVC(C=100, probability=True)
        knn = KNeighborsClassifier(n_neighbors=3)
        lr = LogisticRegression(C=100, random_state=0)
        nn = MLPClassifier((80, 10), early_stopping=False, random_state=0)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=0)
        rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=0)

        models = {'svm': svc,
                'knn': knn,
                'naive bayes': nb,
                'mlp-nn': nn,
                'random forest': rf,
                 'gbm': gb,
                 'logistic': lr,
                }
                
        return models
    
    base_learners = get_models()
    
    ### 元模型 ###
    meta_learner = SVC(C=100, probability=True)
    
    ### 训练模型 & 评估模型效果 ###
    ## Instantiate the ensemble with 10 folds ##
    sl = SuperLearner(
        folds=10,
        random_state=0,
        verbose=2,
        backend="multiprocessing"
                    )
    
    ## Add the base learners and the meta learner ##
    sl.add(list(base_learners.values()), proba=True) 
    sl.add_meta(meta_learner, proba=True)

    ## Train the ensemble ##
    sl.fit(xtrain, ytrain)

    ## Predict the test set ##
    p_sl = sl.predict_proba(xtest)
    pred_sl = sl.predict(xtest)
    roc_auc_score = roc_auc_score(ytest, p_sl[:, 1])
    accuracy_score = accuracy_score(ytest, pred_sl)
    
    ### 测试结果 ###
    print("\nSuper Learner ROC-AUC score: %.3f" %roc_auc_score)
    
    ### 输出1 ###
    pickle.dump(sl, open(outputs.model, 'wb'))
    
    ### 输出2 ###
    meta = \
    {
        "roc_auc_score": roc_auc_score,
        "accuracy_score": accuracy_score
    }
    meta_json = json.dumps(meta, indent=4)
    print("[meta]\n%s"%meta_json)
    with open(outputs.meta_json, 'w') as out:
        out.write(meta_json)