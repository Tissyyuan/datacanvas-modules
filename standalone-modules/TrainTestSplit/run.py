#coding=utf-8
#描述：将数据分为训练集和测试集
#输入：x, y 
#参数：test_size
#输出：xtrain, xtest, ytrain, ytest
#---------------------------------------------------------------
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


def main(params, inputs, outputs):
    
    ### 读入数据 ###
    x = pd.read_pickle(inputs.x)
    y = pd.read_pickle(inputs.y)
    
    ### 读入参数 ###
    test_size = params.test_size
    
    ### 测试 ###
    #test_size = 0.3
    
    ### 将数据分为训练集和测试集 ###
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = test_size, random_state = 0)
   
    ### 测试结果 ###
    print(xtrain.shape)
    print(xtest.shape)
    print(ytrain.shape)
    print(ytest.shape)
   
    ### 输出结果 ###
    pickle.dump(xtrain, open(outputs.xtrain, 'wb'))
    pickle.dump(xtest, open(outputs.xtest, 'wb'))
    pickle.dump(ytrain, open(outputs.ytrain, 'wb'))
    pickle.dump(ytest, open(outputs.ytest, 'wb'))