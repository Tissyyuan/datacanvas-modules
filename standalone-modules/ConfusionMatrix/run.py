#coding=utf-8
#描述：混淆矩阵
#输入：model, xtest, ytest
#参数：None
#输出：cnf_matrix, report, cnf_matrix_plot
#---------------------------------------------------------------
import pandas as pd
import pickle
import itertools
from mlens.ensemble import SuperLearner
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main(params, inputs, outputs):
    
    ### 读入模型和数据 ###
    model = pd.read_pickle(inputs.model)
    xtest = pd.read_pickle(inputs.xtest)
    ytest = pd.read_pickle(inputs.ytest)
    
    ### 预测 ###
    prob = model.predict_proba(xtest)[:, 1]
    pred = (prob > 0.5) * 1
    
    ### classification reprot ###
    report = classification_report(ytest, pred)
    report = str(report)
    
    ### confusion matrix ###
    cnf_matrix = confusion_matrix(ytest, pred)
    
    ### confusion matrix plot ###
    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    ## Normalization can be applied by setting `normalize=True` ##
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    plt.figure(figsize=(5,5))
    class_names = ['Non-Churn', 'Churn']
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    plt.savefig(outputs.cnf_matrix_plot, format = "png")
    
    cnf_matrix = str(cnf_matrix)
    
    ### 输出结果 ###
    with open(outputs.cnf_matrix, 'wb') as out:
        out.write(cnf_matrix)
    
    with open(outputs.report, 'wb') as out:
        out.write(report)
