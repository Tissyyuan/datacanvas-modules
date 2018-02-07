#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datacanvas.new_runtime import BasicRuntime, DataCanvas
dc = DataCanvas(__name__)


import pickle
import pandas as pd
from sklearn.metrics import accuracy_score,\
                            average_precision_score,\
                            f1_score,\
                            log_loss,\
                            precision_score,\
                            recall_score,\
                            roc_auc_score

def get_scores(y_df, y_pred):
    metrics=('accuracy_score',  'f1_score_macro', 'f1_score_micro', 'f1_score_weighted', 'f1_score_None',
             'log_loss', 'precision_score_macro', 'precision_score_micro', 'precision_score_weighted', 'precision_score_None',
             'recall_score_macro', 'recall_score_micro', 'recall_score_weighted', 'recall_score_None',
            )
    score=[]
    score.append(accuracy_score(y_true, y_pred))
    opts = ('macro', 'micro', 'weighted', None)
    for average in opts:
        score.append(f1_score(y_true, y_pred, average=average))
    score.append(log_loss(y_true, y_pred))
    for average in opts:
        score.append(precision_score(y_true, y_pred, average=average))
    for average in opts:
        score.append(recall_score(y_true, y_pred, average=average))
    evaluation = list(zip(metrics, score))
    return pd.DataFrame(data = evaluation, columns=['Metrics', 'Score'])
                 
@dc.basic_runtime(spec_json="spec.json")
def my_module(rt, params, inputs, outputs):
    # TODO : Fill your code here
    y_true = pickle.load(open(inputs.y_true, 'r'))
    y_pred = pickle.load(open(inputs.y_pred, 'r'))
    
    metrics_scores = get_scores(y_true, y_pred)

    pickle.dump(Y, open(outputs.metrics_scores, 'w'))

    print "Done"


if __name__ == "__main__":
    dc.run()
