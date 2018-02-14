#coding=utf-8
#描述：处理不均衡数据
#输入：x, y
#参数：None
#输出：x_resample, y_resample, meta_json
#---------------------------------------------------------------
import pandas as pd
import pickle
import json
from collections import Counter
from imblearn.over_sampling import SMOTE

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    x = pd.read_pickle(inputs.x)
    y = pd.read_pickle(inputs.y)
    
    ### 不均衡处理 ###
    sm = SMOTE(random_state = 0)
    x_resampled, y_resampled = sm.fit_sample(x, y)
    y_resampled = pd.Series(y_resampled)

    ### 输出1 ###
    pickle.dump(x_resampled, open(outputs.x_resample, 'wb'))
    pickle.dump(y_resampled, open(outputs.y_resample, 'wb'))
    
    ### 输出2 ###
    meta = \
    {
        'Original dataset shape': x.shape,
        'Resampled (SMOTE) dateset shape': x_resampled.shape,
        'Original non-churn vs. churn': Counter(y), 
        'Resampled (SMOTE) non-churn vs. churn': Counter(y_resampled)
    }
    meta_json = json.dumps(meta, indent=4)
    print("[meta]\n%s"%meta_json)
    with open(outputs.meta_json, 'w') as out:
        out.write(meta_json)
