#coding=utf-8
#描述：对RFE和SelectFromModel进行特征选择后的变量进行合并(取并集)。
#输入：x, y, rfe_cols, select_cols 
#参数：None
#输出：x_new, y_new, meta_json
#---------------------------------------------------------------
import pandas as pd
import pickle
import json

def main(params, inputs, outputs):
    
    ### 读入数据 ###
    x = pd.read_pickle(inputs.x)
    y = pd.read_pickle(inputs.y)
    rfe_cols = set(pd.read_pickle(inputs.rfe_cols))
    select_cols = set(pd.read_pickle(inputs.select_cols))
    
    ### 对两个变量集取并集 ###
    cols = list(rfe_cols|select_cols)

    ### 生成筛选后的数据集 ###
    x_new = x[cols]
    y_new = y.copy()
    
    ### 输出1 ###
    pickle.dump(x_new, open(outputs.x_new, 'wb'))
    pickle.dump(y_new, open(outputs.y_new, 'wb'))
    
    ### 输出2 ###
    meta = \
    {
        "Number of variables before selection": x.shape[1],
        "Number of variables after union": len(cols),
        "Variables after union": cols
    }
    meta_json = json.dumps(meta, indent=4)
    print("[meta]\n%s"%meta_json)
    with open(outputs.meta_json, 'w') as out:
        out.write(meta_json)
