import pandas as pd
from imblearn.over_sampling import SMOTE

def main(params, inputs, outputs):
    col_x_param = params.col_x
    col_y_param = params.col_y
    data = inputs.data
    data_new = outputs.data_new
    
    data = pd.read_pickle(data)
    exec("data_x = data.iloc[%s]"%col_x_param)
    exec("data_y = data.iloc[%s]"%col_y_param)
    
    sm = SMOTE(random_state=42)
    data_x_new, data_y_new = sm.fit_sample(data_x, data_y)
    
    data_x_new = pd.DataFrame(data_x_new, columns = data_x.columns)
    data_y_new = pd.DataFrame(data_y_new, columns = data_y.columns)
    
    df_new = data_x_new.join(data_y_new)
    df_new.to_pickle(df_new)