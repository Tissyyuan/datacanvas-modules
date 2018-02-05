import pandas as pd
import category_encoders as ce

def main(params, inputs, outputs):
    columns_param = params.columns
    data = inputs.data
    data_new = outputs.data_new
    
    data_0 = pd.read_pickle(data)
    
    encoder = ce.HashingEncoder(cols=[col for col in columns_param.split(",")])
    data_1 = encoder.fit_transform(data_0)
    
    data_1.to_pickle(data_new)