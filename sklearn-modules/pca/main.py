#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datacanvas.new_runtime import DataCanvas
dc = DataCanvas(__name__)

from sklearn.decomposition import PCA
import pandas as pd
import pickle


def do_pca(df, pca_components):
    if pca_components == "#":
        components = None
    else:
        components = int(pca_components)
        
    pca = PCA(components).fit(df)
    explained_variance = pca.explained_variance_.reshape(1,df.size)
    return pd.DataFrame(explained_variance, columns=df.columns), pca.transform(df)

@dc.basic_runtime(spec_json="spec.json")
def my_module(rt, params, inputs, outputs):
    # TODO : Fill your code here
    df = pickle.load(open(inputs.df, 'rb'))

    explained, new_df = do_pca(df, params.pca_components.val)
    print(explained)
    print(new_def.head())

    pickle.dump(explained, open(outputs.explained, 'w'))
    pickle.dump(new_df, open(outputs.new_df, 'w'))

    
    print "Done"


if __name__ == "__main__":
    dc.run()
