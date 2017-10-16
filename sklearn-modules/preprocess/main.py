#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, Binarizer,\
    OneHotEncoder, Imputer, PolynomialFeatures, FunctionTransformer

from datacanvas.new_runtime import DataCanvas
dc = DataCanvas(__name__)

Outputs = None
Params = None

def get_columns(df):
    return df.columns

def standardscaler(df):
    new_ndarry = StandardScaler(Params).fit_transform(df)
    new_df = pd.DataFrame(new_ndarry, columns = get_columns(df))
    pkl.dump(new_df, open(Outputs, 'w'))

def minmaxscaler(df):
    new_ndarry =  MinMaxScaler(Params).fit_transform(df)
    new_df = pd.DataFrame(new_ndarry, columns = get_columns(df))
    pkl.dump(new_df, open(Outputs, 'w'))

def maxabsscaler(df):
    new_ndarry =  MaxAbsScaler(Params).fit_transform(df)
    new_df = pd.DataFrame(new_ndarry, columns = get_columns(df))
    pkl.dump(new_df, open(Outputs, 'w'))

def normalizer(df):
    new_ndarry =  Normalizer(Params).fit_transform(df)
    new_df = pd.DataFrame(new_ndarry, columns = get_columns(df))
    pkl.dump(new_df, open(Outputs, 'w'))

def binarizer(df):
    new_ndarry =  Binarizer(Params).fit_transform(df)
    new_df = pd.DataFrame(new_ndarry, columns = get_columns(df))
    pkl.dump(new_df, open(Outputs, 'w'))

def onehotencoder(df):
    new_ndarry =  OneHotEncoder(Params).fit_transform(df)
    new_df = pd.DataFrame(new_ndarry, columns = get_columns(df))
    pkl.dump(new_df, open(Outputs, 'w'))

def imputer(df):
    new_ndarry =  Imputer(Params).fit_transform(df)
    new_df = pd.DataFrame(new_ndarry, columns = get_columns(df))
    pkl.dump(new_df, open(Outputs, 'w'))

def polynomialfeatures(df):
    new_ndarry = PolynomialFeatures(Params).fit_transform(df)
    new_df = pd.DataFrame(new_ndarry, columns = get_columns(df))
    pkl.dump(new_df, open(Outputs, 'w'))

def pipe_standardscaler_transform(transformer_list):
    transformer_list.append(StandardScaler(Params))
    pkl.dump(transformer_list, open(Outputs, 'w'))

def pipe_minmaxscaler_transform(transformer_list):
    transformer_list.append(MinMaxScaler(Params))
    pkl.dump(transformer_list, open(Outputs, 'w'))

def pipe_maxabsscaler(transformer_list):
    transformer_list.append(MaxAbsScaler(Params))
    pkl.dump(transformer_list, open(Outputs, 'w'))

def pipe_normalizer(transformer_list):
    transformer_list.append(Normalizer(Params))
    pkl.dump(transformer_list, open(Outputs, 'w'))

def pipe_binarizer(transformer_list):
    transformer_list.append(Binarizer(Params))
    pkl.dump(transformer_list, open(Outputs, 'w'))

def pipe_onehotencoder(transformer_list):
    transformer_list.append(OneHotEncoder(Params))
    pkl.dump(transformer_list, open(Outputs, 'w'))

def pipe_imputer(transformer_list):
    transformer_list.append(Imputer(Params))
    pkl.dump(transformer_list, open(Outputs, 'w'))

def pipe_polynomialfeatures(transformer_list):
    transformer_list.append(PolynomialFeatures(Params))
    pkl.dump(transformer_list, open(Outputs, 'w'))

@dc.basic_runtime(spec_json="spec.json")
def my_module(rt, params, inputs, outputs):
    # TODO : Fill your code here
    Outputs = outputs.pkl
    if (params.PipeModel.val == "OFF"):
        df = pickle.load(open(inputs.df, 'rb'))
        if not(params.StandardScaler == None):
            Params = eval(params.StandardScaler)
            standardscaler(df)
        else if not(params.MinMaxScaler == None):
            Params = eval(params.MinMaxScaler)
            minmaxscaler(df)
        else if not(params.MaxAbsScaler == None):
            Params = eval(params.MaxAbsScaler)
            maxabsscaler(df)
        else if not(params.Normalizer == None):
            Params = eval(params.Normalizer)
            normalizer(df)
        else if not(params.Binarizer == None):
            Params = eval(params.Binarizer)
            binarizer(df)
        else if not(params.OneHotEncoder == None):
            Params = eval(params.OneHotEncoder)
            onehotencoder(df)
        else if not(params.Imputer == None):
            Params = eval(params.Imputer)
            imputer(df)
        else if not(params.PolynomialFeatures.val == None):
            Params = eval(params.PolynomialFeatures)
            polynomialFeatures(df)
        else:
            exit("module takes exactly 1 params at lest (0 given)")
    else if (params.pipemodel.val == "ON"):
        transformer_list = pickle.load(open(inputs.Transformer, 'rb'))
        if not(params.StandardScaler == None):
            Params = eval(params.StandardScaler)
            pipe_standardscaler_transform(transformer_list)
        else if not(params.MinMaxScaler == None):
            Params = eval(params.MinMaxScaler)
            pipe_minmaxscaler_transform(transformer_list)
        else if not(params.MaxAbsScaler == None):
            Params = eval(params.MaxAbsScaler)
            pipe_maxabsscaler_transform(transformer_list)
        else if not(params.Normalizer == None):
            Params = eval(params.Normalizer)
            pipe_normalizer_transform(transformer_list)
        else if not(params.Binarizer == None):
            Params = eval(params.Binarizer)
            pipe_binarizer_transform(transformer_list)
        else if not(params.OneHotEncoder == None):
            Params = eval(params.OneHotEncoder)
            pipe_onehotencoder_transform(transformer_list)
        else if not(params.Imputer == None):
            Params = eval(params.Imputer)
            pipe_imputer_transform(transformer_list)
        else if not(params.PolynomialFeatures.val == None):
            Params = eval(params.PolynomialFeatures)
            pipe_PolynomialFeatures_transform(transformer_list)
        else:
            exit("module takes exactly 1 params at lest (0 given)")
    else:
        exit("PipeModel can't be empty")
        
        print "Done"

if __name__ == "__main__":
    dc.run()
