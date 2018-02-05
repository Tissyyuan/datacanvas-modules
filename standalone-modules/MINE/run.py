#coding=utf-8
#最大的基于信息的非参数探索(Maximal Information-based Nonparametric Exploration)。其中最大信息系数(MIE)用于衡量两个变量线性或非线性的强度。
#输入：x, y
#参数：type
#输出：mic, tic
#-----------------------------------------------------------------
import numpy as np
from minepy import pstats, cstats

def main(params, inputs, outputs):
    ### 输入数据 ###
    X = inputs.x 
    Y = inputs.y 
    
    ### 输入参数 ###
    type = params.type
    
    ### 计算X变量间的最大信息系数MIC和总信息系数TIC ###
    if type == 'X之间':
        mic, tic = pstats(X, alpha=9, c=5, est="mic_e")

    ### 计算X与Y间的最大信息系数MIC和总信息系数TIC ###
    if type == 'X与Y之间':
        mic, tic =  cstats(X, Y, alpha=9, c=5, est="mic_e")

    ### 输出结果 ###
    pickle.dump(mic, open(outputs.mic, "wb"))
    pickle.dump(tic, open(outputs.tic, "wb"))
    
    