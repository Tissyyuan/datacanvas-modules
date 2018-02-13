#coding=utf-8
#对变量进行预处理：转换出现错误的变量类型；用0填补部分变量的缺失值(一些变量根据业务定义可以用0进行填补，比如交易金额)；对一些变量进行加工处理。
#输入：df 
#参数：None
#输出：df_new
#---------------------------------------------------------------
import pandas as pd
import pickle

def main(params, inputs, outputs):
  
    ### 读入数据 ###
    df = pd.read_pickle(inputs.df)
  
    ### 对出现错误的变量类型进行转换 ###
    unnormal_columns=['open_time', 'cust_relation_years', 'all_term_deposit_amt', 'all_ln_amt', 
                  'all_financial_amt', 'tmon_buy_fnpd_cnt', 
                  'tmon_buy_fnpd_amt', 'tmon_buy_term_amt', 
                  'last_month_term_amt', 'last_month_finance_amt',
                 'cmon_term_expire_amt', 'cmon_term_expire_amt_af3m', 'cmon_term_expire_amt_af6m', 
                  'cmon_finance_expire_amt', 'cmon_finance_expire_amt_af3m', 'cmon_finance_expire_amt_af6m',
                  'e_dp_bal', 'e_ln_bal', 'all_asset', 'deposit_avg_q','finance_avg_q', 'm0_tx_bal', 'm1_tx_bal', 'm2_tx_bal', 'm3_tx_bal', 'm4_tx_bal', 'm5_tx_bal',
                'm6_tx_bal', 'm7_tx_bal', 'm8_tx_bal', 'm9_tx_bal', 'm10_tx_bal', 'm11_tx_bal', 'm12_tx_bal', 
                  'last_tx_bal', 'ebank_tx_bal_3m', 'mbank_tx_bal_3m', 'pay_tx_bal_3rd_3m', 'ebank_tx_bal_6m', 
                  'mbank_tx_bal_6m', 'pay_tx_bal_3rd_6m', 'ebank_tx_bal_1y', 'mbank_tx_bal_1y', 'pay_tx_bal_3rd_1y'] #异常的连续特征字段，本应该为FLOAT,却标示为OBJECT
    df[unnormal_columns]=df[unnormal_columns].astype('float32')
  
    ### 对部分变量空值替换为0 ###
    tx_amt_null_columns=['last_month_finance_amt', 'cmon_term_expire_amt', 'cmon_term_expire_count', 'all_financial_count',
        'all_financial_amt', 'pay_tx_cnt_3rd_6m', 'pay_tx_bal_3rd_6m', 'cmon_finance_expire_amt_af3m', 'cmon_finance_expire_count_af3m', 
        'cmon_finance_expire_amt_af6m', 'cmon_finance_expire_count_af6m', 'cmon_term_expire_amt_af3m', 'cmon_term_expire_count_af3m', 
        'last_month_term_amt', 'ebank_tx_cnt_3m', 'ebank_tx_bal_3m', 'mbank_tx_cnt_3m', 'mbank_tx_bal_3m', 'pay_tx_bal_3rd_3m',
        'pay_tx_cnt_3rd_3m', 'all_term_deposit_count', 'all_term_deposit_amt', 'mbank_tx_cnt_6m', 'mbank_tx_bal_6m', 'ebank_tx_bal_6m',
        'ebank_tx_cnt_6m', 'pay_tx_bal_3rd_1y', 'pay_tx_cnt_3rd_1y', 'mbank_tx_cnt_1y', 'mbank_tx_bal_1y', 'cmon_term_expire_amt_af6m', 
        'cmon_term_expire_count_af6m', 'ebank_tx_cnt_1y', 'ebank_tx_bal_1y', 'tmon_buy_term_cnt', 'tmon_buy_term_amt', 
        'cmon_finance_expire_amt', 'cmon_finance_expire_count','dep_count', 'prd_count','last_tx_bal'] #涉及到交易笔数和金额的空值替换为0
    df[tx_amt_null_columns] = df[tx_amt_null_columns].fillna(0)
    #mbank_tx_cnt_1y 近一年手机银行交易次数
    #cmon_finance_expire_amt_af3m 未来三个月理财产品到期金额
    #last_tx_bal 最近一次交易金额(近一年或两年)

    ### 根据原有变量添加新变量 ###
    
    ## 是否有产品到期 ##
    df['cmon_term_expire_flag_6m'] = df['cmon_term_expire_count_af6m'].apply(lambda x:1 if x>0 else 0)
    df['cmon_finance_expire_flag_6m'] =df['cmon_finance_expire_count_af6m'].apply(lambda x:1 if x>0 else 0)
    df['cmon_term_expire_flag_3m'] = df['cmon_term_expire_count_af3m'].apply(lambda x:1 if x>0 else 0)
    df['cmon_finance_expire_flag_3m'] = df['cmon_finance_expire_count_af3m'].apply(lambda x:1 if x>0 else 0)
    df['cmon_term_expire_flag'] = df['cmon_term_expire_count'].apply(lambda x:1 if x>0 else 0)
    df['cmon_finance_expire_flag'] = df['cmon_finance_expire_count'].apply(lambda x:1 if x>0 else 0)
    
    ## 累计交易次数 ##
    df['m1_3_tx_cnt'] = df['m1_acc_cnt'] + df['m2_acc_cnt'] + df['m3_acc_cnt'] #前1-3月交易次数
    df['m4_6_tx_cnt'] = df['m4_acc_cnt'] + df['m5_acc_cnt'] + df['m6_acc_cnt'] #前4-6月交易次数
    df['m7_9_tx_cnt'] = df['m7_acc_cnt'] + df['m8_acc_cnt'] + df['m9_acc_cnt'] #前7-9月交易次数
    df['m10_12_tx_cnt'] = df['m10_acc_cnt'] + df['m11_acc_cnt'] + df['m12_acc_cnt'] #前10-12月交易次数
  
    ## 月平均交易金额 ##
    df['m1_3_tx_bal'] = (df['m1_tx_bal'] + df['m2_tx_bal'] + df['m3_tx_bal'])/3 #前1-3月平均交易金额
    df['m4_6_tx_bal'] = (df['m4_tx_bal'] + df['m5_tx_bal'] + df['m6_tx_bal'])/3 #前4-6月平均交易金额
    df['m7_9_tx_bal'] = (df['m7_tx_bal'] + df['m8_tx_bal'] + df['m9_tx_bal'])/3 #前7-9月平均交易金额
    df['m10_12_tx_bal'] = (df['m10_tx_bal'] + df['m11_tx_bal'] + df['m12_tx_bal'])/3 #前10-12月平均交易金额

    ## 删除无用变量 ##
    df.drop(['m1_acc_cnt','m2_acc_cnt','m3_acc_cnt','m4_acc_cnt','m5_acc_cnt','m6_acc_cnt','m7_acc_cnt',
                'm8_acc_cnt','m9_acc_cnt','m10_acc_cnt','m11_acc_cnt','m12_acc_cnt','m1_tx_bal','m2_tx_bal','m3_tx_bal'
                ,'m4_tx_bal','m5_tx_bal','m6_tx_bal','m7_tx_bal','m8_tx_bal','m9_tx_bal','m10_tx_bal','m11_tx_bal','m12_tx_bal'], 1, inplace=True)

    ### 输出结果测试 ###
    print(df.head(10))
  
    ### 输出结果 ###
    df_new = df.copy()
    pickle.dump(df_new, open(outputs.df_new, 'wb'))