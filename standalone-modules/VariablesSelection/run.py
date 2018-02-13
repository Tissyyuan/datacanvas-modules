#coding=utf-8
#选取我们认为对客户流失行为有影响的变量。
#输入：df 
#参数：None
#输出：df_new
#---------------------------------------------------------------
import pandas as pd
import pickle

def main(params, inputs, outputs):
  
  ### 读入数据 ###
  df = pd.read_pickle(inputs.df)
  
  ### 按变量业务解释进行分类 ###
  
  ## 无用字段 ##
  unused_columns=['corporation', 'day_id', 'cust_no']

  ## 预测标签字段 ##
  predict_columns=['Churn'] 

  ## 客户基本信息 ##
  # 未包含'MAX_EDUC_LVL_NAME', 'OCCUP_NAME', 'EMP_FLAG' #
  # 暂时不用'MAX_EDUC_LVL_COD'(空值太多), 'OCCUP_COD'(同unit_busntp) #
  cust_columns = ['OPEN_TIME','CUST_RELATION_YEARS','NATIVE_PLACE','cust_sex','cust_age','unit_busntp','cust_mrg_cond'] 
  cust_columns = [x.lower() for x in cust_columns]
  
  ## 产品持有信息 ##
  # 未包含'NOTICE_DEPOSIT_FLAG', 'CMON_NEW_ACCOUNT_FLAG'无(本月是否有新开理财帐户), 'WX_FLAG'无, 'CMON_NEW_TERM_FLAG'无 #
  ph_columns = ['salary_flag','FINANCE_FLAG','TERM_DEPOSIT_FLAG','TERM_CURR_DEPOSIT_FLAG','WITHHOLDING_CUST_FLAG',
        'PRD_COUNT','DEP_COUNT','LN_COUNT','PB_FLAG','MB_FLAG'] 
  ph_columns = [x.lower() for x in ph_columns]
  
  ## 客户购买行为信息 ##
  # 不用'lst_buy_fnpd_dt', 'lst_buy_term_dt' # 
  cbb_columns = ['ALL_TERM_DEPOSIT_COUNT','ALL_TERM_DEPOSIT_AMT','ALL_LN_COUNT','ALL_LN_AMT',
         'ALL_FINANCIAL_COUNT','ALL_FINANCIAL_AMT','TMON_BUY_FNPD_CNT',
         'TMON_BUY_FNPD_AMT','TMON_BUY_TERM_CNT','TMON_BUY_TERM_AMT' ] 
  cbb_columns = [x.lower() for x in cbb_columns]

  ## 产品行为信息 ##
  # 未包含'cmon_term_expire_flag_6m', 'cmon_finance_expire_flag_6m' #
  pb_columns = ['LAST_MONTH_TERM_AMT','LAST_MONTH_FINANCE_AMT','CMON_TERM_EXPIRE_FLAG','CMON_TERM_EXPIRE_FLAG_3M',
        'CMON_LOAD_EXPIRE_FLAG','CMON_LOAD_EXPIRE_FLAG_3M','CMON_FINANCE_EXPIRE_FLAG',
        'CMON_FINANCE_EXPIRE_FLAG_3M','CMON_NEW_TEMR_FLAG',
        'CMON_NEW_FINANCE_FLAG','CMON_TERM_EXPIRE_AMT','CMON_TERM_EXPIRE_COUNT',
        'CMON_TERM_EXPIRE_AMT_AF3M','CMON_TERM_EXPIRE_COUNT_AF3M','CMON_TERM_EXPIRE_AMT_AF6M',
        'CMON_TERM_EXPIRE_COUNT_AF6M','CMON_FINANCE_EXPIRE_AMT','CMON_FINANCE_EXPIRE_COUNT',
        'CMON_FINANCE_EXPIRE_AMT_AF3M','CMON_FINANCE_EXPIRE_COUNT_AF3M',
        'CMON_FINANCE_EXPIRE_AMT_AF6M','CMON_FINANCE_EXPIRE_COUNT_AF6M'] 
  pb_columns = [x.lower() for x in pb_columns]

  ## 资产信息 ##
  # 暂时不用'S1_AVG_BAL', 'S2_AVG_BAL', 'S3_AVG_BAL', 'S4_AVG_BAL' #
  asset_columns = ['E_DP_BAL','E_LN_BAL','ALL_ASSET','DEPOSIT_AVG_Q','FINANCE_AVG_Q'] 
  asset_columns = [x.lower() for x in asset_columns]

  ## 交易行为信息 ##
  # 不用'last_tx_dt' #
  tb_columns = ['M0_ACC_CNT','M1_ACC_CNT','M2_ACC_CNT','M3_ACC_CNT','M4_ACC_CNT',
        'M5_ACC_CNT','M6_ACC_CNT','M7_ACC_CNT','M8_ACC_CNT','M9_ACC_CNT',
        'M10_ACC_CNT','M11_ACC_CNT','M12_ACC_CNT','M0_TX_BAL','M1_TX_BAL',
        'M2_TX_BAL','M3_TX_BAL','M4_TX_BAL','M5_TX_BAL','M6_TX_BAL','M7_TX_BAL',
        'M8_TX_BAL','M9_TX_BAL','M10_TX_BAL','M11_TX_BAL','M12_TX_BAL',
        'LAST_TX_BAL','EBANK_TX_CNT_3M','EBANK_TX_BAL_3M','MBANK_TX_CNT_3M',
        'MBANK_TX_BAL_3M','PAY_TX_CNT_3RD_3M','PAY_TX_BAL_3RD_3M','EBANK_TX_CNT_6M',
        'EBANK_TX_BAL_6M','MBANK_TX_CNT_6M','MBANK_TX_BAL_6M','PAY_TX_CNT_3RD_6M',
        'PAY_TX_BAL_3RD_6M','EBANK_TX_CNT_1Y','EBANK_TX_BAL_1Y','MBANK_TX_CNT_1Y',
        'MBANK_TX_BAL_1Y','PAY_TX_CNT_3RD_1Y','PAY_TX_BAL_3RD_1Y']
  tb_columns = [x.lower() for x in tb_columns]

  ### 选择模型需要的变量 ###
  feature_columns= predict_columns + cust_columns + ph_columns + cbb_columns + pb_columns + asset_columns + tb_columns  

  ### 生成数据集 ###
  df_new = df[feature_columns]

  ### 输出结果测试 ###
  print(df_new.shape)

  ### 输出结果 ###
  pickle.dump(df_new, open(outputs.df_new, 'wb'))
