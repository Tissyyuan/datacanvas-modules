# Update

## 180209

* [A] [Sample](#sample)
* [A] [ChurnLabel](#CL)
* [A] [ValueCounts](#VC)
* [A] [BucketLowFrequency](#BLF) 

## 180205

* [A] [HashingEncoder](#he)
* [A] [AdaBoost](#Ada)
* [A] [xgboost](#xg)
* [A] [StandardScaler](#sc)
* [A] [MinMaxScaler](#MM)
* [A] [SelectFromModel](#sfm)
* [A] [PearsonCorrelation](#pearson)
* [A] [RFE](#rfe)
* [A] [MINE](#mine)
* [A] [chi2](#chi) 
* [A] [FunctionTransformer](#ft)
* [A] [PolyNomialFeatures](#poly)


# Index

## customer_churn

* [Sample](#sample)
* [ChurnLabel](#CL)
* [ValueCounts](#VC)
* [BucketLowFrequency](#BLF) 
* [VariablesSelection](#VS)
* [DataPreprocessing](#DP)
* [MissingDrop](#MD)
* [DataTypes](#DataT)
* [MissingFill](#MFill)
* [MissingImpute](#MImpute)
* [MissingCheck](#MCheck)
* [AsType](#AsT)
* [ClassMapping](#classmap)
* [Dummy](#dummy)
* [QuantileTransformer](#QTrans)
* [Box](#box)

## calibration

## cluster

## covariance

## dataframe

* [ValueCounts](#VC)
* [BucketLowFrequency](#BLF) 
* [VariablesSelection](#VS)
* [DataTypes](#DataT)
* [MissingCheck](#MCheck)
 
## decomposition 

## discriminant_analysis

## ensemble

* [AdaBoost](#Ada)
* [xgboost](#xg)

## feature_extraction

## feature_selection

* [SelectFromModel](#sfm)
* [PearsonCorrelation](#pearson)
* [RFE](#rfe)
* [MINE](#mine)
* [chi2](#chi) 

## linear_model

## metrics

## model_selection

## multiclass

## naive_bayes

## neighbors

## neural_network

## pipeline

## preprocessing

* [HashingEncoder](#he)
* [StandardScaler](#sc)
* [MinMaxScaler](#MM)
* [FunctionTransformer](#ft)
* [PolyNomialFeatures](#poly)
* [Sample](#sample)
* [DataPreprocessing](#DP)
* [MissingDrop](#MD)
* [MissingFill](#MFill)
* [MissingImpute](#MImpute)
* [AsType](#AsT)
* [Dummy](#dummy)
* [ClassMapping](#classmap)
* [QuantileTransformer](#QTrans)
* [Box](#box)

## svm

## tree

## utils

# Module

## Hive_To_Dataframe

hive表转dataframe

### Param:

* delimiter: 数据分割符

* table_name: hive表名

### Input:

* jdbc_url: hive表地址

### Output:

* dataframe: 输出的df

## FromShow:查看dataframe

### Param:

* head: 显示的行数

### Input:

* dataframe: 要查看的dataframe

### Output:

* Table: 绘制的数据表

## DataPreprocess:数据预处理

### Param:

* frac: 采样百分比如0.1, 0.5 '#'表示取全量
* drop_columns: 待删除的字段如"x1",x2"
* LabelEncode: 要编码的分类字段

### Input:

* dataframe: 待处理的数据

### Output:

* df: 处理后的数据

## select:字段选择

### Param:

* col_name: 按列名选如"x1","x2","x3"
* col_no: 按行号选

### Input:

* data_source: 源数据

### Output:

* columns: 挑选出的字段

## LogisticRegression:逻辑回归

### Param:

* model_save_path: 模型保存路径
* model_name: 模型名称
* CV: 交叉验证次数
* metrics: 模型评估指标:"accuracy","f1_micro","f1_macro"

### Input:

* X: 自变量
* Y: 预测变量

### Output:

* model: 训练出的模型

## DecisionTreeClassifiter:决策树分类

### Param:

* CV: 交叉验证次数
* criterion: The function to measure the quality of a split. eg:gini
* import_feature: 返回的重要特征的阈值(< 1)如0.01
* min_sample_leaf: The minimum number of samples required to be at a leaf node
* max_depth: 最大深度
* metrics: 模型评估指标:"accuracy","f1_micro","f1_macro"
* model_save_path: 模型保存路径
* model_name: 模型名称
* min_samples_split: The minimum number of samples required to split an internal node

### Input:

* X: 自变量
* Y: 预测变量

### Output:

* model: 训练出的模型
* TreeShow: 决策树结构图
* import_X: 重要的特征变量

### MetriceShow:模型指标查看

### Input:

* model: 模型

### Output:

* metrics: 评估指标
* metrics_desribe: 评估指标统计

## ModelSelect:模型选择

### Param:

* metric: 评价指标

### Input:

* model_A: 模型
* model_B: 模型

### Output:

* better_model: 选出的最优模型

## Predict:预测

### Param:

* result_col_name: 预测结果的字段名

### Input:

* model: 模型
* X: 自变量

### Output:

* Y: 预测变量

## ResultEvaluation:预测结果评估

### Param:
* FigureType: 评估方法 如：roc

### Input:

* Real: 真实值
* Predict: 预测值

### Output:

* Show: 评估结果

## df2hive:dataframe存入hive

### Param:

* jdbc_url: hive连接串
* table_name: hive表名

### Input:

* data: 目标数据

## data_describe:字段统计

### input:

* data: 输入的dataframe

### output:

* describes: 数据描述信息

## imputer:缺失值填充

### input:

* data: 输入的dataframe

### output:

* data_new: 填充缺失值后的dataframe

## <a id="he">HashingEncoder</a>

分类变量哈希编码

## Tag:

* preprocessing

### Param:

* columns: 要编码的变量 如: x1,x2,x3 列名

### Input:

* data: 输入的dataframe

### Output:

* data_new: 编码后的dataframe


## OneHotEncoder:onehot编码

### Param:

* columns: 要编码的变量

### Input:

* data: 输入的dataframe

### Output:

* data_new: 编码后的dataframe

## OrdinalEncoder:不知道中文叫啥

### Param:

* columns: 要编码的变量

### Input:

* data: 输入的dataframe

### Output:

* data_new: 编码后的dataframe

## BinaryEncoder

### Param:

* columns: 要编码的变量

### Input:

* data: 输入的dataframe

### Output:

* data_new: 编码后的dataframe

## PolynomialEncoder

### Param:

* columns: 要编码的变量

### Input:

* data: 输入的dataframe

### Output:

* data_new: 编码后的dataframe

## BackwardDifferenceEncoder

### Param:

* columns: 要编码的变量

### Input:

* data: 输入的dataframe

### Output:

* data_new: 编码后的dataframe

## SumEncoder

### Param:

* columns: 要编码的变量

### Input:

* data: 输入的dataframe

### Output:

* data_new: 编码后的dataframe

## HelmertEncoder

### Param:

* columns: 要编码的变量

### Input:

* data: 输入的dataframe

### Output:

* data_new: 编码后的dataframe

## <a id="Ada">AdaBoost::ensemble</a>
集成算法

### Param:

* learning_rate: 学习速率
* n_estimators: 训练模型时的最大因子数量

### Input:

* x_train: 输入的自变量
* y_train: 输入的目标变量

### Output:

* adaboost_classifier: 训练后的adaboost模型

## <a id="xg">xgboost::ensemble</a>
集成算法

### Param:
* None

### Input:

* x_train: 输入的自变量
* y_train: 输入的目标变量

### Output:

* xgboost_classifier: 训练后的xgboost模型

## <a id="sc">StandardScaler::preprocessing</a>
数据标准化

### Param:

* None

### Input:

* df: 输入的dataframe

### Output:

* df_new: 标准化后的dataframe

## <a id="MM">MinMaxScaler::preprocessing</a>
数据归一化

### Param:
* None

### Input:

* df: 输入的dataframe

### Output:

* df_new: 归一化后的dataframe

## <a id="sfm">SelectFromModel::feature_selection</a>
特征选择方法

### Param:

* method: 基于树的特征选择还是基于范数的特征选择

### Input:

* x: 输入的自变量
* y: 输入的因变量

### Output:

* x_new: 特征选择后的自变量
* y_new: 同y

## <a id="pearson">PearsonCorrelation::feature_selection</a>
通过皮尔森相关系数筛选变量

### Param:
* corr_thel: 按输入阈值筛选变量

### Input:

* x: 输入的自变量

### Output:
* x_new: 相关性筛选后的自变量

## <a id="rfe">RFE::feature_selection</a>
递归特征选择法，用于特征筛选

### Param:

* step: 筛选时步长
* n_features: 保留的变量个数

### Input:

* x: 输入的自变量
* y: 输入的因变量

### Output:

* x_new: 特征选择后的自变量
* y_new: 同y

## <a id="mine">MINE::feature_selection</a>
最大信息系数(MIE)用于衡量两个变量线性或非线性的强度

### Param:

* type: 计算自变量间或是自变量与目标变量间的最大信息系数

### Input:

* x: 输入的自变量
* y: 输入的因变量

### Output:

* mic: 最大信息系数
* tic: 总信息系数

## <a id="chi">chi2::feature_selection</a>
特征选择方法，计算自变量与目标变量间的卡方统计量

### Param:

* sample_rate: 抽样样本比例
* n_features: 保留变量个数百分比

### Input:

* x: 输入的自变量
* y: 输入的因变量

### Output:

* x_new: 特征选择后的自变量
* y_new: 同y

## <a id="ft">FunctionTransformer::preprocessing</a>
将x传递给用户自定义的函数，并返回此函数的结果

### Param:

* None

### Input:

* x: 输入的dataframe

### Output:

* x_new: 转换后的dataframe

## <a id="poly">PolyNomialFeaturs</a>
生成多项式和交互变量

### Tag:

* preprocessing

### Param:

* degree: 多项式特征的程度
* interaction_only: 是否只包含交互项

### Input:

* x: 输入的dataframe

### Output:

* x_new: 转换后的dataframe

## <a id="sample">Sample</a>
对数据进行抽样

### Tag:

* customer_churn
* preprocessing

### Param:

* frac: 抽样比例

### Input:

* df: 输入的dataframe

### Output:

* df_new: 抽样后的dataframe

## <a id="CL">ChurnLabel</a>
定义客户流失标签，我们分为三个期：观察期(3个月)，稳定期(2个月)，表现期(3个月)；
先过滤掉稳定期季日均资产下降超过35%以上的客户，再根据表现期季日均同观察期季日均资产相比是否下降超过30%来判定改客户是否为流失客户。


### Tag:

* customer_churn

### Param:

* None

### Input:

* df: 输入的dataframe

### Output:

* df_new: 标签后的dataframe
* df_filter: 聚焦资产变化的dataframe

## <a id="VC">ValueCounts</a>
统计单个变量每一类的数量

### Tag:

* customer_churn
* dataframe

### Param:

* col: 要统计的变量名
 
### Input:

* df: 输入的dataframe

### Output:

* count: 统计结果

## <a id="BLF">BucketLowFrequency</a>
对类别变量进行处理：对单个变量中数量较少的类(百分比小于0.05)合并成一类，统一赋值为99，该步骤应在对变量进行编码之后进行。

### Tag:

* customer_churn
* dataframe

### Param:

* None
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 转换后的dataframe

## <a id="VS">VariablesSelection</a>
选取我们认为对客户流失行为有影响的变量。

### Tag:

* customer_churn
* dataframe

### Param:

* None
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 处理后的dataframe

## <a id="DP">DataPreprocessing</a>
对变量进行预处理：转换错误的变量类型；用0填补部分变量的缺失值(一些变量根据业务定义可以用0进行填补，比如交易金额)；对一些变量进行加工处理

### Tag:

* customer_churn
* preprocessing

### Param:

* None
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 处理后的dataframe

## <a id="MD">MissingDrop</a>
删除几乎拥有唯一值的字段(比如单个变量最大类别个数百分比大于95%)；删除缺失百分比大于一定比率的字段(比如类别变量大于30%，连续变量大于60%)。

### Tag:

* customer_churn
* preprocessing

### Param:

* percent_obj: object型变量删除阈值
* percent_non_obj: 非object型变量删除阈值
* percent_unique: 唯一值变量删除阈值
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 删除变量后的dataframe

## <a id="DataT">DataTypes</a>
探查数据类型

### Tag:

* customer_churn
* dataframe

### Param:

* None
 
### Input:

* df: 输入的dataframe

### Output:

* dtypes: 每个变量的数据类型

## <a id="MFill">MissingFill</a>
小比例缺失值用众数或中位数填充(例如，类别变量缺失小于10%时用众数填充，非类别变量缺失小于30%时用中位数填充)。

### Tag:

* customer_churn
* preprocessing

### Param:

* percent_obj: 类别变量填充阈值
* percent_non_obj：非类别变量填充阈值
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 填充后的dataframe

## <a id="MImpute">MissingImpute</a>
用传播算法对缺失值进行填充

### Tag:

* customer_churn
* preprocessing

### Param:

* lower_null_percent: 类别变量填充阈值下限
* upper_null_percent: 类别变量填充阈值上限 
* lower_null_percent1: 非类别变量填充阈值下限
* upper_null_percent1: 非类别变量填充阈值上限
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 填充后的dataframe

## <a id="MCheck">MissingCheck</a>
统计变量缺失百分比并以柱形图显示。

### Tag:

* customer_churn
* dataframe

### Param:

* None
 
### Input:

* df: 输入的dataframe

### Output:

* df_null: 缺失值百分比统计
* percent_plot: 缺失值百分比柱形图

## <a id="AsT">AsType</a>
转换变量类型

### Tag:

* customer_churn
* preprocessing

### Param:

* None
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 转换后的dataframe
* type: 检查变量数据类型

## <a id="classmap">ClassMapping</a>
将是字符的类别型变量映射为数值, 缺失值仍保持为np.nan。注意：类别型变量若已为数值则不做转换。

### Tag:

* customer_churn
* preprocessing

### Param:

* cols: 待转换的变量
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 转换后的dataframe

## <a id="dummy">Dummy</a>
对类别型变量哑编码(无论是类别中的字符还是数值)，缺失值也做了转换。

### Tag:

* customer_churn
* preprocessing

### Param:

* None
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 转换后的dataframe

## <a id="QTrans">QuantileTransformer</a>
对连续变量进行正态化处理。

### Tag:

* customer_churn
* preprocessing

### Param:

* None
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 转换后的dataframe

## <a id="box">Box</a>
对连续变量进行分箱处理。

### Tag:

* customer_churn
* preprocessing

### Param:

* box_type：按区间分段还是按分位数分段
* box_num：分箱个数
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 分箱后的dataframe