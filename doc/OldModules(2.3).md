## Hive_To_Dataframe

hive表转dataframe

#### Param:

* delimiter: 数据分割符

* table_name: hive表名

#### Input:

* jdbc_url: hive表地址

#### Output:

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

## <a id="sfm">SelectFromModel</a>
特征选择中的一类方法（embedded嵌入类方法）。该方法是基于机器学习模型对特征进行打分的方法。

### Tag:

* feature_selection

### Param:

* None

### Input:

* x: 输入的自变量
* y: 输入的因变量

### Output:

* select_cols: 筛选后的变量
* meta_json: 统计值
* df_support: 是否选择该变量的dataframe

## <a id="pearson">PearsonCorrelation</a>
通过皮尔森相关系数筛选变量

### Tag:

* feature_selection

### Param:
* corr_thel: 按输入阈值筛选变量

### Input:

* x: 输入的自变量

### Output:
* x_new: 相关性筛选后的自变量dataframe
* heatmap: 热力图


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

## <a id="chi">chi2</a>
特征选择方法，计算自变量与目标变量间的卡方统计量

### Tag:

* feature_selection

### Param:

* sample_rate: 抽样样本比例
* percent: 保留变量个数百分比

### Input:

* x: 输入的自变量
* y: 输入的因变量

### Output:

* x_new: 特征选择后的自变量
* y_new: 同y
* stat: 卡方统计后各变量的卡方分数和p值

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

* preprocessing

### Param:

* frac: 抽样比例

### Input:

* df: 输入的dataframe

### Output:

* df_new: 抽样后的dataframe


## <a id="DP">DataPreprocessing</a>
对变量进行预处理：转换错误的变量类型；用0填补部分变量的缺失值(一些变量根据业务定义可以用0进行填补，比如交易金额)；对一些变量进行加工处理

### Tag:

* preprocessing

### Param:

* None
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 处理后的dataframe



## <a id="classmap">ClassMapping</a>
将是字符的类别型变量映射为数值, 缺失值仍保持为np.nan。注意：类别型变量若已为数值则不做转换。

### Tag:

* preprocessing

### Param:

* cols: 待转换的变量
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 转换后的dataframe


## <a id="QTrans">QuantileTransformer</a>
对连续变量进行正态化处理。

### Tag:

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

* preprocessing

### Param:

* box_type：按区间分段还是按分位数分段
* box_num：分箱个数
 
### Input:

* df: 输入的dataframe

### Output:

* df_new: 分箱后的dataframe


## <a id="union">Union</a>
对RFE和SelectFromModel进行特征选择后的变量进行合并(取并集)。

### Tag:

* feature_selection

### Param:

* None
 
### Input:

* x: 输入的自变量
* y: 输入的因变量
* rfe_cols: 递归特征消除筛选的变量
* select_cols: 极端树模型筛选的变量

### Output:

* x_new: 筛选后的自变量
* y_new: 因变量
* meta_json: 统计值

## <a id="imba">Imbalance</a>
处理不均衡数据

### Tag:

* preprocessing

### Param:

* None
 
### Input:

* x: 输入的自变量
* y: 输入的因变量

### Output:

* x_new: 处理后的自变量
* y_new: 处理后的因变量
* meta_json: 统计值

## <a id="split">TrainTestSplit</a>
将数据分为训练集和测试集

### Tag:

* model_selection

### Param:

* test_size: 测试集所占比例
 
### Input:

* x: 输入的自变量
* y: 输入的因变量

### Output:

* xtrain: x训练集
* xtest: x测试集
* ytrain: y训练集
* ytest: y测试集


## <a id="cnf">ConfusionMatrix</a>
混淆矩阵

### Tag:

* metrics

### Param:

* None
 
### Input:

* model: 训练好的模型
* xtest: x测试集
* ytest: y测试集

### Output:

* cnf_matrix: 混淆矩阵
* report: 各个指标报告
* cnf_matrix_plot: 混淆矩阵图

## <a id="pred">Prediction</a>
用训练好的模型生成预测结果

### Tag:

* metrics

### Param:

* None
 
### Input:

* model: 训练好的模型
* xtest: x测试集
* ytest: y测试集

### Output:

* meta_json: 评估值
* pred: 模型预测类别
* prob: 模型预测概率