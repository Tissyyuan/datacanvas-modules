## Hive_To_Dataframe：hive表转dataframe

### Param:

* delimiter:数据分割符

* table_name:hive表名

### Input:

* jdbc_url:hive表地址

### Output:

* dataframe:输出的df

## FromShow:查看dataframe

### Param:

* head:显示的行数

### Input:

* dataframe:要查看的dataframe

### Output:

* Table:绘制的数据表

## DataPreprocess:数据预处理

### Param:

* frac:采样百分比如0.1, 0.5 '#'表示取全量
* drop_columns:待删除的字段如"x1",x2"
* LabelEncode:要编码的分类字段

### Input:

* dataframe:待处理的数据

### Output:

* df:处理后的数据

## select:字段选择

### Param:

* col_name:按列名选如"x1","x2","x3"
* col_no:按行号选

### Input:

* data_source:源数据

### Output:

* columns:挑选出的字段

## LogisticRegression:逻辑回归

### Param:

* model_save_path:模型保存路径
* model_name:模型名称
* CV:交叉验证次数
* metrics:模型评估指标:"accuracy","f1_micro","f1_macro"

### Input:

* X:自变量
* Y:预测变量

### Output:

* model:训练出的模型

## DecisionTreeClassifiter:决策树分类

### Param:

* CV:交叉验证次数
* criterion:The function to measure the quality of a split. eg:gini
* import_feature:返回的重要特征的阈值(< 1)如0.01
* min_sample_leaf:The minimum number of samples required to be at a leaf node
* max_depth:最大深度
* metrics:模型评估指标:"accuracy","f1_micro","f1_macro"
* model_save_path:模型保存路径
* model_name:模型名称
* min_samples_split:The minimum number of samples required to split an internal node

### Input:

* X:自变量
* Y:预测变量

### Output:

* model:训练出的模型
* TreeShow:决策树结构图
* import_X:重要的特征变量

### MetriceShow:模型指标查看

### Input:

* model:模型

### Output:

* metrics:评估指标
* metrics_desribe:评估指标统计

## ModelSelect:模型选择

### Param:

* metric:评价指标

### Input:

* model_A:模型
* model_B:模型

### Output:

* better_model:选出的最优模型

## Predict:预测

### Param:

* result_col_name:预测结果的字段名

### Input:

* model:模型
* X:自变量

### Output:

* Y:预测变量

## ResultEvaluation:预测结果评估

### Param:
* FigureType:评估方法 如：roc

### Input:

* Real:真实值
* Predict:预测值

### Output:

* Show:评估结果

## df2hive:dataframe存入hive

### Param:

* jdbc_url:hive连接串
* table_name:hive表名

### Input:

* data:目标数据

## data_describe:字段统计

### input:

* data:输入的dataframe

### output:

* describes:数据描述信息

## imputer:缺失值填充

### input:

* data:输入的dataframe

### output:

* data_new:填充缺失值后的dataframe

## HashingEncoder:分类变量哈希编码

### Param:

* columns:要编码的变量 如: x1,x2,x3 列名

### Input:

* data:输入的dataframe

### Output:

* data_new:编码后的dataframe


## OneHotEncoder:onehot编码

### Param:

* columns:要编码的变量

### Input:

* data:输入的dataframe

### Output:

* data_new:编码后的dataframe

## OrdinalEncoder:不知道中文叫啥

### Param:

* columns:要编码的变量

### Input:

* data:输入的dataframe

### Output:

* data_new:编码后的dataframe

## BinaryEncoder

### Param:

* columns:要编码的变量

### Input:

* data:输入的dataframe

### Output:

* data_new:编码后的dataframe

## PolynomialEncoder

### Param:

* columns:要编码的变量

### Input:

* data:输入的dataframe

### Output:

* data_new:编码后的dataframe

## BackwardDifferenceEncoder

### Param:

* columns:要编码的变量

### Input:

* data:输入的dataframe

### Output:

* data_new:编码后的dataframe

## SumEncoder

### Param:

* columns:要编码的变量

### Input:

* data:输入的dataframe

### Output:

* data_new:编码后的dataframe

## HelmertEncoder

### Param:

* columns:要编码的变量

### Input:

* data:输入的dataframe

### Output:

* data_new:编码后的dataframe




