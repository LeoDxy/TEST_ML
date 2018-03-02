# 建模流程
1. 类别数据离散数值化，例如 gender 列，Male转化为1，Female转化为0，参考 sklearn.preprocessing.LabelEncoder
2. 所有数据取log
3. 所有特征归一化，参考sklearn.preprocessing.StandardScaler
4. 切分训练集测试集，随机取70%的数据为训练集，30%的数据为测试集
5. 在训练集上训练模型，参考sklearn.tree.DecisionTreeRegressor
6. 在测试集上测试模型，输出误差指标
