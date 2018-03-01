#  -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('income_regression.csv')
target = data.income
features_data = data.drop('income',axis=1)

 # 提取数值类型为整数或浮点数的变量
numeric_features = [c for c in features_data if features_data[c].dtype.kind in ('i', 'f')]
numeric_data = data[numeric_features]

# 类别数据处理
le = preprocessing.LabelEncoder()
def data_encoded(x):
    le.fit(x)
    return le.transform(x)

categorical_data = features_data.drop(numeric_features,axis=1)
categorical_data_encoded = categorical_data.apply(data_encoded)
gender = categorical_data_encoded.gender
categorical_data_encoded = categorical_data_encoded.drop('gender',axis=1)
#  取对数  特征gender没有取对数
def d_log(x):
    if x:
        return np.log2(x)
    else:
        return x
categorical_data_encoded = categorical_data_encoded.applymap(d_log)

features = pd.concat([numeric_data, categorical_data_encoded, gender], axis=1)
features.head()
# 归一化
scaler  = preprocessing.StandardScaler().fit(features)
featrues_trans = scaler.transform(features)

# 切分训练集测试集
x_train, x_test, y_train, y_test=train_test_split(featrues_trans, target, test_size=0.3)
cls = DecisionTreeRegressor()

cls.fit(x_train, y_train)
score = cls.score(x_test, y_test)
print("Trainting score:%f" % (cls.score(x_train, y_train)))
print("预测值:%s" %cls.predict(x_test))
y_hat = cls.predict(x_test)
print(np.mean(np.abs(y_hat - y_test)))