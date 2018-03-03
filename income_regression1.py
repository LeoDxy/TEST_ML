#  -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def data_prepare(file):
    data = pd.read_csv(file)
    target = data.income
    features_data = data.drop('income', axis=1)

    #  提取数值类型为整数或浮点数的变量
    numeric_features = [c for c in features_data if features_data[c].dtype.kind in ('i', 'f')]
    numeric_data = data[numeric_features]

    # 类别特征数据处理
    le = preprocessing.LabelEncoder()
    categorical_data = features_data.drop(numeric_features, axis=1)
    categorical_data_encoded = categorical_data.apply(le.fit_transform)
    gender = categorical_data_encoded.gender
    categorical_data_encoded = categorical_data_encoded.drop('gender', axis=1)

    categorical_data_encoded = categorical_data_encoded.applymap(d_log)
    numeric_data = numeric_data.applymap(d_log)

    features = pd.concat([numeric_data, categorical_data_encoded, gender], axis=1)
    # 归一化
    features_trans = preprocessing.StandardScaler().fit_transform(features)
    # 切分训练集测试集
    x_train, x_test, y_train, y_test = train_test_split(features_trans, target, test_size=0.3)
    return x_train, x_test, y_train, y_test


#  取对数 特征gender没有取对数
def d_log(x):
    if x:
        return np.log10(x)
    else:
        return x


def train_model(x_train, y_train):
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    print("Training score:%f" % (model.score(x_train, y_train)))
    return model

def test_model(x_test, y_test, model):
    # score = model.score(x_test, y_test)
    y_hat = model.predict(x_test)
    print("Testing score:%f" % (model.score(x_test, y_test)))
    print("预测值:%s" % y_hat)
    return y_hat

if __name__ =="__main__":
    file = 'income_regression.csv'
    x_train, x_test, y_train, y_test = data_prepare(file)
    model = train_model(x_train, y_train)
    test_model(x_test, y_test, model)
