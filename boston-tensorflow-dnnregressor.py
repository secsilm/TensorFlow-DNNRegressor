# coding: utf-8
# 详细可以参考 http://blog.csdn.net/u010099080/article/details/72824899

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

boston = load_boston()
boston_df = pd.DataFrame(np.c_[boston.data, boston.target], columns=np.append(boston.feature_names, 'MEDV'))
LABEL_COLUMN = ['MEDV']
FEATURE_COLUMNS = [f for f in boston_df if not f in LABEL_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(boston_df[FEATURE_COLUMNS], boston_df[LABEL_COLUMN], test_size=0.3)
print('训练集：{}\n测试集：{}'.format(X_train.shape, X_test.shape))

# ## 定义 FeatureColumns
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURE_COLUMNS]


# ## 定义 regressor
config = tf.contrib.learn.RunConfig(gpu_memory_fraction=0.3, log_device_placement=False)
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
                                          hidden_units=[10, 128], 
                                          model_dir='./models/dnnregressor', 
                                          config=config)


# ## 定义 input_fn
def input_fn(df, label):
    feature_cols = {k: tf.constant(df[k].values) for k in FEATURE_COLUMNS}
    label = tf.constant(label.values)
    return feature_cols, label


def train_input_fn():
    '''训练阶段使用的 input_fn'''
    return input_fn(X_train, y_train)


def test_input_fn():
    '''测试阶段使用的 input_fn'''
    return input_fn(X_test, y_test)


# 训练
regressor.fit(input_fn=train_input_fn, steps=5000)
# 测试
ev = regressor.evaluate(input_fn=test_input_fn, steps=1)
print('ev: {}'.format(ev))


