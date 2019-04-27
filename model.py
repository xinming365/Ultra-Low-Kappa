from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

label = []
train_data = []

X_train, X_test, y_train, y_test = train_test_split(scaled_train_data, label, test_size=0.2, random_state=4)
# If int, random_state is the seed used by the random number generator;
scaler = preprocessing.StandardScaler().fit(train_data)
mean_ = scaler.mean_  # 保存了均值
scale_ = scaler.scale_  # 保存了标准差
X_train_transformed = scaler.transform(train_data)
X_test_transformed = scaler.transform(X_test)


def dense_net(x):
    length = 0
    batch_size = 500
    inputs = Input(shape=(length,), batch_size=batch_size)
    x = Dense(64, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    y = Dense(1)
