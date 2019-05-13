import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout,BatchNormalization
from tensorflow.keras.models import Model

data_path = './data'
labels = np.load(os.path.join(data_path,'labels.npy'))
train_data = np.load(os.path.join(data_path,'train_version_1.npy'))
# thermal conductivity label(agl_thermal_conductivity_300K)
label = labels[:, 0]
X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.1, random_state=4)
# If int, random_state is the seed used by the random number generator;
scaler = preprocessing.StandardScaler().fit(X_train)
mean_ = scaler.mean_  # 保存了均值
scale_ = scaler.scale_  # 保存了标准差
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)


def dense_net():
    length = 56
    batch_size = 32
    inputs = Input(shape=(length,))
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu' )(x)
    x = Dense(48, activation='relu')(x)
    predictions = Dense(1)(x)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model = Model(inputs = inputs, outputs= predictions)
    model.compile(optimizer= optimizer,loss = 'mse', metrics=['mean_absolute_error'])
    return model
EPOCHS = 300
model = dense_net()
model.summary()
model.fit(x = X_train_transformed, y= y_train, epochs= EPOCHS)
# Returns the loss value & metrics values for the model in test mode.
scalars_train = model.evaluate(X_train_transformed,y_train)
scalars_test = model.evaluate(X_test_transformed,y_test)
result = model.predict(X_test_transformed)
print(result.shape)
print(scalars_train,scalars_test)
print(np.mean(label))

