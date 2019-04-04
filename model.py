from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import keras
from keras.layers import Input, Dense
from keras.layers import Conv2D, Activation, MaxPooling2D, BatchNormalization,Flatten
from keras import Model

label = []
train_data = np.array()

X_train, X_test, y_train, y_test = train_test_split(scaled_train_data, label, test_size=0.33, random_state=4)
# If int, random_state is the seed used by the random number generator;
scaler = preprocessing.StandardScaler().fit(train_data)
mean_ = scaler.mean_  # 保存了均值
scale_ = scaler.scale_  # 保存了标准差
X_train_transformed = scaler.transform(train_data)
X_test_transformed = scaler.transform(X_test)

KFold(n_splits=3)

n_global_feature = train_data.shape[1]
gfeature_input = Input(shape=(n_global_feature,), name='gfeature_input')

force_input = Input(shape=(5, 5), name='force_input')
position_input = Input(shape=(5, 5), name='position_input')


def conv_block(x):
    f = Conv2D(64, (1, 1), padding='same')(x)
    f = BatchNormalization(axis=-1, epsilon=1e-3)(f)
    f = Activation('relu')(f)
    f = Conv2D(64, (1, 1), padding='same')(f)
    f = BatchNormalization(axis=-1, epsilon=1e-3)(f)
    f = Activation('relu')(f)
    f = MaxPooling2D(pool_size=(3, 3))(f)
    return f


f = conv_block(force_input)
f = conv_block(f)
x2 = Flatten(f)
p = conv_block(position_input)
p = conv_block(p)
x3 = Flatten(p)

x = keras.layers.concatenate([x1, x2, x3])
Dense(128)
