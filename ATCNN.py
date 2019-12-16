# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
import sys
import os
from data_preprocessing import decompose_formula
from Gen_atom import atomic_dict
from util import mae, r_square, rmse
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Flatten, Dense, Activation


def get_component_vector(comps):
    namelist, numlist = decompose_formula(comps)
    avector = 100 * [0]
    asum = sum(numlist)
    for i in range(len(namelist)):
        atomic_num = atomic_dict[namelist[i]]
        avector[atomic_num-1] = numlist[i]/asum
    return avector


def read_file():
    train_data = np.load(os.path.join('./data', 'train_data.npy'))
    labels = np.load(os.path.join('./data', 'labels.npy'))
    y_data = labels[:, 0]
    compound_list = train_data[:, 2]
    x_data = []
    for cmp in compound_list:
        x = get_component_vector(cmp)
        x_data.append(x)
    return x_data, y_data


def data_split(x_data, y_data, split_ratio):
    length = len(x_data)
    random_index = np.random.permutation(np.arange(length))
    train_size = np.int(np.rint(length*split_ratio))
    train_index = random_index[:train_size]
    predict_index = random_index[train_size:]
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_train = x_data[train_index]
    y_train = y_data[train_index]
    x_predict = x_data[predict_index]
    y_predict = y_data[predict_index]
    return x_train, y_train, x_predict, y_predict


def ATCNN_model():
    input_shape = (10, 10, 1)
    model = Sequential()

    # layer1
    model.add(Conv2D(64, kernel_size=(5, 5), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # layer2
    model.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # layer3
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # layer4_zx
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # layer5
    model.add(Conv2D(64, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Add_zx
    model.add(Conv2D(64, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # layer6
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(400))

    # # layer7_zx
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(Activation('relu'))
    #
    # model.add(Dense(80))

    # layer8
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # layer9
    model.add(Dense(1))
    # model.add(Activation('relu'))

    model.compile(loss=keras.losses.mean_absolute_error,
                  optimizer=keras.optimizers.Adadelta())
    return model


if __name__=='__main__':
    # x,y=decompose_formula('Mg1B2')
    # print(x,y)
    x_data, y_data = read_file()
    y_data = np.log(y_data)
    x_train, y_train, x_test, y_test = data_split(x_data, y_data, split_ratio=0.8)
    x_train = np.reshape(x_train, (len(x_train), 10, 10, 1))
    x_test = np.reshape(x_test, (len(x_test), 10, 10, 1))
    model = ATCNN_model()
    print(model.summary())
    batch_size = 128

    model.fit(x_train, y_train, validation_split=0.02, batch_size=batch_size, epochs=400)
    # model.save('ATCNN_model.h5')
    loss = model.evaluate(x_test, y_test, batch_size=batch_size)
    ypr
    ed = model.predict(x_test, batch_size=batch_size)
    print([r_square(y_test, ypred), mae(y_test, ypred), mae(np.exp(y_test), np.exp(ypred)), rmse(y_test, ypred)])
    print('test set loss :', loss)



