import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from util import mae, r_square, rmse, split_descriptor
from KRR_SVR import transform_data, read_subset_data, read_equal_subset_data
from util import metric, read_label


class DnnModel:
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        :param x_train: X_train_transformed
        :param y_train: labels for training
        :param x_test: X_test_transformed
        :param y_test: labels for testing
        """
        self.length = x_train.shape[1]
        self.EPOCHS = 200
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def dnn_model(self):
        length = self.length
        inputs = Input(shape=(length,))
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='sigmoid')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='sigmoid')(x)
        predictions = Dense(1)(x)
        optimizer = tf.train.RMSPropOptimizer(0.0001)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])
        return model

    def evaluate(self):
        model = self.dnn_model()
        # model.summary()
        model.fit(x=self.x_train, y=self.y_train, epochs=self.EPOCHS)
        # Returns the loss value & metrics values for the model in test mode.
        scalars_test = model.evaluate(self.x_test, self.y_test)
        result = model.predict(self.x_train)
        result_ = model.predict(self.x_test)
        return result, result_, scalars_test

def train_on_partial_feature_dnn(train_data, label):
    global y_rmse, yerr_rmse_min
    length = len(train_data[0])
    cut_index = np.arange(length)
    output = []
    for i in range(1, length + 1):
        a = 0
        rmse_list = []
        while a < 20:
            a += 1
            np.random.shuffle(cut_index)
            new_train_data = np.copy(train_data[:, cut_index[:i]])
            print(new_train_data)
            x_train, x_test, y_train, y_test = transform_data(new_train_data, label, 0.1, 4)
            dnn_model = DnnModel(x_train, y_train, x_test, y_test)
            result, result_, score = dnn_model.evaluate()
            rmse_metric = rmse(y_test, result_.flatten())
            rmse_list.append(rmse_metric)
        y_rmse = np.mean(rmse_list)
        print(y_rmse)
        yerr_rmse_min = np.min(rmse_list)
        yerr_rmse_max = np.max(rmse_list)
        yerr_rmse_std = np.std(rmse_list)
        output.append([y_rmse, yerr_rmse_min, yerr_rmse_max, yerr_rmse_std])
    return output


def train_on_total_data(train_data, label):
    x_train, x_test, y_train, y_test = transform_data(train_data, label, 0.1, 4)
    dnn_model = DnnModel(x_train, y_train, x_test, y_test)
    result, result_, score = dnn_model.evaluate()
    result_metric = metric(y_test, result_.flatten())
    return result_metric


if __name__ == '__main__':
    result_list=[]
    data_path = './data'
    labels = np.load(os.path.join(data_path, 'labels.npy'))
    i=9 # the 9th category of descriptors
    print('This is the {0}th category descriptor defined in the util.py'.format(i))
    train_data = split_descriptor(filename='train_version_5.npy', category=i)
    train_data, label = read_label(train_data, labels,
        label_index=0)  # 0:thermal conductivity,1:agl_debye,2:Cp at 300K, 3:Cv at 300K,4:thermal expansion
    On_partial = True
    On_subset_data = False
    if On_partial:
        result = train_on_partial_feature_dnn(train_data, label)
        # result =train_on_total_data(train_data, label)
        model_type = 'fc'
        # np.save('./result/accuracy/accuracy_partial_random_' + model_type + '.npy', result)
    if On_subset_data:
        epoch = 0
        while epoch < 20:
            epoch = epoch + 1
            r2_list = []
            # for i in ['1_species', '2_species', '3_species']:
            # for i in ['ort', 'tet', 'hex', 'cub']:
            for i in ['natoms_s', 'natoms_m', 'natoms_l']:
                t, l = read_equal_subset_data(i, 600, train_data, label)
                x_train, x_test, y_train, y_test = transform_data(x_data=t, y_data=l, test_size=0.1, random_state=4)

                kfold = False
                if kfold:
                    kf = KFold(n_splits=5)
                    mae_list = []
                    fold = 0
                    for train_index, validate_index in kf.split(x_train):
                        fold = fold + 1
                        train_x, train_y = x_train[train_index], y_train[train_index]
                        validate_x, validate_y = x_train[validate_index], y_train[validate_index]
                        dnn_model = DnnModel(train_x, train_y, validate_x, validate_y)
                        result, result_, score = dnn_model.evaluate()
                        print("Fold {0} mae: {1}".format(fold, score[1]))
                        mae_list.append(score[1])
                    print(mae_list)
                    mae_outcome = np.mean(mae_list)
                    print("Mae: {0}".format(mae_outcome))

                TEST = True
                if TEST:
                    print('this is the test process')
                    dnn_model = DnnModel(x_train, y_train, x_test, y_test)
                    result, result_, score = dnn_model.evaluate()
                    print(score)
                    r2_metric, mae_metric, rmse_metric = metric(y_test, result_.flatten())
                    # result_list.append([r2_metric, mae_metric,
                    #        mae(np.exp(y_test), np.exp(result_.flatten())), rmse_metric])
                    r2_list.append(r2_metric)
            result_list.append(r2_list)

    print(result_list)
"""
This is the result of Thermal conductivity predictions.
[0.8772933989663311, 0.2767175933086125, 2.3416097294743006, 0.4031145923502155], 
[0.7918775039165272, 0.3678928843004184, 2.9147379307845105, 0.5249936726870987], 
[0.8823843383005736, 0.27430463928775634, 2.4659291186071517, 0.39466365575004186],
 [0.8899444004533537, 0.27299138593003053, 2.3374953092128363, 0.3817689718198876],
  
 [0.6210741716950363, 0.5535339712455267, 4.062358838466344, 0.7083887583438244], 
 [0.745160163496027, 0.43711100249163504, 3.508696006146655, 0.5809358998127028], 
 [0.6477515961669796, 0.4872684178375454, 4.067575563599144, 0.682997435830418],
  [0.5667150838267894, 0.5672829261848272, 4.284316693129993, 0.7574977400099581], 
  
  [0.8839512801410707, 0.2716716177224611, 2.2693935930974054, 0.39202587545903855]]

mae=[2.34, 2.91, 2.46, 2.34, 4.06, 3.51, 4.07, 4.28, 2.27] (original format)

This is the thermal expansion result:
[[0.6956978750357586, 7.470161415911676, inf, 25.49735360664308]]

# n_species performance:
# descriptor of the 9th category
[[0.9331001561156573, 0.3998398113457099, 6.5473021060906405, 0.4553107168829367], [0.8431068109436278, 0.3313673266028911, 2.4087577782409593, 0.4939952836241818], [0.8606407691876001, 0.25788931913878516, 1.5172169250773762, 0.3655829721023384]]

# accuracy_on_category
[[0.8918945552705899, 0.2534030006644068, 1.5536444519255646, 0.3312602540152009], [0.8135292925720039, 0.2650933161743464, 2.205413662105677, 0.3932783986674412], [0.8719702783812269, 0.2797381088559134, 2.3049858817067017, 0.3696286601393471], [0.7202942825594977, 0.3927999085466019, 2.0458554105752422, 0.6108431471289765]]

"""

#
# the following is the DNN model
#
#
# this R2/MAE(log)/MAE = 0.89/0.2654/2.4432
# [0.8881700553899254, 0.2623028988094831, 2.313358586375327, 0.38483415623084516]
# [0.8927512094071135, 0.2689339934965572, 2.25594024430576, 0.3768692974373042]
# def dnn_model(self):
#     length = self.length
#     inputs = Input(shape=(length,))
#     x = Dense(64, activation='relu')(inputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x)
#     x = Dense(128, activation='relu')(inputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x)
#     x = Dense(256, activation='relu')(inputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     x = Dense(512, activation='relu')(inputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x)
#     x = Dense(512, activation='sigmoid')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#     x = Dense(64, activation='sigmoid')(x)
#     predictions = Dense(1)(x)
#     optimizer = tf.train.RMSPropOptimizer(0.0001)
#     model = Model(inputs=inputs, outputs=predictions)
#     model.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])
#     return model

# this R2/MAE(log)/MAE = 0.892/0.2682/2.286
# def dnn_model(self):
#     length = self.length
#     inputs = Input(shape=(length,))
#     x = Dense(64, activation='relu')(inputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x)
#     x = Dense(128, activation='relu')(inputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x)
#     x = Dense(256, activation='relu')(inputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     x = Dense(512, activation='relu')(inputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x)
#     x = Dense(1024, activation='sigmoid')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#     x = Dense(512, activation='sigmoid')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#     x = Dense(64, activation='sigmoid')(x)
#     predictions = Dense(1)(x)
#     optimizer = tf.train.RMSPropOptimizer(0.0001)
#     model = Model(inputs=inputs, outputs=predictions)
#     model.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])
#     return model
"""
result = [0.8037130609557539, 0.8629226426419969, 0.8718019511779229]
epoch = 200

the model of train natoms subset data 
    def dnn_model(self):
        length = self.length
        inputs = Input(shape=(length,))
        x = Dense(256, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='sigmoid')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='sigmoid')(x)
        predictions = Dense(1)(x)
        optimizer = tf.train.RMSPropOptimizer(0.0001)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])
        return model
        
        
"""
"""
    def dnn_model(self):
        length = self.length
        inputs = Input(shape=(length,))
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='sigmoid')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='sigmoid')(x)
        predictions = Dense(1)(x)
        optimizer = tf.train.RMSPropOptimizer(0.0001)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])
        return model
"""

"""
The simplified model used in the accuracy on the partial descriptor.
    def dnn_model(self):
        length = self.length
        inputs = Input(shape=(length,))
        x = Dense(50, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dense(30, activation='sigmoid')(x)
        x = BatchNormalization()(x)
        x = Dense(40, activation='sigmoid')(x)
        predictions = Dense(1)(x)
        optimizer = tf.train.RMSPropOptimizer(0.0001)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])
        return model
"""
"""

class DnnModel:
    def __init__(self, x_train, y_train, x_test, y_test):

        :param x_train: X_train_transformed
        :param y_train: labels for training
        :param x_test: X_test_transformed
        :param y_test: labels for testing

        self.length = x_train.shape[1]
        self.EPOCHS = 200
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def dnn_model(self):
        length = self.length
        inputs = Input(shape=(length,))
        x = Dense(256, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='sigmoid')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='sigmoid')(x)
        predictions = Dense(1)(x)
        optimizer = tf.train.RMSPropOptimizer(0.0001)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])
        return model

    def evaluate(self):
        model = self.dnn_model()
        # model.summary()
        model.fit(x=self.x_train, y=self.y_train, epochs=self.EPOCHS)
        # Returns the loss value & metrics values for the model in test mode.
        scalars_test = model.evaluate(self.x_test, self.y_test)
        result = model.predict(self.x_train)
        result_ = model.predict(self.x_test)
        return result, result_, scalars_test
"""
