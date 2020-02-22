import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from util import mae, load_and_split_descriptor, metric, read_label, r_square
import pickle
import csv
import xgboost as xgb


def krr():
    kr = KernelRidge(alpha=1, kernel='rbf')
    clf = GridSearchCV(kr, scoring='neg_mean_absolute_error', cv=5,
                       param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                   "gamma": np.logspace(-2, 2, 5)})
    return clf


def svr():
    sv = SVR(kernel='rbf')
    clf = GridSearchCV(sv, param_grid={"C": [1e0, 0.1, 1e-2, 1e-3],
                                       "gamma": np.logspace(-2, 2, 5)}, cv=5)
    return clf


def xgb_model():
    cv_params = {'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1]}
    other_params = {'learning_rate': 0.05, 'n_estimators': 800,
                    'max_depth': 5, 'min_child_weight': 3, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.6, 'gamma': 0.1,
                    'reg_alpha': 0.1, 'reg_lambda': 3}
    model = xgb.XGBRegressor(objective='reg:squarederror', **other_params)
    clf = GridSearchCV(model, param_grid=cv_params,
                       scoring='r2', cv=5, verbose=1)
    return clf


def fc(length):
    inputs = Input(shape=(length,))
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='sigmoid')(x)
    predictions = Dense(1)(x)
    optimizer = RMSprop(0.0001)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer, loss='mse', metrics=['mean_absolute_error'])
    return model


class KappaModel:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train_transformed = x_train
        self.x_test_transformed = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, model, epochs=1):
        """
        Args:
            model: the instance of ML model
            epochs: the epochs mainly used in the neural networks model. In the 
            svr/krr/xgboost model, the epochs are not used.
        Returns:
            None
        """
        print(np.shape(self.x_train_transformed))
        print(np.shape(self.y_train))
        if epochs == 1:
            model.fit(self.x_train_transformed, self.y_train)
        else:
            model.fit(self.x_train_transformed, self.y_train, epochs=epochs)

    def predict(self, model, type):
        """
        Args:
            model: the instance of ML model
            type: str. 'train' or 'test'
        Returns:
            None
        """
        if type == 'train':
            y_ = model.predict(self.x_train_transformed)
        elif type == 'test':
            y_ = model.predict(self.x_test_transformed)
        else:
            print('The parameter of <type> must be \'train\' or \'test\'')
        return y_.squeeze()

    @staticmethod
    def save_model(model, filename):
        """
        Args:
            filename: output file name. '*.pkl'
            model: the trained model
        Returns:
            None

        """
        if not os.path.exists('./models'):
            os.makedirs('./models')
        pickle.dump(model, open(os.path.join('./models', filename), 'wb'))

    def save_data(self):
        """
        Save the training data and the test data.
        Returns:
            None
        """
        trains_data_path = './data/trains'
        tests_data_path = './data/tests'
        # Save the data of the train part to allow for reproducibility of results.
        if not os.path.exists(trains_data_path):
            os.makedirs(trains_data_path)
        np.save(os.path.join(trains_data_path, 'train_x.npy'),
                self.x_train_transformed)
        np.save(os.path.join(trains_data_path, 'train_y.npy'),
                self.y_train)

        # Save the data of the test part to allow for reproducibility of results.
        if not os.path.exists(tests_data_path):
            os.makedirs(tests_data_path)
        np.save(os.path.join(tests_data_path, 'test_x.npy'),
                self.x_test_transformed)
        np.save(os.path.join(tests_data_path, 'test_y.npy'),
                self.y_test)


def transform_data(x_data, y_data, test_size, random_state, model_type=None):
    """Preprocess the data set. Just split the data set or combine the standardization with
    the data splitting.

    Args:
        x_data: ndarray with shape '(N, len)',
        y_data: ndarray with shape '(N, )',
        test_size: float,
        random_state: int,
            'random_state' is the seed used by the random number generator; The variable is used
            for the reproducibility of the model.
        model_type: string, optional
    """
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_transformed = scaler.transform(x_train)
    x_test_transformed = scaler.transform(x_test)
    if model_type == 'xgboost':
        return x_train, x_test, y_train, y_test
    else:
        return x_train_transformed, x_test_transformed, y_train, y_test


def read_subset_data(type, train_data, label, equal=False, n=None):
    """Read the subsets from the whole data.

    Args:
        type: a string.
            When the dataset is classified by the lattice system, the type
            belongs to one of ['ort', 'tet', 'hex', 'cub'].
            When it is classified by the 'natoms' (number of atoms per cell), the type
            belongs to one of ['natoms_s', 'natoms_m', 'natoms_l']
            When it is classified by the 'nspecies' (number of species), the type
            belongs to one of ['1_species', '2_species', '3_species']
        train_data: the numpy array.
        label: the numpy array with shape (N, 1), where N is the total number of data.
        equal: bool, optional
            'True' means the same number of different subsets randomly
            collected from the whole dataset. 'False' makes no limitataion on numbers.
        n: integer, optional
            Equal numbers of the dataset. If the 'equal' is True, the 'n' must be
            specified.
    Returns:
        train_data: the numpy array.
        label: the numpy array.
    """
    index_i = np.load(os.path.join('./data/descriptor', type + '.npy'))
    new_train_data = np.copy(train_data[index_i, :])
    new_label = np.copy(label[index_i])
    if equal:
        index = np.arange(n)
        np.random.shuffle(index)
        new_train_data = new_train_data[index, :]
        new_label = new_label[index]
    return new_train_data, new_label


def train_on_different_classes(type, equal=False):
    """ This function must be used in this file because there are some global
    variables '(model, epochs)' are defined in 'model.py'.
    Args:
        type: a string.
            'ls' or 'natoms' in this work.
        equal: bool.
            'True' means the same number of different subsets randomly
            collected from the whole dataset. 'False' makes no limitataion on numbers.
    Returns:
        None
    """
    r2 = []
    if type == 'ls':
        x = ['ort', 'tet', 'hex', 'cub']
        number = 900
    if type == 'natoms':
        x = ['natoms_s', 'natoms_m', 'natoms_l']
        number = 600
    if equal:
        epoch = 0
        while epoch < 20:
            epoch = epoch + 1
            r2_list = []
            for i in x:
                r2_metric = []
                t, l = read_subset_data(i, train_data, label, equal, number)
                x_train, x_test, y_train, y_test = transform_data(x_data=t, y_data=l, test_size=0.1, random_state=4)
                kappa_model = KappaModel(x_train, x_test, y_train, y_test)
                kappa_model.train_model(model, epochs=epochs)
                # predict_train = kappa_model.predict(model, 'train')
                predict_test = kappa_model.predict(model, 'test')
                r2_metric.append(r_square(y_cal=y_test, y_pred=predict_test))
            r2_list.append(r2_metric)
        r2 = np.mean(r2_list, axis=0)
        r2_std = np.std(r2_list, axis=0)
    else:
        for i in x:
            t, l = read_subset_data(i, train_data, label)
            x_train, x_test, y_train, y_test = transform_data(x_data=t, y_data=l, test_size=0.1, random_state=4)
            kappa_model = KappaModel(x_train, x_test, y_train, y_test)
            kappa_model.train_model(model, epochs)
            # predict_train = kappa_model.predict(model, 'train')
            predict_test = kappa_model.predict(model, 'test')
            r2.append(r_square(y_test, predict_test))
            r2_std = []
    return r2, r2_std


if __name__ == '__main__':
    MAEs_train = []
    MAEs_test = []
    metric_matrix = []
    data_path = './data'
    labels = pd.read_csv(os.path.join(data_path, 'labels.2020.1.29.csv')).to_numpy()
    raw_train_data = pd.read_csv(os.path.join(data_path, 'td.2020.1.29.csv')).to_numpy()
    i = 9  # the 9th category of descriptors
    print('This is the {0}th category descriptor defined in the util.py'.format(i))
    train_data = load_and_split_descriptor(raw_train_data, category=i)
    train_data, label = read_label(train_data, labels,
                                   label_index=0)
    # 0:thermal conductivity,1:agl_debye,2:Cp at 300K, 3:Cv at 300K,4:thermal expansion
    length = train_data.shape[1]
    train_on_subset_data = True
    train_on_whole_data = False

    if train_on_subset_data:
        subset_type = 'ls'
        fo = open('./result/subset_result.csv', 'w')
        csv_writer = csv.writer(fo)
        if subset_type == 'ls':
            csv_writer.writerow(['', 'ort', 'tet', 'hex', 'cub'])
        if subset_type == 'natoms':
            csv_writer.writerow(['', 'natoms_s', 'natoms_m', 'natoms_l'])
    if train_on_whole_data:
        fo = open('./result/result.csv', 'w')
        csv_writer = csv.writer(fo)
        csv_writer.writerow(['', 'SVR', 'FC', 'KRR', 'XGBoost'])

    for i in range(4):
        if i == 0:
            epochs = 1
            model = svr()
            model_type = 'SVR'
        elif i == 1:
            epochs = 200
            model = fc(length)
            model_type = 'FC'
        elif i == 2:
            epochs = 1
            model = krr()
            model_type = 'KRR'
        else:
            epochs = 1
            model = xgb_model()
            model_type = 'XGBoost'

        if train_on_subset_data:
            r2, _ = train_on_different_classes(subset_type, equal=False)
            r2.insert(0, model_type)
            csv_writer.writerow(r2)

        if train_on_whole_data:
            x_train, x_test, y_train, y_test = transform_data(x_data=train_data, y_data=label, test_size=0.1,
                                                              random_state=4)
            kappa_model = KappaModel(x_train, x_test, y_train, y_test)
            kappa_model.train_model(model, epochs=epochs)
            predict_train = kappa_model.predict(model, 'train')
            predict_test = kappa_model.predict(model, 'test')

            r2_train, mae_log_train, rmse_train = metric(y_train, predict_train)
            r2_test, mae_log_test, rmse_test = metric(y_cal=y_test, y_pred=predict_test)
            MAEs_train = mae(np.exp(y_train), np.exp(predict_train))
            MAEs_test = mae(np.exp(y_test), np.exp(predict_test))
            print(mae_log_test, r2_test)

            metric_list = [MAEs_train, MAEs_test,
                           r2_train, r2_test,
                           mae_log_train, mae_log_test,
                           rmse_train, rmse_test]
            metric_matrix.append(metric_list)

    if train_on_whole_data:
        metric_matrix.insert(0, ['MAEs of train data', 'MAEs of test data',
                                 'R2 of train data', 'R2 of test data',
                                 'Logarithmic mae of train data', 'Logarithmic mae of test data',
                                 'RMSE_train', 'RMSE_test'])
        metric_matrix = np.transpose(metric_matrix)
        csv_writer.writerows(metric_matrix)
    fo.close()
