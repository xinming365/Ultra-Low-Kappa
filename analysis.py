import numpy as np
import pandas as pd
import pickle
import os
import csv
from util import load_and_split_descriptor, metric, mae, rmse
from model import transform_data, read_label
from model import svr, krr, fc, xgb_model, KappaModel


def read_data():
    x = pd.read_csv(open('./data/td.2020.1.29.csv', 'r')).to_numpy()
    i = 9
    train_data = load_and_split_descriptor(x, category=i)
    labels = pd.read_csv(os.path.join('./data', 'labels.2020.1.29.csv')).to_numpy()
    label = np.log(labels[:, 0])
    return train_data, label


def descriptors_analysis():
    """Make a comparison with different types of descriptors.

    There are nine types of descriptors to test totally. -Crystal part, -CW,
    -Structure, -Statistical part, Crystal part, CW, Structure, Statistical part,
    Crystal+CW and All descriptor. ’-’ indicates the name of the hold-out feature
     category and the descriptors are generated from the ensemble of the other three
     categories.

    Args:
        None
    Returns:
        A numpy array with shape '(10, 4)'.
    """
    data_path = './data'
    labels = pd.read_csv(os.path.join(data_path, 'labels.2020.1.29.csv')).to_numpy()
    raw_train_data = pd.read_csv(os.path.join(data_path, 'td.2020.1.29.csv')).to_numpy()
    result = []
    for descriptors_i in range(10):
        new_train_data = load_and_split_descriptor(raw_train_data, descriptors_i)
        length = new_train_data.shape[1]
        t, l = read_label(new_train_data, labels, label_index=0)
        MAEs_list = []
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
            x_train, x_test, y_train, y_test = transform_data(t, l, 0.1, 4, model_type=model_type)
            kappa_model = KappaModel(x_train, x_test, y_train, y_test)
            kappa_model.train_model(model, epochs=epochs)
            predict_train = kappa_model.predict(model, 'train')
            predict_test = kappa_model.predict(model, 'test')

            MAEs_train = mae(np.exp(y_train), np.exp(predict_train))
            MAEs_test = mae(np.exp(y_test), np.exp(predict_test))
            print(MAEs_test)

            MAEs_list.append(MAEs_test)

        result.append(MAEs_list)
    return result


def accuracy_of_ls():
    x_test = np.load('./data/tests/test_x.npy')
    y_test = np.load('./data/tests/test_y.npy')
    # train_data, label = read_data()
    # x_train, y_train, x_test, y_test = split_data(train_data, label)
    accuracy_list = []
    model_name = 'ptc_ab.pkl'
    optimized_Model = pickle.load(file=open(os.path.join('./models', model_name), 'rb'))
    for lattice_system in range(7):
        lattice_list = []
        y_lattice_list = []
        for index, x in enumerate(x_test):
            if (lattice_system + 1) == x[0]:
                lattice_list.append(x)
                y_lattice_list.append(y_test[index])
        if len(lattice_list) != 0:
            lattice_array = np.array(lattice_list)
            y_lattice_array = np.array(y_lattice_list)
            ypred = optimized_Model.predict(lattice_array)
            accuracy_i = metric(y_lattice_array, ypred)
        else:
            accuracy_i = []
        accuracy_list.append(accuracy_i)
    return accuracy_list


def load_feature_importance():
    optimized_GBM = pickle.load(file=open(os.path.join('./models', 'ptc_ab.pkl'), 'rb'))
    xgb = optimized_GBM.best_estimator_
    feature_importance = xgb.feature_importances_
    return feature_importance


def train_on_partial_feature(model_type):
    """Train the ML models on some parts of the descriptors.

    Args:
        model_type: a string.
            'model_type' designate which model to be used to train a ML model, which
            must belong to one of 'fc', 'xgboost', 'krr', and 'svr'.
    Returns:
        A list with shape '(4,)'. The elements of the list are 'RMSE', 'the minimum of
        the RMSE list', 'the maximum of the RMSE list', and 'the standard deviation of the
        RMSE list'.
    """
    train_data, label = read_data()
    length = len(train_data[0])
    cut_index = np.arange(length)
    result = []
    for i in range(1, length + 1):
        a = 0
        rmse_list = []
        while a < 20:
            a += 1
            np.random.shuffle(cut_index)
            new_train_data = np.copy(train_data[:, cut_index[:i]])
            x_train, x_test, y_train, y_test = transform_data(new_train_data, label, 0.1, 4)
            km = KappaModel(x_train, x_test, y_train, y_test)
            if model_type == 'fc':
                model = fc()
                epochs = 200
            elif model_type == 'xgboost':
                epochs = 1
                model = xgb_model()
            elif model_type == 'krr':
                epochs = 1
                model = krr()
            elif model_type == 'svr':
                epochs = 1
                model = svr()
            else:
                print('The model type must belong to [\'svr\', \'krr\', \'xgboost\', \'fc\']!')
                km.train_model(model, epochs=epochs)
            predict_test = km.predict(model, 'test')
            rmse_metric = rmse(y_test, predict_test)
            rmse_list.append(rmse_metric)
        y_rmse = np.mean(rmse_list)
        print(y_rmse)
        yerr_rmse_min = np.min(rmse_list)
        yerr_rmse_max = np.max(rmse_list)
        yerr_rmse_std = np.std(rmse_list)
        result.append([y_rmse, yerr_rmse_min, yerr_rmse_max, yerr_rmse_std])
    return result


def significance_analysis():
    x = np.load('./data/trains/train_x.npy')
    y = np.load('./data/trains/train_y.npy')
    model = pickle.load(open('./models/ptc_ab.pkl', 'rb'))
    pk_list = []
    for k in range(32):
        sum = 0
        for i in [-0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15]:
            tmp = np.copy(x)
            tmpp = np.copy(x)
            tmp[:, k] = tmp[:, k] * (1 + i)
            tmpp[:, k] = tmp[:, k] * (1 + i + 0.05)
            predict_y = model.predict(tmp)
            predict_yy = model.predict(tmpp)
            delta_ki = mae(predict_y, y)
            delta_kii = mae(predict_yy, y)
            sum = sum + np.abs((delta_kii - delta_ki)) / delta_ki
        pk = sum / 7
        pk_list.append(pk)
    return pk_list


if __name__ == '__main__':
    # accuracy_list = accuracy_of_ls()
    # print(accuracy_list)
    # result = train_on_partial_feature(sort=False)

    # np.save('./result/accuracy/accuracy_partial_unsort.npy', result)
    # model_type='svr'
    # model_type = 'xgboost'
    # model_type = 'krr'
    # model_type = 'fc'
    # result = train_on_partial_feature(model_type)
    # print(result)
    # np.save('./result/accuracy/accuracy_partial_unsort_'+model_type+'.npy', result)
    # np.save('./result/accuracy/accuracy_partial_random_' + model_type + '.npy', result)
    result = descriptors_analysis()
    with open('./result/descriptors_analysis.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['', 'All descriptor', '-Crystal part', ' -CW',
                             '-Structure', ' -Statistical part', 'Crystal part', 'CW',
                             'Structure', 'Statistical part', 'Crystal+CW'])
        result.insert(0, ['SVR', 'FC', 'KRR', 'XGBoost'])
        result = np.transpose(result)
        csv_writer.writerows(result)
"""
1.0774072416661373
0.9682947337113156
0.8623440553497028
0.7519712375554827
0.7212968001880051
0.6662004550743997
0.6416090210595597
0.6319597284807278
0.58512937900294
0.5807036274205497
0.5321536337422506
0.5000436995563206
0.4941387792095182
0.4951586810794656
0.5090253478348833
0.4637741144723856
0.4629616237853565
0.4609231011582128
0.4462057657970225
"""
