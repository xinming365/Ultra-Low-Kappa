import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from util import split_descriptor, metric, mae,rmse
import os
from xgboost_model_tc import xgboost_model
from KRR_SVR import transform_data, KrrSvr
from FCN_model import DnnModel



def read_data():
    i = 9
    train_data = split_descriptor(category=i)
    labels = np.load(os.path.join('./data', 'labels.npy'))
    label = np.log(labels[:, 0])
    return train_data, label


def split_data(train_data, label):
    X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.1, random_state=4)
    return X_train, y_train, X_test, y_test


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


def train_on_partial_feature_2(model_type):
    train_data, label = read_data()
    length = len(train_data[0])
    cut_index = np.arange(length)
    result = []
    for i in range(1, length + 1):
        a=0
        rmse_list = []
        while a < 20:
            a+=1
            np.random.shuffle(cut_index)
            new_train_data = np.copy(train_data[:, cut_index[:i]])
            x_train, x_test, y_train, y_test = transform_data(new_train_data, label, 0.1, 4)
            if model_type=='fc':
                dnn_model = DnnModel(x_train, y_train, x_test, y_test)
                train_y_pred, test_y_pred, score = dnn_model.evaluate()
                rmse_metric = rmse(y_test, test_y_pred.flatten())
                rmse_list.append(rmse_metric)
            elif model_type=='xgboost':
                result_list = xgboost_model(new_train_data,label)
                rmse_metric = result_list[3]
                rmse_list.append(rmse_metric)
            else:
                ml_model = KrrSvr(x_train, x_test, y_train)
                if model_type=='krr':
                    predict_train, predict_test = ml_model.krr()
                if model_type=='svr':
                    predict_train, predict_test = ml_model.svr()
                rmse_metric = rmse(y_test, predict_test)
                rmse_list.append(rmse_metric)
        y_rmse = np.mean(rmse_list)
        print(y_rmse)
        yerr_rmse_min = np.min(rmse_list)
        yerr_rmse_max = np.max(rmse_list)
        yerr_rmse_std = np.std(rmse_list)
        result.append([y_rmse, yerr_rmse_min, yerr_rmse_max, yerr_rmse_std])
    return result


def train_on_partial_feature(sort=True):
    feature_importance = load_feature_importance()
    length = len(feature_importance)
    result = []
    if sort:
        cut_index = np.argsort(feature_importance)
    else:
        cut_index = np.arange(length)
    train_data, label = read_data()
    for i in range(1, length+1):
        new_train_data = np.copy(train_data[:, cut_index[:i]])
        result_i = xgboost_model(new_train_data, label)
        result.append(result_i)
        print(result_i[0])
    return result


def significance_analysis():
    x = np.load('./data/trains/train_x.npy')
    y = np.load('./data/trains/train_y.npy')
    model=pickle.load(open('./models/ptc_ab.pkl','rb'))
    pk_list=[]
    for k in range(32):
        sum=0
        for i in [-0.2, -0.15,-0.1,-0.05,0.05,0.1,0.15]:
            tmp = np.copy(x)
            tmpp=np.copy(x)
            tmp[:,k] = tmp[:,k]*(1+i)
            tmpp[:,k]=tmp[:,k]*(1+i+0.05)
            predict_y=model.predict(tmp)
            predict_yy=model.predict(tmpp)
            delta_ki = mae(predict_y,y)
            delta_kii=mae(predict_yy,y)
            sum=sum+np.abs((delta_kii-delta_ki))/delta_ki
        pk=sum/7
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
    model_type = 'fc'
    result = train_on_partial_feature_2(model_type)
    print(result)
    # np.save('./result/accuracy/accuracy_partial_unsort_'+model_type+'.npy', result)
    np.save('./result/accuracy/accuracy_partial_random_' + model_type + '.npy', result)

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