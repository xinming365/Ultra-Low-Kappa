import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
import xgboost as xgb
from util import mae, r_square, rmse, split_descriptor
from KRR_SVR import read_subset_data, read_equal_subset_data


def xgboost_model(train_data, label, save_models=False, save_results=False):
    X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.2, random_state=4)
    # cv_params = {'n_estimators': [700, 750, 800, 850, 900]}
    # cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}  #0.847
    # cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]} #0.846
    # cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}  # 0.8469
    # cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    # cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    cv_params = {'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1]}

    # Time:2019/10/22/14:22
    other_params = {'learning_rate': 0.05, 'n_estimators': 800, 'max_depth': 5, 'min_child_weight': 3, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.6, 'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 3}
    # other_params = {'n_estimators': 800, 'max_depth': 5, 'min_child_weight': 3, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.6, 'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 3}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1)
    optimized_GBM.fit(X_train, y_train)

    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    evalute_result = optimized_GBM.cv_results_
    print('每轮迭代运行结果:{0}'.format(evalute_result))

    # print(optimized_GBM.feature_importances_)

    if save_models:
        # save the best model
        model_name = 'ptc_ab.pkl'
        if not os.path.exists('./models'):
            os.makedirs('./models')
        # pickle.dump(optimized_GBM, open('./models/ptc.pkl', 'wb'))
        pickle.dump(optimized_GBM, open(os.path.join('./models', model_name), 'wb'))

    save_train_test_descriptor = False
    if save_train_test_descriptor:
        if not os.path.exists('./data/descriptor'):
            os.makedirs('./data/descriptor')
        np.save('./data/descriptor/descriptor_test.npy', X_test)
        np.save('./data/descriptor/descriptor_train.npy', X_train)

    save_tests = False
    if save_tests:
        np.save('./data/tests/test_x.npy', X_test)
        np.save('./data/tests/test_y.npy', y_test)

    ypred = optimized_GBM.predict(X_test)
    ypred_train = optimized_GBM.predict(X_train)
    if save_results:
        # save the results of models' predictions.
        if not os.path.exists('./result'):
            os.makedirs('./result')
        # np.save('./result/ypred_xgboost.npy', ypred)
        # np.save('./result/ytest.npy', y_test)
        np.save('./result/ytrain.npy', y_train)
        np.save('./result/ypred_xgboost_train.npy', ypred_train)
    # print(r_square(y_train, ypred_train))
    # print(r_square(y_test, ypred), mae(y_test, ypred), mae(np.exp(y_test), np.exp(ypred)), rmse(y_test, ypred))
    # plt.scatter(y_test, ypred, s=2)
    # plt.scatter(y_train, ypred_train, s=2)
    # plt.show()
    # return [r_square(y_test, ypred), mae(y_test, ypred), mae(np.exp(y_test), np.exp(ypred)), rmse(y_test, ypred)]
    return [r_square(y_test, ypred),r_square(ypred_train, y_train)]


if __name__ == '__main__':
    data_path = './data'
    labels = np.load(os.path.join(data_path, 'labels.npy'))
    # thermal conductivity label(agl_thermal_conductivity_300K)

    '''
    labels[0]:thermal conductivity
    labels[1]:agl_debye
    labels[2]:Cp at 300K
    labels[3]:Cv at 300K
    labels[4]:thermal expansion
    '''
    label = labels[:,0]
    train = np.load(os.path.join(data_path, 'train_data.npy'))

    result_list = []

    # for i in range(9, 10):
    i = 9
    train_data = split_descriptor(category=i)
    train_on_subset_data = False
    train_on_whole_data = True
    if train_on_subset_data:
        epoch=0
        while epoch < 20:
            epoch=epoch+1
            r2_list=[]
            # for i in ['ort', 'tet', 'hex', 'cub']:
            for i in ['natoms_s', 'natoms_m', 'natoms_l']:
                t, l = read_equal_subset_data(i, 600, train_data, label)
                l=np.log(l)
                result_i = xgboost_model(train_data=t, label=l, save_models=False, save_results=False)
                r2_list.append(result_i[0])
            result_list.append(r2_list)
    if train_on_whole_data:
        result_list = xgboost_model(train_data=train_data, label=label, save_models=False, save_results=False)
    print(result_list)
"""""
-crystal property ; -cw property ; -crystal structure ; -statistical property
[[0.8886073091775165, 0.28194082296872414, 2.5127673231520835, 0.3840810708252689], 
[0.8011290652131111, 0.37672310329615294, 3.0339138819062486, 0.513192397274493],
 [0.903765413896078, 0.25795458754976824, 2.167307067742963, 0.35699336763813466], 
 [0.9007824769091913, 0.2607509918232616, 2.229821411520569, 0.3624839208206655]]
 
 [[0.6438172999066268, 0.529000103639348, 3.843137569211508, 0.6868010765008358], 
 [0.7857774662601121, 0.4077045767661007, 3.4098833619260813, 0.5326318494762965], 
 [0.6447300084034051, 0.5024618290923478, 4.08620087582738, 0.6859205575083548],
  [0.576652417597549, 0.5609816851296134, 4.105776809105287, 0.7487607977019508]]

[0.9018491257809167, 0.2585323586402397, 2.1326042263646934, 0.36053019410540416]

r2=[0.889, 0.801, 0.903, 0.900, 0.643, 0.785, 0.644, 0.577, 0.901]
mae=[2.51, 3.03, 2.17, 2.23, 3.84, 3.41, 4.09, 4.11, 2.13]

# accuracy_on_category:
[[0.9173571560911747, 0.23139430211958198, 1.2514860358073534, 0.28963308727044806], [0.8275425824837026, 0.28248220475390506, 1.9020447139987944, 0.37821236466029273], [0.8832342927156365, 0.25812533711784136, 2.4397612335506333, 0.3529944641465119], [0.7233761844743497, 0.3837634520187851, 1.830503428826911, 0.6074685770046594]]

"""
