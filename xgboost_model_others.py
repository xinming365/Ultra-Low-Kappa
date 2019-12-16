import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
from util import mae, r_square, rmse

data_path = './data'
labels = np.load(os.path.join(data_path, 'labels.npy'))
train_data = np.load(os.path.join(data_path, 'train_version_5.npy'))
train = np.load(os.path.join(data_path, 'train_data.npy'))

# thermal conductivity label(agl_thermal_conductivity_300K)

'''
labels[0]:thermal conductivity 
labels[1]:agl_debye 
labels[2]:Cp at 300K
labels[3]:Cv at 300K
labels[4]:thermal expansion
'''

label_index = 2


label = labels[:, label_index]
natoms = train_data[:, 3]

if (label_index >= 2) and (label_index <= 3):
    label = label / natoms

if label_index == 4:
    label = label * np.power(10,6)


X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.1, random_state=4)
# If int, random_state is the seed used by the random number generator;
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)

# the following codes perform parameter tuning.

# cv_params = {'n_estimators': [700, 750, 800, 850, 900]}
# cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}  #0.847
# cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]} #0.846
# cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}  # 0.8469
# cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
# cv_params = {'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]}
cv_params = {'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1]}
"""
other_params = {'learning_rate': 0.05, 'n_estimators': 800, 'max_depth': 5, 'min_child_weight': 3, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.6, 'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 3}
"""
other_params = {'learning_rate': 0.1, 'n_estimators': 900, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.9, 'colsample_bytree': 0.6, 'gamma': 0.6, 'reg_alpha': 0.05, 'reg_lambda': 2}
model = xgb.XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1)
optimized_GBM.fit(X_train_transformed, y_train)

print('the best value of parameters：{0}'.format(optimized_GBM.best_params_))
print('the score of the best model:{0}'.format(optimized_GBM.best_score_))

TEST = True
SAVE = False
if TEST:
    ypred = optimized_GBM.predict(X_test_transformed)
    print(r_square(y_test, ypred), mae(y_test, ypred), rmse(y_test, ypred))

if SAVE:
    # save the best model
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if label_index == 2:
        models_name = './models/pcp.pkl'
        result_path = './result/cp'
        ypred_name = os.path.join(result_path,'ypred_cp_xgboost.npy')
        ytest_name = os.path.join(result_path, 'ytest_cp.npy')
    if label_index == 3:
        models_name = './models/pcv.pkl'
        result_path = './result/cv'
        ypred_name = os.path.join(result_path, 'ypred_cv_xgboost.npy')
        ytest_name = os.path.join(result_path, 'ytest_cv.npy')
    if label_index == 4:
        models_name = './models/pte.pkl'
        result_path = './result/te'
        ypred_name = os.path.join(result_path, 'ypred_te_xgboost.npy')
        ytest_name = os.path.join(result_path, 'ytest_te.npy')
    # pickle.dump(optimized_GBM, open('./models/pdebye.pkl', 'wb'))
    pickle.dump(optimized_GBM, open(models_name, 'wb'))

    # if not os.path.exists('./result/debye'):
    #     os.makedirs('./result/debye')

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # np.save('./result/debye/ypred_debye_xgboost.npy',ypred)
    # np.save('./result/debye/ytest_debye.npy',y_test)
    np.save(ypred_name, ypred)
    np.save(ytest_name, y_test)
    plt.scatter(y_test, ypred, s=2)
    plt.show()
