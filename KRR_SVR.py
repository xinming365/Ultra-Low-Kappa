import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from util import mae, split_descriptor, metric, read_label


class KrrSvr:
    def __init__(self, x_train, x_test, y_train):
        self.x_train_transformed = x_train
        self.x_test_transformed = x_test
        self.y_train = y_train

    def krr(self):
        kr = KernelRidge(alpha=1, kernel='rbf')
        clf = GridSearchCV(kr, scoring='neg_mean_absolute_error', cv=5, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})
        clf.fit(self.x_train_transformed, self.y_train)
        y = clf.predict(self.x_train_transformed)
        y_ = clf.predict(self.x_test_transformed)
        return y, y_

    def svr(self):
        svr = SVR(kernel='rbf')
        clf = GridSearchCV(svr, param_grid={"C": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)}, cv=5)
        clf.fit(self.x_train_transformed, self.y_train)
        y = clf.predict(self.x_train_transformed)
        y_ = clf.predict(self.x_test_transformed)
        return y, y_


def transform_data(x_data, y_data, test_size, random_state):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_transformed = scaler.transform(x_train)
    x_test_transformed = scaler.transform(x_test)
    return x_train_transformed, x_test_transformed, y_train, y_test


def read_subset_data(type, train_data, label):
    """
    type: lattice_system:['ort', 'tet', 'hex', 'cub'];
    nspecies:['1_species', '2_species', '3_species'];
    natoms:['natoms_s', 'natoms_m', 'natoms_l']
    :return: train_data, label
    """
    index_i = np.load(os.path.join('./data/descriptor', type + '.npy'))
    new_train_data = np.copy(train_data[index_i, :])
    new_label = np.copy(label[index_i])
    return new_train_data, new_label

def read_equal_subset_data(type, n, train_data, label):
    """
    type: lattice_system:['ort', 'tet', 'hex', 'cub'];
    nspecies:['1_species', '2_species', '3_species'];
    natoms:['natoms_s', 'natoms_m', 'natoms_l']
    n: equal numbers of the dataset
    :return: train_data, label
    """
    index=np.arange(n)
    np.random.shuffle(index)

    index_i = np.load(os.path.join('./data/descriptor', type + '.npy'))
    new_train_data = np.copy(train_data[index_i, :])
    new_label = np.copy(label[index_i])

    x=new_train_data[index,:]
    y=new_label[index]
    return x, y


if __name__ == '__main__':
    result_list=[]
    data_path = './data'
    labels = np.load(os.path.join(data_path, 'labels.npy'))
    i=9 # the 9th category of descriptors
    print('This is the {0}th category descriptor defined in the util.py'.format(i))
    train_data = split_descriptor(filename='train_version_5.npy', category=i)
    train_data, label = read_label(train_data, labels, label_index=1) # 0:thermal conductivity,1:agl_debye,2:Cp at 300K, 3:Cv at 300K,4:thermal expansion
    epoch=0
    train_on_subset_data = False
    train_on_whole_data = True
    if train_on_subset_data:
        while epoch < 20:
            epoch=epoch+1
            r2_list = []
            # for i in ['ort', 'tet', 'hex', 'cub']:
            # for i in ['1_species', '2_species', '3_species']:
            for i in ['natoms_s', 'natoms_m', 'natoms_l']:
                t, l = read_equal_subset_data(i, 600, train_data, label)
                x_train, x_test, y_train, y_test = transform_data(x_data=t, y_data=l, test_size=0.1, random_state=4)
                ml_model = KrrSvr(x_train, x_test, y_train)
                # predict_train, predict_test = ml_model.krr()
                predict_train, predict_test = ml_model.svr()
                r2_metric, mae_metric, rmse_metric = metric(y_cal=y_test, y_pred=predict_test)
                mae2 = mae(np.exp(y_test), np.exp(predict_test))
                # result_list.append([r2_metric, mae_metric, mae2, rmse_metric])
                r2_list.append(r2_metric)
            result_list.append(r2_list)
    if train_on_whole_data:
        x_train, x_test, y_train, y_test = transform_data(x_data=train_data, y_data=label, test_size=0.1, random_state=4)
        ml_model = KrrSvr(x_train, x_test, y_train)
        predict_train, predict_test = ml_model.svr()
        # predict_train, predict_test = ml_model.krr()
        r2_metric, mae_metric, rmse_metric = metric(y_cal=y_test, y_pred=predict_test)
        #result_list=[mae(np.exp(y_train),np.exp(predict_train)),mae(np.exp(y_test), np.exp(predict_test))]

    print(mae_metric)

"""
KRR result
[0.9093637336518622, 0.22885934839174016, [1.1524167847221307], 0.30331685911995],
[0.8078166601452073, 0.2880316886193797, [1.9358172551941415], 0.39925710362684264],
[0.8923666273338905, 0.23754279703848738, [1.9446800421796862], 0.3389094759539846], 
[0.7439852928100251, 0.38571805348632016, [1.923018657804092], 0.5844017281072322]]

SVR result
[0.8866525324888326, 0.2546317919421483, [1.2842541644858956], 0.33919657417206245],
[0.7374347851829923, 0.33606921298409087, [2.166764694061658], 0.46667375584236787], 
[0.7769923700740443, 0.35324028520918094, [3.3792331061262697], 0.48783179046680636],
[0.6480240946879124, 0.441754034544339, [2.1841366274846603], 0.6852286318912958]]


# n_species performance:
# 9th category descriptor
SVR result
[[0.8169590123488525, 0.5905355265765119, 7.507486745708519, 0.7531292165109436], [0.7689943244620766, 0.41295383691437015, 2.9698287958979934, 0.5994211772324232], [0.8152087705971964, 0.2938680452139048, 1.583224626516713, 0.4209773852622055]]
KRR result
[[0.8218344240959772, 0.5633357443654716, 7.795214001021999, 0.7430314878544134], [0.8344143466323544, 0.3433991296047896, 2.2858622013740413, 0.50749540014687], [0.8758964483507363, 0.2502419294769539, 1.4550982302872564, 0.3449929292796441]]

# natoms performance:
# 9th category descriptor
SVR result
[0.7729204878212963, 0.7890204851492184, 0.8772619912007676]

KRR result
[0.8212165862855878, 0.865736823141431, 0.8929842190033598]

"""