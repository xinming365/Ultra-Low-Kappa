from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

file_path = './data/'
seed = 331 #random seed used by the random number generator;

def data_standardize(file_path):
    train_data = np.load(file_path+'train_data.npy')
    labels = np.load(file_path + 'labels_data.npy')
    X_train,X_test,y_train,y_test = train_test_split(scaled_train_data,label,test_size=0.15,random_state=seed)
    scaler = preprocessing.StandardScaler().fit(X_train)
    #mean_ = scaler.mean_ #保存了均值
    #scale_ = scaler.scale_ #保存了标准差
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    return X_train_transformed,y_train,X_test_transformed,y_test
X_train_transformed, y_train, X_test_transformed, y_test = data_standardize(file_path)







