import numpy as np
import os

file_path = "./data/"
train_data = np.load(file_path+'train_data.npy')
auid = np.load(file_path+'auid.npy')
print(train_data[0])
print(auid[0])


