# Ultra-Low-Kappa: Identification of Crystalline Materials with Ultra-Low Thermal Conductivity Based on Machine Learning Study

## Introduction
This repository represents the work we did for accelerating the search of low thermal conducvitiy materials in materials science.


## Dataset
The dataset contains 5486 crystalline materials with thermal conductivity property in total, downloaded from the [AFLOW](http://www.aflowlib.org/) repository. 
The original data was transformed into four categories with 75 dimensions to describe the crystal materials by data preprocessing. They are saved as csv format for 
the sake of accessibility and reproducibility. In addition to the thermal conductivity, other targets of thermal properties like Debye temperature
, heat capacity at constant volume, heat capacity at constant pressure, and thermal expansion coefficient are proviede in the *lables.2020.1.29.csv* file.

## Usage
The training data of csv file can be loaded as 
```python
import pandas as pd
import os
data_path='./data'
labels = pd.read_csv(os.path.join(data_path, 'labels.2020.1.29.csv')).to_numpy()
raw_train_data = pd.read_csv(os.path.join(data_path, 'td.2020.1.29.csv')).to_numpy()
 ```
In our work, the results with best performance of ML models are saved in the [result](https://github.com/xinming365/Ultrow-Low-Kappa/tree/master/result) file. The following is an example to load *ptc_ab.pkl* model
```python
import pickle
with open('./models/ptc_ab.pkl', 'r') as f:
  optimized_model=pickle.load(f)
```

## Prerequisites
The ML models were developed using
* -[Sklearn](https://scikit-learn.org/stable/)
* -[Keras](https://keras.io/) 
* -[Tensorflow](http://www.tensorflow.org/)
* -[XGBoost](https://xgboost.readthedocs.io/en/latest/) 

Other packages used in the work including
* -[Pandas](https://pandas.pydata.org/)
* -[Numpy](https://numpy.org/)
