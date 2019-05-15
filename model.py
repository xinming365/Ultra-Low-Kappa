import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow_estimator import E

data_path = './data'
labels = np.load(os.path.join(data_path,'labels.npy'))
train_data = np.load(os.path.join(data_path,'train_version_1.npy'))
# thermal conductivity label(agl_thermal_conductivity_300K)
label = labels[:, 0]
X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.1, random_state=4)
# If int, random_state is the seed used by the random number generator;
scaler = preprocessing.StandardScaler().fit(X_train)
mean_ = scaler.mean_  # 保存了均值
scale_ = scaler.scale_  # 保存了标准差
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)


def dnn_model():
    length = 56
    batch_size = 32
    inputs = Input(shape=(length,))
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu' )(x)
    x = Dense(48, activation='relu')(x)
    predictions = Dense(1)(x)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model = Model(inputs = inputs, outputs= predictions)
    model.compile(optimizer= optimizer,loss = 'mse', metrics=['mean_absolute_error'])
    return model

def dnn_predict():
    EPOCHS = 300
    model = dnn_model()
    model.summary()
    model.fit(x = X_train_transformed, y= y_train, epochs= EPOCHS)
    # Returns the loss value & metrics values for the model in test mode.
    scalars_train = model.evaluate(X_train_transformed,y_train)
    scalars_test = model.evaluate(X_test_transformed,y_test)
    result = model.predict(X_test_transformed)
    print(result.shape)
    print(scalars_train,scalars_test)
    print(np.mean(label))

def cnn_model(features,labels,mode):
    """
    :param features: the descriptors of materials
    :param labels: the thermotics property
    :param mode: tf.estimator.Modekeys.TRAIN OR tf.estimator.Modekeys.PREDICT
    :return: cnn_model for training, evaluating, and prediction

    """
    # [batch_size, image_height, image_width, channels]
    input_layer = []

    # Convolutional Layer
    conv1 = tf.layers.conv2d(inputs= input_layer,filters= 32,kernel_size=[3,3],padding='same',activation=tf.nn.relu)

    # Pooling Layer
    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

    # Convolutional Layer and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding='same', activation = tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2,[-1,7*7*64])
    dense = tf.layers.dense(inputs = pool2_flat,units=1024,activation=tf.nn.relu,name='layer_fc1')
    net = tf.layers.dense(inputs=dense,units=1, name ='layer_fc2')

    # output of the neural network
    predictions = net

    if mode == tf.estimator.Modekeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions = predictions)
    loss = tf.losses.mean_squared_error(labels=labels,predictions=net)
    # Configure the training Op
    if mode == tf.estimator.Modekeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss = loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss=loss, train_op=train_op)
    # Add evaluation metrics
    eval_metric_ops = {"accuracy":tf.metrics.mean_absolute_error(labels=labels,predictions=predictions)}
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

def cnn_predict()
    train_data = []
    train_labels = y_train

    eval_data = []
    eval_labels = y_train

    # Create the Estimator
    cnn_predict = tf.estimator.Estimator(model_fn = cnn_model, model_dir = './tmp')

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=train_data,y=train_labels
                                                        ,batch_size = 100, shuffle=True)
    cnn_predict.train(input_fn = train_input_fn,steps=2000)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=eval_data,y=eval_labels,num_epochs=1,shuffle=False)
    eval_results = cnn_predict.evaluate(input_fn=eval_input_fn)
    print(eval_results)


















