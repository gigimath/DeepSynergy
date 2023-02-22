import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip

import matplotlib.pyplot as plt

import keras as K
import tensorflow as tf
from keras import backend
from keras.backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout




def get_model(X_tr, eta, layers, act_func, input_dropout, dropout):
    model = Sequential()
    for i in range(len(layers)):
        if i==0:
            model.add(Dense(layers[i], input_shape=(X_tr.shape[1],), activation=act_func, 
                            kernel_initializer='he_normal'))
            model.add(Dropout(float(input_dropout)))
        elif i==len(layers)-1:
            model.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
        else:
            model.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
            model.add(Dropout(float(dropout)))
        model.compile(loss='mean_squared_error', optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))
    return model

def train_1(model, X_tr, y_tr, X_val, y_val, epochs):
    hist = model.fit(X_tr, y_tr, epochs=epochs, shuffle=True, batch_size=64, validation_data=(X_val, y_val))
    val_loss = hist.history['val_loss']
    model.reset_states()
    return hist, val_loss, model

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def smooth(val_loss):
    average_over = 15
    mov_av = moving_average(np.array(val_loss), average_over)
    smooth_val_loss = np.pad(mov_av, int(average_over/2), mode='edge')
    epo = np.argmin(smooth_val_loss)
    return epo, smooth_val_loss

def train_2(model, X_train, y_train, X_test, y_test, epo):
    hist = model.fit(X_train, y_train, epochs=epo, shuffle=True, batch_size=64, validation_data=(X_test, y_test))
    test_loss = hist.history['val_loss']
    return hist, test_loss

def plot_performance(fold, val_loss, smooth_val_loss, test_loss):
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(val_loss, label='validation loss')
    ax.plot(smooth_val_loss, label='smooth validation loss')
    ax.plot(test_loss, label='test loss')
    ax.legend()
    plt.savefig('performances'+os.sep+'performance_'+fold+'.png')
    #plt.show()
    return

def save_model(fold, model):
    name_model = 'modelsDir'+os.sep+'model_'+fold
    model.save(name_model)
    return








hyperparameter_file = 'hyperparameters' # textfile which contains the hyperparameters of the model
exec(open(hyperparameter_file).read())

path_data = os.getcwd()+os.sep+'data_cv'+os.sep

# 0, 1, 2, 3
deviceNumb = 2
# 0, 1, 2, 3 ,4
foldNumb = 2

os.environ["CUDA_VISIBLE_DEVICES"]=str(deviceNumb)

fold = 'fold'+str(foldNumb)
file_name = path_data+'data_test_'+fold+'_tanh_norm.p'
file = open(file_name,'rb')
X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = pickle.load(file)
file.close()

config = tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True))
set_session(tf.compat.v1.Session(config=config))


model = get_model(X_tr, eta, layers, act_func, input_dropout, dropout)
hist1, val_loss, model = train_1(model, X_tr, y_tr, X_val, y_val, epochs)
epo, smooth_val_loss = smooth(val_loss)
hist2, test_loss = train_2(model, X_train, y_train, X_test, y_test, epo)
plot_performance(fold, val_loss, smooth_val_loss, test_loss)
save_model(fold, model)

