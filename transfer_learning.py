# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:45:25 2017

@author: servad
"""
import numpy as np
import pylab as plt
from keras import initializations
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras import optimizers
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.layers.advanced_activations import LeakyReLU as LRelu
from sklearn.utils import shuffle
import sys


# Defining the model that has been used for HGG patients.
num_classes = 5
def hgg_model():
     # create model
     model = Sequential()
     model.add(Conv2D(64, 3, 3, border_mode='same', input_shape=(4, 33, 33), init='glorot_uniform'))
     model.add(LRelu(alpha=0.333))
     model.add(Conv2D(64, 3, 3, border_mode='same', input_shape=(64, 33, 33), init='glorot_uniform'))
     model.add(LRelu(alpha=0.333))     
     model.add(Conv2D(64, 3, 3, border_mode='same', input_shape=(64, 33, 33), init='glorot_uniform'))
     model.add(LRelu(alpha=0.333))
     model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
     model.add(Conv2D(128, 3, 3, border_mode='same', input_shape=(64, 16, 16), init='glorot_uniform'))
     model.add(LRelu(alpha=0.333))
     model.add(Conv2D(128, 3, 3, border_mode='same', input_shape=(128, 16, 16), init='glorot_uniform'))     
     model.add(LRelu(alpha=0.333))
     model.add(Conv2D(128, 3, 3, border_mode='same', input_shape=(128, 16, 16), init='glorot_uniform'))
     model.add(LRelu(alpha=0.333))
     model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
     model.add(Dropout(0.1))
     model.add(Flatten())
     model.add(Dense(256, init='glorot_uniform'))
     model.add(LRelu(alpha=0.333))
     model.add(Dense(256, init='glorot_uniform'))
     model.add(LRelu(alpha=0.333))
     model.add(Dense(num_classes, init='glorot_uniform', activation='softmax'))
     # Compile model
     sgd = optimizers.SGD(lr=0.003, decay=15e-5, momentum=0.9, nesterov=True)
     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
     return model

# Removing the last three FC layers and one max-pooling and one convolution layer from the HGG model.
def model_main():
    model = hgg_model()
    # Loading the weights learned for the HGG model.
    model.load_weights('Weights_withtumor25_same/weights-19.hdf5')
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

   
if __name__ == "__main__":
    print(" -> Preparing model")
    model = model_main()
    print(" -> Model ready. Starting predictions")

    # Loading the Patch library for LGG patients
    X_train = np.load("patch_temp_lgg.npy")
    y_train = np.load("class_temp_lgg.npy")
    
    X_train = X_train/float(32767)

    y_train = np_utils.to_categorical(y_train)
  
    # Saving a vector of length 128x16x16 which will be given as output from 'model'
    np.save("transfer_features.npy", model.predict(X_train))
    np.save("transfer_classes.npy", y_train)	

    print '...........Done\n\n'






































