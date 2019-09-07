# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:47:41 2017

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


# Loading training vectors for the LGG model
X_train = np.load('transfer_features.npy')
y_train = np.load('transfer_classes.npy')

X_train, y_train = shuffle(X_train, y_train) 
X_train = X_train.reshape(X_train.shape[0], 128, 16, 16).astype('float32')

# Defining the model for the LGG patients
num_classes = 5
model = Sequential()
model.add(Conv2D(128, 3, 3, border_mode='same', input_shape=(128, 16, 16), init='glorot_uniform'))
model.add(LRelu(alpha=0.333))

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(256, init='glorot_uniform'))
model.add(LRelu(alpha=0.333))

model.add(Dense(256, init='glorot_uniform'))
model.add(LRelu(alpha=0.333))
    
model.add(Dense(num_classes, init='glorot_uniform', activation='softmax'))
# Compile model
sgd = optimizers.SGD(lr=0.003, decay=15e-5, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.summary()

# checkpoint, it helps to store the weights learned during each epoch
filepath="Weight_transfer_conv_included/weights-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_weights_only=True)
callbacks_list = [checkpoint]

# Fit data into model
model.fit(X_train, y_train, nb_epoch=20, batch_size=128, verbose=2, callbacks=callbacks_list)

