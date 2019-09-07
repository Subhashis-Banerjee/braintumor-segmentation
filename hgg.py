# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 09:03:07 2017

@author: Subhashis Banerjee
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
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

''' Good website read '''
''' http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/ '''

# Load data, Pre-process, CNN
X_train = np.load("patch_combine_hgg.npy")
y_train = np.load("class_combine_hgg.npy")

print X_train.shape[0]


# We have tried this pre-processing, but it doesnot work.
#seq1 = a1[:, 0, :, :]
#seq2 = a1[:, 1, :, :]
#seq3 = a1[:, 2, :, :]
#seq4 = a1[:, 3, :, :]
#
#if (np.std(seq1)!=0):
#    seq1 = (seq1 - np.mean(seq1)) / np.std(seq1)
#
#if (np.std(seq2)!=0):
#    seq2 = (seq2 - np.mean(seq2)) / np.std(seq2)
#
#if (np.std(seq3)!=0):
#    seq3 = (seq3 - np.mean(seq3)) / np.std(seq3)
#
#if (np.std(seq4)!=0):
#    seq4 = (seq4 - np.mean(seq4)) / np.std(seq4)
#
#for i in range(0, seq1.shape[0]):
#    a1[i][0] = seq1[i]
#    a1[i][1] = seq2[i]
#    a1[i][2] = seq3[i]
#    a1[i][3] = seq4[i]    

# Data normalisation
X_train = X_train/float(32767)

# Shuffling the data for better training
X_train, y_train = shuffle(X_train, y_train)

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 4, 33, 33).astype('float32')

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
num_classes = 5

# CNN model for HGG patients
def baseline_model():
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

# build the model
model = baseline_model()

# Real time data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    vertical_flip=True)

datagen.fit(X_train)

# checkpoint
filepath="Weights_withtumor25_same/weights-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_weights_only=True)
callbacks_list = [checkpoint]

# Fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, y_train, batch_size=128), samples_per_epoch=len(X_train), nb_epoch=20, verbose=2)
#history = model.fit(X_train, y_train, nb_epoch=20, validation_split=0.10, batch_size=128, verbose=2)

# Final evaluation of the model
#%%

# list all data in history
#print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

# Save weights
#%%
#fname = "weights-hgg.hdf5"
#model.save_weights(fname, overwrite=True)

# Load weights
#%%
#fname = "weights-hgg.hdf5"
#model.load_weights(fname)


# Segmentation slice by slice
#%%
#test1 = np.load("Patches_segmentation/patches_for_slice16.npy")
#np.save("predict16.npy", model.predict_classes(test1))
#plt.imshow(np.load("predict1.npy"))




