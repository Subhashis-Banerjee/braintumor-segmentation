# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:22:37 2017

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
import random
import SimpleITK as sitk
import os

seed = 7
np.random.seed(seed)

# CNN model for LGG patients
num_classes = 5
def model_transfer():
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

    print ' -> Loading weights'
    # Load the weight learned for this CNN
    fname = "Weight_transfer_conv_included/weights-19.hdf5"
    model.load_weights(fname)
    return model

# CNN model for HGG patients
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

# Removing the last 3 FC layer, one max-pooling layer and one convolution layer from the HGG model
def model_main():
    model = hgg_model()
    # Load the weights learned for this CNN
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
    model_final = model_transfer()
    print(" -> Model ready. Starting predictions")

    # Loop for each patient in the training dataset
    patient = 0
    for f1 in os.listdir("BRATS2015_Nyul/LGG"):
        for f2 in os.listdir("BRATS2015_Nyul/LGG/"+f1):
            for f3 in os.listdir("BRATS2015_Nyul/LGG/"+f1+"//"+f2):
                if f3.endswith(".mha"):
                    if "Flair" in f3:
                        originalFlairPath = "BRATS2015_Nyul/LGG/"+f1+"//"+f2+"//"+f3
                    if "T1c" in f3:
                        originalT1cPath = "BRATS2015_Nyul/LGG/"+f1+"//"+f2+"//"+f3
                    if "T1" in f3:
                        originalT1Path = "BRATS2015_Nyul/LGG/"+f1+"//"+f2+"//"+f3
                    if "T2" in f3:
                        originalT2Path = "BRATS2015_Nyul/LGG/"+f1+"//"+f2+"//"+f3
    
        imgT1Original = sitk.ReadImage(originalT1Path)
        imgT2Original = sitk.ReadImage(originalT2Path)
        imgT1cOriginal = sitk.ReadImage(originalT1cPath)
        imgFlairOriginal = sitk.ReadImage(originalFlairPath)
    
        arrayT1 = sitk.GetArrayFromImage(imgT1Original)
        arrayT2 = sitk.GetArrayFromImage(imgT2Original)
        arrayT1c = sitk.GetArrayFromImage(imgT1cOriginal)
        arrayFlair = sitk.GetArrayFromImage(imgFlairOriginal)
    
        write_path = "BRATS2015_Nyul/LGG/segmented_"+f1+".mha"
    
        patient = patient + 1
        b = np.load("black_patch.npy")
        black = b[0] # 'black' will basically contain a 4x33x33 matrix where all the value will be 0
    
        i=0
        j=0
        t = 0
        print("\n Finding patches for patient no. ", patient)
	# Loop through each slice
        for z in range(0, arrayT1.shape[0]):
	    # variable to store all the patches for a slice
            patches = []   
            print ' -> Slice reached: ', z
                
            i = 0
            sliceT1 = arrayT1[z]
            sliceT2 = arrayT2[z]
            sliceT1c = arrayT1c[z]
            sliceFlair = arrayFlair[z]
            while i<arrayT1.shape[1]:
                j = 0
                while j<arrayT1.shape[2]:
                    if (i-16)<0 or (i+16+1)>arrayT1.shape[1] or (j-16)<0 or (j+16+1)>arrayT1.shape[2]:
                        patches.append(black)
                    else:
                        temp = []
                        temp.append(sliceT1[i-16:i+16+1, j-16:j+16+1])
                        temp.append(sliceT2[i-16:i+16+1, j-16:j+16+1])
                        temp.append(sliceT1c[i-16:i+16+1, j-16:j+16+1])
                        temp.append(sliceFlair[i-16:i+16+1, j-16:j+16+1])
                            
                        patches.append(temp) 
                    j = j+1
                i = i+1
            
            arr0 = np.zeros(len(patches))
            arr0 = np.array(patches)
            arr0 = arr0/float(32767)
    		
	    # Pass the patches through the HGG model to get vector of size 128x16x16
            features = model.predict(arr0)        
            
	    # Pass the vectors through the LGG model to generate the class level for that patch
            file_name = 'output/predict' + str(z) + '.npy'
            np.save(file_name, model_final.predict_classes(features, verbose=0))
    
    	# Combine all the segmentation files generated for each brain slice
        print ' -> Generating output....'            
        a0 = np.load("output/predict0.npy")
        for z in range(1, arrayT1.shape[0]):
            file_name = 'output/predict' + str(z) + '.npy'
            a1 = np.load(file_name)
            a0 = np.concatenate((a0, a1), axis=0)
        
        arr = np.zeros((arrayT1.shape[0], arrayT1.shape[1], arrayT1.shape[2]))
        i=0
        j=0
        k=0
        for t in a0:
            arr[k][i][j] = t
            j=j+1
            if j%arrayT1.shape[2]==0:
                j=0
                i=i+1
                if i%arrayT1.shape[1]==0:
                    i=0
                    k=k+1
        sitk.WriteImage(sitk.GetImageFromArray(arr), write_path)








    
