# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 07:31:40 2017
@author: Subhashis Banerjee
"""

''' Header files '''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers
K.set_image_dim_ordering('th')
from keras.layers.advanced_activations import LeakyReLU as LRelu
import sys
import os
from PIL import Image
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import random

seed = 7
np.random.seed(seed)

# Defining the model for HGG patients    
num_classes=5    
def baseline_model():
     # create model
     model = Sequential()
     
     model.add(Conv2D(64, 3, 3, border_mode='same', input_shape=(4, 33, 33), init='glorot_uniform'))
     model.add(LRelu(alpha=0.333))
     
     model.add(Conv2D(64, 3, 3, border_mode='same', input_shape=(64, 33, 33), init='glorot_uniform'))
     model.add(LRelu(alpha=0.333))     

     model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))    
    
     model.add(Conv2D(64, 3, 3, border_mode='same', input_shape=(64, 16, 16), init='glorot_uniform'))
     model.add(LRelu(alpha=0.333))
     
     model.add(Conv2D(128, 3, 3, border_mode='same', input_shape=(64, 16, 16), init='glorot_uniform'))
     model.add(LRelu(alpha=0.333))
     
     model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
     
     model.add(Conv2D(128, 3, 3, border_mode='same', input_shape=(128, 7, 7), init='glorot_uniform'))     
     model.add(LRelu(alpha=0.333))
     
     model.add(Conv2D(128, 3, 3, border_mode='same', input_shape=(128, 7, 7), init='glorot_uniform'))
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

# Loading the weights for the model
print ' -> Loading weights'
fname = "Weights_new_model/weights-18.hdf5"
model.load_weights(fname)

''' Opening the files '''
originalFlairPath = ""
originalT1cPath = ""
originalT1Path = ""
originalT2Path = ""
originalTruthPath = ""

patient = 0
max = 0
min = 0

# Looping through the testing dataset
for f1 in os.listdir("BRATS2015_Training_Nyul/segment_test"):
#    if patient == 5:
#        break
    for f2 in os.listdir("BRATS2015_Training_Nyul/segment_test/"+f1):
        for f3 in os.listdir("BRATS2015_Training_Nyul/segment_test/"+f1+"//"+f2):
            if f3.endswith(".mha"):
                if "Flair" in f3:
                    originalFlairPath = "BRATS2015_Training_Nyul/segment_test/"+f1+"//"+f2+"//"+f3
                if "T1c" in f3:
                    originalT1cPath = "BRATS2015_Training_Nyul/segment_test/"+f1+"//"+f2+"//"+f3
                if "T1" in f3:
                    originalT1Path = "BRATS2015_Training_Nyul/segment_test/"+f1+"//"+f2+"//"+f3
                if "T2" in f3:
                    originalT2Path = "BRATS2015_Training_Nyul/segment_test/"+f1+"//"+f2+"//"+f3
                if "OT" in f3:
                    originalTruthPath = "BRATS2015_Training_Nyul/segment_test/"+f1+"//"+f2+"//"+f3
    
    imgT1Original = sitk.ReadImage(originalT1Path)
    imgT2Original = sitk.ReadImage(originalT2Path)
    imgT1cOriginal = sitk.ReadImage(originalT1cPath)
    imgFlairOriginal = sitk.ReadImage(originalFlairPath)
    imgTruthOriginal = sitk.ReadImage(originalTruthPath)

    arrayT1 = sitk.GetArrayFromImage(imgT1Original)
    arrayT2 = sitk.GetArrayFromImage(imgT2Original)
    arrayT1c = sitk.GetArrayFromImage(imgT1cOriginal)
    arrayFlair = sitk.GetArrayFromImage(imgFlairOriginal)
    arrayTruth = sitk.GetArrayFromImage(imgTruthOriginal)

    write_path = "BRATS2015_Training_Nyul/segment_test/segmented_"+f1+".mha"

                        
    patient = patient + 1
    b = np.load("black_patch.npy")
    black = b[0] # 'black' will basically contain a 4x33x33 matrix where all the value will be 0

    i=0
    j=0
    t = 0
    print("\n Finding patches for patient no. ", patient)
    for z in range(0, 155):
	# variable to store all the patches for a brain slice
        patches = []
        if z%10==0:
            print ' -> Slice reached: ', z
            
        i = 0
        sliceT1 = arrayT1[z]
        sliceT2 = arrayT2[z]
        sliceT1c = arrayT1c[z]
        sliceFlair = arrayFlair[z]
        sliceTruth = arrayTruth[z]
        while i<240:
            j = 0
            while j<240:
                if (i-16)<0 or (i+16+1)>240 or (j-16)<0 or (j+16+1)>240:
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
        
	# Pass the patches through the model to get the class level for that patch
        file_name = 'output/predict' + str(z) + '.npy'
        np.save(file_name, model.predict_classes(arr0, verbose=0))
    
    # Combine all the segmentation files generated for each brain slice
    print ' -> Generating output....'            
    a0 = np.load("output/predict0.npy")
    for z in range(1, 155):
        file_name = 'output/predict' + str(z) + '.npy'
        a1 = np.load(file_name)
        a0 = np.concatenate((a0, a1), axis=0)
    
    arr = np.zeros((155, 240, 240))
    i=0
    j=0
    k=0
    for t in a0:
        arr[k][i][j] = t
        j=j+1
        if j%240==0:
            j=0
            i=i+1
            if i%240==0:
                i=0
                k=k+1
    sitk.WriteImage(sitk.GetImageFromArray(arr), write_path)

  
print('done')
