# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 20:33:38 2017

@author: Subhashis Banerjee
"""

import numpy as np
import pylab as plt
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import signal
from sklearn.utils import shuffle
import random
from skimage.morphology import remove_small_objects
import SimpleITK as sitk
from skimage.measure import regionprops
from skimage import measure
from scipy.ndimage.morphology import binary_erosion
import sys
import cv2
from skimage.morphology import erosion
from skimage.morphology import ball

# Reading the image over which post processing have to be applied.
img = sitk.ReadImage("BRATS2015_Training_Nyul/segment_test/segmented_brats_2013_pat0011_1.mha")
img = sitk.GetArrayFromImage(img)

# Creating a binary image using the input image
binary_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        for k in range(0, img.shape[2]):
            if img[i][j][k]!=0:
                binary_img[i][j][k] = 1

label_img = measure.label(binary_img, background=0)

# Printing all the regions and the no. of pixels in each region
unique, counts = np.unique(label_img, return_counts=True)
print unique, counts

# Stroing all the regions which has less than 10,000 pixels
temp = []
for i in range(0, len(counts)):
    if counts[i] > 10000:
        temp.append(unique[i])

# Printing all regions which have more than 10,000 pixels
print 'this: ', temp

# Removing regions from the binary image
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        for k in range(0, img.shape[2]):
            if label_img[i][j][k] not in temp:
                label_img[i][j][k] = 0

# Updating the input image according to the binary image, this will remove regions which are less than 10,000 pixels from the original input image.
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        for k in range(0, img.shape[2]):
            if label_img[i][j][k]==0:
                img[i][j][k]=0
       
# Uncomment this code if you wish to apply Errosion also.                 
#new_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
#for i in range(0, img.shape[0]):
#    for j in range(0, img.shape[1]):
#        for k in range(0, img.shape[2]):
#            if img[i][j][k]!=0:
#                new_img[i][j][k]=1
#
#temp = erosion(new_img, ball(2))
#
#for i in range(0, img.shape[0]):
#    for j in range(0, img.shape[1]):
#        for k in range(0, img.shape[2]):
#            if temp[i][j][k]==0:
#                img[i][j][k]=0

sitk.WriteImage(sitk.GetImageFromArray(img), "BRATS2015_Training_Nyul/segment_test/temp.mha")


 
