# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:14:05 2017

@author: servad
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

# Loading data           
check = sitk.ReadImage("BRATS2015_Training_Nyul/segment_test/pat11.mha")
img = sitk.ReadImage("BRATS2015_Training_Nyul/segment_test/temp.mha")

check = sitk.GetArrayFromImage(check)
img = sitk.GetArrayFromImage(img)

# Dice score
o_binary = np.zeros((155, 240, 240))
for i in range(0, 155):
    for j in range(0, 240):
        for k in range(0, 240):
            if check[i][j][k]==1 or check[i][j][k]==2 or check[i][j][k]==3 or check[i][j][k]==4:                
                o_binary[i][j][k] = 1
                
m_binary = np.zeros((155, 240, 240))
for i in range(0, 155):
    for j in range(0, 240):
        for k in range(0, 240):
            if img[i][j][k]==1 or img[i][j][k]==2 or img[i][j][k]==3 or img[i][j][k]==4:
                m_binary[i][j][k] = 1

temp = np.logical_and(o_binary, m_binary)
z = 2*np.count_nonzero(temp)/float((np.count_nonzero(m_binary)+np.count_nonzero(o_binary)))
print ' -> Dice score Complete(1,2,3,4): ', z*100

# Score Core
for i in range(0, 155):
    for j in range(0, 240):
        for k in range(0, 240):
            if check[i][j][k]==1 or check[i][j][k]==3 or check[i][j][k]==4:                
                o_binary[i][j][k] = 1
            else:
                o_binary[i][j][k] = 0
                
for i in range(0, 155):
    for j in range(0, 240):
        for k in range(0, 240):
            if img[i][j][k]==1 or img[i][j][k]==3 or img[i][j][k]==4:
                m_binary[i][j][k] = 1
            else:
                m_binary[i][j][k] = 0
temp = np.logical_and(o_binary, m_binary)
z = 2*np.count_nonzero(temp)/float((np.count_nonzero(m_binary)+np.count_nonzero(o_binary)))
print ' -> Dice score Core(1,3,4): ', z*100

# Score Enhancing
#for i in range(0, 155):
#    for j in range(0, 240):
#        for k in range(0, 240):
#            if check[i][j][k]==4:                
#                o_binary[i][j][k] = 1
#            else:
#                o_binary[i][j][k] = 0
#                
#for i in range(0, 155):
#    for j in range(0, 240):
#        for k in range(0, 240):
#            if img[i][j][k]==4:
#                m_binary[i][j][k] = 1
#            else:
#                m_binary[i][j][k] = 0
#temp = np.logical_and(o_binary, m_binary)
#z = 2*np.count_nonzero(temp)/float((np.count_nonzero(m_binary)+np.count_nonzero(o_binary)))
#print ' -> Dice score Enhancing(4): ', z*100


