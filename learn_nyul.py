# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 07:31:40 2017
@author: Subhashis Banerjee
"""

''' Header files '''
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
from scipy import interpolate 

imgPath = ""

# Sequence no.s, Flair = 0, T1c = 1, T1 = 2, T2 = 3

# Variables
total_patient = 15
patient = 0
seq = -1

# Initials values that are considered, read Nyul et al. paper for details
s1 = 1
s2 = 4095
pc1 = 0
pc2 = 99.8/float(100)

final_landmarks = np.zeros((4, 11))
m_k = np.zeros((4, total_patient, 11))
temp = np.zeros((11))

for f1 in os.listdir("Nyul_train/LGG"):
    print '\nPateint no: ', patient+1

    for f2 in os.listdir("Nyul_train/LGG/"+f1):
        for f3 in os.listdir("Nyul_train/LGG/"+f1+"//"+f2):
            if "Flair" in f3:
                seq = 0
            elif "T1c" in f3:
                seq = 1
            elif "T1" in f3:
                seq = 2
            elif "T2" in f3:
                seq = 3
            elif "OT" in f3:
                seq = -1
            imgPath = "Nyul_train/LGG/"+f1+"//"+f2+"//"+f3
            
            if seq!=(-1):
                img = sitk.ReadImage(imgPath)
                arrayImage = sitk.GetArrayFromImage(img)
            
		# Removing very small intensity values, read paper for details.
                template_brainmask = arrayImage > 0.05
                count = np.count_nonzero(template_brainmask)
                template = np.zeros(count)
                
                ct = 0
		# Getting all the intensity values than 0.05 in an array
                for i in range(0, 155):
                    for j in range(0, 240):
                        for k in range(0, 240):
                            if template_brainmask[i][j][k]==1:
                                template[ct] = arrayImage[i][j][k] 
                                ct = ct+1
                
                T = np.sort(template)
		
		# Finding the landmark intensity values
                temp[0] = T[int(math.ceil(pc1*T.shape[0]))]
                temp[1] = T[int(math.ceil(0.1*T.shape[0]))]
                temp[2] = T[int(math.ceil(0.2*T.shape[0]))]
                temp[3] = T[int(math.ceil(0.3*T.shape[0]))]
                temp[4] = T[int(math.ceil(0.4*T.shape[0]))]
                temp[5] = T[int(math.ceil(0.5*T.shape[0]))]
                temp[6] = T[int(math.ceil(0.6*T.shape[0]))]
                temp[7] = T[int(math.ceil(0.7*T.shape[0]))]
                temp[8] = T[int(math.ceil(0.8*T.shape[0]))]
                temp[9] = T[int(math.ceil(0.9*T.shape[0]))]
                temp[10] = T[int(math.ceil(pc2*T.shape[0]))]
                
                y = [s1, s2]
                x = [temp[0], temp[10]]

		# Mapping the intensity values onto the standard scale
                f = interpolate.interp1d(x, y)
            
                i = 0
                for t in temp:
                    m_k[seq][patient][i] = f(t)
                    i = i+1
    patient = patient + 1

# Taking the avg. for all the values in each sequence
for i in range(0, 4):
    sum0=0; sum1=0; sum2=0; sum3=0; sum4=0; sum5=0; sum6=0; sum7=0; sum8=0; sum9=0; sum10=0;
    for j in range(0, total_patient):
        sum0 = sum0 + m_k[i][j][0]
        sum1 = sum1 + m_k[i][j][1]
        sum2 = sum2 + m_k[i][j][2]
        sum3 = sum3 + m_k[i][j][3]
        sum4 = sum4 + m_k[i][j][4]
        sum5 = sum5 + m_k[i][j][5]
        sum6 = sum6 + m_k[i][j][6]
        sum7 = sum7 + m_k[i][j][7]
        sum8 = sum8 + m_k[i][j][8]
        sum9 = sum9 + m_k[i][j][9]
        sum10 = sum10 + m_k[i][j][10]
        
    final_landmarks[i][0] = sum0/float(total_patient)        
    final_landmarks[i][1] = sum1/float(total_patient)  
    final_landmarks[i][2] = sum2/float(total_patient)  
    final_landmarks[i][3] = sum3/float(total_patient)  
    final_landmarks[i][4] = sum4/float(total_patient)  
    final_landmarks[i][5] = sum5/float(total_patient)  
    final_landmarks[i][6] = sum6/float(total_patient)  
    final_landmarks[i][7] = sum7/float(total_patient)  
    final_landmarks[i][8] = sum8/float(total_patient)  
    final_landmarks[i][9] = sum9/float(total_patient)  
    final_landmarks[i][10] = sum10/float(total_patient)

np.save("nyul_landmarks_lgg.npy", final_landmarks)
print final_landmarks

        
 
    
