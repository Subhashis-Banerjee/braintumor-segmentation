# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 07:31:40 2017
@author: Mohit Singhaniya
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

''' Opening the files '''
originalFlairPath = ""
originalT1cPath = ""
originalT1Path = ""
originalT2Path = ""

patient = 0

m_k = np.zeros((4, 11))

# Initial values, read Nyul et al. paper for details
s1 = 1
s2 = 4095
pc1 = 0
pc2 = 99.8/float(100)

# Loading the learned intensity landmarks
m_k = np.load("nyul_landmarks_lgg.npy")
temp = np.zeros((11))

# Creating directory
directory = "BRATS2015_Nyul/LGG" 
if not os.path.exists(directory):
    os.makedirs(directory)

for f1 in os.listdir("BRATS2015_Corrected/LGG"):    
    print 'Doing for patient: ', patient+1

    # Creating directory
    directory = "BRATS2015_Nyul/LGG/" + f1
    if not os.path.exists(directory):
        os.makedirs(directory)

    for f2 in os.listdir("BRATS2015_Corrected/LGG/"+f1):
	# Creating directory
        directory = "BRATS2015_Nyul/LGG/" +f1+"//"+f2
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        for f3 in os.listdir("BRATS2015_Corrected/LGG/"+f1+"//"+f2):
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

            write_path = "BRATS2015_Nyul/LGG/"+f1+"//"+f2+"//"+f3
            imgPath = "BRATS2015_Corrected/LGG/"+f1+"//"+f2+"//"+f3

	    # if the file is for the ground truth values, then simply copy the same
            if seq==(-1):
                inputImage = sitk.ReadImage(imgPath)
                sitk.WriteImage(inputImage, write_path) 

	    # apply transformation as defined in the paper
            else:
                img = sitk.ReadImage(imgPath)
                arrayImage = sitk.GetArrayFromImage(img)
            
		# Removing very small intensity values, read paper for details.
                template_brainmask = arrayImage > 0.05
                count = np.count_nonzero(template_brainmask)
                template = np.zeros(count)
            
                ct = 0
		# Getting all the intensity values than 0.05 in an array
                for i in range(0, arrayImage.shape[0]):
                    for j in range(0, arrayImage.shape[1]):
                        for k in range(0, arrayImage.shape[2]):
                            if template_brainmask[i][j][k]==1:
                                template[ct] = arrayImage[i][j][k] 
                                ct = ct+1
                
                T = np.sort(template)
                
                min = np.amin(T)
                    
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
                
                max = np.amax(T)
                
                mask = arrayImage < temp[0];
                
                SscaleExtremeMin= m_k[seq][0] + (min - temp[0]) / (temp[1] - temp[0]) * (m_k[seq][1] - m_k[seq][0]);
                SscaleExtremeMax= m_k[seq][9] + (max - temp[9]) / (temp[10] - temp[9]) * (m_k[seq][10] - m_k[seq][9])
                
                x = np.zeros(12)
                i = 0
                for t in temp:
                    x[i] = t
                    i = i+1
                x[i] = max
                
                y = np.zeros(12)
                i = 0
                for t in m_k[seq]:
                    y[i] = t
                    i = i+1
                y[i] = SscaleExtremeMax
		
		# Do the mapping
                f = interpolate.interp1d(x, y)
                
		# Generate the transformed image
                for i in range(0, arrayImage.shape[0]):
                    for j in range(0, arrayImage.shape[1]):
                        for k in range(0, arrayImage.shape[2]):
                            if mask[i][j][k]==1:
                                arrayImage[i][j][k] = 0
                            else:
                                arrayImage[i][j][k] = f(arrayImage[i][j][k])

                sitk.WriteImage(sitk.GetImageFromArray(arrayImage), write_path)
    patient = patient + 1
                
