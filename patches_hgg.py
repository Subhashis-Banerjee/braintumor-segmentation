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
import random

''' Global variables '''
#Threshold for PATCHES
threshold = 60
thresholdBackground = 75
patchLength = 33

''' Function to find non-zero background pixels in a patch '''
def entropyBackground(patchT1):
    count = 0
    for i in range(0, patchLength):
        for j in range(0, patchLength):
            if patchT1[i][j]!=0:
                count = count+1
    return (count*100)/(patchLength*patchLength)
    
''' Funtion to find the percentage entrophy '''
def findEntropy(patchTruth, patchT1):
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    for i in range(0, patchLength):
        for j in range(0, patchLength):
            if patchTruth[i][j]==1:
                b = b+1
            elif patchTruth[i][j]==2:
                c = c+1
            elif patchTruth[i][j]==3:
                d = d+1
            elif patchTruth[i][j]==4:
                e = e+1
            elif patchT1[i][j]!=0:
                a = a+1
    return ((b+c+d+e)*100)/(patchLength*patchLength)

''' Function to check if the present point is already taken in some patch or not '''
def check(points, i, j):
    for t in points:
        i0 = t[0]
        i1 = t[1]
        j0 = t[2]
        j1 = t[3]
        if i>=i0 and i<=i1 and j>=j0 and j<=j1:
            return True
    return False
    
allPatches0 = []
allPatches1 = []
allPatches2 = []
allPatches3 = []
allPatches4 = []

totalClass1 = 0
totalClass2 = 0
totalClass3 = 0
totalClass4 = 0

''' Opening the files '''
originalFlairPath = ""
originalT1cPath = ""
originalT1Path = ""
originalT2Path = ""
originalTruthPath = ""

patient = 0

# Looping through all the training data
# We have divided the dataset in some parts and we are running multiple instance of spyder using different terminals over each part to do the processing parallely
for f1 in os.listdir("BRATS2015_Training_Nyul/LGG/part6"):    
    for f2 in os.listdir("BRATS2015_Training_Nyul/LGG/part6/"+f1):
        for f3 in os.listdir("BRATS2015_Training_Nyul/LGG/part6/"+f1+"//"+f2):
            if f3.endswith(".mha"):
                if "Flair" in f3:
                    originalFlairPath = "BRATS2015_Training_Nyul/LGG/part6/"+f1+"//"+f2+"//"+f3
                if "T1c" in f3:
                    originalT1cPath = "BRATS2015_Training_Nyul/LGG/part6/"+f1+"//"+f2+"//"+f3
                if "T1" in f3:
                    originalT1Path = "BRATS2015_Training_Nyul/LGG/part6/"+f1+"//"+f2+"//"+f3
                if "T2" in f3:
                    originalT2Path = "BRATS2015_Training_Nyul/LGG/part6/"+f1+"//"+f2+"//"+f3
                if "OT" in f3:
                    originalTruthPath = "BRATS2015_Training_Nyul/LGG/part6/"+f1+"//"+f2+"//"+f3

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
    
    unique, counts = np.unique(arrayTruth, return_counts=True)
    print(unique, counts)  
    
    patient = patient + 1
    patches0_0 = []
    patches0_1 = []
    patches0_2 = []
    patches0 = []
    patches1 = []
    patches2 = []
    patches3 = []
    patches4 = []
    region = 0
    endz = 0

    # We are dividing the image into 3 regions, where region 0 is the region from the first brain slice to the slice where tumor is seen for the first time
    # Region 1 is the region where tumor is present
    # Region 2 is the region from the slice just after the slice where tumor was seen last to the end of the brain image    

    for z in range(154, 0, -1):
        if endz!=0:
            break
        sliceTruth = arrayTruth[z]
        for i in range(0, 240):
            for j in range(0, 240):
                if sliceTruth[i][j]!=0:
                    endz = z
                    break
            if endz!=0:
                break
        if endz!=0:
                break
    
    i=0
    j=0
    ok = 0
    print("\n Finding patches for patient no. ", patient)
    # Looping through each brain slice
    for z in range(0, 155):
        patch0Points = []
        patch1Points = []
        patch2Points = []
        patch3Points = []
        patch4Points = []
        i = 15
        if z%25==0:
            print(" Slice reached. ", z)        
        if z == endz:
            region = 2
        sliceT1 = arrayT1[z]
        sliceT2 = arrayT2[z]
        sliceT1c = arrayT1c[z]
        sliceFlair = arrayFlair[z]
        sliceTruth = arrayTruth[z]
        while i<223:
            i = i+1
            j = 15
            while j<223:
                j = j+1
                if sliceT1[i][j]!=0 and sliceTruth[i][j]==0 and not(check(patch0Points, i, j)):
		    # We are taking only 200 images from region 0, 2000 from region 1 and 200 from region 2
		    # Refer method 3 from the report to understand this algorithm better
                    if (region==0 and len(patches0_0)<200) or (region==1 and len(patches0_1)<2000) or (region==2 and len(patches0_2)<200):
                        truthPatch =  sliceTruth[i-16:i+16+1, j-16:j+16+1] 

                        if(not(np.any(truthPatch[:, :] != 0))):
#                        if np.count_nonzero(truthPatch)<=544:
                            patchT1 = sliceT1[i-16:i+16+1, j-16:j+16+1]
                            
                            if entropyBackground(patchT1) >= thresholdBackground:
                                t = []; 
                                t.append(i-8); t.append(i+8); t.append(j-8); t.append(j+8)
                                patch0Points.append(t)
                                    
                                temp = []
                                temp.append(patchT1)
                                temp.append(sliceT2[i-16:i+16+1, j-16:j+16+1])
                                temp.append(sliceT1c[i-16:i+16+1, j-16:j+16+1])
                                temp.append(sliceFlair[i-16:i+16+1, j-16:j+16+1])
                                    
                                if region==0:
                                    patches0_0.append(temp)
                                elif region==1:
                                    patches0_1.append(temp)
                                else:
                                    patches0_2.append(temp)
                            
                if sliceTruth[i][j]!=0:
                    if region==0:
                        region = 1
                    if sliceTruth[i][j]==1 and not(check(patch1Points, i, j)):
                        ok = 1
                    if sliceTruth[i][j]==2 and not(check(patch2Points, i, j)):
                        ok = 1
                    elif sliceTruth[i][j]==3 and not(check(patch3Points, i, j)):
                        ok = 1
                    elif sliceTruth[i][j]==4 and not(check(patch4Points, i, j)):
                        ok = 1
                    if ok==1:
                        ok=0
                        region = 1
                        truthPatch =  sliceTruth[i-16:i+16+1, j-16:j+16+1]  
                        patchT1 = sliceT1[i-16:i+16+1, j-16:j+16+1]
                        entropyVal = findEntropy(truthPatch, patchT1)
                        if entropyVal >= threshold:
                            t = []    
                            
                            patchT2 = sliceT2[i-16:i+16+1, j-16:j+16+1]
                            patchT1c = sliceT1c[i-16:i+16+1, j-16:j+16+1]
                            patchFlair = sliceFlair[i-16:i+16+1, j-16:j+16+1]
                            temp = []
                            temp.append(patchT1)
                            temp.append(patchT2)
                            temp.append(patchT1c)
                            temp.append(patchFlair)
    
                            if sliceTruth[i][j]==1:
                                t.append(i-3); t.append(i+3); t.append(j-3); t.append(j+3)
                                patch1Points.append(t)
                                patches1.append(temp)
                            elif sliceTruth[i][j]==2:
                                t.append(i-5); t.append(i+5); t.append(j-5); t.append(j+5)
                                patch2Points.append(t)
                                patches2.append(temp)
                            elif sliceTruth[i][j]==3:
                                t.append(i-5); t.append(i+5); t.append(j-5); t.append(j+5)
                                patch3Points.append(t)
                                patches3.append(temp)
                            else: 
                                t.append(i-1); t.append(i+1); t.append(j-1); t.append(j+1)
                                patch4Points.append(t)
                                patches4.append(temp)
                

    for i in range(0, len(patches0_0)):
        allPatches0.append(patches0_0[i])
    for i in range(0, len(patches0_1)):
        allPatches0.append(patches0_1[i])
    for i in range(0, len(patches0_2)):
        allPatches0.append(patches0_2[i])
        
    for i in range(0, len(patches1)):
        allPatches1.append(patches1[i])
        
    for i in range(0, len(patches2)):
        allPatches2.append(patches2[i])
        
    for i in range(0, len(patches3)):
        allPatches3.append(patches3[i])
        
    for i in range(0, len(patches4)):
        allPatches4.append(patches4[i])
                           
    print(len(allPatches0), len(allPatches1), len(allPatches2), len(allPatches3), len(allPatches4))
    
''' Save as a numpy array '''    
print(" Saving the patches")
arr0 = np.zeros(len(allPatches0))
arr1 = np.zeros(len(allPatches1))
arr2 = np.zeros(len(allPatches2))
arr3 = np.zeros(len(allPatches3))
arr4 = np.zeros(len(allPatches4))

arr0 = np.array(allPatches0)
arr1 = np.array(allPatches1)
arr2 = np.array(allPatches2)
arr3 = np.array(allPatches3)
arr4 = np.array(allPatches4)

np.save("patches_part6_3551_lgg_hgg0.npy", arr0)
#np.save("patches_part6_3551_lgg_hgg1.npy", arr1)
#np.save("patches_part6_3551_lgg_hgg2.npy", arr2)
#np.save("patches_part6_3551_lgg_hgg3.npy", arr3)
#np.save("patches_part6_3551_lgg_hgg4.npy", arr4)
