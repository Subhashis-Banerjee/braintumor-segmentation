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

''' Global variables '''
#Threshold for PATCHES
threshold = 75
patchLength = 33

''' Function to find the entrophy of a 1D array '''
def entropy(img):
    hist = np.histogram(img)
    ent = 0
    for i in hist[0]:
        if i!=0:
            ent = ent + i * math.log(abs(i), 2)
    return abs(-ent)
        
''' Average entropy for the four MRI sequences '''
def totalEntropy(patch):
    sum = 0
    sum = entropy(patch[0].flatten()) + entropy(patch[1].flatten()) + entropy(patch[2].flatten()) + entropy(patch[3].flatten())
    return sum/4

''' Function to find Mean Square Error '''
def ncc(im1, im2):
    dist_ncc = sum((im1 - np.mean(im1))*(im2 - np.mean(im2))) / ((im1.size)*np.std(im1)*np.std(im2)) 
    return dist_ncc

''' Average Mean Square Erro for the four MRI sequences '''
def totalNcc(patch1, patch2):
    sum = 0
    sum = ncc(patch1[0].flatten(), patch2[0].flatten()) + ncc(patch1[1].flatten(), patch2[1].flatten()) + ncc(patch1[2].flatten(), patch2[2].flatten()) + ncc(patch1[3].flatten(), patch2[3].flatten())
    return sum/4

'''
    Funtion to find the percentage entrophy
'''
def findEntropy(patchTruth, patchT1):
    a=0
    b=0
    c=0
    d=0
    e=0
    for i in range(0, patchLength):
        for j in range(0, patchLength):
            if patchTruth[i][j]==1:
                b=b+1
            elif patchTruth[i][j]==2:
                c=c+1
            elif patchTruth[i][j]==3:
                d=d+1
            elif patchTruth[i][j]==4:
                e=e+1
            elif patchT1[i][j]!=0:
                a=a+1
    return ((a+b+c+d+e)*100)/(patchLength*patchLength)

''' Function to display '''
def display(patch):
    plt.imshow(patch, cmap='gray')
    return

''' Funtion to sort '''
def sort(entropy, patch):
    for i in range(0, patchLength):
        for j in range(0, patchLength):
            if entropy[j]>entropy[j+1]:
                temp = entropy[j]
                entropy[j] = entropy[j+1]
                entropy[j+1] = temp
                
                temp = patch[j]
                patch[j] = patch[j+1]
                patch[j+1] = patch[j]

''' Function to check if a slice has some brain pixels or not '''
def check(slice):
    for i in range(0, 240):
        for j in range(0, 240):
            if slice[i][j]!=0:
                return 1
    return 0
                
                

patches = []
classes = []

''' Opening the files '''
originalFlairPath = ""
originalT1cPath = ""
originalT1Path = ""
originalT2Path = ""
originalTruthPath = ""
patient = 0
max = 0
min = 65536

# Looping through all the training data
# We have divided the dataset in some parts and we are running multiple instance of spyder using different terminals over each part to do the processing parallely
for f1 in os.listdir("BRATS2015_Training/LGG/train/set1"):
    if patient == 1:
        break
    for f2 in os.listdir("BRATS2015_Training/LGG/train/set1/"+f1):
        for f3 in os.listdir("BRATS2015_Training/LGG/train/set1/"+f1+"//"+f2):
            if f3.endswith(".mha"):
                if "Flair" in f3:
                    originalFlairPath = "BRATS2015_Training/LGG/train/set1/"+f1+"//"+f2+"//"+f3
                if "T1c" in f3:
                    originalT1cPath = "BRATS2015_Training/LGG/train/set1/"+f1+"//"+f2+"//"+f3
                if "T1" in f3:
                    originalT1Path = "BRATS2015_Training/LGG/train/set1/"+f1+"//"+f2+"//"+f3
                if "T2" in f3:
                    originalT2Path = "BRATS2015_Training/LGG/train/set1/"+f1+"//"+f2+"//"+f3
                if "OT" in f3:
                    originalTruthPath = "BRATS2015_Training/LGG/train/set1/"+f1+"//"+f2+"//"+f3

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

    append = 1
    not_appended = 0
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
    
    i=0
    j=0
    print("\n Finding patches for patient no. ", patient)
    # Looping through each brain slice
    for z in range(0, 155):
        i = 15
        print("Slice no. ", z)        
        if z == endz:
            region = 2
        sliceT1 = arrayT1[z]
        sliceT2 = arrayT2[z]
        sliceT1c = arrayT1c[z]
        sliceFlair = arrayFlair[z]
        sliceTruth = arrayTruth[z]
        while i<224:
            i = i+1
            j = 15
            while j<224:
                j = j+1
                append = 1
                if sliceT1[i][j]!=0 and sliceTruth[i][j]==0:
                    # We are taking only 250 images from region 0, 4000 from region 1 and 250 from region 2
		    # Refer method 1 from the report to understand this algorithm better
                    if region==0 and len(patches0_0)<250:
                        temp = []
                        temp.append(sliceT1[i-16:i+16+1, j-16:j+16+1])
                        temp.append(sliceT2[i-16:i+16+1, j-16:j+16+1])
                        temp.append(sliceT1c[i-16:i+16+1, j-16:j+16+1])
                        temp.append(sliceFlair[i-16:i+16+1, j-16:j+16+1])
                        i=i+4
                        j=j+4
                        for x in range(0, len(patches0_0)): # checking if the patch is matching to the any of the extracted patch of the same class
                            if totalNcc(patches0_0[x], temp) > 0.5:
                                append = 0
                                break
                        if append!=0:
                            patches0_0.append(temp)
                        #else:   
                    elif region==1 and len(patches0_1)<4000:
                        temp = []
                        temp.append(sliceT1[i-16:i+16+1, j-16:j+16+1])
                        temp.append(sliceT2[i-16:i+16+1, j-16:j+16+1])
                        temp.append(sliceT1c[i-16:i+16+1, j-16:j+16+1])
                        temp.append(sliceFlair[i-16:i+16+1, j-16:j+16+1])
                        j=j+16
                        for x in range(0, len(patches0_1)): # checking if the patch is matching to the any of the extracted patch of the same class
                            if totalNcc(patches0_1[x], temp) > 0.5:
                                append = 0
                                break
                        if append!=0:
                            patches0_1.append(temp)
                    elif region==2 and len(patches0_2)<250:
                        temp = []
                        temp.append(sliceT1[i-16:i+16+1, j-16:j+16+1])
                        temp.append(sliceT2[i-16:i+16+1, j-16:j+16+1])
                        temp.append(sliceT1c[i-16:i+16+1, j-16:j+16+1])
                        temp.append(sliceFlair[i-16:i+16+1, j-16:j+16+1])
                        i=i+4
                        j=j+4
                        for x in range(0, len(patches0_2)): # checking if the patch is matching to the any of the extracted patch of the same class
                            if totalNcc(patches0_2[x], temp) > 0.5:
                                append = 0
                                break 
                        if append!=0:
                            patches0_2.append(temp)
                ##do append part
                        
                if sliceTruth[i][j]!=0:
                    region = 1
                    truthPatch =  sliceTruth[i-16:i+16+1, j-16:j+16+1]  
                    patchT1 = sliceT1[i-16:i+16+1, j-16:j+16+1]
                    patchT2 = sliceT2[i-16:i+16+1, j-16:j+16+1]
                    patchT1c = sliceT1c[i-16:i+16+1, j-16:j+16+1]
                    patchFlair = sliceFlair[i-16:i+16+1, j-16:j+16+1]
                    j = j+4
                    entrophy = findEntropy(truthPatch, patchT1)
                    if entrophy > threshold:
                        temp = []
                        temp.append(patchT1)
                        temp.append(patchT2)
                        temp.append(patchT1c)
                        temp.append(patchFlair)
                        if sliceTruth[i][j]==1:    
                            for x in range(0, len(patches1)): # checking if the patch is matching to the any of the extracted patch of the same class
                                if totalNcc(patches1[x], temp) > 0.90:
                                    not_appended = not_appended + 1
                                    append = 0
                                    break
                        elif sliceTruth[i][j]==2:    
                            for x in range(0, len(patches2)): # checking if the patch is matching to the any of the extracted patch of the same class
                                if totalNcc(patches2[x], temp) > 0.90:
                                    append = 0
                                    break
                        elif sliceTruth[i][j]==0:    
                            for x in range(0, len(patches3)): # checking if the patch is matching to the any of the extracted patch of the same class
                                if totalNcc(patches3[x], temp) > 0.90:
                                    append = 0
                                    break
                        else:    
                            for x in range(0, len(patches4)): # checking if the patch is matching to the any of the extracted patch of the same class
                                if totalNcc(patches4[x], temp) > 0.90:
                                    append = 0
                                    break        
                        
                        if append!=0:
                            if sliceTruth[i][j]==1:
                                patches1.append(temp)
                            elif sliceTruth[i][j]==2:
                                patches2.append(temp)
                            elif sliceTruth[i][j]==3:
                                patches3.append(temp)
                            else: 
                                patches4.append(temp)
    print(not_appended)
    patches0 = patches0_0 + patches0_1 + patches0_2
    list = [len(patches0), len(patches1), len(patches2), len(patches3), len(patches4)]
    entropy0 = []
    entropy1 = []
    entropy2 = []
    entropy3 = []
    entropy4 = []
    
    for i in range(0, len(patches0)): 
        entropy0.append(totalEntropy(patches0[i]))
    for i in range(0, len(patches1)):
        entropy1.append(totalEntropy(patches1[i]))
    for i in range(0, len(patches2)):
        entropy2.append(totalEntropy(patches2[i]))
    for i in range(0, len(patches3)):
        entropy3.append(totalEntropy(patches3[i]))
    for i in range(0, len(patches4)):
        entropy4.append(totalEntropy(patches4[i]))        

    min = list[0]  
    ind = 0
    for i in range(0, 5):
        if min==0 or min>list[i]:
            min = list[i]
            ind = i
    print(list)
    print(list[ind], ind)

    # sorting the patches based on entropy
    sort(entropy0, patches0)
    sort(entropy1, patches1)
    sort(entropy2, patches2)
    sort(entropy3, patches3)
    sort(entropy4, patches4)

	
    # We are taking only a maximum of '1125' patches from each class 1, 2, 3, 4 and '4500' patches for class 0, this we are doing as the no. of patches extracted from this method is very high
    if min>1125:
        min = 1125
    
    if(len(patches0)>=min):
        t = len(patches0)
        if t>4500:
            t = 4500
        for i in range(0, t):
            patches.append(patches0[i])
            classes.append(0)
    if(len(patches1)>=min):
        for i in range(0, min):
            patches.append(patches1[i])
            classes.append(1)
    if(len(patches2)>=min):
        for i in range(0, min):
            patches.append(patches2[i])
            classes.append(2)
    if(len(patches3)>=min):        
        for i in range(0, min):
            patches.append(patches3[i])
            classes.append(3)
    if(len(patches4)>=min):
        for i in range(0, min):
            patches.append(patches4[i])
            classes.append(4)
   
#    ''' Delete based on similarity '''
#    for i in range(0, len(patches)):
#        for j in range(0, len(patches)):
#            if totalMse(patches[i], patches[j]) < 25:
#                np.delete(patches, j)
#                
    

#''' Save as a numpy array '''
arr = np.zeros(len(patches))
arr_class = np.zeros(len(classes))

arr = np.array(patches)
arr_class = np.array(classes)

np.save("patch.npy", arr)
np.save("class.npy", arr_class)
