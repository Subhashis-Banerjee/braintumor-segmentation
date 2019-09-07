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
threshold = 75
thresholdBackground = 50
patchLength = 33
maxExtraPatchesLength = 5000

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

''' Function to find non-zero background pixels in a patch '''
def entropyBackground(patchT1):
    count = 0
    for i in range(0, patchLength):
        for j in range(0, patchLength):
            if patchT1[i][j]!=0:
                count = count+1
    return (count*100)/(patchLength*patchLength)
    
''' Funtion to find the percentage entrophy '''
def findEntropy(patchTruth, patchT1, center):
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    bonus = 0
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
    if center==1 or center==4:
        bonus = 25
    if (bonus + ((a+b+c+d+e)*100)/(patchLength*patchLength)) >= threshold:
        return (bonus + ((a+b+c+d+e)*100)/(patchLength*patchLength))
    
    # Boundary patches
    if center==1 or center==4:
        if b>5 and c>5 and d>5 and e>5:
            return 100
    elif b>110 and c>110 and d>110 and e>110:
            return 100
    return 0
        
''' Function to display '''
def display(patch):
    plt.imshow(patch, cmap='gray')
    return

''' Funtion to sort '''
def sort(entropy, patch):
    for i in range(0, len(patch)):
        for j in range(0, len(patch)-1):
            if entropy[j]>entropy[j+1]:
                temp1 = entropy[j]
                entropy[j] = entropy[j+1]
                entropy[j+1] = temp1
                
                temp2 = patch[j]
                patch[j] = patch[j+1]
                patch[j+1] = temp2                          

patches = []
classes = []

extraPatches1 = []
extraPatches4 = []

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
max = 0
min = 0

# Looping through all the training data
# We have divided the dataset in some parts and we are running multiple instance of spyder using different terminals over each part to do the processing parallely
for f1 in os.listdir("BRATS2015_Training/LGG/set2_new"):
#    if patient == 1:
#        break
    for f2 in os.listdir("BRATS2015_Training/LGG/set2_new/"+f1):
        for f3 in os.listdir("BRATS2015_Training/LGG/set2_new/"+f1+"//"+f2):
            if f3.endswith(".mha"):
                if "Flair" in f3:
                    originalFlairPath = "BRATS2015_Training/LGG/set2_new/"+f1+"//"+f2+"//"+f3
                if "T1c" in f3:
                    originalT1cPath = "BRATS2015_Training/LGG/set2_new/"+f1+"//"+f2+"//"+f3
                if "T1" in f3:
                    originalT1Path = "BRATS2015_Training/LGG/set2_new/"+f1+"//"+f2+"//"+f3
                if "T2" in f3:
                    originalT2Path = "BRATS2015_Training/LGG/set2_new/"+f1+"//"+f2+"//"+f3
                if "OT" in f3:
                    originalTruthPath = "BRATS2015_Training/LGG/set2_new/"+f1+"//"+f2+"//"+f3

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
    not_appended0 = 0
    not_appended1 = 0
    not_appended2 = 0
    not_appended3 = 0
    not_appended4 = 0
    patient = patient + 1
    patches0_0 = []
    patches0_1 = []
    patches0_2 = []
    patches0 = []
    patches1 = []
    patches2 = []
    patches3 = []
    patches4 = []
    boundaryPatches = []
    boundaryClasses = []
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
    count = 0
    t = 0
    print("\n Finding patches for patient no. ", patient)
    bpfs = 0
    # Looping through each brain slice
    for z in range(0, 155):
        bpfs = 0 # to keep track of patches of class level 0 extracted from region 1, we arenot extracting more than 100 patches of level 0 from a particular brain slice of region 1
        count = 0
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
        while i<224:
            i = i+1
            j = 15
            while j<224:
                j = j+1
                append = 1
                if sliceT1[i][j]!=0 and sliceTruth[i][j]==0:
		    # We are taking only 250 images from region 0, 4000 from region 1 and 250 from region 2
		    # Refer method 2 from the report to understand this algorithm better
                    if (region==0 and len(patches0_0)<250) or (region==1 and bpfs<100 and len(patches0_1)<4000) or (region==2 and len(patches0_2)<250):
                        patchT1 = sliceT1[i-16:i+16+1, j-16:j+16+1]
                        if entropyBackground(patchT1) > thresholdBackground:
                            temp = []
                            temp.append(patchT1)
                            temp.append(sliceT2[i-16:i+16+1, j-16:j+16+1])
                            temp.append(sliceT1c[i-16:i+16+1, j-16:j+16+1])
                            temp.append(sliceFlair[i-16:i+16+1, j-16:j+16+1])
                            
                            if region==0:
                                if len(patches0_0)!=0:
                                    if totalNcc(patches0_0[len(patches0_0)-1], temp) > 0.5: # checking if the patch is matching to the last extracted patch of the same class
                                        append = 0  
                                if append!=0:
                                    patches0_0.append(temp)
                            elif region==1:
                                if len(patches0_1)!=0:
                                    if totalNcc(patches0_1[len(patches0_1)-1], temp) > 0.5: # checking if the patch is matching to the last extracted patch of the same class
                                        not_appended0 = not_appended0 + 1
                                        append = 0
                                if append!=0:
                                    bpfs = bpfs+1
                                    patches0_1.append(temp)
                            else:
                                if len(patches0_2)!=0:
                                    if totalNcc(patches0_2[len(patches0_2)-1], temp) > 0.5: # checking if the patch is matching to the last extracted patch of the same class
                                        append = 0
                                if append!=0:
                                    patches0_2.append(temp)
                        
                if sliceTruth[i][j]!=0:
                    region = 1
                    truthPatch =  sliceTruth[i-16:i+16+1, j-16:j+16+1]  
                    patchT1 = sliceT1[i-16:i+16+1, j-16:j+16+1]
                    entropyVal = findEntropy(truthPatch, patchT1, sliceTruth[i][j])
                    if entropyVal >= threshold:    
                        patchT2 = sliceT2[i-16:i+16+1, j-16:j+16+1]
                        patchT1c = sliceT1c[i-16:i+16+1, j-16:j+16+1]
                        patchFlair = sliceFlair[i-16:i+16+1, j-16:j+16+1]
                        temp = []
                        temp.append(patchT1)
                        temp.append(patchT2)
                        temp.append(patchT1c)
                        temp.append(patchFlair)
       
			# boundary patches can be included by uncommenting this code
                        #if entropyVal==100:
                        #    boundaryPatches.append(temp)
                        #    boundaryClasses.append(sliceTruth[i][j])
                        #    append = 0
                        
                        elif sliceTruth[i][j]==1:  
                            if len(patches1)!=0:
                                if totalNcc(patches1[len(patches1)-1], temp) > 0.90: # checking if the patch is matching to the last extracted patch of the same class
                                    not_appended1 = not_appended1 + 1
                                    append = 0
                                    
                        elif sliceTruth[i][j]==2:  
                            if len(patches2)!=0:
                                if totalNcc(patches2[len(patches2)-1], temp) > 0.60: # checking if the patch is matching to the last extracted patch of the same class
                                    not_appended2 = not_appended2 + 1
                                    append = 0
                                
                        elif sliceTruth[i][j]==3:  
                            if len(patches3)!=0:
                                if totalNcc(patches3[len(patches3)-1], temp) > 0.60: # checking if the patch is matching to the last extracted patch of the same class
                                    not_appended3 = not_appended3 + 1
                                    append = 0
                                
                        else:
                            if len(patches4)!=0:
                                if totalNcc(patches4[len(patches4)-1], temp) > 0.90: # checking if the patch is matching to the last extracted patch of the same class
                                    not_appended4 = not_appended4 + 1
                                    append = 0        
                        
                        if append!=0:
                            if sliceTruth[i][j]==1:
                                patches1.append(temp)
                            elif sliceTruth[i][j]==2:
                                patches2.append(temp)
                            elif sliceTruth[i][j]==3:
                                patches3.append(temp)
                            else: 
                                patches4.append(temp)                    
    
    unique, counts = np.unique(boundaryClasses, return_counts=True)
    print(unique, counts)   
                                    
    #print(not_appended0, not_appended1, not_appended2, not_appended3, not_appended4)
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

    # We are taking only a maximum of '1125' patches from each class 1, 2, 3, 4 and '4500' patches for class 0, this we are doing as the no. of patches extracted from this method is very high
    min = 1125 

    print(list)
    
    # sorting the patches based on entropy
    sort(entropy0, patches0)
    sort(entropy1, patches1)
    sort(entropy2, patches2)
    sort(entropy3, patches3)
    sort(entropy4, patches4)
     

    # Append boundary patches (this I tried to include boundary patches of tumor)
#    var2 = 0
#    var3 = 0
#    for i in range(0, len(boundaryClasses)):
#        if boundaryClasses[i]==1:
#            totalClass1 = totalClass1+1
#            patches.append(boundaryPatches[i])
#            classes.append(1)
#        elif boundaryClasses[i]==2 and var2<50:
#            var2 = var2+1
#            patches.append(boundaryPatches[i])
#            classes.append(2)
#        elif boundaryClasses[i]==3 and var3<50:
#            var3 = var3+1
#            patches.append(boundaryPatches[i])
#            classes.append(3)
#        elif boundaryClasses[i]==4:
#            totalClass4 = totalClass4+1
#            patches.append(boundaryPatches[i])
#            classes.append(4)
#    totalClass2 = totalClass2 + var2
#    totalClass3 = totalClass3 + var3        


    # We are keeping track of the no. of patient and the total patches that have been extracted for a perticular class.
    # By this, if we are getting very less patch for a perticular class i.e. less than our desired 1125, then we can recover that from another patient
    # We are also storing some extra patches for the underrepresented classes, so that we can recover if the patches for such classes are less      
        
    # Append patch with center truth value 0
    t = len(patches0)
    for i in range(0, t):
        patches.append(patches0[i])
        classes.append(0)
    
    # Append patch with center truth value 1
    count = 0
    if len(patches1)>=min:
        reqd = patient*min - totalClass1
        if len(patches1)>=reqd:
            count = reqd
            # Store extra patches
            if len(extraPatches1) < maxExtraPatchesLength:
                extra = len(patches1) - reqd
                for i in range(0, extra):
                    extraPatches1.append(patches1[count+i])
        else:
            count = len(patches1)
    else:
        count = len(patches1)
    totalClass1 = totalClass1 + count
    for i in range(0, count):
        patches.append(patches1[i])
        classes.append(1)
    
    # Append patch with center truth value 2        
    count = 0
    if len(patches2)>=min:
        reqd = patient*min - totalClass2
        if len(patches2)>=reqd:
            count = reqd
        else:
            count = len(patches2)
    else:
        count = len(patches2)
    totalClass2 = totalClass2 + count
    for i in range(0, count):
        patches.append(patches2[i])
        classes.append(2)
    
    # Append patch with center truth value 3       
    count = 0
    if len(patches3)>=min:
        reqd = patient*min - totalClass3
        if len(patches3)>=reqd:
            count = reqd
        else:
            count = len(patches3)
    else:
        count = len(patches3)
    totalClass3 = totalClass3 + count
    for i in range(0, count):
        patches.append(patches3[i])
        classes.append(3)
    
    # Append patch with center truth value 4  
    count = 0
    if len(patches4)>=min:
        reqd = patient*min - totalClass4
        if len(patches4)>=reqd:
            count = reqd
            # Store extra patches
            if len(extraPatches4) < maxExtraPatchesLength:
                extra = len(patches4) - reqd
                for i in range(0, extra):
                    extraPatches4.append(patches4[count+i])
        else:
            count = len(patches4)
    else:
        count = len(patches4)
    totalClass4 = totalClass4 + count
    for i in range(0, count):
        patches.append(patches4[i])
        classes.append(4)     
                 

# Balance for class 1 and 4, since patches for necrosis and enhanching tumor are less for LGG patients
if classes.count(1)<(min*patient):
    extra = min*patient - classes.count(1)
    if len(extraPatches1)>=extra:
        for i in range(0, extra):
            patches.append(extraPatches1[i])
            classes.append(1)
    else:
        for i in range(0, len(extraPatches1)):
            patches.append(extraPatches1[i])
            classes.append(1)
if classes.count(4)<(min*patient):
    extra = min*patient - classes.count(4)
    if len(extraPatches4)>=extra:
        for i in range(0, extra):
            patches.append(extraPatches4[i])
            classes.append(4)
    else:
        for i in range(0, len(extraPatches4)):
            patches.append(extraPatches4[i])
            classes.append(4)

''' Save as a numpy array '''
# Saving
print(" Saving the patches")
arr = np.zeros(len(patches))
arr_class = np.zeros(len(classes))

arr = np.array(patches)
arr_class = np.array(classes)

np.save("patch_set2_new.npy", arr)
np.save("class_set2_new.npy", arr_class)
