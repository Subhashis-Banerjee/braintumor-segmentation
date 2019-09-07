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
import sys

# Combine by balancing classes. Read report for why we have to balance classes.

#a0 = np.load("patch_temp_hgg.npy")
#c0 = np.load("class_temp_hgg.npy")
#
#a1 = np.load("patch_temp_lgg.npy")
#c1 = np.load("class_temp_lgg.npy")

d0 = np.load("Patches_nyul/patches_part1_withtumor25_hgg0.npy")
d1 = np.load("Patches_nyul/patches_part1_hgg1.npy")
d2 = np.load("Patches_nyul/patches_part1_hgg2.npy")
d3 = np.load("Patches_nyul/patches_part1_hgg3.npy")
d4 = np.load("Patches_nyul/patches_part1_hgg4.npy")

e0 = np.load("Patches_nyul/patches_part2_withtumor25_hgg0.npy")
e1 = np.load("Patches_nyul/patches_part2_hgg1.npy")
e2 = np.load("Patches_nyul/patches_part2_hgg2.npy")
e3 = np.load("Patches_nyul/patches_part2_hgg3.npy")
e4 = np.load("Patches_nyul/patches_part2_hgg4.npy")

f0 = np.load("Patches_nyul/patches_part3_withtumor25_hgg0.npy")
f1 = np.load("Patches_nyul/patches_part3_hgg1.npy")
f2 = np.load("Patches_nyul/patches_part3_hgg2.npy")
f3 = np.load("Patches_nyul/patches_part3_hgg3.npy")
f4 = np.load("Patches_nyul/patches_part3_hgg4.npy")

g0 = np.load("Patches_nyul/patches_part4_withtumor25_hgg0.npy")
g1 = np.load("Patches_nyul/patches_part4_hgg1.npy")
g2 = np.load("Patches_nyul/patches_part4_hgg2.npy")
g3 = np.load("Patches_nyul/patches_part4_hgg3.npy")
g4 = np.load("Patches_nyul/patches_part4_hgg4.npy")

h0 = np.load("Patches_nyul/patches_part5_withtumor25_hgg0.npy")
h1 = np.load("Patches_nyul/patches_part5_hgg1.npy")
h2 = np.load("Patches_nyul/patches_part5_hgg2.npy")
h3 = np.load("Patches_nyul/patches_part5_hgg3.npy")
h4 = np.load("Patches_nyul/patches_part5_hgg4.npy")
#
#i0 = np.load("Patches_nyul/patches_part6_hgg0.npy")
i1 = np.load("Patches_nyul/patches_part6_hgg1.npy")
i2 = np.load("Patches_nyul/patches_part6_hgg2.npy")
i3 = np.load("Patches_nyul/patches_part6_hgg3.npy")
i4 = np.load("Patches_nyul/patches_part6_hgg4.npy")
#
#j0 = np.load("Patches_nyul/patches_part7_withtumor25_hgg0.npy")
j1 = np.load("Patches_nyul/patches_part7_hgg1.npy")
j2 = np.load("Patches_nyul/patches_part7_hgg2.npy")
j3 = np.load("Patches_nyul/patches_part7_hgg3.npy")
j4 = np.load("Patches_nyul/patches_part7_hgg4.npy")

#k0 = np.load("Patches_nyul/patches_part8_withtumor25_hgg0.npy")
k1 = np.load("Patches_nyul/patches_part8_hgg1.npy")
k2 = np.load("Patches_nyul/patches_part8_hgg2.npy")
k3 = np.load("Patches_nyul/patches_part8_hgg3.npy")
k4 = np.load("Patches_nyul/patches_part8_hgg4.npy")

#l0 = np.load("Patches_nyul/patches_part9_withtumor25_hgg0.npy")
l1 = np.load("Patches_nyul/patches_part9_hgg1.npy")
l2 = np.load("Patches_nyul/patches_part9_hgg2.npy")
l3 = np.load("Patches_nyul/patches_part9_hgg3.npy")
l4 = np.load("Patches_nyul/patches_part9_hgg4.npy")

#m0 = np.load("Patches_nyul/patches_part9_withtumor25_hgg0.npy")
m1 = np.load("Patches_nyul/patches_part10_hgg1.npy")
m2 = np.load("Patches_nyul/patches_part10_hgg2.npy")
m3 = np.load("Patches_nyul/patches_part10_hgg3.npy")
m4 = np.load("Patches_nyul/patches_part10_hgg4.npy")
#
#
z0 = np.concatenate((d0, e0, f0, g0, h0), axis=0)
z1 = np.concatenate((d1, e1, f1, g1, h1, i1, j1, k1, l1, m1), axis=0)
z2 = np.concatenate((d2, e2, f2, g2, h2, i2, j2, k2, l2, m2), axis=0)
z3 = np.concatenate((d3, e3, f3, g3, h3, i3, j3, k3, l3, m3), axis=0)
z4 = np.concatenate((d4, e4, f4, g4, h4, i4, j4, k4, l4, m4), axis=0)
#

l0 = z0.shape[0]
l1 = z1.shape[0]
l2 = z2.shape[0]
l3 = z3.shape[0]
l4 = z4.shape[0]
#
print(l0, l1, l2, l3, l4)

classes = []
patches = []

list1_shuf = []
index_shuf = range(l0)
random.shuffle(index_shuf)
k = 0
for i in index_shuf:
    k = k+1
    list1_shuf.append(z0[i])
    if k==113187:
        break

list2_shuf = []
index_shuf = range(l1)
random.shuffle(index_shuf)
k = 0
for i in index_shuf:
    k = k+1
    list2_shuf.append(z1[i])
    if k==42445:
        break
    
list3_shuf = []
index_shuf = range(l2)
random.shuffle(index_shuf)
k = 0
for i in index_shuf:
    k = k+1
    list3_shuf.append(z2[i])
    if k==42445:
        break
    
list4_shuf = []
index_shuf = range(l4)
random.shuffle(index_shuf)
k = 0
for i in index_shuf:
    k = k+1
    list4_shuf.append(z4[i])
    if k==42445:
        break
#
#list5_shuf = []
#index_shuf = range(l4)
#random.shuffle(index_shuf)
#k = 0
#for i in index_shuf:
#    k = k+1
#    list5_shuf.append(z4[i])
#    if k==19598:
#        break
#
for t in list1_shuf:
    patches.append(t)
    classes.append(0)

for t in list2_shuf:
    patches.append(t)
    classes.append(1)

for t in list3_shuf:
    patches.append(t)
    classes.append(2)
    
for i in range(l3):
    patches.append(z3[i])
    classes.append(3)

for t in list4_shuf:
    patches.append(t)
    classes.append(4)
#
arr = np.zeros(len(patches))
arr_class = np.zeros(len(classes))
##
arr = np.array(patches)
arr_class = np.array(classes)

#x = np.concatenate((a1, a2, arr), axis=0)
#y = np.concatenate((c1, c2, arr_class), axis=0)
#
np.save("patch_combine_hgg.npy", arr)
np.save("class_combine_hgg.npy", arr_class)

#x, y = shuffle(t, c)




