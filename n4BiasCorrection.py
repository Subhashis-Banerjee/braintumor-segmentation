import SimpleITK as sitk
import pylab as plt
import numpy as np
import os

# We have divided the dataset in some parts and we are running multiple instance of spyder using different terminals over each part to do the processing parallely.
# The input to this code is the 'BRATS2015' directory and the output is 'BRATS2015_Corrected' directory.

# Creating directory
directory = "BRATS2015_Corrected/LGG/part4" 
if not os.path.exists(directory):
    os.makedirs(directory)

# Variable use to display how many patients have been completed
patient=0

for f1 in os.listdir("BRATS2015/LGG/part4"):
    # Creating directory 
    directory = "BRATS2015_Corrected/LGG/part4/" + f1
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for f2 in os.listdir("BRATS2015/LGG/part4/"+f1):
	# Creating directory
        directory = "BRATS2015_Corrected/LGG/part4/" +f1+"//"+f2
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        for f3 in os.listdir("BRATS2015/LGG/part4/"+f1+"//"+f2):
            if f3.endswith(".mha"):
                print(" -> Correcting patient no = ", patient, ", file = ", f3)
                t = 0
                write_path = "BRATS2015_Corrected/LGG/part4/"+f1+"//"+f2+"//"+f3
                read_path = "BRATS2015/LGG/part4/"+f1+"//"+f2+"//"+f3
                if "OT" in f3:
                    t = 1
		
		# Simply copying the ground truth values file
                if t==1:
                    inputImage = sitk.ReadImage(read_path)
                    sitk.WriteImage(inputImage, write_path)                     
		
		# Applying N4ITK bias correction to the MRI files.                 
		else:
                    inputImage = sitk.ReadImage(read_path)
                    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
                    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
                    corrector = sitk.N4BiasFieldCorrectionImageFilter();
                    output = corrector.Execute(inputImage, maskImage)    
                    sitk.WriteImage(output, write_path)
    patient = patient + 1         

