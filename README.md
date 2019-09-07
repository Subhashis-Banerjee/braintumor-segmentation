# braintumor-segmentation
Convolutional Neural Network based brain tumor segmentation from multimodal MRI
Documentation

This documentation is prepared so that the work done on Brain Tumor Segmentation could be further extended. The python files used and what are the python file doing is explained in this documentation. The code has been commented so that it is easy to understand it.

All this files discussed here are used for the project. I have also included some extra files which were also created during the project, although not useful but can be used to learn some syntax.

1. n4BiasCorrection.py
The pupose of this file is to apply N4ITK bias correction over all the data (training and testing).

2. learn_nyul.py
The purpose of this file is to learn the landmarks from the data. We have taken 50 random patients to learn the landmarks in case of HGG and 15 patients in case of LGG.

3. transform_nyul.py
The purpose of this file is to transform the image according to the landmarks learned from the file “learn_nyul.py”

4. patch.py
This file is for the patch extraction method 1. 

5. patch_new.py
This file is for the patch extraction method 2.

6. patches_hgg.py
This file is for the patch extraction method 3. 

7. hgg.py
The CNN model for HGG patients is defined in this file and is trained using real time data augmentation.

8. post_processing.py
The purpose of this file is to apply post processing i.e. removing clusters of size less than 10,000 in case of HGG and removing clusters of size less than 3,000 in case of LGG.

9. segment.py
The purpose of this file is to generate a segmented file for a HGG patient.

10. transfer_learning.py
The purpose of this file is to remove the last 5 layers of the HGG model and get a vector of size 128x16x16 as the output and to store the vector as a numpy array.

11. lgg_model_transfer.py
The purpose of this file is to define and train the model defined for the LGG patients using the vectors obtained from “transfer_learning.py”.

12. transfer_test.py
The purpose of this file is to generate segmented files for the LGG patients.



13. segmentation_scores.py
The purpose of this file is to generate the final dice score between the ground truth and the segmented file generated by the model.

14. test.py
The purpose of this file was to help with the trick we used during patch extraction. The trick is explained below.

Trick: The time required for patch extraction is huge. The GPU server has 24 CPU cores and since patch extraction code need only CPU, so I used to divide the datasets into 10 parts and run 10 different instances of spyder using the terminal (command: spyder –new-instance). In this 10 different instances of spyder I used to extract patches for each of the 10 parts of the dataset and then finally I used to combine the patches using the code in 'test.py'. This simple trick helped me to extract patches 10x faster.  


Work left
1. We can test different CNN architecture to see if the accuracy increases.
2. Try removing more layers from the HGG model to see if the segmentation results for the LGG patients can be further increased.
3. Static data augmentation can be applied to LGG patches and then vectors can be generated to train the LGG model.
4. Improvements in the patch extraction method 3 that has been proposed by us.
