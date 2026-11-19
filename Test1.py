import numpy as np 
import matplotlib.pyplot as plt 
import nibabel as nib
import os 
import cv2
from  Preprocess import *
from Segmentation import *
from Optimization import *
from tkinter.filedialog import askopenfilename
from keras.models import load_model 
import warnings 
warnings.filterwarnings('ignore') 
from absl import logging as absl_logging
import tensorflow as tf
absl_logging.set_verbosity(absl_logging.ERROR)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import random




def Test():
    # Reading the data
    def normalize(image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    
    def load_patient(path):
        flair = normalize(nib.load(path + '_flair.nii').get_fdata())
        t1 = normalize(nib.load(path + '_t1.nii').get_fdata())
        t1ce = normalize(nib.load(path + '_t1ce.nii').get_fdata())
        t2 = normalize(nib.load(path + '_t2.nii').get_fdata())
        seg = nib.load(path + '_seg.nii').get_fdata()
    
        # Slice-wise preparation (example: axial view)
        X_slices = []
        y_slices = []
        actual_images=[]
    
        for i in range(flair.shape[2]):
            # Stack modalities as channels
            img = np.stack([flair[:, :, i], t1[:, :, i], t1ce[:, :, i], t2[:, :, i]], axis=-1)
            mask = seg[:, :, i]
    
            # Resize (optional)
            img = cv2.resize(img, (128, 128))
            gray = np.mean(img, axis=-1)
            preprocessed_image= preprocess_image(gray)
            mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
    
            X_slices.append(preprocessed_image)
            
            y_slices.append(mask)
            actual_images.append(img)
    
        return np.array(actual_images),np.array(X_slices), np.array(y_slices)
    
    images_fold_path="MICCAI_BraTS_2019_Data_Training"
    print("Select The Input ... ")
    image_path = askopenfilename(initialdir=images_fold_path)
    images_file_path=image_path.split("/")[:-1]
    images_file_path.append(images_file_path[-1])
    images_path='/'.join(images_file_path)
    images,pre_images,masks=load_patient(images_path)
    
    reshaped_images=pre_images.reshape(pre_images.shape[0],pre_images.shape[1],pre_images.shape[2],1)
    '''segmentation_model=load_model("Models/Dataset1/AE_model.h5")'''
    
    predicted_images=[]
    cmt=0
    for ind,(i,j,k,l) in enumerate(zip(reshaped_images,masks,images,pre_images)):
        image_uint8 = (k * 255).astype(np.uint8) if k.max() <= 1.0 else k.astype(np.uint8)
        rgb_image = cv2.cvtColor(image_uint8, cv2.COLOR_RGBA2RGB)
        
        cv2.imwrite('Results/Dataset1/Images/'+"image_"+str(ind)+'.png',rgb_image)
        cv2.imwrite('Results/Dataset1/Preprocessed/'+"image_"+str(ind)+'.png',l)
        segmented_image = segment_with_residual_autoencoder(i, segmentation_model, n_clusters=4)
        
        predicted_images.append(segmented_image)
      
    plot(image_path,images,pre_images,masks,np.array(predicted_images))
Test()