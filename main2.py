import numpy as np 
import matplotlib.pyplot as plt 
import nibabel as nib
import os 
import cv2
from  Preprocess import *
from Segmentation import *


def Train():
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
    
        for i in range(flair.shape[2]):
            # Stack modalities as channels
            img = np.stack([flair[:, :, i], t1[:, :, i], t1ce[:, :, i], t2[:, :, i]], axis=-1)
            mask = seg[:, :, i]
    
            # Resize (optional)
            img = cv2.resize(img, (128, 128))
            mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
    
            X_slices.append(img)
            y_slices.append(mask)
    
        return np.array(X_slices), np.array(y_slices)
    
    
    overall_images,overall_masks,preprocessed_images=[],[],[]
    folder_path="Brats2021"   
    path_list=os.listdir(folder_path)
    
    for i in path_list:
        sub_dir_path=folder_path+"/"+i
        
        
       
        images,masks=load_patient(sub_dir_path+'/'+i)
        for img,msk in zip(images,masks):
            gray = np.mean(img, axis=-1)
            preprocessed_image= preprocess_image(gray)
           
            overall_images.append(img)
            overall_masks.append(msk)
            preprocessed_images.append(preprocessed_image)
                
    

    
    images=preprocessed_images.reshape(preprocessed_images.shape[0],preprocessed_images.shape[1],preprocessed_images.shape[2],1)
    masks=overall_masks.reshape(overall_masks.shape[0],overall_masks.shape[1],overall_masks.shape[2],1)
    
    
    autoencoder = build_residual_autoencoder((128, 128,3))
    opt=SBOA()
    autoencoder.compile(optimizer=opt, loss='mse')
    autoencoder.fit(images, masks, epochs=100,batch_size=32)
