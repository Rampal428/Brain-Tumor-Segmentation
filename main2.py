import numpy as np 
import matplotlib.pyplot as plt 
import nibabel as nib
import os 
import cv2
from  Preprocess import *
from Segmentation import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from Optimization import *
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
      
    
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        images,
        masks,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    # 10% of total data = 1/3 of remaining 30%
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp,
        Y_temp,
        test_size=2/3,
        random_state=42,
        shuffle=True
    )

    #Segmenting
    # Residual Autoencoder-based Enhanced K-Means Clustering 
    # Secretary Bird Optimization (SBO) algorithm
    
    optimizer = SBOA(
    pop_size=30,
    max_iter=50
    )
    
    best_solution, best_fitness = optimizer.optimize(
        X_train, Y_train,
        X_val, Y_val
    )
    BEST_FILTERS = int(
        np.clip(round(best_solution[0] / 16) * 16, 16, 64)
    )
    
    BEST_BATCH_SIZE = int(
        np.clip(round(best_solution[1] / 8) * 8, 8, 64)
    )
    
    BEST_LR = 10 ** np.clip(
        best_solution[2], -4, -2
    )

    
    autoencoder = residual_autoencoder(
        (128,128,3),
        BEST_FILTERS
    )
    
    autoencoder.compile(
        optimizer=Adam(learning_rate=BEST_LR),
        loss='mse',
        metrics=['mae']
    )

    autoencoder.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=100,
        batch_size=BEST_BATCH_SIZE,
        verbose=1
    )
        
