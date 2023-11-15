"""
Helper functions from https://github.com/zhangjun001/ICNet.

Some functions has been modified.
"""

import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import os
from os import listdir
from os.path import join
import itertools
import pandas as pd

def load_train_pair(data_path, filename1, filename2):
    # Load images and labels
    nim1 = nib.load(os.path.join(data_path, 'hyperdata', filename1, 'aligned_norm.nii.gz'))
    image1 = nim1.get_data()
    image1 = np.array(image1, dtype='float32')

    nim2 = nib.load(os.path.join(data_path, 'hyperdata', filename2, 'aligned_norm.nii.gz'))
    image2 = nim2.get_data()
    image2 = np.array(image2, dtype='float32')
    
    nim5 = nib.load(os.path.join(data_path, 'hyperdata', filename1, 'aligned_seg35.nii.gz'))
    image5 = nim5.get_data()
    image5 = np.array(image5, dtype='float32')
    nim6 = nib.load(os.path.join(data_path, 'hyperdata', filename2, 'aligned_seg35.nii.gz'))
    image6 = nim6.get_data()
    image6 = np.array(image6, dtype='float32') 
    
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    image5 = np.reshape(image5, (1,) + image5.shape)
    image6 = np.reshape(image6, (1,) + image6.shape)
    return image1, image2, image5, image6

class TrainDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path, img_file=None, trainingset = 1):
        'Initialization'
        super(TrainDataset, self).__init__()
        self.data_path = data_path
        self.names = np.loadtxt(os.path.join(self.data_path, img_file),dtype='str')
        if trainingset == 1:
            self.filename = list(zip(self.names[:-1], self.names[1:]))
            assert len(self.filename) == 200, "# of images != 200."
        elif trainingset == 2:
            self.filename = list(zip(self.names[1:], self.names[:-1]))
            assert len(self.filename) == 200, "# of images != 200."
        elif trainingset == 3:
            self.zip_filename_1 = list(zip(self.names[:-1], self.names[1:]))
            self.zip_filename_2 = list(zip(self.names[1:], self.names[:-1]))
            self.filename = self.zip_filename_1 + self.zip_filename_2
            assert len(self.filename) == 400, "# of images != 400."
        elif trainingset == 4:
            self.filename = list(itertools.permutations(self.names, 2))
            print(len(self.names))
            print(len(self.filename))
            assert len(self.filename) == 154842, "# of images != 154842."
        
        else:
             assert 0==1, print('TrainDataset Invalid!')
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #print('Total # of Images:   ', len(self.filename))
        # 154842
        #print(self.filename)
        mov_img, fix_img, mov_lab, fix_lab = load_train_pair(self.data_path, self.filename[index][0], self.filename[index][1])
        return  mov_img, fix_img, mov_lab, fix_lab

class ValidationDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path, transform=None):
        'Initialization'
        super(ValidationDataset, self).__init__()
        self.data_path = data_path
        self.filename = pd.read_csv(os.path.join(data_path,'pairs_val.csv')).values
        #print(self.filename)
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        assert len(self.filename)==19, "Oh no! # of images != 19."
        # Select sample
        img_A, img_B, label_A, label_B = load_validation_pair(self.data_path, self.filename[index][0], self.filename[index][1])
        return self.filename[index][0],self.filename[index][1], img_A, img_B, label_A, label_B
def load_validation_pair(data_path, fixed, moving):
    # Load images and labels
    nim1 = nib.load(os.path.join(data_path, 'hyperdata',  'OASIS_OAS1_0'+str(fixed)+'_MR1', 'aligned_norm.nii.gz'))
    image1 = nim1.get_data()[:, :, :]
    image1 = np.array(image1, dtype='float32')

    nim2 = nib.load(os.path.join(data_path, 'hyperdata',  'OASIS_OAS1_0'+str(moving)+'_MR1', 'aligned_norm.nii.gz'))
    image2 = nim2.get_data()[:, :, :]
    image2 = np.array(image2, dtype='float32')
    
    nim3 = nib.load(os.path.join(data_path, 'hyperdata',  'OASIS_OAS1_0'+str(fixed)+'_MR1', 'aligned_seg35.nii.gz'))
    image3 = nim3.get_data()[:, :, :]
    image3 = np.array(image3, dtype='float32')

    nim4 = nib.load(os.path.join(data_path, 'hyperdata',  'OASIS_OAS1_0'+str(moving)+'_MR1', 'aligned_seg35.nii.gz'))
    image4 = nim4.get_data()[:, :, :]
    image4 = np.array(image4, dtype='float32')
    #preprocessing
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    image3 = np.reshape(image3, (1,) + image3.shape)
    image4 = np.reshape(image4, (1,) + image4.shape)
    return image1, image2,image3,image4
