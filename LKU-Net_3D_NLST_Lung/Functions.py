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
    frame = np.random.choice(['imagesTr_400.0_-1000.0', 'imagesTr_450.0_-1000.0', 'imagesTr_500.0_-1000.0','imagesTr_400.0_-900.0', 'imagesTr_450.0_-900.0', 'imagesTr_500.0_-900.0'])
    nim1 = nib.load(os.path.join(data_path, 'NLST2023', frame, filename1))
    image1 = nim1.get_data()
    image1 = np.array(image1, dtype='float32')
    # print(image1.shape)
    nim2 = nib.load(os.path.join(data_path, 'NLST2023', frame, filename2))
    image2 = nim2.get_data()
    image2 = np.array(image2, dtype='float32')
    # print(np.max(image1-image2))
    nim3 = nib.load(os.path.join(data_path, 'NLST2023', 'masksTr', filename1))
    image3 = nim3.get_data()
    image3 = np.array(image3, dtype='float32')
    nim4 = nib.load(os.path.join(data_path, 'NLST2023', 'masksTr', filename2))
    image4 = nim4.get_data()
    image4 = np.array(image4, dtype='float32')
    
    nim5 = nib.load(os.path.join(data_path, 'NLST2023', 'keypointsTr_1', filename1))
    image5 = nim5.get_data()
    image5 = np.array(image5, dtype='float32')
    nim6 = nib.load(os.path.join(data_path, 'NLST2023', 'keypointsTr_1', filename2))
    image6 = nim6.get_data()
    image6 = np.array(image6, dtype='float32')
    
    image3[image5 ==1] = 2
    image4[image6 ==1] = 2
    
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    image3 = np.reshape(image3, (1,) + image3.shape)
    image4 = np.reshape(image4, (1,) + image4.shape)
    
    return image1, image2, image3, image4

class TrainDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path, transform=None):
        'Initialization'
        super(TrainDataset, self).__init__()
        self.data_path = data_path
        self.filename = pd.read_csv(os.path.join(data_path,'train_200.csv'), encoding= 'unicode_escape').values
        #print(self.filename)
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        assert len(self.filename) == 200
        img_A, img_B, label_A, label_B = load_train_pair(self.data_path, self.filename[index][0], self.filename[index][1])
        return img_A, img_B, label_A, label_B#, pts_A, pts_B# , pts_A1, pts_B1


def load_validation_pair(data_path, filename1, filename2):
    # Load images and labels
    nim1 = nib.load(os.path.join(data_path, 'NLST2023', 'imagesTr_450.0_-1000.0', filename1))
    image1 = nim1.get_data()
    image1 = np.array(image1, dtype='float32')
    nim2 = nib.load(os.path.join(data_path, 'NLST2023', 'imagesTr_450.0_-1000.0', filename2))
    image2 = nim2.get_data()
    image2 = np.array(image2, dtype='float32')
    
    nim3 = nib.load(os.path.join(data_path, 'NLST2023', 'masksTr', filename1))
    image3 = nim3.get_data()
    image3 = np.array(image3, dtype='float32')
    nim4 = nib.load(os.path.join(data_path, 'NLST2023', 'masksTr', filename2))
    image4 = nim4.get_data()
    image4 = np.array(image4, dtype='float32')
    
    nim5 = nib.load(os.path.join(data_path, 'NLST2023', 'keypointsTr_1', filename1))
    image5 = nim5.get_data()
    image5 = np.array(image5, dtype='float32')
    nim6 = nib.load(os.path.join(data_path, 'NLST2023', 'keypointsTr_1', filename2))
    image6 = nim6.get_data()
    image6 = np.array(image6, dtype='float32')
    
    
    nim7 = nib.load(os.path.join(data_path, 'NLST2023', 'imagesTr_450.0_-900.0', filename1))
    image7 = nim7.get_data()
    image7 = np.array(image7, dtype='float32')
    nim8 = nib.load(os.path.join(data_path, 'NLST2023', 'imagesTr_450.0_-900.0', filename2))
    image8 = nim8.get_data()
    image8 = np.array(image8, dtype='float32')
    
    
    
    # image7[image5 ==1] = 2
    # image8[image6 ==1] = 2
    
    image3[image5 ==1] = 2
    image4[image6 ==1] = 2
    
    
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    image3 = np.reshape(image3, (1,) + image3.shape)
    image4 = np.reshape(image4, (1,) + image4.shape)
    # image5 = np.reshape(image5, (1,) + image5.shape)
    # image6 = np.reshape(image6, (1,) + image6.shape)
    image7 = np.reshape(image7, (1,) + image7.shape)
    image8 = np.reshape(image8, (1,) + image8.shape)
    # return image1, image2,image3, image4, image5, image6,image7, image8
    return image1, image2, image3, image4, image7, image8

class ValidationDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path, transform=None):
        'Initialization'
        super(ValidationDataset, self).__init__()
        self.data_path = data_path
        self.filename = pd.read_csv(os.path.join(data_path,'val.csv'), encoding= 'unicode_escape').values
        #print(self.filename)
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        assert len(self.filename) == 10
        img_A, img_B, pts_A, pts_B, imagA, imagB = load_validation_pair(self.data_path, self.filename[index][0], self.filename[index][1])
        return img_A, img_B, pts_A, pts_B, imagA, imagB# , pts_A1, pts_B1
        

class SubmissionDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path, transform=None):
        'Initialization'
        super(SubmissionDataset, self).__init__()
        self.data_path = data_path
        self.filename = pd.read_csv(os.path.join(data_path,'submission_val.csv'), encoding= 'unicode_escape').values
        #print(self.filename)
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_A, img_B, label_A, label_B, imagA, imagB  = load_Submission_pair(self.data_path, self.filename[index][0], self.filename[index][1])
        return self.filename[index][0],self.filename[index][1], img_A, img_B, label_A, label_B, imagA, imagB 

def load_Submission_pair(data_path, filename1, filename2):
    # Load images and labels
    nim1 = nib.load(os.path.join(data_path, 'NLST2023', 'imagesTr_450.0_-1000.0', filename1))
    image1 = nim1.get_data()
    image1 = np.array(image1, dtype='float32')
    nim2 = nib.load(os.path.join(data_path, 'NLST2023', 'imagesTr_450.0_-1000.0', filename2))
    image2 = nim2.get_data()
    image2 = np.array(image2, dtype='float32')
    
    nim3 = nib.load(os.path.join(data_path, 'NLST2023', 'masksTr', filename1))
    image3 = nim3.get_data()
    image3 = np.array(image3, dtype='float32')
    nim4 = nib.load(os.path.join(data_path, 'NLST2023', 'masksTr', filename2))
    image4 = nim4.get_data()
    image4 = np.array(image4, dtype='float32')
    
    
    nim7 = nib.load(os.path.join(data_path, 'NLST2023', 'imagesTr_450.0_-900.0', filename1))
    image7 = nim7.get_data()
    image7 = np.array(image7, dtype='float32')
    nim8 = nib.load(os.path.join(data_path, 'NLST2023', 'imagesTr_450.0_-900.0', filename2))
    image8 = nim8.get_data()
    image8 = np.array(image8, dtype='float32')
    
    
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    image3 = np.reshape(image3, (1,) + image3.shape)
    image4 = np.reshape(image4, (1,) + image4.shape)
    image7 = np.reshape(image7, (1,) + image7.shape)
    image8 = np.reshape(image8, (1,) + image8.shape)
    # return image1, image2,image3, image4, image5, image6,image7, image8
    return image1, image2, image3, image4, image7, image8