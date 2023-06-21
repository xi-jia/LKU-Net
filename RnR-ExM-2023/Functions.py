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
from volumentations import *
import torch.nn.functional as nnF
import h5py
from scipy.spatial.transform import Rotation as R
def validation_crop_and_pad(img, sizex):
    # print(img.shape)
    # print(y.shape)
    sizey, sizez = 256, 256
    img_new = np.zeros((sizex, sizey, sizez))#.float().cuda()
    # assert sizex ==559
    h = np.amin([sizex,img.shape[0]])
    w = np.amin([sizey,img.shape[1]])
    d = np.amin([sizez,img.shape[2]])
    img_new[sizex//2-h//2:sizex//2+h//2,sizey//2-w//2:sizey//2+w//2,sizez//2-d//2:sizez//2+d//2]=img[img.shape[0]//2-h//2:img.shape[0]//2+h//2,img.shape[1]//2-w//2:img.shape[1]//2+w//2,img.shape[2]//2-d//2:img.shape[2]//2+d//2]
    return img_new
    
def rigid_augment(image1):
    # Rotation
    
    angle = np.random.choice(np.linspace(-5, 5, 10), 3)
    r = R.from_rotvec(angle * np.array([1, 1, 1]), degrees=True)
    r = torch.tensor(r.as_matrix())
    

    # if np.random.choice([0, 1], p=[0.3, 0.7]):
    # Translation
    translation = np.random.choice(np.arange(-0.1, .1, 0.01), 3)
    t = torch.tensor([translation[0], translation[1], translation[2]])
    t = torch.unsqueeze(t,dim=1)

    # Tranformation matrix
    theta = torch.cat((r,t),axis = 1)
    theta = torch.unsqueeze(theta,axis = 0)
    # print(theta.shape)
    # print(theta)# Define grid
    Z_out, H_out, W_out = image1.shape[0], image1.shape[1], image1.shape[2] 
    grid = nnF.affine_grid(theta, (1,3,Z_out,H_out,W_out), align_corners=True)
    # print('grid shape', grid.shape)
    output= nnF.grid_sample(torch.from_numpy(image1).float().unsqueeze(0).unsqueeze(0), grid.float(), align_corners=True)
    output = output[0,0,:,:,:]
    return output.numpy()

def get_augmentation(patch_size):
    return Compose([
        Rotate((-5, 5), (-5, 5), (-5, 5), p=0.8),
        RandomCropFromBorders(crop_value=0.1, p=0.8),
        # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        # Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
        # Flip(0, p=0.5),
        # Flip(1, p=0.5),
        # Flip(2, p=0.5),
        # RandomRotate90((1, 2), p=0.5),
        GaussianNoise(var_limit=(0.0, 0.0002), p=0.2),
        # RandomGamma(gamma_limit=(80, 120), p=0.2),
    ], p=1.0)


def rescale_intensity(image, thres=(0.1, 99.9)):
    """ Rescale the image intensity to the range of [0, 1] """
    image = image.astype(np.float32)
    val_l, val_h = np.percentile(image, thres)
    # print(val_l, val_h)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2

def load_train_pair(filename):
    
    
    name = filename[-5:]

    nim1 = nib.load(filename +"_mov.nii.gz")
    image1 = nim1.get_data()[:]
    img1 = np.array(image1, dtype='float32')
    
    nim2 = nib.load(filename +"_fix.nii.gz")
    image2 = nim2.get_data()[:]
    img2 = np.array(image2, dtype='float32')
    
    
    # if name == 'pair4':
        # nim3 = nib.load(filename +"_mov_seg.nii.gz")
        # image3 = nim3.get_data()[:]
        # img3 = np.array(image3, dtype='float32')
        
        # nim4 = nib.load(filename +"_fix_seg.nii.gz")
        # image4 = nim4.get_data()[:]
        # img4 = np.array(image4, dtype='float32')
    
    
    if np.random.choice([0, 1], p=[0.1, 0.9]):
        image1 = rigid_augment(img1)
    else:
        image1 = img1
    
    
    if np.random.choice([0, 1], p=[0.1, 0.9]):
        image2 = rigid_augment(img2)
    else:
        image2 = img2
    
    image1 = validation_crop_and_pad(image1, 358)
    image2 = validation_crop_and_pad(image2, 559)
    
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    # if name == 'pair4':
        # image3 = validation_crop_and_pad(image3, 358)
        # image4 = validation_crop_and_pad(image4, 559)
        # image3 = np.reshape(image3, (1,) + image3.shape)
        # image4 = np.reshape(image4, (1,) + image4.shape)
    # else:
        # image3 = image1
        # image4 = image2
    return image1, image2#, image3, image4#, Moving, Fixed, MaskMoving, MaskFixed

class TrainDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, img_file=None):
        'Initialization'
        super(TrainDataset, self).__init__()
        self.filename = np.loadtxt(os.path.join(img_file),dtype='str')
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        assert len(self.filename) == 7
        image1, image2 = load_train_pair(self.filename[index])
        return self.filename[index][-5:], image1, image2#, image3, image4


def load_validation_pair(filename):
    
    nim1 = nib.load(filename + "_mov.nii.gz")
    image1 = nim1.get_data()[:]
    image1 = np.array(image1, dtype='float32')
    
    nim2 = nib.load(filename + "_fix.nii.gz")
    image2 = nim2.get_data()[:]
    image2 = np.array(image2, dtype='float32')
    
    xv_seg = nib.load(filename + "_mov_seg.nii.gz")
    x_seg = xv_seg.get_data()[:]
    x_seg = np.array(x_seg, dtype='float32')
    
    yv_seg = nib.load(filename + "_fix_seg.nii.gz")
    y_seg = yv_seg.get_data()[:]
    y_seg = np.array(y_seg, dtype='float32')
    
    image1 = validation_crop_and_pad(image1, 358)
    image2 = validation_crop_and_pad(image2, 559)
    x_seg = validation_crop_and_pad(x_seg, 358)
    y_seg = validation_crop_and_pad(y_seg, 559)
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    
    
    # image3 = np.reshape(image3, (1,) + image3.shape)
    # image4 = np.reshape(image4, (1,) + image4.shape)
    
    x_seg = np.reshape(x_seg, (1,) + x_seg.shape)
    y_seg = np.reshape(y_seg, (1,) + y_seg.shape)
    
    return image1, image2, x_seg, y_seg

class ValidationDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, img_file=None):
        'Initialization'
        super(ValidationDataset, self).__init__()
        self.filename = np.loadtxt(os.path.join(img_file),dtype='str')
        # print(len(self.filename))
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        assert len(self.filename) == 2
        img_A, img_B, mask_A, mask_B = load_validation_pair(self.filename[index])
        return img_A, img_B, mask_A, mask_B#, x_small, y_small# , pts_A1, pts_B1
        

class SubmissionDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, img_file=None):
        'Initialization'
        super(SubmissionDataset, self).__init__()
        self.filename = np.loadtxt(os.path.join(img_file),dtype='str')
        #print(self.filename)
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # assert len(self.filename) == 3
        img_A, img_B, x_small, y_small = load_Submission_pair(self.filename[index])
        return self.filename[index], img_A, img_B, x_small, y_small 

def load_Submission_pair(filename):
    
    
    name = filename[-5:]
    
    nim1 = nib.load(filename + "_mov.nii.gz")
    image1 = nim1.get_data()[:]
    image1 = np.array(image1, dtype='float32')
    
    nim2 = nib.load(filename + "_fix.nii.gz")
    image2 = nim2.get_data()[:]
    image2 = np.array(image2, dtype='float32')
    
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    # vol_to_align = np.reshape(vol_to_align, (1,) + vol_to_align.shape)
    # vol_to_align = np.array(vol_to_align, dtype='float32')
    # fixed_vol = np.reshape(fixed_vol, (1,) + fixed_vol.shape)
    # fixed_vol = np.array(fixed_vol, dtype='float32')
    return image1, image2, image1, image2
    
    
