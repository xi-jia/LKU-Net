
import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import os
from os import listdir
from os.path import join
import itertools
import h5py

filename = './Celegan/Test/c_elegan_pair7'
# '''
# def validation_crop_and_pad(img):
#     # print(img.shape)
#     # print(y.shape)
#     img_new = np.zeros((559, 256, 256))#.float().cuda()
#     sizex, sizey, sizez = 559, 256, 256
#     # assert sizex ==559
#     h = np.amin([sizex,img.shape[0]])
#     w = np.amin([sizey,img.shape[1]])
#     d = np.amin([sizez,img.shape[2]])

#     img_new[sizex//2-h//2:sizex//2+h//2,sizey//2-w//2:sizey//2+w//2,sizez//2-d//2:sizez//2+d//2]=img[img.shape[0]//2-h//2:img.shape[0]//2+h//2,img.shape[1]//2-w//2:img.shape[1]//2+w//2,img.shape[2]//2-d//2:img.shape[2]//2+d//2]
#     return img_new

def rescale_intensity(image, thres=(0.01, 99.99)):
    """ Rescale the image intensity to the range of [0, 1] """
    image = image.astype(np.float32)
    val_l, val_h = np.percentile(image, thres)
    # print(val_l, val_h)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2



hf = h5py.File(filename + '.h5', 'r')
fixed_vol = hf.get('fixed')[:]
vol_to_align = hf.get('move')[:]

img1 = np.zeros(vol_to_align.shape)
img2 = np.zeros(fixed_vol.shape)

# for i in range(0, len(vol_to_align)):
    # img1[i] = rescale_intensity(vol_to_align[i])
# for i in range(0, len(fixed_vol)):
    # img2[i] = rescale_intensity(fixed_vol[i])


img1 = rescale_intensity(vol_to_align)
img2 = rescale_intensity(fixed_vol)

x = torch.from_numpy(img1.astype(np.float32))
y = torch.from_numpy(img2.astype(np.float32))

print(x.shape)
print(y.shape)

# x_small = torch.nn.functional.interpolate(x.unsqueeze(0).unsqueeze(0), size=[img1.shape[0], 256, 256], mode='trilinear', align_corners=True)
# y_small = torch.nn.functional.interpolate(y.unsqueeze(0).unsqueeze(0), size=[img2.shape[0], 256, 256], mode='trilinear', align_corners=True)
x_small = torch.nn.functional.interpolate(x.unsqueeze(0).unsqueeze(0), size=[img1.shape[0], 256, 256], mode='nearest')
y_small = torch.nn.functional.interpolate(y.unsqueeze(0).unsqueeze(0), size=[img2.shape[0], 256, 256], mode='nearest')
# y_small = validation_crop_and_pad(y_small.squeeze(0).squeeze(0).numpy())

source = nib.Nifti1Image(x_small.squeeze(0).squeeze(0).numpy(), affine=np.eye(4))
nib.save(source, filename +"_mov.nii.gz")
target = nib.Nifti1Image(y_small.squeeze(0).squeeze(0).numpy(), affine=np.eye(4))
nib.save(target, filename +"_fix.nii.gz")

# '''
'''
hf_seg = h5py.File(filename + '_segmentation.h5', 'r')
fixed_vol_seg = hf_seg.get('fixed')[:]
vol_to_align_seg = hf_seg.get('move')[:]

XX = torch.from_numpy(vol_to_align_seg.astype(np.float32))
YY = torch.from_numpy(fixed_vol_seg.astype(np.float32))

print('Before len: ', len(list(np.unique(vol_to_align_seg))))
print('Before len: ', len(list(np.unique(fixed_vol_seg))))

image3 = torch.nn.functional.interpolate(XX.unsqueeze(0).unsqueeze(0), size=[img1.shape[0], 256, 256], mode='nearest').squeeze(0).squeeze(0).numpy()
image4 = torch.nn.functional.interpolate(YY.unsqueeze(0).unsqueeze(0), size=[img2.shape[0], 256, 256], mode='nearest').squeeze(0).squeeze(0).numpy()

print('After len: ', len(list(np.unique(image3))))
print('After len: ', len(list(np.unique(image4))))


source = nib.Nifti1Image(image3, affine=np.eye(4))
nib.save(source, filename +"_mov_seg.nii.gz")
target = nib.Nifti1Image(image4, affine=np.eye(4))
nib.save(target, filename +"_fix_seg.nii.gz")
'''
