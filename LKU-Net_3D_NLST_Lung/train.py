import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from Models import *
from Functions import *
import torch.utils.data as Data
from natsort import natsorted
import csv
import matplotlib.pyplot as plt
from losses import *
import random
parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=200001,#309685,
                    help="number of total iterations")
parser.add_argument("--mask_labda", type=float,
                    dest="mask_labda", default=0.0,
                    help="mask_labda loss: suggested range 0.1 to 10")
parser.add_argument("--pts_labda", type=float,
                    dest="pts_labda", default=1.0,
                    help="pts_labda loss: suggested range 0.1 to 10")
parser.add_argument("--data_labda", type=float,
                    dest="data_labda", default=1.0,
                    help="data_labda loss: suggested range 0.1 to 10")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=1.0,
                    help="labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=200,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='../L2R-2023-NLST/PreProcessed_2023/',
                    help="data path for training images")
parser.add_argument("--trainingset", type=int,
                    dest="trainingset", default=4,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=1,
                    help="using l2 or not")
parser.add_argument("--using_smooth", type=int,
                    dest="using_smooth",
                    default=1,
                    help="using_smooth or not")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
datapath = opt.datapath
mask_labda = opt.mask_labda
data_labda = opt.data_labda
trainingset = opt.trainingset
using_l2 = opt.using_l2
using_smooth = opt.using_smooth


def dice(pred1, truth1):
    dice_temp=np.zeros(2)
    for k in range(1,3,1):
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred[truth==1.0]) * 2.0
        dice_temp[k-1]=intersection / (np.sum(pred) + np.sum(truth))
    return np.mean(dice_temp)

def save_checkpoint(state, save_dir, save_filename, max_model_num=21):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    # print(model_lists)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def train():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model = CascadeLKUNet(2, 3, start_channel).cuda()
    if using_l2 == 1:
        loss_similarity = MSE().loss
    elif using_l2 == 0:
        loss_similarity = SAD().loss
    elif using_l2 == 2:
        loss_similarity = NCC()
    elif using_l2 == 3:
        loss_similarity = LNCCLoss()
    elif using_l2 == 4:
        loss_similarity = MutualInformation()
    elif using_l2 == 5:
        loss_similarity = localMutualInformation()
    if using_smooth ==1:
        loss_smooth = smoothloss
    elif using_smooth==2:
        loss_smooth = DisplacementRegularizer(energy_type = 'bending')
    loss_dice = DiceLoss(num_class=3)
    
    transform = SpatialTransform().cuda()
    transform_seg = SpatialTransform_1().cuda()
    
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    lossall = np.zeros((4, iteration))
    train_set = TrainDataset(datapath)
    training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
    test_set = ValidationDataset(datapath)#,img_file='val_list.txt')
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
    model_dir = './Fth_L2ss_{}_{}_Chan_{}_LRB_{}_{}_DataSmthSeg_{}_{}_{}/'.format(using_l2, using_smooth, start_channel, lr, bs, data_labda, smooth, mask_labda)#, pts_labda)#, pts_labda1)
    csv_name = 'FSV_L2ss_{}_{}_Chan_{}_LRB_{}_{}_DataSmthSeg_{}_{}_{}.csv'.format(using_l2, using_smooth, start_channel, lr, bs, data_labda, smooth, mask_labda)#, pts_labda)#, pts_labda1)
    
    
    assert os.path.exists(csv_name) ==0
    assert os.path.isdir(model_dir) ==0
   
   
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
    step = 1

    while step <= iteration:
        for mov_img, fix_img, mov_lab, fix_lab in training_generator:

            mov_img = mov_img.cuda().float()
            fix_img = fix_img.cuda().float()
            mov_lab = mov_lab.cuda().float()
            fix_lab = fix_lab.cuda().float()
            
            mov_Seg = nn.functional.one_hot(mov_lab.long(), num_classes=3)
            mov_Seg = mov_Seg.squeeze(1).permute(0, 4, 1, 2, 3)
            f_xy = model(mov_img, fix_img)
            
            
            warped_mov = transform(mov_img, f_xy.permute(0, 2, 3, 4, 1))
            warped_mov_Seg = transform(mov_Seg.float(), f_xy.permute(0, 2, 3, 4, 1))
           
            loss1 = loss_similarity(fix_img, warped_mov) # GT shall be 1st Param
            loss2 = loss_smooth(f_xy)
            loss3 = loss_dice(warped_mov_Seg, fix_lab.long())
            
            loss = data_labda * loss1 + smooth * loss2 + mask_labda * loss3 #+ pts_labda * loss4# + pts_labda * loss5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:,step] = np.array([loss.item(), loss1.item(), loss2.item(), loss3.item()])
            sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim "{2:.4f}" -smo "{3:.4f}" -seg "{4:.4f}"'.format(step, loss.item(),loss1.item(),loss2.item(),loss3.item()))
            sys.stdout.flush()
            
            if (step % n_checkpoint == 0) or (step == 1):
                modelname = 'Step_{:09d}.pth'.format(step)
                torch.save(model.state_dict(), model_dir + modelname)
                
            step += 1
            
            if step > iteration:
                break
        print("one epoch pass")
    np.save('Loss.npy', lossall)

train()