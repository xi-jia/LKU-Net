import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch.nn.functional as F
import torch
from torchvision import transforms
from Models import *
from Functions import *
import torch.utils.data as Data
from natsort import natsorted
import csv
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=4001,
                    help="number of total iterations")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=4,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=0,
                    help="using l2 or not")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
n_checkpoint = opt.checkpoint
using_l2 = opt.using_l2

def dice(pred1, truth1, k=1):
    dice_avg=0
    labels = list(np.unique(truth1))
    # print(labels)
    labels.remove(0.0) # remove background
    for k in labels:
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        dice_avg+=intersection / (np.sum(pred) + np.sum(truth))
    return dice_avg/len(labels)

def save_checkpoint(state, save_dir, save_filename, max_model_num=10):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    # print(model_lists)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def train():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_set = TrainDataset(img_file='./celegan_train_list.txt')
    val_set = ValidationDataset(img_file='./celegan_val_list.txt')
    train_loader = Data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = Data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)


    model = UNet(2, start_channel).cuda()
    if using_l2 == 1:
        loss_similarity = MSE().loss
    else:
        loss_similarity = NCC()
    # loss_dice = DiceLoss()
    
    transform_seg = ApplyAffine(mode='nearest').cuda()
    transform_img = ApplyAffine().cuda()
    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model_dir = './L2ss_{}_Chan_{}_LR_{}/'.format(using_l2,start_channel,lr)
    # model_png_dir = './L2ss_{}_Chan_{}_LR_{}_Png/'.format(using_l2,start_channel,lr)
    csv_name = 'L2ss_{}_Chan_{}_LR_{}.csv'.format(using_l2,start_channel,lr)
    assert os.path.exists(csv_name) ==0
    assert os.path.isdir(model_dir) ==0
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # if not os.path.isdir(model_png_dir):
    #     os.mkdir(model_png_dir)

    lossall = np.zeros((1, iteration))
    
    step = 1
    epoch = 1
    while step <= iteration:
        for  imgName, X, Y in train_loader:
        
            X = X.cuda().float()
            Y = Y.cuda().float()
            
            Warped_X, mat = model(X, Y)
            
            
            # if imgName[0] =='pair4':
                # X_Seg = nn.functional.one_hot(mov_lab.long(), num_classes=62) 
                # X_Seg = X_Seg.squeeze(1).permute(0, 4, 1, 2, 3)
            
                # __, warped_mov_lab = transform_img(X_Seg.float().to(device), mat, outputsize = fix_lab.shape[2])
                # loss3 = loss_dice(warped_mov_lab, fix_lab.long().to(device))
                
                # loss = loss_similarity(Y, Warped_X) + loss3
            # else:
            loss = loss_similarity(Y, Warped_X) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:,step] = np.array([loss.item()])
            sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" '.format(step, loss.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0) or (step == 1):
                with torch.no_grad():
                    Dices_Validation = []
                    for data in val_loader:
                        model.eval()
                        xv_small = data[0]
                        yv_small = data[1]
                        xv_seg = data[2]
                        yv_seg = data[3]
                        
                        __, vmat = model(xv_small.float().to(device), yv_small.float().to(device))
                        
                        
                        grid, warped_xv_seg= transform_seg(xv_seg.float().to(device), vmat, outputsize = yv_seg.shape[2])
                        
                        dice_bs=dice(warped_xv_seg[0,...].data.cpu().numpy().copy(),yv_seg[0,...].data.cpu().numpy().copy())
                        Dices_Validation.append(dice_bs)
                        
                    modelname = 'DiceVal_{:.4f}_Epoch_{:09d}.pth'.format(np.mean(Dices_Validation), epoch)
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, np.mean(Dices_Validation)])
                    save_checkpoint(model.state_dict(), model_dir, modelname)
                np.save(model_dir + 'Loss.npy', lossall)
            step += 1

            if step > iteration:
                break
        print("one epoch pass")
        epoch = epoch + 1
    np.save(model_dir + '/Loss.npy', lossall)
train()