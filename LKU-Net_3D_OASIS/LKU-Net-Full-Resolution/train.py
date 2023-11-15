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

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=309685,
                    help="number of total iterations")
parser.add_argument("--mask_labda", type=float,
                    dest="mask_labda", default=1.0,
                    help="mask_labda loss: suggested range 0.1 to 10")
parser.add_argument("--data_labda", type=float,
                    dest="data_labda", default=0.02,
                    help="data_labda loss: suggested range 0.1 to 10")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=1.0,
                    help="labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=394,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/export/local/L2R/Data/',
                    help="data path for training images")
parser.add_argument("--trainingset", type=int,
                    dest="trainingset", default=4,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=1,
                    help="using l2 or not")
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

def dice(pred1, truth1):
    dice_35=np.zeros(35)
    for k in range(1,36,1):
        #print(k)
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred[truth==1.0]) * 2.0
        # print(intersection)
        dice_35[k-1]=intersection / (np.sum(pred) + np.sum(truth))
    return np.mean(dice_35)

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
    model = UNet(2, 3, start_channel).cuda()
    if using_l2 == 1:
        loss_similarity = MSE().loss
    elif using_l2 == 0:
        loss_similarity = SAD().loss
    elif using_l2 == 2:
        loss_similarity = NCC()
    loss_smooth = smoothloss
    loss_dice = DiceLoss()

    transform = SpatialTransform().cuda()
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lossall = np.zeros((4, iteration))
    train_set = TrainDataset(datapath,img_file='trainList.txt', trainingset = trainingset)
    training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
    test_set = ValidationDataset(opt.datapath)#,img_file='val_list.txt')
    test_generator = Data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)
    model_dir = './L2ss_{}_Set_{}_Chan_{}_LR_{}_Smth_{}_Seg_{}/'.format(using_l2, trainingset, start_channel, lr, smooth, mask_labda)
    csv_name = 'L2ss_{}_Set_{}_Chan_{}_LR_{}_Smth_{}_Seg_{}.csv'.format(using_l2, trainingset, start_channel, lr, smooth, mask_labda)
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
    
    step = 1

    while step <= iteration:
        for mov_img, fix_img, mov_lab, fix_lab in training_generator:

            fix_img = fix_img.cuda().float()

            mov_img = mov_img.cuda().float()

            fix_lab = fix_lab.cuda().float()

            mov_lab = mov_lab.cuda().float()
            
            X_Seg = nn.functional.one_hot(mov_lab.long(), num_classes=36)
            X_Seg = X_Seg.squeeze(1).permute(0, 4, 1, 2, 3)
            f_xy = model(mov_img, fix_img)
            # D_f_xy = diff_transform(f_xy)
            
            warped_mov = transform(mov_img, f_xy.permute(0, 2, 3, 4, 1))
            X_Y_Seg = transform(X_Seg.float(), f_xy.permute(0, 2, 3, 4, 1))
           
            loss1 = loss_similarity(fix_img, warped_mov) # GT shall be 1st Param
            loss2 = loss_smooth(f_xy)
            loss3 = loss_dice(X_Y_Seg, fix_lab.long())
            
            loss = loss1 + smooth * loss2 + mask_labda * loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:,step] = np.array([loss.item(), loss1.item(), loss2.item(), loss3.item()])
            sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim "{2:.4f}" -smo "{3:.4f}" -seg "{4:.4f}" '.format(step, loss.item(),loss1.item(),loss2.item(),loss3.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0):
                with torch.no_grad():
                    Dices_Validation = []
                    for __, __, mov_img, fix_img, mov_lab, fix_lab in test_generator:
                        model.eval()
                        V_xy = model(mov_img.float().to(device), fix_img.float().to(device))
                        # D_V_xy = diff_transform(V_xy)
                        warped_mov_lab = transform(mov_lab.float().to(device), V_xy.permute(0, 2, 3, 4, 1), mod = 'nearest')
                        for bs_index in range(bs):
                            dice_bs = dice(warped_mov_lab[bs_index,...].data.cpu().numpy().copy(),fix_lab[bs_index,...].data.cpu().numpy().copy())
                            Dices_Validation.append(dice_bs)
                    modelname = 'DiceVal_{:.4f}_Step_{:09d}.pth'.format(np.mean(Dices_Validation), step)
                    csv_dice = np.mean(Dices_Validation)
                    save_checkpoint(model.state_dict(), model_dir, modelname)
                    np.save(model_dir + 'Loss.npy', lossall)
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([step, csv_dice])
            step += 1
            
            if step > iteration:
                break
        print("one epoch pass")
    np.save(model_dir + '/Loss.npy', lossall)

train()
