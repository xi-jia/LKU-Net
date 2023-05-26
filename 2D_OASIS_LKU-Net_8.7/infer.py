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
                    dest="iteration", default=320001,
                    help="number of total iterations")
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=1000.0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--magnitude", type=float,
                    dest="magnitude", default=0.001,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument("--mask_labda", type=float,
                    dest="mask_labda", default=0.25,
                    help="mask_labda loss: suggested range 0.1 to 10")
parser.add_argument("--data_labda", type=float,
                    dest="data_labda", default=0.02,
                    help="data_labda loss: suggested range 0.1 to 10")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=0.02,
                    help="labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=4000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    #default='/export/local/xxj946/AOSBraiCN2',
                    default='/bask/projects/d/duanj-ai-imaging/Accreg/brain/OASIS_AffineData/',
                    help="data path for training images")
parser.add_argument("--trainingset", type=int,
                    dest="trainingset", default=4,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=1,
                    help="using l2 or not")
opt = parser.parse_args()


def dice(pred1, truth1):
    
    mask4_value1 = np.unique(pred1)
    mask4_value2 = np.unique(truth1)
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    
    # print(mask_value4)
    # assert 0 ==1
    dice_35=np.zeros(len(mask_value4)-1)
    index = 0
    for k in mask_value4[1:]:
        #print(k)
        truth = truth1 == k
        pred = pred1 == k
        # pred = pred1.copy()
        # truth[truth!=k]=0
        # pred[pred!=k]=0
        # truth=truth/k
        # pred=pred/k
        intersection = np.sum(pred * truth) * 2.0
        # print(intersection)
        dice_35[index]=intersection / (np.sum(pred) + np.sum(truth))
        index = index + 1
    return np.mean(dice_35)

def test(model_dir):
    bs = 1
    model = UNet(2, 2, opt.start_channel).cuda()
    
    
    model_idx = -1
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])#['state_dict']
    model.load_state_dict(best_model)
    
    torch.backends.cudnn.benchmark = True
    transform = SpatialTransform().cuda()
    # model.load_state_dict(torch.load(modelpath))
    #model_lambda = model.ic_block.labda.data.cpu().numpy()
    #model_odr = model.ic_block.odr.data.cpu().numpy()
    model.eval()
    transform.eval()
#    diff_transform.eval()
#    com_transform.eval()
#    Dices_before=[]
    Dices_35=[]
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    test_set = ValidationDataset(opt.datapath,img_file='test_list.txt')
    test_generator = Data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)
    for __, __, mov_img, fix_img, mov_lab, fix_lab in test_generator:
        with torch.no_grad():
            V_xy = model(mov_img.float().to(device), fix_img.float().to(device))
            __,warped_mov_lab = transform(mov_lab.float().to(device), V_xy.permute(0, 2, 3, 1), mod = 'nearest')
            
            #print('V_xy.shape . . . ', V_xy.shape)  #([1, 3, 160, 192, 224])
            #print('warped_mov_lab.shape . . . ', warped_mov_lab.shape) #([1, 1, 160, 192, 224])
            
            for bs_index in range(bs):
                dice_bs = dice(warped_mov_lab[bs_index,...].data.cpu().numpy().copy(),fix_lab[bs_index,...].data.cpu().numpy().copy())
                Dices_35.append(dice_bs)
    print(np.mean(Dices_35))
    print(np.std(Dices_35))

    return Dices_35


if __name__ == '__main__':
    DICESCORES4=[]
    DICESCORES35=[]
    
    csvname = 'Infer_L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}.csv'.format(opt.using_l2, opt.start_channel, opt.smth_labda, opt.trainingset, opt.lr)
    f = open(csvname, 'w')

    with f:
        fnames = ['Dice35']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    # try:
        # for i in range(opt.checkpoint,opt.iteration,opt.checkpoint):
            # model_path='./L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}/SYMNet_{}.pth'.format(opt.using_l2, opt.start_channel, opt.smth_labda, opt.trainingset, opt.lr, i)
            # print(model_path)
    model_dir = './L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_Pth/'.format(opt.using_l2, opt.start_channel, opt.smth_labda, opt.trainingset, opt.lr)
    print(model_dir)
    dice35_temp= test(model_dir)
    f = open(csvname, 'a')
    with f:
        writer = csv.writer(f)
        # dice35_temp = np.array(dice35_temp)
        writer.writerow(dice35_temp)
    # DICESCORES35.append(dice35_temp)
    # except:
        # print(np.argmax(DICESCORES35))
        # print(np.max(DICESCORES35))
    # print(np.argmax(DICESCORES35))
    # print(np.max(DICESCORES35))