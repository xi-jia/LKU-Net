import os
from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import ValidationDataset
import torch.utils.data as Data
import csv
from scipy.ndimage.interpolation import map_coordinates, zoom
import torch.nn.functional as F
from natsort import natsorted

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
                    default='/export/local/L2R/Data_T3',
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


#savepath = opt.savepath
#fixed_path = opt.fixed
#moving_path = opt.moving

#if not os.path.isdir(savepath):
#    os.mkdir(savepath)

def dice(pred, truth, k = 1):
    truth[truth!=k]=0
    pred[pred!=k]=0
    truth=truth/k
    pred=pred/k
    intersection = np.sum(pred[truth==1.0]) * 2.0
#    print(intersection)
    dice = intersection / (np.sum(pred) + np.sum(truth))
    return dice
def convert_pytorch_grid2scipy(grid):
    
    _, H, W, D = grid.shape
    grid_x = (grid[0, ...] + 1) * (D -1)/2
    grid_y = (grid[1, ...] + 1) * (W -1)/2
    grid_z = (grid[2, ...] + 1) * (H -1)/2
    
    grid = np.stack([grid_z, grid_y, grid_x])
    
    identity_grid = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    grid = grid - identity_grid
    
    # Simple ITK to nibabel grid
    #grid = grid[::-1, ...] 
    #grid = grid.swapaxes(1, 3)
    
    return grid
def submit(modelpath, savepath):
    bs = 1
    model = UNet(2, 3, opt.start_channel).cuda()
    torch.backends.cudnn.benchmark = True
    transform = SpatialTransform_1().cuda()
    model.load_state_dict(torch.load(modelpath))
    #model_lambda = model.ic_block.labda.data.cpu().numpy()
    #model_odr = model.ic_block.odr.data.cpu().numpy()
    model.eval()
    transform.eval()
#    diff_transform.eval()
    Dices_35=[]
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    test_set = ValidationDataset(opt.datapath)
    test_generator = Data.DataLoader(dataset=test_set, batch_size=bs,
                                         shuffle=False, num_workers=2)
    for fix_name,mov_name,fix_img, mov_img, fix_lab, mov_lab in test_generator:
        with torch.no_grad():
            
            XX, YY, ZZ = mov_lab.squeeze(0).squeeze(0).data.numpy().shape
            identity = np.meshgrid(np.arange(XX), np.arange(YY), np.arange(ZZ), indexing='ij')
            
            mask_value1 = np.unique(mov_lab.data.cpu().numpy())
            mask_value2 = np.unique(fix_lab.data.cpu().numpy())
            mask_value = list(set(mask_value1) & set(mask_value2))
            
            
            V_xy = model(mov_img.float().to(device), fix_img.float().to(device))
            pytorch_Warped, pytorch_grid = transform(mov_lab.float().to(device), V_xy.permute(0, 2, 3, 4, 1),mod = 'nearest')
            
            dice_bs=[]
            for i in mask_value[1:]:
                dice_bs.append(dice(pytorch_Warped.squeeze(0).squeeze(0).data.cpu().numpy().copy(),fix_lab.squeeze(0).squeeze(0).data.numpy().copy(), k = i))
            print('Full res pytorch_grid :   ',np.mean(dice_bs))
            
            pytorch_grid = pytorch_grid.squeeze(0).permute(3,0,1,2) # Discard the first channel DHW3 -> 3DHW
            
            scipy_disp = convert_pytorch_grid2scipy(pytorch_grid.data.cpu().numpy())

            moving_warped = map_coordinates(mov_lab.squeeze(0).squeeze(0).data.numpy(), identity + scipy_disp, order=0)
            
            dice_bs=[]
            for i in mask_value[1:]:
                dice_bs.append(dice(moving_warped.copy(),fix_lab.squeeze(0).squeeze(0).data.numpy().copy(), k = i))
            print('Full res scipy_disp :   ',np.mean(dice_bs))
            
            
            save_npz_name = os.path.join(savepath,'disp_{:04d}_{:04d}.npz'.format(int(fix_name[0]),int(mov_name[0])))
            
            # Here the order is 0-5, we set the order to be 2, matching the Challenge Evaluation Code.
            disp1 = np.array([zoom(scipy_disp[i], 0.5, order=2) for i in range(3)])
            # disp1 = np.array([zoom(scipy_disp[i], 0.5, order=3, prefilter=False, grid_mode= False) for i in range(3)])
            
            # disp1 = torch.from_numpy(disp1).unsqueeze(0).unsqueeze(0)
            # print(disp1.shape)
            # print(scipy_disp[:,:,0,...].shape)
            # downsample_scipy_disp = [torch.nn.functional.interpolate(scipy_disp[:,:,i,...], size=[80, 96, 112], mode='area', align_corners=None, recompute_scale_factor=None).squeeze(0).squeeze(0).data.numpy() for i in range(3)]
            # print(np.array(downsample_scipy_disp).shape)
            # np.savez(save_npz_name, np.array(downsample_scipy_disp).astype(np.float16))
            np.savez(save_npz_name, np.array(disp1).astype(np.float16))
            
            disp_field = np.load(save_npz_name)['arr_0'].astype('float32')
            # Here the order is 2, matching the Challenge Evaluation Code.
            disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])
            
            
            moving_warped = map_coordinates(mov_lab.squeeze(0).squeeze(0).data.numpy(), identity + disp_field, order=0)
            
            dice_bs=[]
            for i in mask_value[1:]:
                dice_bs.append(dice(moving_warped.copy(),fix_lab.squeeze(0).squeeze(0).data.numpy().copy(), k = i))
            print('Half res scipy_disp :   ',np.mean(dice_bs))
            
            
            
            Dices_35.append(np.mean(dice_bs))
    print(np.mean(Dices_35))


if __name__ == '__main__':
    DICESCORES35=[]
    # i = 1970
    
    model_dir = './L2ss_{}_Set_{}_Chan_{}_LR_{}_Smth_{}_Seg_{}/'.format(using_l2, trainingset, start_channel, lr, smooth, mask_labda)
    model_idx = -2
    print(model_dir)
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])#['state_dict']
    best_model = model_dir + natsorted(os.listdir(model_dir))[model_idx] #['state_dict']
    savepath = './submission/L2ss_{}_Set_{}_Chan_{}_LR_{}_Smth_{}_Seg_{}'.format(using_l2, trainingset, start_channel, lr, smooth, mask_labda)
    
    if not os.path.exists('./submission'):
        os.makedirs('./submission')
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    submit(best_model, savepath)
    print(model_dir)
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))