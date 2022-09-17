#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
import argparse
import json

def norm_img(image):
    image400 = (image + 1000) / 1400
    image400[image400 > 1] = 1.
    image400[image400 < 0] = 0.
    image500 = (image + 1000) / 1500
    image500[image500 > 1] = 1.
    image500[image500 < 0] = 0.
    return image400, image500

class UNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        bias_opt = True
        super(UNet, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        # self.dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                # stride=1, bias=bias_opt)
        # self.dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=5, stride=1, padding=2, bias=False)

        self.up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        # self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            # nn.Dropout(0.1),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        f_xy = self.dc9(d2)
        #f_yx = self.dc10(d3)
        f_xy_0 = torch.nn.functional.interpolate(f_xy[:,0:1,...], size=[224, 192, 224], mode='trilinear')
        f_xy_1 = torch.nn.functional.interpolate(f_xy[:,1:2,...], size=[224, 192, 224], mode='trilinear')
        f_xy_2 = torch.nn.functional.interpolate(f_xy[:,2:3,...], size=[224, 192, 224], mode='trilinear')
        f_xy = torch.cat((f_xy_0, f_xy_1, f_xy_2), 1)

        return f_xy#, f_yx

class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, 224), torch.linspace(-1, 1, 192), torch.linspace(-1, 1, 224)])
        #d2, h2, w2 = 224, 192, 224
        self.grid_h = grid_h.cuda().float()
        self.grid_d = grid_d.cuda().float()
        self.grid_w = grid_w.cuda().float()
    def forward(self, mov_image, flow, mod = 'bilinear', output_grid=False):
        # self.grid_d = nn.Parameter(grid_d, requires_grad=False)
        # self.grid_w = nn.Parameter(grid_w, requires_grad=False)
        # self.grid_h = nn.Parameter(grid_h, requires_grad=False)
        # Remove Channel Dimension
        disp_d = (self.grid_d + (flow[:,:,:,:,0])).squeeze(1)
        disp_h = (self.grid_h + (flow[:,:,:,:,1])).squeeze(1)
        disp_w = (self.grid_w + (flow[:,:,:,:,2])).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)
        if output_grid==True:
            return sample_grid
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True)
        return warped
class CascadeUNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(CascadeUNet, self).__init__()
        self.net = UNet(self.in_channel, self.n_classes, self.start_channel)
        self.warp = SpatialTransform()
    def forward(self, x, y):
        # in_pair = torch.cat((x, y), 1)
        fxy_1 = self.net(x, y)
        x2 = self.warp(x, fxy_1.permute(0, 2, 3, 4, 1))
        # in_pair2 = torch.cat((x2, y), 1)
        fxy_2 = self.net(x2, y)
        fxy_2_ = self.warp(fxy_1, fxy_2.permute(0, 2, 3, 4, 1))
        
        fxy_2_ = fxy_2_ + fxy_2
        x3 = self.warp(x, fxy_2_.permute(0, 2, 3, 4, 1))
        # in_pair3 = torch.cat((x3, y), 1)
        fxy_3 = self.net(x3, y)
        fxy_3_ = self.warp(fxy_2_, fxy_3.permute(0, 2, 3, 4, 1))
        fxy_3_ = fxy_3_ + fxy_3
        return fxy_3_
def convert_pytorch_grid2scipy(grid):
    _, H, W, D = grid.shape
    grid_x = (grid[0, ...] + 1) * (D -1)/2
    grid_y = (grid[1, ...] + 1) * (W -1)/2
    grid_z = (grid[2, ...] + 1) * (H -1)/2
    
    grid = np.stack([grid_z, grid_y, grid_x])
    
    identity_grid = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    grid = grid - identity_grid
    return grid
def main(dataset_json):
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    transform = SpatialTransform().cuda()
    
    model_400 = CascadeUNet(2, 3, 32).to(device)
    model_400.load_state_dict(torch.load('./400_Avg.pth'))
    model_500 = CascadeUNet(2, 3, 32).to(device)
    model_500.load_state_dict(torch.load('./500_Avg.pth'))
    model_400.eval()
    model_500.eval()
    transform.eval()

    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
    # D, H, W = dataset['tensorImageShape']['0']

    num_pairs = dataset['numRegistration_test']
    pairs = dataset['registration_test']
    with torch.no_grad():
        for i in range(num_pairs):
            pair = pairs[i]
            fix_path = os.path.join(args.test_data_dir, pair['fixed'])
            mov_path = os.path.join(args.test_data_dir, pair['moving'])
            fix_id = os.path.basename(fix_path).split('_')[1]
            mov_id = os.path.basename(mov_path).split('_')[1]
            disp_path = os.path.join(args.save_disp_dir,'disp_{}_{}.nii.gz'.format(fix_id, mov_id))
            
            fix = nib.load(fix_path)
            mov = nib.load(mov_path)
            fix = fix.get_data()
            mov = mov.get_data()
            fix = np.array(fix, dtype='float32')
            mov = np.array(mov, dtype='float32')
            fix400, fix500 = norm_img(fix)
            mov400, mov500 = norm_img(mov)
            
            mov400 = torch.from_numpy(mov400).unsqueeze(0).unsqueeze(0)
            mov500 = torch.from_numpy(mov500).unsqueeze(0).unsqueeze(0)
            fix500 = torch.from_numpy(fix500).unsqueeze(0).unsqueeze(0)
            fix400 = torch.from_numpy(fix400).unsqueeze(0).unsqueeze(0)
            # disp = np.zeros((D, H, W, 3))
            
            V_xy_400 = model_400(mov400.to(device), fix400.to(device))
            V_xy_500 = model_500(mov500.to(device), fix500.to(device))
            V_xy = (V_xy_400 + V_xy_500) / 2
            pytorch_grid = transform(0, V_xy.permute(0, 2, 3, 4, 1), output_grid=True)
            scipy_disp = convert_pytorch_grid2scipy(pytorch_grid.squeeze(0).permute(3,0,1,2).data.cpu().numpy())
            scipy_disp = np.moveaxis(scipy_disp, 0, -1)
            # print(np.max(scipy_disp))
            nib.save(nib.Nifti1Image(scipy_disp, np.eye(4)), disp_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Docker Submission')
    parser.add_argument('dataset_json', help='path to dataset_json')
    parser.add_argument("--test_data_dir", type=str, dest="test_data_dir", default='/l2r/data')
    parser.add_argument("--save_disp_dir", type=str, dest="save_disp_dir", default='/l2r/output')
    args = parser.parse_args()
    if not os.path.exists(args.save_disp_dir):
        os.makedirs(args.save_disp_dir)
    # import time
    # start = time.time()
    main(args.dataset_json)
    # print(time.time()-start)
