import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class LK_encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        padding=2,
        bias=False,
        batchnorm=False,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.batchnorm = batchnorm

        super(LK_encoder, self).__init__()

        self.layer_regularKernel = self.encoder_LK_encoder(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.bias,
            batchnorm=self.batchnorm,
        )
        self.layer_largeKernel = self.encoder_LK_encoder(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            batchnorm=self.batchnorm,
        )
        self.layer_oneKernel = self.encoder_LK_encoder(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.bias,
            batchnorm=self.batchnorm,
        )
        self.layer_nonlinearity = nn.PReLU()

    def encoder_LK_encoder(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        batchnorm=False,
    ):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        return layer

    def forward(self, inputs):
        regularKernel = self.layer_regularKernel(inputs)
        largeKernel = self.layer_largeKernel(inputs)
        oneKernel = self.layer_oneKernel(inputs)
        outputs = regularKernel + largeKernel + oneKernel + inputs
        return self.layer_nonlinearity(outputs)


class ApplyAffine(nn.Module):
    """
    3-D Affine Transformer
    This function is originally from TransMorph
    https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/tree/main/TransMorph_affine
    """
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, mat, outputsize):
        grid = F.affine_grid(mat, [src.shape[0], 3, outputsize, src.shape[3], src.shape[4]], align_corners=True)
        return grid, F.grid_sample(src, grid, align_corners=True, mode=self.mode, padding_mode='border')


class AffineTransform(nn.Module):
    """
    3-D Affine Transformer
    This function is originally from TransMorph
    https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/tree/main/TransMorph_affine
    """

    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, src, target_z_size, rotation, scale, shear, translation):

        theta_x = rotation[:, 0]
        theta_y = rotation[:, 1]
        theta_z = rotation[:, 2]
        scale_x = scale[:, 0]
        scale_y = scale[:, 1]
        scale_z = scale[:, 2]
        shear_xy = shear[:, 0]
        shear_xz = shear[:, 1]
        shear_yx = shear[:, 2]
        shear_yz = shear[:, 3]
        shear_zx = shear[:, 4]
        shear_zy = shear[:, 5]
        trans_x = translation[:, 0]
        trans_y = translation[:, 1]
        trans_z = translation[:, 2]

        rot_mat_x = torch.stack(
            [
                torch.stack(
                    [
                        torch.ones_like(theta_x),
                        torch.zeros_like(theta_x),
                        torch.zeros_like(theta_x),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        torch.zeros_like(theta_x),
                        torch.cos(theta_x),
                        -torch.sin(theta_x),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [torch.zeros_like(theta_x), torch.sin(theta_x), torch.cos(theta_x)],
                    dim=1,
                ),
            ],
            dim=2,
        ).to(src.device)
        rot_mat_y = torch.stack(
            [
                torch.stack(
                    [torch.cos(theta_y), torch.zeros_like(theta_y), torch.sin(theta_y)],
                    dim=1,
                ),
                torch.stack(
                    [
                        torch.zeros_like(theta_y),
                        torch.ones_like(theta_x),
                        torch.zeros_like(theta_x),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        -torch.sin(theta_y),
                        torch.zeros_like(theta_y),
                        torch.cos(theta_y),
                    ],
                    dim=1,
                ),
            ],
            dim=2,
        ).to(src.device)
        rot_mat_z = torch.stack(
            [
                torch.stack(
                    [
                        torch.cos(theta_z),
                        -torch.sin(theta_z),
                        torch.zeros_like(theta_y),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [torch.sin(theta_z), torch.cos(theta_z), torch.zeros_like(theta_y)],
                    dim=1,
                ),
                torch.stack(
                    [
                        torch.zeros_like(theta_y),
                        torch.zeros_like(theta_y),
                        torch.ones_like(theta_x),
                    ],
                    dim=1,
                ),
            ],
            dim=2,
        ).to(src.device)

        scale_mat = torch.stack(
            [
                torch.stack(
                    [scale_x, torch.zeros_like(theta_z), torch.zeros_like(theta_y)],
                    dim=1,
                ),
                torch.stack(
                    [torch.zeros_like(theta_z), scale_y, torch.zeros_like(theta_y)],
                    dim=1,
                ),
                torch.stack(
                    [torch.zeros_like(theta_y), torch.zeros_like(theta_y), scale_z],
                    dim=1,
                ),
            ],
            dim=2,
        ).to(src.device)

        shear_mat = torch.stack(
            [
                torch.stack(
                    [
                        torch.ones_like(theta_x),
                        torch.tan(shear_xy),
                        torch.tan(shear_xz),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        torch.tan(shear_yx),
                        torch.ones_like(theta_x),
                        torch.tan(shear_yz),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        torch.tan(shear_zx),
                        torch.tan(shear_zy),
                        torch.ones_like(theta_x),
                    ],
                    dim=1,
                ),
            ],
            dim=2,
        ).to(src.device)

        trans = torch.stack([trans_x, trans_y, trans_z], dim=1).unsqueeze(dim=2)

        mat = torch.bmm(
            shear_mat,
            torch.bmm(
                scale_mat, torch.bmm(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))
            ),
        )

        inv_mat = torch.inverse(mat)
        mat = torch.cat([mat, trans], dim=-1)

        inv_trans = torch.bmm(-inv_mat, trans)
        inv_mat = torch.cat([inv_mat, inv_trans], dim=-1)

        grid = F.affine_grid(
            mat,
            [src.shape[0], 3, target_z_size, src.shape[3], src.shape[4]],
            align_corners=True,
        )
        warped_src = F.grid_sample(src, grid, align_corners=True, mode=self.mode)
        return warped_src, mat


class UNet(nn.Module):
    def __init__(self, in_channel, start_channel):
        self.in_channel = in_channel
        self.start_channel = start_channel

        bias_opt = True

        super(UNet, self).__init__()

        ## Backbone

        self.eninput = self.encoder(self.in_channel, self.start_channel, stride=(2,1,1), bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, stride=(2,2,2), bias=bias_opt)
        self.ec2 = self.encoder(
            self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt
        )
        self.ec3 = LK_encoder(
            self.start_channel * 2,
            self.start_channel * 2,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=bias_opt,
        )
        self.ec4 = self.encoder(
            self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt
        )
        self.ec5 = LK_encoder(
            self.start_channel * 4,
            self.start_channel * 4,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=bias_opt,
        )
        self.ec6 = self.encoder(
            self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt
        )
        self.ec7 = LK_encoder(
            self.start_channel * 8,
            self.start_channel * 8,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=bias_opt,
        )
        self.ec8 = self.encoder(
            self.start_channel * 8, self.start_channel * 8, stride=2, bias=bias_opt
        )
        self.ec9 = LK_encoder(
            self.start_channel * 8,
            self.start_channel * 8,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=bias_opt,
        )

        ## AffineHead
        ## (4 Heads)
        ## A Global Feature Integration Layer (gloFeaInt)
        ## and A Transformation Parameter Estimation Layer (transParaEst)

        # self.rot_gloFeaInt = nn.Linear(self.start_channel * 8 * 9 * 8 * 8, 100)
        self.rot_gloFeaInt = nn.Linear(self.start_channel * 8 * 9 * 8 * 8, 100)
        self.act_rot = nn.PReLU()
        self.rot_transParaEst = nn.Linear(100, 3)
        # self.rot_transParaEst = nn.Linear(10, 3)

        # self.scl_gloFeaInt = nn.Linear(self.start_channel * 8 * 9 * 8 * 8, 100)
        self.scl_gloFeaInt = nn.Linear(self.start_channel * 8 * 9 * 8 * 8, 100)
        self.act_scl = nn.PReLU()
        self.scl_transParaEst = nn.Linear(100, 3)
        # self.scl_transParaEst = nn.Linear(10, 3)

        # self.shear_gloFeaInt = nn.Linear(self.start_channel * 8 * 9 * 8 * 8, 100)
        self.shear_gloFeaInt = nn.Linear(self.start_channel * 8 * 9 * 8 * 8, 100)
        self.act_shear = nn.PReLU()
        self.shear_transParaEst = nn.Linear(100, 6)

        # self.trans_gloFeaInt = nn.Linear(self.start_channel * 8 * 9 * 8 * 8, 100)
        self.trans_gloFeaInt = nn.Linear(self.start_channel * 8 * 9 * 8 * 8, 100)
        self.act_trans = nn.PReLU()
        self.trans_transParaEst = nn.Linear(100, 3)

        self.affine_trans = AffineTransform()

    def encoder(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        batchnorm=False,
    ):
        layer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.PReLU(),
        )
        return layer

    def forward(self, x, y):

        # pad images to same shape
        p3d = (0, 0, 0, 0, 101, 100)
        x_new = F.pad(x, p3d, "constant", 0)
        # y_new = F.pad(y, p3d, "constant", 0)
        # print(f"---------------")
        # print(f"x new shape: {x_new.shape}")
        # print(f"y new shape: {y.shape}")

        x_in = torch.cat((x_new, y), 1)

        # x_in = torch.cat((x, y), 1)
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
        
        e5 = torch.flatten(e4, start_dim=1)
        
        rot = self.rot_gloFeaInt(e5)
        rot = self.act_rot(rot)
        rot = self.rot_transParaEst(rot)

        scl = self.scl_gloFeaInt(e5)
        scl = self.act_scl(scl)
        scl = self.scl_transParaEst(scl)
        
        

        shr = self.shear_gloFeaInt(e5)
        shr = self.act_shear(shr)
        shr = self.shear_transParaEst(shr)

        trans = self.trans_gloFeaInt(e5)
        trans = self.act_trans(trans)
        trans = self.trans_transParaEst(trans)

        rot = torch.clamp(rot, min=-1, max=1) * np.pi

        # scl = scl + 1
        scl = torch.clamp(scl, min=0, max=3)

        shr = torch.clamp(shr, min=-1, max=1) * np.pi

        # print(f"x shape: {x.shape}")
        warped_x, affine_mat = self.affine_trans(x, y.shape[2], rot, scl, shr, trans)
        # print(f"warped_x shape: {warped_x.shape}")
        
        return warped_x, affine_mat
    

"""
NCC is from SYMNet.
https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks/blob/master/Code/Models.py
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=9, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones(
            (1, 1, weight_win_size, weight_win_size, weight_win_size),
            device=I.device,
            requires_grad=False,
        )
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size / 2))
        J_sum = conv_fn(J, weight, padding=int(win_size / 2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size / 2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size / 2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size / 2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class SAD:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))


if __name__ == "__main__":
    from torchsummary import summary

    # The first para 2 is the input channel.
    # The second para 8 is the number of channels in first conv layers, enlarging the value returns larger networks.
    model = UNet(2, 8).cuda()
    # summary(model, [(1, 81, 128, 128), (1, 81, 128, 128)])
    summary(model, [(1, 358, 256, 256), (1, 559, 256, 256)])
