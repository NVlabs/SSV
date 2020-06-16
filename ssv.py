# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Siva Karthik Mustikovela.
# --------------------------------------------------------

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import torch.nn.utils.spectral_norm as spectralnorm
from torch.autograd import Variable

from easydict import EasyDict as edict
from collections import OrderedDict as odict
from utils import ssv
import numpy as np
from math import sqrt
import random
import code
from utils.network_blocks import *
from extern.network_blocks import PixelNorm

eps = 1e-6

# Generator module for VP aware synthesizer
class VPASGenerator(nn.Module):
    def __init__(self, code_dim):
        super().__init__()

        self.progression1 = nn.ModuleList(
            [
                StyledConvBlock3(512, 512, 3, 1, style_dim=code_dim, initial=True),
                StyledConvBlock3(512, 512, 3, 1, style_dim=code_dim,),
                StyledConvBlock3(512, 256, 3, 1, style_dim=code_dim,),
            ]
        )

        self.progression2 = nn.ModuleList(
            [
                StyledConvBlock3_noAdaIN(256, 128, 3, 1),
                StyledConvBlock3_noAdaIN(128, 64, 3, 1),
            ]
        )

        self.projection_unit = projection_unit(64*16, 64*16)

        self.scb1 = StyledConvBlock2(1024, 512, 3, 1, style_dim=code_dim)
        self.scb2 = StyledConvBlock2(512, 512, 3, 1, style_dim=code_dim)
        self.scb3 = StyledConvBlock2(512, 256, 3, 1, style_dim=code_dim)
        self.scb4 = StyledConvBlock2(256, 128, 3, 1, style_dim=code_dim)

        self.to_rgb = EqualConv2d(128, 3, 1) 

    def forward(self, style, rots, batch_size):

        for i,conv in enumerate(self.progression1):
            if i==0:
                out = conv(batch_size, style[0])
            else:
                upsample = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)
                out = conv(upsample, style[0])
 
        flow = F.affine_grid(rots, torch.Size([batch_size, 256, 16, 16, 16]))  
        out = F.grid_sample(out, flow) 

        for i,conv in enumerate(self.progression2):
            out = conv(out)

        out = self.projection_unit(out)

        out = self.scb1(out,style[1]) 
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.scb2(out,style[1]) 
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.scb3(out,style[1]) 
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.scb4(out,style[1]) 

        out = self.to_rgb(out)

        return out

# Viewpoint aware synthesizer
class VPAwareSynthesizer(nn.Module):
    def __init__(self, code_dim=128, n_mlp=8):
        super().__init__()

        # Generator network
        self.generator = VPASGenerator(code_dim)

        # Style network
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, input, rots=None):

        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        return self.generator(styles, rots, batch_size=input[0].shape[0])


# Generator module for VP aware synthesizer
class VPASGenerator_ssv(nn.Module):
    def __init__(self, code_dim):
        super().__init__()

        self.progression1 = nn.ModuleList(
            [
                StyledConvBlock3(512, 512, 3, 1, style_dim=code_dim, initial=True),
                StyledConvBlock3(512, 256, 3, 1, style_dim=code_dim,),
                StyledConvBlock3(256, 128, 3, 1, style_dim=code_dim,),
            ]
        )

        self.progression2 = nn.ModuleList(
            [
                StyledConvBlock3_noAdaIN(128, 64, 3, 1),
                StyledConvBlock3_noAdaIN(64, 64, 3, 1),
            ]
        )

        self.projection_unit = projection_unit(64*16, 512)

        self.scb1 = StyledConvBlock2(512, 256, 3, 1, style_dim=code_dim)
        self.scb2 = StyledConvBlock2(256, 64, 3, 1, style_dim=code_dim)
        self.scb3 = StyledConvBlock2(64, 32, 3, 1, style_dim=code_dim)

        self.to_rgb = EqualConv2d(32, 3, 1) 

    def forward(self, style, rots, batch_size):

        for i,conv in enumerate(self.progression1):
            if i==0:
                out = conv(batch_size, style[0])
            else:
                upsample = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)
                out = conv(upsample, style[0])
 
        flow = F.affine_grid(rots, torch.Size([batch_size, 256, 16, 16, 16]))  
        out = F.grid_sample(out, flow, mode='nearest') 

        for i,conv in enumerate(self.progression2):
            out = conv(out)

        out = self.projection_unit(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.scb1(out,style[1]) 
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.scb2(out,style[1]) 
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.scb3(out,style[1]) 
        out = self.to_rgb(out)

        return out

# Viewpoint aware synthesizer
class VPAwareSynthesizer_ssv(nn.Module):
    def __init__(self, code_dim=128, n_mlp=8):
        super().__init__()

        # Generator network
        self.generator = VPASGenerator_ssv(code_dim)

        # Style network
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, input, rots=None):

        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        return self.generator(styles, rots, batch_size=input[0].shape[0])




# Viewpoint network
class VPNet(nn.Module):

    @staticmethod
    def head_seq(in_size, num_fc=1024, init_weights=True):      
        """
        Creates a head with fc layer and outputs for magnitute of [sine, cosine] and direction {--, -+, +-, ++}
        """
        seq_fc8 = nn.Sequential(
                        EqualLinear(in_size, num_fc),                             
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                    )
        seq_ccss= EqualLinear(num_fc, 2)                     # magnitude of sin and cos
        seq_sgnc= EqualLinear(num_fc, 4)                     # direction/quadrant of sine and cos

        return seq_fc8, seq_ccss, seq_sgnc

    def __init__(self, code_dim=128, instance_norm=False):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1, instance_norm=instance_norm),
                ConvBlock(32, 64, 3, 1, instance_norm=instance_norm),
                ConvBlock(64, 128, 3, 1, instance_norm=instance_norm),
                ConvBlock(128, 256, 3, 1, instance_norm=instance_norm),
                ConvBlock(256, 512, 3, 1, instance_norm=instance_norm),
                ConvBlock(512, 512, 3, 1, instance_norm=instance_norm),
                ConvBlock(512, 512, 3, 1, instance_norm=instance_norm),
                ConvBlock(512, 512, 3, 1, instance_norm=instance_norm),
                ConvBlock(513, 512, 3, 1, 4, 0, last=True, instance_norm=instance_norm),
            ]
        )

        self.from_rgb = nn.ModuleList(
            [
                EqualConv2d(3, 16, 1),
                EqualConv2d(3, 32, 1),
                EqualConv2d(3, 64, 1),
                EqualConv2d(3, 128, 1),
                EqualConv2d(3, 256, 1),
                EqualConv2d(3, 512, 1),
                EqualConv2d(3, 512, 1),
                EqualConv2d(3, 512, 1),
                EqualConv2d(3, 512, 1),
             
            ]
        )

        self.n_layer = len(self.progression)

        #  FC layer for classification
        self.class_linear = EqualLinear(512, 1)

        # FC layer for reconstruction of z
        self.z_linear = EqualLinear(512, code_dim)

        # Head for viewpoint estimation
        self.head_fc_a, self.head_x2_y2_mag_a, self.head_sin_cos_direc_a = self.head_seq(512, num_fc=256)
        self.head_fc_e, self.head_x2_y2_mag_e, self.head_sin_cos_direc_e = self.head_seq(512, num_fc=256)
        self.head_fc_t, self.head_x2_y2_mag_t, self.head_sin_cos_direc_t = self.head_seq(512, num_fc=256)

        # For the magnitute part
        self.logsoftmax = nn.LogSoftmax(dim=2).cuda()  

        # Setup the loss here. 
        self.loss_mag = negDotLoss()
        self.loss_direc = CELoss()
        self.balance_weight = 1.0

    def forward(self, input):
        step = 5
        alpha = 0
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    skip_rgb = self.from_rgb[index + 1](input)
                    out = F.interpolate(skip_rgb, scale_factor=0.5, mode='bilinear', align_corners=False)

        # Output from trunk.
        trunk_out = out.squeeze(2).squeeze(2)
        out = out.squeeze(2).squeeze(2)

        batchsize = out.size(0)

        # Outputs
        # Class output
        class_out = self.class_linear(out)

        # z output
        z_out = self.z_linear(out)

        # Now the viewpoint part
        x_a = self.head_fc_a(out)
        x_e = self.head_fc_e(out)
        x_t = self.head_fc_t(out)

        # Get magnitude outputs {MAGNITUDE}        
        mag_x2_y2_a = self.head_x2_y2_mag_a(x_a).view(batchsize, 1, 2)
        mag_x2_y2_e = self.head_x2_y2_mag_e(x_e).view(batchsize, 1, 2)        
        mag_x2_y2_t = self.head_x2_y2_mag_t(x_t).view(batchsize, 1, 2)        

        # Log Softmax on mag outputs {MAGNITUDE}
        logsoftmax_x2_y2_a = self.logsoftmax(mag_x2_y2_a)
        logsoftmax_x2_y2_e = self.logsoftmax(mag_x2_y2_e)
        logsoftmax_x2_y2_t = self.logsoftmax(mag_x2_y2_t)

        # Signs/Directions of outputs {SIGN}
        sign_x_y_a = self.head_sin_cos_direc_a(x_a).view(batchsize,1,4)
        sign_x_y_e = self.head_sin_cos_direc_e(x_e).view(batchsize,1,4)
        sign_x_y_t = self.head_sin_cos_direc_t(x_t).view(batchsize,1,4)

        viewpoint_op = odict(# log probability of xx, yy   (xx+yy=1 or x^2+y^2=1)
                            logprob_xxyy = odict(   a = logsoftmax_x2_y2_a,
                                                    e = logsoftmax_x2_y2_e,
                                                    t = logsoftmax_x2_y2_t,),
                            sign_x_y =      odict(  a = sign_x_y_a,
                                                    e = sign_x_y_e,
                                                    t = sign_x_y_t,))
       

        return class_out, z_out, viewpoint_op, trunk_out

    def compute_vp_loss(self, pred, GT):
        """
        Compute loss for magnitude heads using negdot
        Compute loss for direction heads using crossentropy
        """
        Loss_c2s2 = self.loss_mag.compute_loss(['a','e','t'], pred['logprob_xxyy'], dict(a=GT['ccss_a'],e=GT['ccss_e'],t=GT['ccss_t']))
        Loss_direc = self.loss_direc.compute_loss(['a','e','t'], pred['sign_x_y'], dict(a=GT['sign_a'],e=GT['sign_e'],t=GT['sign_t']))
        Loss = odict( ccss_a=Loss_c2s2['a'] * self.balance_weight,
                      ccss_e=Loss_c2s2['e'] * self.balance_weight,
                      ccss_t=Loss_c2s2['t'] * self.balance_weight,
                      #
                      sign_a=Loss_direc['a'],
                      sign_e=Loss_direc['e'],
                      sign_t=Loss_direc['t'],)
        return Loss


    @staticmethod
    def compute_vp_pred(network_op):
        lmap = torch.FloatTensor([[ 1, 1],
                                         [ 1,-1],
                                         [-1, 1],
                                         [-1,-1]])
        lmap = Variable(lmap).cuda()
        bsize = network_op['logprob_xxyy']['a'].size(0)

        vp_pred = odict()
        for tgt in network_op['logprob_xxyy'].keys():
            # Get the magnitude from outputs
            logprob_xx_yy    = network_op['logprob_xxyy'][tgt]
            abs_cos_sin  = torch.sqrt(torch.exp(logprob_xx_yy))
            vp_pred['ccss_'+tgt] = torch.exp(logprob_xx_yy)

            # Get the direction from outputs
            sign_ind = torch.argmax(network_op['sign_x_y'][tgt].view(network_op['sign_x_y'][tgt].shape[0],4), dim=1)
            vp_pred['sign_'+tgt] = sign_ind
            i_inds    = torch.from_numpy(np.arange(bsize)).cuda()
            direc_cos_sin = lmap.expand(bsize,4,2)[i_inds, sign_ind]
            cos_sin      = abs_cos_sin.view(abs_cos_sin.shape[0],2)*direc_cos_sin
            vp_pred[tgt]    = torch.atan2(cos_sin[:,1], cos_sin[:,0]) #
        return vp_pred  

    @staticmethod
    def compute_gt_flip(network_op, dtach=False):
        """
        Takes a prediction for an image and computes the GT for the corresponding flipped image. 
        For a flipped image, the magnitude of azimuth, elevation and tilt have to be the same.
        The signs/ directions for azimuth and tilt are flipped.
        So, for correct image : [ a,  e,  t] (from the input)
        For flipped image :     [-a,  e, -t] (produce GT representation for this)
        MAP :
        +, +  ->  +, - | 0 -> 1
        +, -  ->  +, + | 1 -> 0
        -, +  ->  -, - | 2 -> 3
        -, -  ->  -, + | 3 -> 2  
        """
        lmap = torch.FloatTensor([[ 1, 1],
                                            [ 1,-1],
                                            [-1, 1],
                                            [-1,-1]])

        lmap = Variable(lmap).cuda()
        batchsize = network_op['logprob_xxyy']['a'].size(0)

        vp_pred = edict()
        for tgt in network_op['logprob_xxyy'].keys():
            # Get the magnitude from outputs
            logprob_xx_yy    = network_op['logprob_xxyy'][tgt]
            abs_cos_sin  = torch.sqrt(torch.exp(logprob_xx_yy))

            vp_pred['ccss_'+tgt] = torch.exp(logprob_xx_yy)

            # Get the direction from outputs
            sign_ind = torch.argmax(network_op['sign_x_y'][tgt].view(network_op['sign_x_y'][tgt].shape[0],4), dim=1)
            if tgt=='a' or tgt=='t':
                sign_ind_flipped = (1 - sign_ind%2)+ (2*(sign_ind//2))
            else:
                sign_ind_flipped = sign_ind
            vp_pred['sign_'+tgt] = sign_ind_flipped
            item_inds    = torch.from_numpy(np.arange(batchsize)).cuda()
            sign_cos_sin = lmap.expand(batchsize,4,2)[item_inds, sign_ind]
            cos_sin      = abs_cos_sin.view(abs_cos_sin.shape[0],2)*sign_cos_sin
            vp_pred[tgt]    = torch.atan2(cos_sin[:,1], cos_sin[:,0]) #
        return vp_pred    
