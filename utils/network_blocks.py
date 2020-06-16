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

from collections import OrderedDict as odict
import numpy as np
from math import sqrt
import random

from extern.network_blocks import EqualConv2d, AdaptiveInstanceNorm, EqualLinear, equal_lr

eps = 1e-6



class EqualConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv3d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)



class ConstantInput3(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size, size))

    def forward(self, batch_size):
        out = self.input.repeat(batch_size, 1, 1, 1, 1)

        return out



class AdaptiveInstanceNorm3(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm3d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out



class StyledConvBlock3(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput3(in_channel) # 3D constant representation
        else:
            self.conv1 = EqualConv3d(in_channel, out_channel, kernel_size, padding=padding) # 3D convolutions.

        self.adain1 = AdaptiveInstanceNorm3(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv3d(out_channel, out_channel, kernel_size, padding=padding)
        self.adain2 = AdaptiveInstanceNorm3(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, batch_size, style):
        out = self.conv1(batch_size)
        out = self.adain1(out, style)
        out = self.lrelu1(out)

        out = self.conv2(out)
        out = self.adain2(out, style)
        out = self.lrelu2(out)

        return out



class StyledConvBlock3_noAdaIN(nn.Module):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
    ):
        super().__init__()

        self.conv1 = EqualConv3d(in_channel, out_channel, kernel_size, padding=padding)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv3d(out_channel, out_channel, kernel_size, padding=padding)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input):

        out = self.conv1(input)
        out = self.lrelu1(out)

        out = self.conv2(out)
        out = self.lrelu2(out)

        return out


class projection_unit(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=1,
    ):
        super().__init__()

        self.conv = EqualConv2d(in_channel, out_channel, kernel_size, padding=0)
        self.lrelu = nn.PReLU(512)         

    def forward(self, input):
        batch = input.shape[0]
        out = input.view(batch, input.shape[1]*input.shape[2], input.shape[3], input.shape[4])
        out = self.conv(out)
        out = self.lrelu(out)

        return out   


class StyledConvBlock2(nn.Module):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
    ):
        super().__init__()
        self.conv1 = EqualConv2d(
            in_channel, out_channel, kernel_size, padding=padding
        )

        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style):
        out = self.conv1(input)
        out = self.adain1(out, style)
        out = self.lrelu1(out)

        out = self.conv2(out)
        out = self.adain2(out, style)
        out = self.lrelu2(out)

        return out      


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        pixel_norm=True,
        spectral_norm=False,
        instance_norm=False,
        last=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        if instance_norm and last==True:
            self.conv = nn.Sequential(
                EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
                nn.InstanceNorm2d(out_channel),
                nn.LeakyReLU(0.2),
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )
        elif instance_norm and last==False:
            self.conv = nn.Sequential(
                EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
                nn.InstanceNorm2d(out_channel),
                nn.LeakyReLU(0.2),
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.InstanceNorm2d(out_channel),
                nn.LeakyReLU(0.2),
            )            
        else:
            self.conv = nn.Sequential(
                EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
                nn.LeakyReLU(0.2),
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv(input)

        return out        


class negDotLoss:
    def __init_(self):
        pass

    def compute_loss(self, tgts, Pred, GT):
        Loss = odict()
        for tgt in tgts:
            Loss[tgt] = torch.mean(-torch.bmm(GT[tgt].view(GT[tgt].shape[0],1,2).float(), Pred[tgt].view(Pred[tgt].shape[0],2,1).float()))
        return Loss


class CELoss:
    def __init__(self):
        self.CELoss = nn.CrossEntropyLoss().cuda()

    def compute_loss(self, tgts, Pred, GT):
        Loss = odict()
        for tgt in tgts:
            Loss[tgt] = self.CELoss(Pred[tgt].view(Pred[tgt].size()[0],4), GT[tgt].view(Pred[tgt].size()[0],))
        return Loss