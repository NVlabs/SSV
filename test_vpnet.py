# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Siva Karthik Mustikovela.
# --------------------------------------------------------

import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

import os
import random
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm

from ssv import VPNet
from utils.dataset import ImageFolderWithPaths

import code

os.environ['CUDA_VISIBLE_DEVICES']="1,"

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--model_path', type=str)

args = parser.parse_args()


test_data_root = args.data_dir
code_size = 64
vpnet = nn.DataParallel(VPNet((code_size*2), instance_norm=True)).cuda()
vpnet.load_state_dict(torch.load(args.model_path))
vpnet.eval()

transform = transforms.Compose(
    [   transforms.CenterCrop(192),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]) 

dataset = ImageFolderWithPaths(test_data_root, transform=transform)
test_data_loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

pred_a = []; pred_e = []; pred_t = []
gt_a = []; gt_e = []; gt_t = []
fnames = []
for i, (sampled_batch, _, fname) in tqdm(enumerate(test_data_loader)):
    fname = fname[0]
    op_cls, op_z, op_vp, _ = vpnet(sampled_batch.cuda())

    vp = vpnet.module.compute_vp_pred(op_vp)

    pred_a.append(vp['a'].cpu().detach().numpy())
    pred_e.append(vp['e'].cpu().detach().numpy())
    pred_t.append(vp['t'].cpu().detach().numpy())
    
    gt_a.append(float(fname.split('/')[-1].split('_')[6]))
    gt_e.append(float(fname.split('/')[-1].split('_')[4]))
    gt_t.append(float(fname.split('/')[-1].split('_')[8]))
    
    fnames.append(fname)

pred_a = np.rad2deg(np.asarray(pred_a))
pred_e = np.rad2deg(np.asarray(pred_e))
pred_t = np.rad2deg(np.asarray(pred_t))

gt_a = np.asarray(gt_a)
gt_e = np.asarray(gt_e)
gt_t = np.asarray(gt_t)

sample_inds = np.load('model/lin_sample_inds.npy')
mae=0

az_x = pred_a[sample_inds]
az_y = gt_a[sample_inds]
linreg = LinearRegression().fit(az_x.reshape(az_x.shape[0],1), az_y.reshape(az_x.shape[0],1))         
linreg_pred_a = linreg.predict(pred_a.reshape(gt_a.shape[0],-1))
linreg_pred_a = np.delete(linreg_pred_a, sample_inds)
gt_a = np.delete(gt_a, sample_inds)
err_a = (abs(linreg_pred_a - gt_a)).mean()
print('Azimuth error: ',err_a)
mae+=err_a

el_x = pred_e[sample_inds]
el_y = gt_e[sample_inds]
linreg = LinearRegression().fit(el_x.reshape(el_x.shape[0],1), el_y.reshape(el_x.shape[0],1))         
linreg_pred_e = linreg.predict(pred_e.reshape(gt_e.shape[0],-1))
linreg_pred_e = np.delete(linreg_pred_e, sample_inds)
gt_e = np.delete(gt_e, sample_inds)
err_e = (abs(linreg_pred_e - gt_e)).mean()
print('Elevation error: ',err_e)
mae+=err_e

ct_x = pred_t[sample_inds]
ct_y = gt_t[sample_inds]
linreg = LinearRegression().fit(ct_x.reshape(ct_x.shape[0],1), ct_y.reshape(ct_x.shape[0],1)) 
linreg_pred_t = linreg.predict(pred_t.reshape(gt_t.shape[0],-1))
linreg_pred_t = np.delete(linreg_pred_t, sample_inds)
gt_t = np.delete(gt_t, sample_inds)

err_t = (abs(linreg_pred_t - gt_t)).mean()
print('Tilt error: ',err_t)
mae+=err_t

print('MAE: ',mae/3)