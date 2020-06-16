# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Siva Karthik Mustikovela.
# --------------------------------------------------------

import numpy as np
import random
from PIL import Image
import cv2
import math
from math import sin,cos


import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torchvision
from torchvision import models

from collections import OrderedDict as odict
import os
import imageio


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return np.asarray(img.convert('RGB')).copy()


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class AlexNetConv4(nn.Module):
    def __init__(self):
        super(AlexNetConv4, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = nn.Sequential(
            # stop at conv4
            *list(original_model.features.children())[:4]
        )
    def forward(self, x):
        x = self.features(x)
        return x

        
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size=2, image_size=128, num_workers=2):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, drop_last=True, batch_size=batch_size, num_workers=num_workers)
    return loader


def get_az_rots(b_size, az_range):
    """
    Get azimuth angle rotation matrices.
    """
    azs = torch.FloatTensor(1,b_size).uniform_(-az_range,az_range).cuda()
    cos_azs = torch.cos(azs)
    sin_azs = torch.sin(azs)
    rot_azs = torch.zeros(b_size,3,4).cuda()
    rot_azs[:,0,0] = cos_azs
    rot_azs[:,0,2] = sin_azs
    rot_azs[:,1,1] = 1
    rot_azs[:,2,0] = -sin_azs
    rot_azs[:,2,2] = cos_azs
    rot_azs = rot_azs.float()  
    azs = azs.view(b_size,1)
    cos_azs = torch.cos(azs)
    sin_azs = torch.sin(azs)    

    ccss_az = torch.cat((cos_azs**2, sin_azs**2), 1)

    sign_cls_az =    (((cos_azs>=0) & (sin_azs>=0))*0) \
                + (((cos_azs>=0) & (sin_azs<0))*1) \
                + (((cos_azs<0) & (sin_azs>=0))*2) \
                + (((cos_azs<0) & (sin_azs<0))*3)
    sign_cls_az = sign_cls_az.type(torch.long)

    return rot_azs, azs, ccss_az, sign_cls_az

def get_el_rots(b_size, el_range):
    """
    Get elevation angle rotation matrices.
    """

    els = torch.FloatTensor(1,b_size).uniform_(-el_range,el_range).cuda()
    cos_els = torch.cos(els)
    sin_els = torch.sin(els)
    rot_els = torch.zeros(b_size,3,4).cuda()
    rot_els[:,0,0] = 1
    rot_els[:,1,1] = cos_els
    rot_els[:,1,2] = -sin_els
    rot_els[:,2,1] = sin_els
    rot_els[:,2,2] = cos_els
    rot_els = rot_els.float()  
    els = els.view(b_size,1)
    cos_els = torch.cos(els)
    sin_els = torch.sin(els)    
    
    ccss_el = torch.cat((cos_els**2, sin_els**2), 1)

    sign_cls_el = (((cos_els>=0) & (sin_els>=0))*0) \
                + (((cos_els>=0) & (sin_els<0))*1) \
                + (((cos_els<0) & (sin_els>=0))*2) \
                + (((cos_els<0) & (sin_els<0))*3)
    sign_cls_el = sign_cls_el.type(torch.long)

    return rot_els, els, ccss_el, sign_cls_el


def get_ct_rots(b_size, ct_range):
    """
    Get elevation angle rotation matrices.
    """

    cts = torch.FloatTensor(1,b_size).uniform_(-ct_range,ct_range).cuda()
    cos_cts = torch.cos(cts)
    sin_cts = torch.sin(cts)
    rot_cts = torch.zeros(b_size,3,4).cuda()
    rot_cts[:,0,0] = cos_cts
    rot_cts[:,0,1] = -sin_cts
    rot_cts[:,1,0] = sin_cts
    rot_cts[:,1,1] = cos_cts
    rot_cts[:,2,2] = 1
    rot_cts = rot_cts.float()  
    cts = cts.view(b_size,1)
    cos_cts = torch.cos(cts)
    sin_cts = torch.sin(cts)    
    
    ccss_ct = torch.cat((cos_cts**2, sin_cts**2), 1)

    sign_cls_ct = (((cos_cts>=0) & (sin_cts>=0))*0) \
                + (((cos_cts>=0) & (sin_cts<0))*1) \
                + (((cos_cts<0) & (sin_cts>=0))*2) \
                + (((cos_cts<0) & (sin_cts<0))*3)
    sign_cls_ct = sign_cls_ct.type(torch.long)

    return rot_cts, cts, ccss_ct, sign_cls_ct


def get_az_el_ct_rots(b_size, az_range, el_range, ct_range):
    """
    Get multiplied rotation matrices of azs, els and cts.
    rot_mat =( az * el )* ct
    """
    rot_azs, azs, ccss_azs, sign_cls_azs = get_az_rots(b_size, az_range)
    rot_els, els, ccss_els, sign_cls_els = get_el_rots(b_size, el_range)
    rot_cts, cts, ccss_cts, sign_cls_cts = get_ct_rots(b_size, ct_range)
    rot_mats = torch.bmm(rot_azs[:,:,0:3],rot_els[:,:,0:3])
    rot_mats_final = torch.cat( (torch.bmm(rot_mats[:,:,0:3],rot_cts[:,:,0:3]), torch.zeros((b_size,3,1)).cuda()), 2)
    vp_biternion = odict(
                    ccss_a = ccss_azs.cuda(),
                    ccss_e = ccss_els.cuda(),
                    ccss_t = ccss_cts.cuda(),
                    sign_a = sign_cls_azs.cuda(),
                    sign_e = sign_cls_els.cuda(),
                    sign_t = sign_cls_cts.cuda(),)

    return rot_mats_final, azs.cuda(), els.cuda(), cts.cuda(), vp_biternion 


def gen_az_rots(b_size,az=None):
    """
    Get azimuth angle rotation matrices.
    """
    if az is None:
        print('az is None')
        azs = torch.FloatTensor(1,b_size).uniform_(-2,2)
    else:
        # print(az)
        azs = torch.ones(1,b_size).cuda()*az
        azs = azs.float()
    cos_azs = torch.cos(azs)
    sin_azs = torch.sin(azs)
    rot_azs = torch.zeros(b_size,3,4).cuda()
    rot_azs[:,0,0] = cos_azs
    rot_azs[:,0,2] = sin_azs
    rot_azs[:,1,1] = 1
    rot_azs[:,2,0] = -sin_azs
    rot_azs[:,2,2] = cos_azs
    rot_azs = rot_azs.float()  
    return rot_azs

def gen_el_rots(b_size,el=None):
    """
    Get azimuth angle rotation matrices.
    """
    if el is None:
        print('el is None')
        els = torch.FloatTensor(1,b_size).uniform_(-2,2)
    else:
        # print(el)
        els = torch.ones(1,b_size).cuda()*el
        els = els.float()
    cos_els = torch.cos(els)
    sin_els = torch.sin(els)
    rot_els = torch.zeros(b_size,3,4).cuda()
    rot_els[:,0,0] = 1
    rot_els[:,1,1] = cos_els
    rot_els[:,1,2] = -sin_els
    rot_els[:,2,1] = sin_els
    rot_els[:,2,2] = cos_els
    rot_els = rot_els.float()    
    return rot_els    

def gen_ct_rots(b_size,ct=None):
    """
    Get camera tilt angle rotation matrices.
    """
    if ct is None:
        print('ct is None')
        cts = torch.FloatTensor(1,b_size).uniform_(-2,2)
    else:
        # print(el)
        cts = torch.ones(1,b_size).cuda()*ct
        cts = cts.float()
    cos_cts = torch.cos(cts)
    sin_cts = torch.sin(cts)
    rot_cts = torch.zeros(b_size,3,4).cuda()
    rot_cts[:,0,0] = cos_cts
    rot_cts[:,0,1] = -sin_cts
    rot_cts[:,1,0] = sin_cts
    rot_cts[:,1,1] = cos_cts
    rot_cts[:,2,2] = 1
    rot_cts = rot_cts.float()      
    return rot_cts


def generate_samples(generator, azs, els, cts, args, itr, tag, z_sampling='uniform', num_samples=16):

    os.makedirs(os.path.join(args.exp_root,args.exp_name,'gen_samples',str(itr)), exist_ok=True)
    os.makedirs(os.path.join(args.exp_root,args.exp_name,'gen_samples',str(itr), tag), exist_ok=True)

    gen_in11, gen_in12 = torch.FloatTensor(2, num_samples, args.code_size).uniform_(-1,1).chunk(2, 0)  
    gen_in11 = gen_in11.cuda(); gen_in12 = gen_in12.cuda()     
    style_code = [gen_in11.squeeze(0), gen_in12.squeeze(0)]

    ind=0

    for ct in cts:
        rot_cts_test = gen_ct_rots(num_samples, ct)
        for el in els:    
            rot_els_test = gen_el_rots(num_samples, el)
            for az in azs:
                rot_azs_test = gen_az_rots(num_samples, az)
                rot_mats = torch.bmm(rot_azs_test[:,:,0:3],rot_els_test[:,:,0:3])
                rot_azs_test[:,:,0:3] = rot_mats            

                rot_mats = torch.bmm(rot_azs_test[:,:,0:3],rot_cts_test[:,:,0:3])
                rot_azs_test[:,:,0:3] = rot_mats    

                image = generator(style_code, rot_azs_test.cuda())
                ind+=1
                img_name = os.path.join(args.exp_root, args.exp_name, 'gen_samples', str(itr), tag, str(ind)+'.png')
                utils.save_image(image, img_name, nrow=4, normalize=True, range=(-1, 1))

    gif_images = []
    for i in range(1,ind):
        img_name = os.path.join(args.exp_root, args.exp_name, 'gen_samples', str(itr), tag, str(i)+'.png')
        gif_images.append(imageio.imread(img_name))
    imageio.mimsave(os.path.join(args.exp_root, args.exp_name, 'gen_samples', str(itr), tag, 'gen_gif.gif'), gif_images)        


def drawPose(img, r, t, cam, dist):
    modelAxes = np.array([
        np.array([0., -20., -40.]).reshape(1,3),
        np.array([50., -20., 0.]).reshape(1,3),
        np.array([0., -70., 0.]).reshape(1,3),
        np.array([0., -20., -50.]).reshape(1,3)
    ])

    projAxes, jac = cv2.projectPoints(modelAxes, r, t, cam, dist)

    cv2.line(img, (int(projAxes[0,0,0]), int(projAxes[0,0,1])), \
             (int(projAxes[1,0,0]), int(projAxes[1,0,1])), (0,255,255), 2)
    cv2.line(img, (int(projAxes[0,0,0]), int(projAxes[0,0,1])), \
             (int(projAxes[2,0,0]), int(projAxes[2,0,1])), (255,0,255), 2)
    cv2.line(img, (int(projAxes[0,0,0]), int(projAxes[0,0,1])), \
             (int(projAxes[3,0,0]), int(projAxes[3,0,1])), (255,255,0), 2)
             
    return img, projAxes


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 40):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    # return img
    return tdx, tdy, np.array([x1, x2, x3]), np.array([y1, y2, y3]), img     


class Saver():
	def __init__(self, opt):
		self.opt = opt
		self.model_list = {} 
		self.modeldir = os.path.join(opt.exp_root,opt.exp_name, 'checkpoint')
		os.makedirs(self.modeldir,exist_ok=True)

	def add_model(self, modelname, model):
		self.model_list[modelname] = model 

	def save_model(self, modelname, epoch):
		torch.save(self.model_list[modelname].state_dict(), 
					os.path.join(self.modeldir,'%s_%08i_%s_checkpoint.pt'%(modelname, epoch, self.opt.model_name)) ) 
	
	def save_all_models(self,epoch):
		for modelname in self.model_list.keys():
			self.save_model(modelname, epoch)     