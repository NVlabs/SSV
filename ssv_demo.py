import torch
import os
import glob2
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import sys
import PIL
import math
from math import sin,cos
import pickle   
import argparse
from tqdm import tqdm 
from torchvision import datasets, transforms
import torch.nn as nn
from ssv import VPNet
from utils.ssv import pil_loader, draw_axis, drawPose
sys.path.append('data_preprocessing/mtcnn-pytorch')
from src import detect_faces
sys.path.append('extern')
from transformImage import transformImage
from data_preprocessing.preprocess_data import readCalibrationFile, readPoseFile, detect_face

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

transform = transforms.Compose(
    [   transforms.CenterCrop(192),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]) 

if __name__ == "__main__":

    code_size = 64
    vpnet = nn.DataParallel(VPNet((code_size*2), instance_norm=True)).cuda()
    vpnet.load_state_dict(torch.load('model/vpnet_ssv.pt'))
    vpnet.eval()

    linreg_a, linreg_e, linreg_t = pickle.load(open('model/linreg_coeff','rb'))

    session_dir = os.path.join('demo_images/')
    images = glob2.glob(session_dir + '/*rgb.png')
    calibration = readCalibrationFile(os.path.join(session_dir,'rgb.cal'))

    for image in tqdm(images[:10]):
        print(image)
        cv_orig_im = cv2.imread(image)
        mask = cv2.imread(os.path.join(image[:-8] + '_depth_mask.png'))
        fig = plt.figure()

        landmarks = detect_face(cv_orig_im, mask)
        if landmarks==[]:
            continue

        pose = readPoseFile(image[:-7]+'pose.txt')
        R = calibration['R']
        T = calibration['T']
        R_rgb = np.dot(pose['R'], R.transpose())
        T_rgb = np.dot(pose['T'].transpose(), R.transpose()).transpose() + np.dot(R, T)
                

        pil_im = Image.fromarray(cv2.cvtColor(cv_orig_im, cv2.COLOR_BGR2RGB))
        trans_pil_im, R_trans = transformImage(pil_im, cv_orig_im, landmarks)

        in_im = transform(trans_pil_im)
        op_cls, op_z, op_vp, _ = vpnet(in_im[None,:,:].cuda())
        vp = vpnet.module.compute_vp_pred(op_vp)
        pred_a = np.rad2deg(vp['a'].cpu().detach().numpy())
        pred_e = np.rad2deg(vp['e'].cpu().detach().numpy())
        pred_t = np.rad2deg(vp['t'].cpu().detach().numpy())  
        
        linreg_pred_a = linreg_a.predict(pred_a.reshape(pred_a.shape[0],-1))
        linreg_pred_e = linreg_e.predict(pred_e.reshape(pred_e.shape[0],-1))
        linreg_pred_t = linreg_t.predict(pred_t.reshape(pred_t.shape[0],-1))  

        pred_R = cv2.Rodrigues(np.asarray([linreg_pred_a, linreg_pred_e, linreg_pred_a]))[0]
        pred_R = pred_R.astype(np.float64)    

        cv_im = cv_orig_im.copy()
        imgn, proj_points = drawPose(cv_orig_im.copy(), R_rgb, T_rgb, calibration['intrinsics'], calibration['dist'])

        tdx_p, tdy_p, xp, yp, im_axes = draw_axis(cv_im,linreg_pred_a, linreg_pred_e, linreg_pred_t, tdx=proj_points[0,0,0], tdy=proj_points[0,0,1])
        dx_p = xp-tdx_p
        dy_p = yp-tdy_p

        plt.imshow(cv_im[:,:,::-1]) 
        plt.arrow(tdx_p, tdy_p, dx_p[0], dy_p[0], head_width=5, head_length=2.5, linewidth=5, color='#B21117', length_includes_head=True,rasterized=None, overhang=0)
        plt.arrow(tdx_p, tdy_p, dx_p[1], dy_p[1], head_width=5, head_length=2.5, linewidth=5, color='#5A9E27', length_includes_head=True,rasterized=None, overhang=0)
        plt.arrow(tdx_p, tdy_p, dx_p[2], dy_p[2], head_width=5, head_length=2.5, linewidth=5, color='#1F58A8', length_includes_head=True,rasterized=None, overhang=0) 
        plt.axis('off')
        fig.subplots_adjust(left=0,bottom=0, right=1, top=1)
        fpath = 'demo_images/plots'
        os.makedirs(fpath, exist_ok=True)
        plt.savefig(os.path.join(fpath,image[12:]), dpi=200)
        plt.close()