# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Shalini De Mello.
# --------------------------------------------------------

import os
import glob2
import numpy as np
import cv2
from PIL import Image
import sys
import PIL
import scipy
import scipy.ndimage
import scipy.io as sio
import math
import argparse
import pandas as pd
import pickle
from tqdm import tqdm 
sys.path.append('./mtcnn-pytorch')
from src import detect_faces
sys.path.append('../extern')
from transformImage import transformImage


def readCalibrationFile(calibration_file):
    """
    Reads the calibration parameters
    """
    cal  = {}
    fh = open(calibration_file, 'r')
    # Read the [intrinsics] section
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(' ')])
    cal['intrinsics'] = np.array(vals).reshape(3,3)
    
    # Read the [intrinsics] section
    fh.readline().strip()
    vals = []
    vals.append([float(val) for val in fh.readline().strip().split(' ')])
    cal['dist'] = np.array(vals).reshape(4,1)
    
    # Read the [R] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(' ')])
    cal['R'] = np.array(vals).reshape(3,3)
    
     # Read the [T] section
    fh.readline().strip()
    vals = []
    vals.append([float(val) for val in fh.readline().strip().split(' ')])
    cal['T'] = np.array(vals).reshape(3,1)

    # Read the [resolution] section
    fh.readline().strip()
    cal['size'] = [int(val) for val in fh.readline().strip().split(' ')]
    cal['size'] = cal['size'][0], cal['size'][1]    
    
    fh.close()
    return cal

    
def readPoseFile(pose_file):
    """
    Reads the calibration parameters
    """
    pose  = {}
    fh = open(pose_file, 'r')

    # Read the [R] section
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(' ')])
    pose['R'] = np.array(vals).reshape(3,3)
    
     # Read the [T] section
    fh.readline().strip()
    vals = []
    vals.append([float(val) for val in fh.readline().strip().split(' ')])
    pose['T'] = np.array(vals).reshape(3,1)
    fh.close()
    
    return pose

def detect_face(cv_orig_im, mask):
    final_landmarks = []
    x,y,w,h = cv2.boundingRect(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    top = int(max(math.ceil(y-h*0.1), 0))
    bottom = int(max(math.floor(y+h*0.1), cv_orig_im.shape[0]))
    left = int(min(math.ceil(x-w*0.1), 0))
    right = int(max(math.floor(x+w*0.1), cv_orig_im.shape[1]))
    pil_im = Image.fromarray(cv2.cvtColor(cv_orig_im[top:bottom,left:right,:], cv2.COLOR_BGR2RGB))
        
    bboxes, landmarks = detect_faces(pil_im, in_weights_dir = 'data_preprocessing/mtcnn-pytorch/src/weights')

    if len(bboxes)>1:
        max_prob = 0
        max_idx = 0
        for j in range(len(bboxes)):
            if bboxes[j,4] > max_prob:
                max_prob = bboxes[j,4]
                max_idx = j
        final_bbox = bboxes[max_idx,:]
        if max_prob>0.91:
            final_landmarks = landmarks[max_idx,:]
    else:
        if bboxes[0,4]>0.91:
            final_bbox = bboxes
            final_landmarks = landmarks

    if final_landmarks==[]:
        return final_landmarks
        
    final_bbox = np.squeeze(final_bbox)
    final_bbox[0] += left
    final_bbox[2] += left
    final_bbox[1] += top
    final_bbox[3] += top
    
    landmarks = np.array(final_landmarks).reshape(2,5).transpose() 
    landmarks[:,0] += left
    landmarks[:,1] += top   

    return landmarks     


parser = argparse.ArgumentParser(description='Pre-process data')

parser.add_argument('--weights-dir', type=str, default='mtcnn-pytorch/src/weights',
                    help='directory in which the pretrained face detector weights are saved')
parser.add_argument('--dataset', type=str, default='BIWI',
                    help='dataset, default: BIWI, options: BIWI/300W_LP')
parser.add_argument('--src-dir', type=str, default='../data',
                    help='path to raw images')
parser.add_argument('--dst-dir', type=str, default='./cropped1024x1024',
                    help='destination directory')
parser.add_argument('--start', default=0, type=int, help='starting index')
parser.add_argument('--end', default=0, type=int, help='ending index')

args = parser.parse_args()
                    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"



if __name__ == "__main__":
    
    if args.dataset == '300WLP':

        images = open(args.src_dir + '/images.txt').readlines()
        for image in tqdm(images[args.start:args.end]):
            cv_orig_im = cv2.imread(os.path.join(args.src_dir, image[:-1]))
            pil_im = Image.fromarray(cv2.cvtColor(cv_orig_im, cv2.COLOR_BGR2RGB))
            bboxes, landmarks = detect_faces(pil_im, min_face_size = 30, in_weights_dir = args.weights_dir)   
            if len(bboxes) < 1:
                print('no face found')
                continue
            elif len(bboxes)>1:
                max_prob = 0
                max_idx = 0
                for j in range(len(bboxes)):
                    if bboxes[j,4] > max_prob:
                        max_prob = bboxes[j,4]
                        max_idx = j
                final_bbox = bboxes[max_idx,:]
                final_landmarks = landmarks[max_idx,:]
            else:
                final_bbox = bboxes
                final_landmarks = landmarks
            final_bbox = np.squeeze(final_bbox)
            
            landmarks = np.array(final_landmarks).reshape(2,5).transpose()
            
            if not os.path.exists(os.path.join(args.src_dir, image[:-4]+ 'mat')):
                print('no mat file')
                continue       
            
            data = sio.loadmat(os.path.join(args.src_dir, image[:-4]+ 'mat'))
            R = data['Pose_Para'][0,0:3]        
            T = data['Pose_Para'][0,3:6]
            
            # compute the similarity transform:
            pil_im = Image.fromarray(cv2.cvtColor(cv_orig_im, cv2.COLOR_BGR2RGB))
            trans_pil_im, R_trans = transformImage(pil_im, cv_orig_im, landmarks)
            
            # Save aligned image.
            os.makedirs(os.path.join(args.dst_dir, args.dataset), exist_ok=True)
            trans_pil_im.save(os.path.join(os.path.join(args.dst_dir, args.dataset), os.path.dirname(image[:-1]) + '_' + os.path.basename(image)[:-4] + '_pitch_%f_yaw_%f_roll_%f_deg.png' 
                                % (180*R[0]/np.pi, 180*R[1]/np.pi, 180*R[2]/np.pi)))


    elif args.dataset == 'BIWI':
        for i in range(24):

            session_dir = os.path.join(args.src_dir, 'hpdb/', '%02d' % (i+1))
            images = glob2.glob(session_dir + '/*.png')
            calibration = readCalibrationFile(os.path.join(session_dir,'rgb.cal'))
        
            for image in tqdm(images[args.start:args.end]):
                cv_orig_im = cv2.imread(image)
     
                mask = cv2.imread(os.path.join(args.src_dir, 'head_pose_masks/', '%02d' % (i+1), image[-19:-8] + '_depth_mask.png'))
                if mask is None:
                    continue
                x,y,w,h = cv2.boundingRect(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
                top = int(max(math.ceil(y-h*0.1), 0))
                bottom = int(max(math.floor(y+h*0.1), cv_orig_im.shape[0]))
                left = int(min(math.ceil(x-w*0.1), 0))
                right = int(max(math.floor(x+w*0.1), cv_orig_im.shape[1]))
                pil_im = Image.fromarray(cv2.cvtColor(cv_orig_im[top:bottom,left:right,:], cv2.COLOR_BGR2RGB))
                    
                bboxes, landmarks = detect_faces(pil_im, in_weights_dir = args.weights_dir)
                if len(bboxes) < 1:
                    continue
                elif len(bboxes)>1:
                    max_prob = 0
                    max_idx = 0
                    for j in range(len(bboxes)):
                        if bboxes[j,4] > max_prob:
                            max_prob = bboxes[j,4]
                            max_idx = j
                    final_bbox = bboxes[max_idx,:]
                    final_landmarks = landmarks[max_idx,:]
                    if max_prob<0.91:
                        continue
                else:
                    if bboxes[0,4]<0.91:
                        continue
                    final_bbox = bboxes
                    final_landmarks = landmarks

                final_bbox = np.squeeze(final_bbox)
                final_bbox[0] += left
                final_bbox[2] += left
                final_bbox[1] += top
                final_bbox[3] += top
                
                landmarks = np.array(final_landmarks).reshape(2,5).transpose() 
                landmarks[:,0] += left
                landmarks[:,1] += top

                pose = readPoseFile(image[:-7]+'pose.txt')
                R = calibration['R']
                T = calibration['T']
                R_rgb = np.dot(pose['R'], R)
                T_rgb = np.dot(pose['T'].transpose(), R).transpose() - np.dot(R.transpose(), T)
                        
                # compute the similarity transform:
                pil_im = Image.fromarray(cv2.cvtColor(cv_orig_im, cv2.COLOR_BGR2RGB))
                trans_pil_im, R_trans = transformImage(pil_im, cv_orig_im, landmarks)
        
                R_rgb = cv2.Rodrigues(R_rgb)[0]*180/np.pi
                
                # Save aligned image.
                dst_subdir = os.path.join(args.dst_dir, args.dataset, '%02d' % (i+1))
                os.makedirs(dst_subdir, exist_ok=True)
                trans_pil_im.save(os.path.join(dst_subdir, image[-19:-4] + 
                                    '_pitch_%f_yaw_%f_roll_%f_deg.png' % (R_rgb[0], R_rgb[1], R_rgb[2])))                                