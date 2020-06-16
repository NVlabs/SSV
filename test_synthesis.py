# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Siva Karthik Mustikovela.
# --------------------------------------------------------

import os
from tqdm import tqdm
import numpy as np
import imageio

import torch
from torch import nn
from torchvision import utils

from ssv import VPAwareSynthesizer_ssv
from utils.ssv import gen_az_rots


if __name__=='__main__':

    os.environ['CUDA_VISIBLE_DEVICES']="1,"

    num_samples = 4
    code_size = 64
    az_range = 1.5

    synthesizer = VPAwareSynthesizer_ssv(code_size).cuda()
    synthesizer.load_state_dict(torch.load('model/synthesis_net.pt'))
    
    synthesizer.eval()    

    gen_in11, gen_in12 = torch.FloatTensor(2, num_samples, code_size).uniform_(-1,1).chunk(2, 0)  
    gen_in11 = gen_in11.cuda(); gen_in12 = gen_in12.cuda()     
    style_code = [gen_in11.squeeze(0), gen_in12.squeeze(0)]

    os.makedirs('synth_images', exist_ok=True)

    for ind, az in tqdm(enumerate(np.arange(-az_range, az_range, 0.08))):
        rot_azs_test = gen_az_rots(num_samples, az)
        image = synthesizer(style_code, rot_azs_test.cuda())
        img_name = os.path.join('synth_images', str(ind)+ '.png')
        utils.save_image(image, img_name, nrow=4, normalize=True, range=(-1, 1))

    gif_images = []
    for i in range(1,ind):
        img_name = os.path.join('synth_images', str(i)+'.png')
        gif_images.append(imageio.imread(img_name))
    imageio.mimsave(os.path.join('synth_images', 'gen_gif.gif'), gif_images)