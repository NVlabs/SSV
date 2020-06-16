# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Siva Karthik Mustikovela.
# --------------------------------------------------------

import argparse
import os
from tqdm import tqdm
import numpy as np
from collections import OrderedDict as odict

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torchvision

from ssv import  VPNet,  VPAwareSynthesizer_ssv
from utils.dataset import lmdbDataset, lmdbDataset_withGT
from utils.ssv import get_az_el_ct_rots, gen_az_rots, gen_el_rots, gen_ct_rots, generate_samples
from utils.ssv import AlexNetConv4, accumulate, sample_data, requires_grad
from utils.ssv import Saver

def train(args, dataset, generator, discriminator, saver):

    loader = sample_data(dataset, args.batch_size, 128, args.num_workers)
    data_loader = iter(loader)

    vg = AlexNetConv4().cuda()
    requires_grad(vg, False)
    cosd = torch.nn.CosineSimilarity().cuda()
    cosd.requires_grad = True

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    used_sample = 0

    prog = tqdm(range(500_000))
    for i in prog:
        try:
            real_image, fl_real_image = next(data_loader)
        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, fl_real_image = next(data_loader)

        used_sample += real_image.shape[0]
        b_size = real_image.size(0)
        real_image1 = real_image.cuda()
        real_image1_fl = fl_real_image.cuda()
     
        gen_in11, gen_in12, gen_in21, gen_in22, gen_in31, gen_in32 = torch.FloatTensor(6, b_size, code_size).uniform_(-1,1).cuda().chunk(6, 0)
        gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
        gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)] 
        gen_in3 = [gen_in31.squeeze(0), gen_in32.squeeze(0)]

        gen_in2[0] = torch.cat((gen_in2[0], gen_in2[0], gen_in3[0]))
        gen_in2[1] = torch.cat((gen_in2[1], gen_in2[1], gen_in3[1]))

        z_recn_gt1 = torch.cat((gen_in1[0], gen_in1[1]),1)
        z_recn_gt2 = torch.cat((gen_in2[0], gen_in2[1]),1)

        # Get random viewpoints and form rotation matrices
        rot_mats1, azs1, els1, cts1, vp_biternions1 = get_az_el_ct_rots(b_size, args.az_range, args.el_range, args.ct_range)
        rot_mats2, azs2, els2, cts2, vp_biternions2 = get_az_el_ct_rots(b_size, args.az_range, args.el_range, args.ct_range)
        rot_mats3, azs3, els3, cts3, vp_biternions3 = get_az_el_ct_rots(b_size, args.az_range, args.el_range, args.ct_range)

        rot_mats2 = torch.cat((rot_mats2, rot_mats3, rot_mats2), dim=0)

        for key in vp_biternions2.keys():
            vp_biternions2[key] = torch.cat((vp_biternions2[key], vp_biternions3[key], vp_biternions2[key]))    

        flipped_azs2_rot = gen_az_rots(b_size,-azs2.squeeze())
        flipped_els2_rot = gen_el_rots(b_size,els2.squeeze())
        flipped_cts2_rot = gen_ct_rots(b_size,-cts2.squeeze())        
        rot_mats_flipped = torch.cat((torch.bmm(torch.bmm(flipped_azs2_rot[:,:,0:3],flipped_els2_rot[:,:,0:3]), flipped_cts2_rot[:,:,0:3]), torch.zeros((b_size,3,1)).cuda()),2)        

        #--------Train VPNet-------#
        discriminator.zero_grad()
        real_predict1, real_z_pred1, real_vp_pred1, _ = discriminator(real_image1)
        
        real_predict1 = real_predict1.mean() - 0.001 * (real_predict1 ** 2).mean()
        real_predict1_loss = -real_predict1

        # L_imc
        vpp1 = discriminator.module.compute_vp_pred(real_vp_pred1)
        vpp_az_rot1 = gen_az_rots(b_size,vpp1['a'])
        vpp_el_rot1 = gen_el_rots(b_size,vpp1['e'])
        vpp_ct_rot1 = gen_ct_rots(b_size,vpp1['t'])
        vpp_rot_mats = torch.cat((torch.bmm(torch.bmm(vpp_az_rot1[:,:,0:3],vpp_el_rot1[:,:,0:3]), vpp_ct_rot1[:,:,0:3]), torch.zeros((b_size,3,1)).cuda()),2)
        reconstructed_image1 = generator([real_z_pred1[:,:code_size].reshape(b_size,code_size), real_z_pred1[:,code_size:].reshape(b_size,code_size)], vpp_rot_mats)

        rreal_image1 = F.interpolate(real_image1, size=(224,224), mode='bilinear', align_corners=False)
        real_image_feats = vg(rreal_image1) 
        real_image_feats = real_image_feats.reshape(b_size, real_image_feats.shape[1]*real_image_feats.shape[2]*real_image_feats.shape[3])

        rreconstructed_image_resized = F.interpolate(reconstructed_image1, size=(224,224), mode='bilinear', align_corners=False)
        rec_image_feats = vg(rreconstructed_image_resized) 
        rec_image_feats_reshaped = rec_image_feats.reshape(b_size, rec_image_feats.shape[1]*rec_image_feats.shape[2]*rec_image_feats.shape[3])
        
        cos_similarity = -args.img_recn_weight * cosd(real_image_feats, rec_image_feats_reshaped)
        im_consistency_loss = cos_similarity.mean()

        # L_flip
        _, _, real_vp_pred1_fl, _ = discriminator(real_image1_fl)

        real_vp_pred1_for_pGT = real_vp_pred1
        for t in real_vp_pred1_for_pGT.keys():
            for t1 in real_vp_pred1_for_pGT[t].keys():
                real_vp_pred1_for_pGT[t][t1] = real_vp_pred1_for_pGT[t][t1].detach()

        pGT_flip = discriminator.module.compute_gt_flip(real_vp_pred1_for_pGT)
        fc_loss = discriminator.module.compute_vp_loss(real_vp_pred1_fl, pGT_flip)
        flip_consistency_loss=0
        for key in fc_loss.keys():
            flip_consistency_loss += fc_loss[key]

        # Pass Z through generator for fake images
        fake_image1 = generator(gen_in1, rot_mats1.cuda())

        # Pass fake images through D for predictions. 
        fake_predict1, fake_z_predict1, fake_vp_pred1, _ = discriminator(fake_image1.detach())

        # classification
        fake_predict1 = fake_predict1.mean()

        # z recn
        z_recn_loss1 = z1_recn_loss(fake_z_predict1, z_recn_gt1)

        # Viewpoint reconstruction
        vpr_loss1 = discriminator.module.compute_vp_loss(fake_vp_pred1, vp_biternions1)
        vp_recn_loss1=0
        for key in vpr_loss1.keys():
            vp_recn_loss1 += vpr_loss1[key]

        # Backprop for the sum of both classification loss and reconstruction loss.
        fake_im_loss1 = real_predict1_loss \
                        + fake_predict1 \
                        + (args.img_recn_weight * im_consistency_loss) \
                        + (args.flip_cons_weight * flip_consistency_loss) \
                        + (args.z_recn_weight * z_recn_loss1) \
                        + (args.vp_recn_weight * vp_recn_loss1)\

        fake_im_loss1.backward()

        # Calculate gradient penalty
        ################################
        eps = torch.rand(b_size, 1, 1, 1).cuda()
        x_hat = eps * real_image1.data + (1 - eps) * fake_image1.data
        x_hat.requires_grad = True
        hat_predict, _, _, _ = discriminator(x_hat)
        grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val = grad_penalty.data
        ################################

        disc_loss_val = (real_predict1 - fake_predict1).data
        disc_z_loss_val = (args.z_recn_weight * z_recn_loss1).data
        disc_vp_loss_val = (args.vp_recn_weight * vp_recn_loss1).data
        d_optimizer.step()

        #----------Train VPASynthesizer---------#

        generator.zero_grad()

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        fake_image2 = generator(gen_in2, rot_mats2.cuda())

        cls_predict2, z_pred2, vp_pred2, _ = discriminator(fake_image2)

        # Classification
        cln_loss = -cls_predict2.mean()

        # Z reconstruction
        z_recn_loss2 = z1_recn_loss(z_pred2, z_recn_gt2)

        # VP Reconstruction
        vpr_loss2 = discriminator.module.compute_vp_loss(vp_pred2, vp_biternions2)
        vp_recn_loss2=0
        for key in vpr_loss2.keys():
            vp_recn_loss2 += vpr_loss2[key]

        # Flip image consistency loss     
        z_for_flip = [gen_in21.squeeze(0), gen_in22.squeeze(0)]
        flipped_img_gen = generator(z_for_flip, rot_mats_flipped)
        
        flipped_fake_img = torch.flip(fake_image2[:b_size,:,:,:], [3]).detach()
        flip_consistency_loss_G = flipc_G_loss(flipped_fake_img,flipped_img_gen)
        
        gen_loss =    cln_loss \
                    + (args.z_recn_weight * z_recn_loss2) \
                    + (args.vp_recn_weight * vp_recn_loss2) \
                    + (args.flipc_recn_weight_G*flip_consistency_loss_G)    
        gen_loss.backward()        
        
        g_optimizer.step()
        accumulate(g_running, generator.module)

        gen_loss_val = cln_loss.data
        gen_z_loss_val = (args.z_recn_weight * z_recn_loss2).data
        gen_vp_loss_val = (args.vp_recn_weight * vp_recn_loss2).data

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        #----------End training cycle---------#        

        if (i + 1) % args.sample_interval == 0:
            images = []
            gen_i = 2; gen_j = 3
            with torch.no_grad():
                for _ in range(gen_i):

                    gen_in31, gen_in32 = torch.FloatTensor(2, gen_j, code_size).uniform_(-1,1).cuda().chunk(2, 0)
                    gen_in3 = [gen_in31.squeeze(0), gen_in32.squeeze(0)]
                    # rot_mats_test, _, _, _ = get_az_el_rots(gen_j, args.az_range, args.el_range)
                    rot_mats_test, _, _, _, _ = get_az_el_ct_rots(gen_j, args.az_range, args.el_range, args.ct_range)
                    images.append(g_running(gen_in3, rot_mats_test.cuda()).data.cpu())

            utils.save_image(torch.cat(images, 0), f'{args.exp_root}/{args.exp_name}/sample/{str(i + 1).zfill(6)}.png', nrow=gen_i, normalize=True, range=(-1, 1),)

        if (i + 1) % args.save_interval == 0:
            saver.save_all_models(i+1)

            # generate with zero elevation and var azimuths
            gen_azs = list(np.arange(-args.az_range, args.az_range, 0.08))
            gen_els = [0.0]
            gen_cts = [0.0]
            generate_samples(g_running, gen_azs, gen_els, gen_cts, args, i+1, 'el0_ct0_var_az', num_samples=2)
            # generate with const az and var elevations
            gen_azs = [1.2]
            gen_els = list(np.arange(args.el_range, -args.el_range, -0.08))
            gen_cts = [0.0]
            generate_samples(g_running, gen_azs, gen_els, gen_cts, args, i+1, 'var_el_az1.2_ct0', num_samples=2)
            # generate with const az and var elevations
            gen_azs = [0.0]
            gen_els = list(np.arange(args.el_range, -args.el_range, -0.08))
            gen_cts = [0.0]
            generate_samples(g_running, gen_azs, gen_els, gen_cts, args, i+1, 'var_el_az0_ct0', num_samples=2)
            # generate with zero az, zero el and varying camera tilts
            gen_azs = [0.0]
            gen_els = [0.0]
            gen_cts = list(np.arange(-args.ct_range, args.ct_range, 0.08))
            generate_samples(g_running, gen_azs, gen_els, gen_cts, args, i+1, 'el0_az0_var_ct', num_samples=2)            

        state_msg = (f'G: {gen_loss_val:.1f}; G_z: {gen_z_loss_val:.1f}; G_vp: {gen_vp_loss_val:.1f}; D: {disc_loss_val:.1f}; D_z: {disc_z_loss_val:.1f}; D_vp: {disc_vp_loss_val:.1f}; CosD: {im_consistency_loss.data:.3f}; FL_c: {flip_consistency_loss.sum().data:.3f}; Grad: {grad_loss_val:.1f}')
        prog.set_description(state_msg)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SSV')

    parser.add_argument('--data_path', type=str, help='path of specified dataset')

    parser.add_argument('--exp_root', default='/4tb/experiments', type=str, help='experiments root')
    parser.add_argument('--exp_name', default='ssv_debug', type=str, help='name of current experiment')
    parser.add_argument('--model_name', default='ssv', type=str, help='prefix for model name')
    
    parser.add_argument('--save_interval', default=5000, type=int, help='interval to save models')        
    parser.add_argument('--sample_interval', default=5000, type=int, help='interval to generate samples')            
    
    parser.add_argument('--gpus', type=str, default='0', help='GPU numbers used for training')
    parser.add_argument('--num_workers', type=int, default=16, help='num workers for data loader')
    
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')    

    parser.add_argument('--z_recn_loss', type=str, default='l2', choices=['l2', 'l1'], help='loss type for z reconstruction.')
    parser.add_argument('--z_recn_weight', type=float, default=1, help='weight for z reconstruction loss')
    parser.add_argument('--vp_recn_weight', type=float, default=1, help='weight for vp reconstruction loss')
    parser.add_argument('--img_recn_weight', type=float, default=0, help='weight for image reconstruction loss')
    parser.add_argument('--flipc_recn_weight_G', type=float, default=0, help='weight for image reconstruction loss for generator')    
    parser.add_argument('--flip_cons_weight', type=float, default=0, help='weight for image reconstruction loss')

    parser.add_argument('--code_size', type=int, default=128, help='style code size')    
    parser.add_argument('--az_range', type=float, default=1, help='range for azimuth')
    parser.add_argument('--el_range', type=float, default=0.3, help='range for elevation')
    parser.add_argument('--ct_range', type=float, default=0.78, help='range for azimuth')    

    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------------------------------#

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    code_size = args.code_size

    generator = nn.DataParallel(VPAwareSynthesizer(code_size)).cuda()
    g_running = VPAwareSynthesizer(code_size).cuda()    
    g_running.train(False)
    
    discriminator = nn.DataParallel(VPNet(2*code_size, instance_norm=True)).cuda()

    if args.z_recn_loss == 'l2':
        z1_recn_loss = nn.MSELoss().cuda()
    else:
        z1_recn_loss = nn.SmoothL1Loss().cuda()

    flipc_G_loss = nn.SmoothL1Loss().cuda()

    # Set optimizers
    g_optimizer = optim.Adam(generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group({'params': generator.module.style.parameters(), 'lr': args.lr * 0.01, 'mult': 0.01})
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator.module, 0)

    saver = Saver(args)
    saver.add_model('generator',generator)
    saver.add_model('g_running',g_running)
    saver.add_model('discriminator',discriminator)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    dataset = lmdbDataset(args.data_path, transform)

    args.lr = {}

    # make all directories
    os.makedirs(os.path.join(args.exp_root,args.exp_name,'sample'), exist_ok=True)
    os.makedirs(os.path.join(args.exp_root,args.exp_name,'checkpoint'), exist_ok=True)
    os.makedirs(os.path.join(args.exp_root,args.exp_name,'log'), exist_ok=True)    
    os.makedirs(os.path.join(args.exp_root,args.exp_name,'gen_samples'), exist_ok=True)    

    # Start training
    train(args, dataset, generator, discriminator, saver)