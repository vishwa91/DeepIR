#!/usr/bin/env python

'''
    Routines for dealing with thermal images
'''
import tqdm
import copy

import cv2

import numpy as np
from skimage.metrics import structural_similarity as ssim_func

import torch
import kornia
import torch.nn.functional as F

import utils
import losses
import motion
import deep_prior

def get_metrics(gt, estim, pad=True):
    '''
        Compute SNR, PSNR, SSIM, and LPIP between two images.
        
        Inputs:
            gt: Ground truth image
            estim: Estimated image
            lpip_func: CUDA function for computing lpip value
            pad: if True, remove boundaries when computing metrics
            
        Outputs:
            metrics: dictionary with following fields:
                snrval: SNR of reconstruction
                psnrval: Peak SNR 
                ssimval: SSIM
                lpipval: VGG perceptual metrics
    '''
    if min(gt.shape) < 50:
        pad = False
    if pad:
        gt = gt[20:-20, 20:-20]
        estim = estim[20:-20, 20:-20]
        
    snrval = utils.asnr(gt, estim)
    psnrval = utils.asnr(gt, estim, compute_psnr=True)
    ssimval = ssim_func(gt, estim)
    
    metrics = {'snrval': snrval,
               'psnrval': psnrval,
               'ssimval': ssimval}
    return metrics

def create_fpn(imsize, vmin=0.9, vmax=1, method='col', rank=1):
    '''
        Generate fixed pattern noise for microbolometer-type sensors
        
        Inputs:
            imsize: (H, W) tuple
            vmin, vmax: Minimum and maximum value of gain
            method:
                'col' -- generate column only noise
                'both' -- generate rank-k noise
                'corr_col' -- correlated columns
                'corr_both' -- correlated rows and columns
            rank: if method is 'both' generate noise with this rank.
            
        Outputs:
            fpn: (H, W)-sized fixed pattern noise
    '''
    
    H, W = imsize

    if method == 'col':
        fpn = np.ones((H, 1)).dot(vmin + (vmax-vmin)*np.random.rand(1, W))
    elif method == 'both':
        fpn = 0
        
        for idx in range(rank):
            col = vmin + (vmax - vmin)*np.random.rand(H, 1)
            row = vmin + (vmax - vmin)*np.random.rand(1, W)
            
            fpn += col.dot(row)
        fpn /= rank
    elif method == 'corr_col':
        row = vmin + (vmax-vmin)*np.random.rand(W)
        row = np.convolve(row, np.ones(5)/5, mode='same')
        
        fpn = np.ones((H, 1)).dot(row.reshape(1, W))
    elif method == 'corr_both':
        row = vmin + (vmax-vmin)*np.random.rand(W)
        row = np.convolve(row, np.ones(5)/5, mode='same')
        
        col = vmin + (vmax-vmin)*np.random.rand(H)
        col = np.convolve(col, np.ones(5)/5, mode='same')
        
        fpn = col.reshape(H, 1).dot(row.reshape(1, W))
        
    return fpn

def reg_avg_denoise(imstack, ecc_mats=None):
    '''
        Denoise a thermal stack by registering and averaging.
        
        Inputs:

        Outputs:
            im_denoised: Denoised image
    '''
    nimg, H, W = imstack.shape
    # if ecc_mats is none, register the stack
    if ecc_mats is None:
        ecc_mats = motion.register_stack(imstack, (H, W))[:, :2, :]
    
     # Now warp image back to reference and average
    ecc_inv = motion.invert_regstack(ecc_mats)
    imten = torch.tensor(imstack.astype(np.float32))[:, None, ...]
    ecc_ten = torch.tensor(ecc_inv.astype(np.float32))
    
    imwarped = kornia.warp_affine(imten, ecc_ten, (H, W), flags='bilinear')
    im_denoised = imwarped.mean(0)[0, ...].numpy()
    weights = (imwarped > 0).type(torch.float32).mean(0)[0, ...].numpy()
    weights[weights == 0] = 1
    
    im_denoised /= weights

    return im_denoised
    
def interp_DIP(imstack, reg_stack, hr_size, params_dict):
    '''
        Super resolve from a stack of images using deep image prior
        
        Inputs:
            imstack: (nimg, Hl, Wl) stack of low resolution images
            reg_stack: (nimg, 2, 3) stack of affine matrices
            hr_size: High resolution image size
            params_dict: Dictionary containing parameters for optimization
                kernel_type: Type of downsampling
                input_type: Type of input
                input_depth: Depth of input data (number of channels)
                skip_n33d: Parameter for the neural network
                skip_n33u: Parameter for the neural network
                skip_n11: Parameter for the neural network
                num_scales: Parameter for the neural network
                upsample_mode: Parameter for the neural network
                niters: Number of DIP iterations
                batch_size: Batch size of data
                num_workers: Workers for data loading
                learning_rate: Learning rate for optimization
                prior_type: tv, or hessian
                lambda_prior: Prior weight
                optimize_reg: If True, optimize registration parameters
                visualize: If True, visualize reconstructions at each iteration
                gt: If visualize is true, gt is the ground truth image
                reg_final: If True, register the final result to gt
                lpip_func: If gt is true, evaluate perceptual similarity with
                    this function
            
        Returns:
            im_hr: High resolution image
            profile: Dictionary containing the following:
                loss_array: Array with loss at each iteration
                trained_model: State dictionary for best model
                metrics: if gt is provided, this is a dictionary with:
                    snrval: SNR of reconstruction
                    psnrval: Peak SNR 
                    ssimval: SSIM
                    lpipval: VGG perceptual metrics
                    
    '''
    nimg, Hl, Wl = imstack.shape
    H, W = hr_size
    
    scale_sr = 0.5*(H/Hl + W/Wl)
    
    # Internal constant
    img_every = 2 
    
    if params_dict['mul_gain']:
        lambda_offset = 10
    else:
        lambda_offset = 0
    
    # Create loss functions
    criterion_fidelity = losses.L1Norm()
    criterion_offset = losses.TVNorm(mode='l2')
    if params_dict['prior_type'] == 'tv':
        criterion_prior = losses.TVNorm()
    elif params_dict['prior_type'] == 'hessian':
        criterion_prior = losses.HessianNorm()
    else:
        raise ValueError('Prior not implemented')
    
    # Create input
    model_input = deep_prior.get_noise(params_dict['input_depth'],
                                       params_dict['input_type'],
                                       (H, W)).cuda().detach()
    
    # Create the network
    if params_dict['predmode'] == 'combined':
        nchan = 3
    else:
        nchan = 1
    model = deep_prior.get_net(params_dict['input_depth'], 'skip',
                               'reflection', n_channels=nchan,
                               skip_n33d=params_dict['skip_n33d'],
                               skip_n33u=params_dict['skip_n33u'],
                               skip_n11=params_dict['skip_n11'],
                               num_scales=params_dict['num_scales'],
                               upsample_mode=params_dict['upsample_mode']
                               ).cuda()
    # Set it to training
    model.train()
    
    if params_dict['integrator'] == 'learnable':
        kernel_size = (int(scale_sr), int(scale_sr))
        integrator = torch.nn.Conv2d(1, 1, kernel_size=kernel_size,
                                     stride=int(scale_sr), bias=False).cuda()
        
        with torch.no_grad():
            integrator.weight.fill_(1.0/(scale_sr*scale_sr))
    
    # Create parameters from affine matrices
    affine_mat = torch.tensor(reg_stack).cuda()
    affine_var = torch.autograd.Variable(affine_mat, requires_grad=True).cuda()
    affine_param = torch.nn.Parameter(affine_var)
    
    # Create gain parameter
    vmin = params_dict['fpn_vmin']
    params = list(model.parameters())
    
    if params_dict['predmode'] != 'combined':
        gain = torch.ones(1, 1, Hl, Wl).cuda()
        gain_var = torch.autograd.Variable(gain, requires_grad=True).cuda()
        gain_param = torch.nn.Parameter(gain_var)
        
        offset = torch.ones(1, 1, Hl, Wl).cuda()*1e-1
        offset_var = torch.autograd.Variable(offset, requires_grad=True).cuda()
        offset_param = torch.nn.Parameter(offset_var)
        
        params += [gain_param] + [offset_param]
        
    if params_dict['integrator'] == 'learnable':
        params += integrator.parameters()
        
    # Create an ADAM optimizer
    optimizer = torch.optim.Adam(lr=params_dict['learning_rate'],
                                 params=params)
    
    # Affine transform requires a separate optimizer
    reg_optimizer = torch.optim.Adam(lr=params_dict['affine_learning_rate'],
                                     params=[affine_param])
        
    loss_array = np.zeros(params_dict['niters'])
    best_loss = float('inf')
    best_state_dict = None

    # We will just use all data 
    gt = torch.tensor(imstack).cuda()[:, None, ...]
    for epoch in tqdm.tqdm(range(params_dict['niters'])):
        train_loss = 0
        img_and_gain = model(model_input)
        img_hr = img_and_gain[:, [0], ...]
        
        if params_dict['predmode'] == 'combined':
            gain_param = img_and_gain[:, [1], ...]
            offset_param = img_and_gain[:, [2], ...]
        
            if scale_sr > 1:
                gain_param = F.interpolate(gain_param, (Hl, Wl))
                offset_param = F.interpolate(offset_param, (Hl, Wl))
        
        # Generate low resolution images
        img_hr_cat = torch.repeat_interleave(img_hr, nimg, 0)
        
        if params_dict['integrator'] == 'area':
            img_hr_affine = kornia.warp_affine(img_hr_cat, affine_param,
                                            (H, W), align_corners=True)
            img_lr = F.interpolate(img_hr_affine, (Hl, Wl), mode='area')
        elif params_dict['integrator'] == 'learnable':
            img_hr_affine = kornia.warp_affine(img_hr_cat, affine_param,
                                            (H, W), align_corners=True)
            img_lr = integrator(img_hr_affine)
        else:
            img_lr = kornia.warp_affine(img_hr_cat,
                                        affine_param/scale_sr,
                                        (Hl, Wl), align_corners=False)
                    
        # Multiply with the gain term
        mask = img_lr  > 0
        
        if params_dict['add_offset']:
            img_lr = img_lr + offset_param
        
        if params_dict['mul_gain']:
            img_lr = gain_param * img_lr

        mse_loss = criterion_fidelity(img_lr*mask, gt*mask)
        prior_loss = params_dict['lambda_prior']*criterion_prior(img_hr)
        
        loss = mse_loss + prior_loss
        
        if params_dict['add_offset']:
            offset_loss = lambda_offset*criterion_offset(offset_param)
            loss = loss + offset_loss
        
        optimizer.zero_grad()
        
        if params_dict['optimize_reg']:
            reg_optimizer.zero_grad()
    
        loss.backward()
        optimizer.step()
        
        if params_dict['optimize_reg']:
            reg_optimizer.step()
                    
        train_loss = loss.item()
            
        # Find if we have the best mode
        if train_loss < best_loss:
            best_loss = train_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            
        loss_array[epoch] = train_loss
        
        if params_dict['visualize']:
            if epoch%img_every == 0:
                with torch.no_grad():
                    img_hr_cpu = img_hr.cpu().detach().numpy().reshape(H, W)
                v_idx = np.random.randint(nimg)    
                img_lr_cpu = img_lr[v_idx, ...].cpu().detach().reshape(Hl, Wl)
                
                snrval = utils.asnr(params_dict['gt'], img_hr_cpu,
                                    compute_psnr=True)
                ssimval = ssim_func(params_dict['gt'], img_hr_cpu)
                
                txt = 'PSNR: %.1f | SSIM: %.2f'%(snrval, ssimval)
                gain = gain_param.cpu().detach().numpy().reshape(Hl, Wl)
                offset = offset_param.cpu().detach().numpy().reshape(Hl, Wl)
                img_hr_ann = utils.textfunc(img_hr_cpu/img_hr_cpu.max(), txt)
                
                imtop = np.hstack((imstack[v_idx, ...], img_lr_cpu.numpy()))
                imbot = np.hstack((gain/gain.max(), offset/offset.max()))
                imcat = np.vstack((imtop, imbot))
                imcat_full = np.hstack((params_dict['gt'], img_hr_ann))
                
                cv2.imshow('Recon LR', np.clip(imcat, 0, 1))
                cv2.imshow('Recon HR', np.clip(imcat_full, 0, 1))
                cv2.waitKey(1)
                
    # We are done, obtain the best model
    model.eval()
    with torch.no_grad():
        model.load_state_dict(best_state_dict)
        img_and_gain = model(model_input)
        img_hr = img_and_gain[[0], [0], ...].reshape(1, 1, H, W)
        img_hr = kornia.warp_affine(img_hr,
                                    affine_param[[0], ...], (H, W))
        img_hr = img_hr.cpu().detach().numpy().reshape(H, W)
        
        if params_dict['predmode'] == 'combined':
            gain_param = img_and_gain[0, 1, ...]
            offset_param = img_and_gain[0, 2, ...]
        
    # In case there's a shift in reconstruction
    if params_dict['reg_final'] and 'gt' in params_dict:
        try:
            img_hr = motion.ecc_flow(params_dict['gt'], img_hr)[1]
        except:
            pass
        
    # If ground truth is provided, return metrics
    if 'gt' in params_dict:
        metrics = get_metrics(params_dict['gt'], img_hr)
        
    gain = gain_param.detach().cpu().numpy().reshape(Hl, Wl)
    offset = offset_param.detach().cpu().numpy().reshape(Hl, Wl)
    profile = {'loss_array': loss_array,
               'trained_model': best_state_dict,
               'metrics': metrics,
               'ecc_mats': affine_param.detach().cpu().numpy(),
               'gain': gain,
               'offset': offset}
    
    return img_hr, profile
