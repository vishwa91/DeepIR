#!/usr/bin/env python

import os
import sys
from pprint import pprint

# Pytorch requires blocking launch for proper working
if sys.platform == 'win32':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
from scipy import io

import torch
import torch.nn

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

sys.path.append('modules')
import utils
import motion
import dataset
import thermal

if __name__ == '__main__':
    imname = 'test1'                # Name of the test file name
    camera = 'sim'                  # 'sim', 'boson' or 'lepton' 
    scale_sr = 1                    # 1 for denoising/NUC, 2, 3, .. for SR
    nimg = 5                        # Number of input images
    
    # Load config file -- 
    config = dataset.load_config('configs/dip_%s.ini'%camera)
    config['batch_size'] = nimg
    config['num_workers'] = (0 if sys.platform=='win32' else 4)
    config['lambda_prior'] *= (scale_sr/nimg)
        
    # Load data
    if not config['real']:
        # This is simulated data
        im = utils.get_img(imname, 1)
        minval = 0
        maxval = 1
    else:
        # This is real data
        im, minval, maxval = utils.get_real_im(imname, camera)
    
    # Get data for SR -- this will also get an initial estimate for registration
    im, imstack, ecc_mats = motion.get_SR_data(im, scale_sr, nimg, config)
    ecc_mats[:, :, 2] *= scale_sr
    H, W = im.shape
    
    # Load LPIPs function
    config['gt'] = im
        
    # Now run denoising
    im_dip, profile_dip = thermal.interp_DIP(imstack.astype(np.float32),
                                             ecc_mats.astype(np.float32),
                                             (H, W), config)
    # Save data
    mdict = {'gt': im,
             'rec': im_dip,
             'gain': profile_dip['gain'],
             'offset': profile_dip['offset'],
             'snr': profile_dip['metrics']['snrval'],
             'psnr': profile_dip['metrics']['psnrval'],
             'ssim': profile_dip['metrics']['ssimval'],
             'minval': minval,
             'maxval': maxval}
    io.savemat('%s_%s_%dx_%d.mat'%(imname, camera, scale_sr, nimg), mdict)
    
    pprint(profile_dip['metrics'])
