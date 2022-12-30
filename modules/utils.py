#!/usr/bin/env python

'''
    Miscellaneous utilities that are extremely helpful but cannot be clubbed
    into other modules.
'''
import torch

# Scientific computing
import numpy as np
import scipy.linalg as lin
from scipy import io

# Plotting
import cv2
import matplotlib.pyplot as plt

def nextpow2(x):
    '''
        Return smallest number larger than x and a power of 2.
    '''
    logx = np.ceil(np.log2(x))
    return pow(2, logx)

def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x
    
    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin)/(xmax - xmin)

    return xnormalized

def asnr(x, xhat, compute_psnr=False):
    '''
        Compute affine SNR, which accounts for any scaling and shift between two
        signals

        Inputs:
            x: Ground truth signal(ndarray)
            xhat: Approximation of x

        Outputs:
            asnr_val: 20log10(||x||/||x - (a.xhat + b)||)
                where a, b are scalars that miminize MSE between x and xhat
    '''
    mxy = (x*xhat).mean()
    mxx = (xhat*xhat).mean()
    mx = xhat.mean()
    my = x.mean()
    

    a = (mxy - mx*my)/(mxx - mx*mx)
    b = my - a*mx

    if compute_psnr:
        return psnr(x, a*xhat + b)
    else:
        return rsnr(x, a*xhat + b)

def rsnr(x, xhat):
    '''
        Compute reconstruction SNR for a given signal and its reconstruction.

        Inputs:
            x: Ground truth signal (ndarray)
            xhat: Approximation of x

        Outputs:
            rsnr_val: RSNR = 20log10(||x||/||x-xhat||)
    '''
    xn = lin.norm(x.reshape(-1))
    en = lin.norm((x-xhat).reshape(-1)) + 1e-12
    rsnr_val = 20*np.log10(xn/en)

    return rsnr_val

def psnr(x, xhat):
    ''' Compute Peak Signal to Noise Ratio in dB

        Inputs:
            x: Ground truth signal
            xhat: Reconstructed signal

        Outputs:
            snrval: PSNR in dB
    '''
    err = x - xhat
    denom = np.mean(pow(err, 2)) + 1e-12

    snrval = 10*np.log10(np.max(x)/denom)

    return snrval

def embed(im, embedsize):
    '''
        Embed a small image centrally into a larger window.

        Inputs:
            im: Image to embed
            embedsize: 2-tuple of window size

        Outputs:
            imembed: Embedded image
    '''

    Hi, Wi = im.shape
    He, We = embedsize

    dH = (He - Hi)//2
    dW = (We - Wi)//2

    imembed = np.zeros((He, We), dtype=im.dtype)
    imembed[dH:Hi+dH, dW:Wi+dW] = im

    return imembed

def measure(x, noise_snr=40, tau=100):
    ''' Realistic sensor measurement with readout and photon noise

        Inputs:
            noise_snr: Readout noise in electron count
            tau: Integration time. Poisson noise is created for x*tau.
                (Default is 100)

        Outputs:
            x_meas: x with added noise
    '''
    x_meas = np.copy(x)

    noise = pow(10, -noise_snr/20)*np.random.randn(x_meas.size).reshape(x_meas.shape)

    # First add photon noise, provided it is not infinity
    if tau != float('Inf'):
        x_meas = x_meas*tau

        x_meas[x > 0] = np.random.poisson(x_meas[x > 0])
        x_meas[x <= 0] = -np.random.poisson(-x_meas[x <= 0])

        x_meas = (x_meas + noise)/tau

    else:
        x_meas = x_meas + noise

    return x_meas

def grid_plot(imdata):
    '''
        Plot 3D set of images into a 2D grid using subplots.

        Inputs:
            imdata: N x H x W image stack

        Outputs:
            None
    '''
    N, H, W = imdata.shape

    nrows = int(np.sqrt(N))
    ncols = int(np.ceil(N/nrows))

    for idx in range(N):
        plt.subplot(nrows, ncols, idx+1)
        plt.imshow(imdata[idx, :, :], cmap='gray')
        plt.xticks([], [])
        plt.yticks([], [])
        
def build_montage(images):
    '''
        Build a montage out of images
    '''
    nimg, H, W = images.shape
    
    nrows = int(np.ceil(np.sqrt(nimg)))
    ncols = int(np.ceil(nimg/nrows))
    
    montage_im = np.zeros((H*nrows, W*ncols), dtype=np.float32)
    
    cnt = 0
    for r in range(nrows):
        for c in range(ncols):
            h1 = r*H
            h2 = (r+1)*H
            w1 = c*W
            w2 = (c+1)*W

            if cnt == nimg:
                break

            montage_im[h1:h2, w1:w2] = images[cnt, ...]
            cnt += 1
    
    return montage_im

def ims2rgb(im1, im2):
    '''
        Concatenate images into RGB
        
        Inputs:
            im1, im2: Two images to compare
    '''
    H, W = im1.shape
    
    imrgb = np.zeros((H, W, 3))
    imrgb[..., 0] = im1
    imrgb[..., 2] = im2

    return imrgb

def textfunc(im, txt):
    return cv2.putText(im, txt, (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (1, 1, 1),
                        2,
                        cv2.LINE_AA)
    
def get_img(imname, scaling):
    # Read image
    im = cv2.resize(plt.imread('data/%s.png'%imname), None,
                    fx=scaling, fy=scaling)
    
    if im.ndim == 2:
        im = im[:, :, np.newaxis]
        im = im[:, :, [0, 0, 0]]
        im = np.copy(im, order='C')
    H, W, _ = im.shape
    
    return np.copy(im[..., 1], order='C').astype(np.float32)

def get_real_im(imname, camera):
    im = io.loadmat('data/%s/%s.mat'%(camera, imname))['imstack']
    minval = im.min()
    maxval = im.max()
    
    if camera == 'rgb':
        im = normalize(im[:, ::2, ::2], True)
    else:
        im = normalize(im, True).astype(np.float32)
        
    return im, minval, maxval

def boxify(im, topleft, boxsize, color=[1, 1, 1], width=2):
    '''
        Generate a box around a region.
    '''
    h, w = topleft
    dh, dw = boxsize
    
    im[h:h+dh+1, w:w+width, :] = color
    im[h:h+width, w:w+dh+width, :] = color
    im[h:h+dh+1, w+dw:w+dw+width, :] = color
    im[h+dh:h+dh+width, w:w+dh+width, :] = color

    return im

def get_inp(tensize, const=10.0):
    '''
        Wrapper to get a variable on graph
    '''
    inp = torch.rand(tensize).cuda()/const
    inp = torch.autograd.Variable(inp, requires_grad=True).cuda()
    inp = torch.nn.Parameter(inp)
    
    return inp
