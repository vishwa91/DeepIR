#!/usr/bin/env python

import os
import sys
import tqdm
import pdb
import math
import configparser

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

import skimage
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def xy_mgrid(H, W):
    '''
        Generate a flattened meshgrid for heterogenous sizes
        
        Inputs:
            H, W: Input dimensions
        
        Outputs:
            mgrid: H*W x 2 meshgrid
    '''
    Y, X = torch.meshgrid(torch.linspace(-1, 1, H),
                          torch.linspace(-1, 1, W))
    mgrid = torch.stack((X, Y), dim=-1).reshape(-1, 2)
    
    return mgrid

class ImageDataset(Dataset):
    def __init__(self, img):
        super().__init__()
        H, W, nchan = img.shape
        img = torch.tensor(img)[..., None]
        self.pixels = img.view(-1, nchan)
        self.coords = xy_mgrid(H, W)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels
    
class Image3x3Dataset(Dataset):
    def __init__(self, img):
        super().__init__()
        H, W, nchan = img.shape
        img = torch.tensor(img)[..., None]
        self.pixels = img.view(-1, nchan)
        self.coords = xy_mgrid(H, W)
        
        # Stack coordinates in the 3x3 neighborhood
        coords_stack = []
        for xshift in [0, 1]:
            for yshift in [0, 1]:
                shift_array = np.array([xshift/W, yshift/H]).reshape(1, 2)
                coords_stack.append(self.coords + shift_array)
                
        self.coords = np.hstack(coords_stack).astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels
    
class ImageFlowDataset(Dataset):
    def __init__(self, img1, img2):
        super().__init__()
        H, W = img1.shape
        img1 = torch.tensor(img1)[..., None]
        img2 = torch.tensor(img2)[..., None]
        self.pixels1 = img1.view(-1, 1)
        self.pixels2 = img2.view(-1, 1)
        
        self.coords = xy_mgrid(H, W)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels1, self.pixels2
    
class ImageRegDataset(Dataset):
    def __init__(self, imstack):
        super().__init__()
        
        self.imstack = imstack
        self.nimg, H, W = imstack.shape
        
    def __len__(self):
        return self.nimg

    def __getitem__(self, idx):
        img = torch.tensor(self.imstack[idx, ...])[None, ...]
        
        return img, idx
    
class ImageStackDataset(Dataset):
    def __init__(self, imstack):
        super().__init__()
        
        self.imstack = imstack
        self.nimg, H, W = imstack.shape
        
        self.coords = xy_mgrid(H, W)
        
    def __len__(self):
        return self.nimg

    def __getitem__(self, idx):
        img = torch.tensor(self.imstack[idx, ...])
        pixels = img[None, ...].permute(1, 2, 0).view(-1, 1)
        
        return self.coords, pixels
    
class ImageSRDataset(Dataset):
    def __init__(self, imstack, Xstack=None, Ystack=None, masks=None,
                 jitter=False, xjitter=None, yjitter=None, get_indices=False):
        super().__init__()
        
        self.imstack = imstack
        self.Xstack = Xstack
        self.Ystack = Ystack
        self.masks = masks
        self.jitter = jitter
        self.get_indices = get_indices
        
        self.nimg, self.H, self.W = imstack.shape
        
        if xjitter is None:
            self.xjitter = 1/self.W
            self.yjitter = 1/self.H
        else:
            self.xjitter = xjitter
            self.yjitter = yjitter

    def __len__(self):
        return self.nimg

    def __getitem__(self, idx):
        img = torch.tensor(self.imstack[idx, ...])
        
        
        # If Jitter is enabled, return stratified sampled coordinates
        pixels = img[None, ...].permute(1, 2, 0).view(-1, 1)
        
        if self.masks is not None:
            mask = torch.tensor(self.masks[idx, ...])
            mask = mask[None, ...].permute(1, 2, 0).view(-1, 1)
        else:
            mask = torch.zeros(1)
            
        if self.Xstack is not None:
            coords = torch.stack((torch.tensor(self.Xstack[idx, ...]),
                              torch.tensor(self.Ystack[idx, ...])),
                             dim=-1).reshape(-1, 2)
        else:
            coords = torch.zeros(1)
        
        if self.get_indices:
            return coords, pixels, mask, idx
        else:
            return coords, pixels, mask

class ImageChunkDataset(Dataset):
    def __init__(self, imstack, patchsize):
        super().__init__()
        
        self.imstack = imstack
        self.nimg, self.H, self.W = imstack.shape
        self.patchsize = patchsize
        
        self.patch_coords = xy_mgrid(patchsize[0], patchsize[1])
        
        self.nH = int(np.ceil(self.H/patchsize[0]))
        self.nW = int(np.ceil(self.W/patchsize[1]))
        
    def __len__(self):
        return (self.nH * self.nW)
    
    def __getitem__(self, idx):        
        w_idx = int(idx%self.nH)
        h_idx = int((idx - w_idx)//self.nH)
        
        h1 = h_idx*self.patchsize[0]
        h2 = h_idx*self.patchsize[0] + self.patchsize[0]
        
        w1 = w_idx*self.patchsize[1]
        w2 = w_idx*self.patchsize[1] + self.patchsize[1]
        
        if h2 > self.H:
            h1 = self.H - self.patchsize[0]
            h2 = self.H
        
        if w2 > self.W:
            w1 = self.W - self.patchsize[1]
            w2 = self.W
            
        img = torch.tensor(self.imstack[:, h1:h2, w1:w2])
        pixels = img.reshape(-1, 1)
        
        coords = torch.clone(self.patch_coords)
        coords[:, 0] = coords[:, 0] + w1
        coords[:, 1] = coords[:, 1] + h1
        
        coords = torch.repeat_interleave(coords, self.nimg, 0)
        
        return coords, pixels
    
def load_config(configpath):
    '''
        Load configuration file
    '''
    parser = configparser.ConfigParser()
    parser.read(configpath)
    
    params_dict = dict()
    
    for section in parser.keys():
        for key in parser[section].keys():
            token = parser[section][key]
            if token == 'False':
                params_dict[key] = False
            elif token == 'True':
                params_dict[key] = True
            elif '.' in token:
                params_dict[key] = float(token)
            else:
                try:
                    params_dict[key] = int(token)
                except:
                    params_dict[key] = token
                    
    return params_dict