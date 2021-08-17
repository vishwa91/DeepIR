#!/usr/bin/env python

import torch

class TVNorm():
    def __init__(self, mode='l1'):
        self.mode = mode
    def __call__(self, img):
        grad_x = img[..., 1:, 1:] - img[..., 1:, :-1]
        grad_y = img[..., 1:, 1:] - img[..., :-1, 1:]
        
        if self.mode == 'isotropic':
            #return torch.sqrt(grad_x.abs().pow(2) + grad_y.abs().pow(2)).mean()
            return torch.sqrt(grad_x**2 + grad_y**2).mean()
        elif self.mode == 'l1':
            return abs(grad_x).mean() + abs(grad_y).mean()
        else:
            return (grad_x.pow(2) + grad_y.pow(2)).mean()     

class HessianNorm():
    def __init__(self):
        pass
    def __call__(self, img):
        # Compute Individual derivatives
        fxx = img[..., 1:-1, :-2] + img[..., 1:-1, 2:] - 2*img[..., 1:-1, 1:-1]
        fyy = img[..., :-2, 1:-1] + img[..., 2:, 1:-1] - 2*img[..., 1:-1, 1:-1]
        fxy = img[..., :-1, :-1] + img[..., 1:, 1:] - \
              img[..., 1:, :-1] - img[..., :-1, 1:]
          
        return torch.sqrt(fxx.abs().pow(2) +\
                          2*fxy[..., :-1, :-1].abs().pow(2) +\
                          fyy.abs().pow(2)).mean()
    
class L1Norm():
    def __init__(self):
        pass
    def __call__(self, x1, x2):
        return abs(x1 - x2).mean()        
    
class PoissonNorm():
    def __init__(self):
        pass
    def __call__(self, x1, x2):
        return (x1 - torch.log(x1 + 1e-12)*x2).mean()

class L2Norm():
    def __init__(self):
        pass
    def __call__(self, x1, x2):
        return ((x1 - x2).pow(2)).mean()    
