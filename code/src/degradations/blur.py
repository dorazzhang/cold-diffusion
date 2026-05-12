from .base import BaseDegradation
import torch
import numpy as np
from torch import nn
import torchgeometry as tgm

class BlurDegradation(BaseDegradation, nn.Module):
    def __init__(self, image_size, channels=3, timesteps=20, blur_size=11, blur_std=7.0, blur_routine='Constant'):
        nn.Module.__init__(self)

        self.image_size = image_size
        self.channels = channels
        self.num_timesteps = int(timesteps)
        self.blur_size = blur_size
        self.blur_std = blur_std
        self.blur_routine = blur_routine

        self.gaussian_kernels = nn.ModuleList(self.get_kernels())
    
    def blur(self, dims, std):
        return tgm.image.get_gaussian_kernel2d(dims, std)

    def get_conv(self, dims, std, mode='circular'):
        kernel = self.blur(dims, std)
        conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=dims, padding=int((dims[0]-1)/2), padding_mode=mode,
                         bias=False, groups=self.channels)
        with torch.no_grad():
            kernel = torch.unsqueeze(kernel, 0)
            kernel = torch.unsqueeze(kernel, 0)
            kernel = kernel.repeat(self.channels, 1, 1, 1)
            conv.weight = nn.Parameter(kernel)

        return conv

    def get_kernels(self):
        kernels = []
        for i in range(self.num_timesteps):
            if self.blur_routine == 'Constant':
                kernels.append(self.get_conv((self.blur_size, self.blur_size), (self.blur_std, self.blur_std) ) )
            elif self.blur_routine == 'Exponential_reflect':
                ks = self.blur_size
                kstd = np.exp(self.blur_std * i)
                kernels.append(self.get_conv((ks, ks), (kstd, kstd), mode='reflect'))
            elif self.blur_routine == 'Exponential':
                ks = self.blur_size
                kstd = np.exp(self.blur_std * i)
                kernels.append(self.get_conv((ks, ks), (kstd, kstd)))
            elif self.blur_routine == 'Special_6_routine':
                ks = 11
                kstd = i/100 + 0.35
                kernels.append(self.get_conv((ks, ks), (kstd, kstd), mode='reflect'))
        return kernels
    
    def __call__(self, x0, t):
        degraded = torch.zeros_like(x0)
        batch_size = x0.shape[0]
        
        for b in range(batch_size):
            img = x0[b].unsqueeze(0) 
            
            target_step = t[b].item()
            
            with torch.no_grad():
                for i in range(target_step):
                    img = self.gaussian_kernels[i](img)
            
            degraded[b] = img.squeeze(0)
            
        return degraded