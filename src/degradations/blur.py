from .base import BaseDegradation
import torch
import numpy as np
from torch import nn
import torchgeometry as tgm

class BlurDegradation(BaseDegradation):
    def __init__(self, image_size, channels=3, time_steps=20, blur_size=10, blur_std=0.1, blur_routine='Constant'):
        self.image_size = image_size
        self.channels = channels
        self.num_timesteps = int(time_steps)
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
        for i in range(t):
            with torch.no_grad():
                x0 = self.gaussian_kernels[i](x0)
        degraded = x0
        return degraded