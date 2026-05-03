from .base import BaseDegradation
import torch
import torch.nn.functional as F

class PixelateDegradation(BaseDegradation):
    def __init__(self, image_size, channels=3, time_steps=20, resolution_routine='Incremental'):
        self.image_size = image_size
        self.channels = channels
        self.num_timesteps = int(time_steps)
        self.resolution_routine = resolution_routine

        self.func = self.get_funcs()
    
    def transform_func(self, img, dec_size, mode):
        img_1 = F.interpolate(img, size=img.shape[2] - dec_size, mode=mode, antialias=False)
        img_1 = F.interpolate(img_1, size=img.shape[2], mode='nearest-exact', antialias=False)
        return img_1

    def get_funcs(self):
        all_funcs = []
        for i in range(self.num_timesteps):
            if self.resolution_routine == 'Incremental':
                all_funcs.append((lambda img, d=i, mode='bicubic': self.transform_func(img, d, mode)))
            elif self.resolution_routine == 'Incremental_factor_2':
                all_funcs.append((lambda img, d=self.image_size -self.image_size // 2**(i+1), mode='bicubic': self.transform_func(img, d, mode)))
            
        return all_funcs

    def __call__(self, x0, t):
        for i in range(t):
            with torch.no_grad():
                x0 = self.func[i](x0)
        degraded = x0
        return degraded