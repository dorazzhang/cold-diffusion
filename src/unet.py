import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal positional embeddings for the given time steps.
        Args:
            time (torch.Tensor): A tensor of shape (batch_size,) containing the time steps for which to generate embeddings.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, embedding_dim) containing the sinusoidal positional embeddings for the input time steps.
        """
        device = time.device
        half_dim = self.embedding_dim // 2
        # Calculate the frequencies for the sine/cosine waves
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        # Interleave sine and cosine waves
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        return embeddings
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()


    def forward(self, x, t_emb):
        # First convolutional layer
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        # Add time embedding
        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        x = x + t_emb
        # Second convolutional layer
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, out_channels=None, time_emb_dim=256):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU()
        )
        self.down_block1 = Block(in_channels, base_channels, time_emb_dim * 4)
        self.down_block2 = Block(base_channels, base_channels * 2, time_emb_dim * 4)
        self.down_block3 = Block(base_channels * 2, base_channels * 4, time_emb_dim * 4)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up_block1 = Block(base_channels * 4, base_channels * 2, time_emb_dim * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up_block2 = Block(base_channels * 2, base_channels, time_emb_dim * 4)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        # Downsample
        d1 = self.down_block1(x, t_emb)
        p1 = self.pool(d1)
        d2 = self.down_block2(p1, t_emb)
        p2 = self.pool(d2)
        d3 = self.down_block3(p2, t_emb)
        # Upsample
        u1 = self.up1(d3)
        s1 = torch.cat([u1, d2], dim=1)
        u1_out = self.up_block1(s1, t_emb)
        u2 = self.up2(u1_out)
        s2 = torch.cat([u2, d1], dim=1)
        u2_out = self.up_block2(s2, t_emb)
        out = self.final_conv(u2_out)
        return out