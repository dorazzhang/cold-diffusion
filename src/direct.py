import torch
from tqdm import tqdm

class Sampler:
    """
    Implements direct R restoration prediction.
    """
    def __init__(self, model, degradation, device='cpu'):
        self.model = model
        self.degradation = degradation
        self.device = device

    @torch.no_grad()
    def sample(self, x_T, t, save_every=5):
        print("Sampling with direct R restoration prediction")
        self.model.eval()
        x_t = x_T.to(self.device)
        batch_size = x_t.shape[0]

        t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        x0_hat = self.model(x_t, t_batch)

        return x0_hat, [], []