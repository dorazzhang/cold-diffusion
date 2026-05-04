import torch
from tqdm import tqdm

class Sampler:
    """
    Implements Algorithm 2 from the Cold Diffusion paper.
    """
    def __init__(self, model, degradation, device='cpu'):
        self.model = model
        self.degradation = degradation
        self.device = device

    @torch.no_grad()
    def sample(self, x_T, t, save_every=5):
        self.model.eval()
        x_t = x_T.to(self.device)
        batch_size = x_t.shape[0]

        x_t_history = []
        x0_hat_history = []

        for s in tqdm(range(t, 0, -1), desc="Sampling"):
            t_batch = torch.full((batch_size,), s, device=self.device, dtype=torch.long)
            x0_hat = self.model(x_t, t_batch)

            if t % save_every == 0 or t == t or t == 1:
                # We clone and move to CPU to avoid blowing up the GPU VRAM
                x_t_history.append((t, x_t.clone().cpu()))
                x0_hat_history.append((t, x0_hat.clone().cpu()))


            if s == 1:
                x_t = x0_hat
                break
            D_x0_t = self.degradation(x0_hat, t_batch)

            t_minus_1_batch = torch.full((batch_size,), s-1, device=self.device, dtype=torch.long)
            D_x0_t_minus_1 = self.degradation(x0_hat, t_minus_1_batch)

            x_t = x_t - D_x0_t + D_x0_t_minus_1
 
        return x_t, x_t_history, x0_hat_history