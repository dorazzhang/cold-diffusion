import torch
import torch.nn.functional as F
from tqdm import tqdm

def project_consistency(x, y, block_size):
    """
    Project x onto the constraint that its block means match the target blocky image y.
    x: (B, C, H, W) Current step prediction
    y: (B, C, H, W) Target blocky image -> D(x0_hat, s-1)
    block_size: int, side length of each pixelation block
    """
    # 1. Get the average of the current hallucinated pixels in each block
    current_means = F.avg_pool2d(x, kernel_size=block_size)
    
    # 2. Blow that up to full resolution so it matches y's dimensions
    current_means_full = F.interpolate(current_means, scale_factor=block_size, mode='nearest')
    
    # 3. Calculate the difference between the target blocky image and our current means
    correction_full = y - current_means_full
    
    # 4. Apply the exact correction to the original image
    return x + correction_full


class Sampler:
    """
    Implements Consistency-Projected Algorithm 2 for discrete degradations.
    """
    def __init__(self, model, degradation, device='cpu'):
        self.model = model
        self.degradation = degradation
        self.device = device

    @torch.no_grad()
    def sample(self, x_T, t, save_every=5, use_projection=True):
        print(f"Sampling with Algorithm 2 (Projection: {use_projection})")
        self.model.eval()
        x_t = x_T.to(self.device)
        batch_size = x_t.shape[0]

        x_t_history = []
        x0_hat_history = []

        for s in tqdm(range(t, 0, -1), desc="Sampling"):
            t_batch = torch.full((batch_size,), s, device=self.device, dtype=torch.long)
            x0_hat = self.model(x_t, t_batch)

            # FIXED: Swapped 't' back to 's' so intermediate frames save correctly
            if s % save_every == 0 or s == t or s == 1:
                # We clone and move to CPU to avoid blowing up the GPU VRAM
                x_t_history.append((s, x_t.clone().cpu()))
                x0_hat_history.append((s, x0_hat.clone().cpu()))

            if s == 1:
                x_t = x0_hat
            else:
                # Forward degradations
                D_x0_t = self.degradation(x0_hat, t_batch)

                t_minus_1_batch = torch.full((batch_size,), s-1, device=self.device, dtype=torch.long)
                D_x0_t_minus_1 = self.degradation(x0_hat, t_minus_1_batch)

                # Standard Update rule
                x_t = x_t - D_x0_t + D_x0_t_minus_1
              
                # --- CONSISTENCY PROJECTION ---
                # Only run if toggled ON and the degradation is actually Pixelation (has resolution_routine)
                if use_projection and hasattr(self.degradation, 'resolution_routine'):
                    
                    # Dynamically calculate the target resolution dimension
                    dec_size = self.degradation.image_size - self.degradation.image_size // 2**(s)
                    target_res = self.degradation.image_size - dec_size
                    
                    # Calculate block size
                    block_size = self.degradation.image_size // target_res
                    
                    # Apply the mathematical cage
                    x_t = project_consistency(x_t, D_x0_t_minus_1, block_size)
 
        return x_t, x_t_history, x0_hat_history