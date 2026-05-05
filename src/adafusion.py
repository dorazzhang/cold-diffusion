import torch
from tqdm import tqdm

class Sampler:
    """
    Implements Cold Diffusion Sampling algorithms.
    """
    def __init__(self, model, degradation, device='cpu'):
        self.model = model
        self.degradation = degradation
        self.device = device

    @torch.no_grad()
    def sample(self, x_T, t, save_every=5, beta1=0.9, beta2=0.999, lam=0.01, eps=1e-8):
        print("Sampling with Adafusion")
        self.model.eval()
        x_t = x_T.to(self.device)
        batch_size = x_t.shape[0]

        x_t_history = []
        x0_hat_history = []

        # Initialize Adam variables (m and v start as tensors of zeros matching x_t)
        m = torch.zeros_like(x_t)
        v = torch.zeros_like(x_t)
        k = 0

        for s in tqdm(range(t, 0, -1), desc="Sampling (Adam Residual)"):
            k += 1 # Increment step counter for bias correction
            
            t_batch = torch.full((batch_size,), s, device=self.device, dtype=torch.long)
            
            # Predict clean image: R(x_s, s)
            x0_hat = self.model(x_t, t_batch)

            if s % save_every == 0 or s == t or s == 1:
                x_t_history.append((s, x_t.clone().cpu()))
                x0_hat_history.append((s, x0_hat.clone().cpu()))

            if s == 1:
                x_t = x0_hat
            else:
                # Forward degradations
                D_x0_t = self.degradation(x0_hat, t_batch)

                t_minus_1_batch = torch.full((batch_size,), s-1, device=self.device, dtype=torch.long)
                D_x0_t_minus_1 = self.degradation(x0_hat, t_minus_1_batch)

                # Calculate the "gradient" step for the pixels: g_s
                g_s = -D_x0_t + D_x0_t_minus_1

                # Update first moment (momentum)
                m = beta1 * m + (1 - beta1) * g_s
                
                # Update second moment (variance), using element-wise multiplication
                v = beta2 * v + (1 - beta2) * (g_s * g_s)

                # Bias correction for early steps
                m_hat = m / (1 - beta1**k)
                v_hat = v / (1 - beta2**k)

                # Calculate Adam adjustment term: a_s
                a_s = m_hat / (torch.sqrt(v_hat) + eps)

                # Final update rule: x_{s-1} = x_s + g_s + \lambda a_s
                x_t = x_t + g_s + (lam * a_s)
 
        return x_t, x_t_history, x0_hat_history