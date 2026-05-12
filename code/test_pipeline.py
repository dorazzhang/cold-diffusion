import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import os
import yaml
import argparse

from src.unet import UNet
from src.dataset import get_dataloader

def evaluate_pipeline(config, weights_path, algorithm, max_batches=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing Full Pipeline Evaluation on {device}...")

    # Initialize model
    model = UNet(
        in_channels=config['model']['in_channels'], 
        base_channels=config['model']['base_channels'], 
        time_emb_dim=config['model']['time_emb_dim']
    ).to(device)
    
    # Load weights (handling both raw state_dicts and checkpoints)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Cannot find weights at {weights_path}.")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded weights from {weights_path}")

    # Initialize degradation
    timesteps = config['degradation']['timesteps']
    # NOTE: Ensure your degradation parameters match what you want to test!
    degradation = Degradation(
        image_size=config['dataset']['image_size'],
        channels=config['model']['in_channels'],
        timesteps=timesteps,
        # Set your routine here, e.g., resolution_routine='Incremental_factor_2'
    ).to(device)

    # Initialize the chosen sampler
    if algorithm == 'adafusion':
        from src.adafusion import Sampler
        print("Using Adafusion (Algorithm 3) Sampler...")
    else:
        from src.sampler import Sampler
        print("Using Standard (Algorithm 2) Sampler...")
        
    sampler = Sampler(model, degradation, device)

    # Initialize Metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    # FID feature size 64 is faster for quick evaluation. Use 2048 for publication-ready metrics.
    fid_metric = FrechetInceptionDistance(feature=64)

    # Data pipeline
    dataset_name = config['dataset']['name']
    print(f"Loading {dataset_name.upper()} TEST dataset...")
    dataloader = get_dataloader(
        dataset_name=dataset_name,
        root=config['dataset']['root'],
        train=False, 
        image_size=config['dataset']['image_size'],
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers']
    )

    model.eval()
    
    total_rmse = 0.0
    total_ssim = 0.0
    batches_processed = 0
    
    # Calculate how many batches to run
    limit = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    progress_bar = tqdm(dataloader, total=limit, desc="Evaluating Full Pipeline")
    
    # Helper to convert [-1, 1] floats to [0, 255] uint8 for FID
    def to_uint8_image(tensor):
        img = (tensor + 1) / 2 # move to [0, 1]
        img = torch.clamp(img, 0, 1)
        return (img * 255).to(torch.uint8)

    with torch.no_grad():
        for batch_x0, _ in progress_bar:
            if batches_processed >= limit:
                break
                
            batch_x0 = batch_x0.to(device)
            current_batch_size = batch_x0.shape[0]

            # 1. Apply maximum degradation to create starting state (x_T)
            t_max = torch.full((current_batch_size,), timesteps, device=device, dtype=torch.long)
            x_T = degradation(batch_x0, t_max)

            # 2. Run the full reverse sampler pipeline
            # Note: We don't need the history lists for metrics, just the final image
            generated_x0, _, _ = sampler.sample(x_T, timesteps, save_every=timesteps+1)

            # 3. Calculate batch RMSE and SSIM (using the raw [-1, 1] tensors)
            rmse = torch.sqrt(F.mse_loss(generated_x0, batch_x0))
            ssim = ssim_metric(generated_x0, batch_x0)

            total_rmse += rmse.item()
            total_ssim += ssim.item()

            # 4. Format images for FID
            real_uint8 = to_uint8_image(batch_x0)
            fake_uint8 = to_uint8_image(generated_x0)

            # FID requires 3 channels. If MNIST (1 channel), repeat it across RGB.
            if real_uint8.shape[1] == 1:
                real_uint8 = real_uint8.repeat(1, 3, 1, 1)
                fake_uint8 = fake_uint8.repeat(1, 3, 1, 1)

            # Update FID internal states (real=True for ground truth, real=False for generated)
            fid_metric.update(real_uint8.cpu(), real=True)
            fid_metric.update(fake_uint8.cpu(), real=False)
            
            batches_processed += 1
            progress_bar.set_postfix({
                "RMSE": f"{rmse.item():.4f}", 
                "SSIM": f"{ssim.item():.4f}"
            })
            
    # Calculate final averages
    avg_rmse = total_rmse / batches_processed
    avg_ssim = total_ssim / batches_processed
    
    print("\nComputing final FID score (this may take a moment)...")
    final_fid = fid_metric.compute()
    
    print("\n" + "="*50)
    print("FINAL PIPELINE GENERATION METRICS")
    print("="*50)
    print(f"Total Images Evaluated: {batches_processed * config['dataset']['batch_size']}")
    print(f"Average RMSE: {avg_rmse:.6f}")
    print(f"Average SSIM: {avg_ssim:.6f}")
    print(f"FID Score:    {final_fid.item():.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the full Cold Diffusion pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--degradation", type=str, default="blur")
    parser.add_argument("--algorithm", type=str, default="alg2", choices=['alg2', 'adafusion'], help="Which sampler to use")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights/checkpoint")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit number of batches for faster evaluation")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    weights_path = args.weights if args.weights else os.path.join(config['training']['output_dir'], config['training']['save_name'])

    if args.degradation == "blur":
        from src.degradations.blur import BlurDegradation as Degradation
    elif args.degradation == "pixelate":
        from src.degradations.pixelate import PixelateDegradation as Degradation
    else:
        raise ValueError("Unsupported degradation")

    evaluate_pipeline(config, weights_path, args.algorithm, args.max_batches)