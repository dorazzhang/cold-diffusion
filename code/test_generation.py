import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import os

from src.dataset import get_dataloader
import utils

def evaluate_pipeline(config, weights_path, algorithm, max_batches=None):
    device = utils.get_device()
    print(f"Initializing Full Pipeline Evaluation on {device}...")
    print(f"Algorithm: {algorithm.upper()}")

    model = utils.build_model(config, device)
    utils.load_weights(model, weights_path, device)
    print(f"Loaded weights from {weights_path}")

    degradation = utils.build_degradation(config, device)
    timesteps = config['degradation']['timesteps']

    if algorithm == 'adafusion':
        from src.adafusion import Sampler
    elif algorithm == 'direct':
        from src.direct import Sampler
    elif algorithm == 'projection':
        from src.projection import Sampler
    elif algorithm != 'degraded':
        from src.sampler import Sampler
        
    if algorithm != 'degraded':
        sampler = Sampler(model, degradation, device)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048)

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
    total_rmse, total_ssim, batches_processed = 0.0, 0.0, 0
    limit = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    progress_bar = tqdm(dataloader, total=limit, desc="Evaluating Full Pipeline")
    
    def to_uint8_image(tensor):
        img = torch.clamp((tensor + 1) / 2, 0, 1)
        return (img * 255).to(torch.uint8)

    with torch.no_grad():
        for batch_x0, _ in progress_bar:
            if batches_processed >= limit:
                break
                
            batch_x0 = batch_x0.to(device)
            current_batch_size = batch_x0.shape[0]

            t_max = torch.full((current_batch_size,), timesteps, device=device, dtype=torch.long)
            x_T = degradation(batch_x0, t_max)

            if algorithm != 'degraded':
                generated_x0, _, _ = sampler.sample(x_T, timesteps, save_every=timesteps+1)
            else:
                generated_x0 = x_T

            rmse = torch.sqrt(F.mse_loss(generated_x0, batch_x0))
            ssim = ssim_metric(generated_x0, batch_x0)

            total_rmse += rmse.item()
            total_ssim += ssim.item()

            real_uint8 = to_uint8_image(batch_x0)
            fake_uint8 = to_uint8_image(generated_x0)

            if real_uint8.shape[1] == 1:
                real_uint8 = real_uint8.repeat(1, 3, 1, 1)
                fake_uint8 = fake_uint8.repeat(1, 3, 1, 1)

            fid_metric.update(real_uint8.cpu(), real=True)
            fid_metric.update(fake_uint8.cpu(), real=False)
            
            batches_processed += 1
            progress_bar.set_postfix({"RMSE": f"{rmse.item():.4f}", "SSIM": f"{ssim.item():.4f}"})
            
    avg_rmse = total_rmse / batches_processed
    avg_ssim = total_ssim / batches_processed
    
    print("\nComputing final FID score (this may take a moment)...")
    final_fid = fid_metric.compute()
    
    print("FINAL PIPELINE GENERATION METRICS")
    print(f"Total Images Evaluated: {batches_processed * config['dataset']['batch_size']}")
    print(f"Average RMSE: {avg_rmse:.6f}")
    print(f"Average SSIM: {avg_ssim:.6f}")
    print(f"FID Score:    {final_fid.item():.4f}")

if __name__ == "__main__":
    parser = utils.setup_basic_parser("Evaluate the full Cold Diffusion pipeline")
    parser.add_argument("--algorithm", type=str, default="alg2", choices=['alg2', 'adafusion', 'direct', 'projection', 'degraded'])
    parser.add_argument("--weights", type=str, default=None, help="Path to weights/checkpoint")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit number of batches")
    args = parser.parse_args()

    config = utils.load_config(args.config)
    weights_path = args.weights or os.path.join(config['training']['output_dir'], config['training']['save_name'])

    evaluate_pipeline(config, weights_path, args.algorithm, args.max_batches)