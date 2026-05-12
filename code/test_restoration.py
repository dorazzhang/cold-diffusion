import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import os

from src.dataset import get_dataloader
import utils

def test(config, weights_path):
    device = utils.get_device()
    print(f"Initializing testing on {device}...")

    model = utils.build_model(config, device)
    utils.load_weights(model, weights_path, device)
    print(f"Successfully loaded weights from {weights_path}")
    
    degradation = utils.build_degradation(config, device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

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
    total_l1_loss, total_rmse, total_ssim = 0.0, 0.0, 0.0
    
    progress_bar = tqdm(dataloader, desc="Evaluating Test Set")
    
    with torch.no_grad():
        for batch_x0, _ in progress_bar:
            batch_x0 = batch_x0.to(device)
            current_batch_size = batch_x0.shape[0]

            t = torch.randint(1, config['degradation']['timesteps'] + 1, (current_batch_size,), device=device)
            x_t = degradation(batch_x0, t)
            predicted_x0 = model(x_t, t)

            loss = F.l1_loss(predicted_x0, batch_x0)
            rmse = torch.sqrt(F.mse_loss(predicted_x0, batch_x0))
            ssim = ssim_metric(predicted_x0, batch_x0)

            total_l1_loss += loss.item()
            total_rmse += rmse.item()
            total_ssim += ssim.item()
            
            progress_bar.set_postfix({
                "L1": f"{loss.item():.4f}", 
                "RMSE": f"{rmse.item():.4f}", 
                "SSIM": f"{ssim.item():.4f}"
            })
            
    avg_l1 = total_l1_loss / len(dataloader)
    avg_rmse = total_rmse / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    
    print("FINAL TEST SET METRICS")
    print(f"Average L1 Loss: {avg_l1:.6f}")
    print(f"Average RMSE:    {avg_rmse:.6f}")
    print(f"Average SSIM:    {avg_ssim:.6f}")

if __name__ == "__main__":
    parser = utils.setup_basic_parser("Evaluate Cold Diffusion Model on Test Set")
    parser.add_argument("--weights", type=str, default=None, help="Specific path to a .pt weights file")
    args = parser.parse_args()

    config = utils.load_config(args.config)
    weights_path = args.weights or os.path.join(config['training']['output_dir'], config['training']['save_name'])
    
    test(config, weights_path)