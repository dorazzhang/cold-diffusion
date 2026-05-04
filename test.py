import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import os
import yaml
import argparse

from src.unet import UNet
from src.dataset import get_dataloader

def test(config, weights_path):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing testing on {device}...")

    # Model components
    model = UNet(
        in_channels=config['model']['in_channels'], 
        base_channels=config['model']['base_channels'], 
        time_emb_dim=config['model']['time_emb_dim']
    ).to(device)
    
    # Load the trained weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Cannot find weights at {weights_path}.")
    
    # Handle loading either a final state_dict or a saved checkpoint dictionary
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    print(f"Successfully loaded weights from {weights_path}")
    
    # Initialize degradation
    degradation = Degradation(
        image_size=config['dataset']['image_size'],
        channels=config['model']['in_channels'],
        timesteps=config['degradation']['timesteps']
    ).to(device)

    # Initialize Metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    # Data pipeline: CRITICAL - train=False loads the unseen test split
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

    # Testing loop
    model.eval() # Disable dropout and batch norm tracking
    
    total_l1_loss = 0.0
    total_rmse = 0.0
    total_ssim = 0.0
    
    progress_bar = tqdm(dataloader, desc="Evaluating Test Set")
    
    # Disable gradient calculation entirely to save memory and speed up inference
    with torch.no_grad():
        for batch_x0, _ in progress_bar:
            batch_x0 = batch_x0.to(device)
            current_batch_size = batch_x0.shape[0]

            # Sample random timestamps for the test batch
            t = torch.randint(1, config['degradation']['timesteps'], (current_batch_size,), device=device)

            # Apply degradation
            x_t = degradation(batch_x0, t)

            # Unet prediction
            predicted_x0 = model(x_t, t)

            # Compute metrics
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
            
    # Calculate final averages over the entire test set
    avg_l1 = total_l1_loss / len(dataloader)
    avg_rmse = total_rmse / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    
    print("\n" + "="*50)
    print("FINAL TEST SET METRICS")
    print("="*50)
    print(f"Average L1 Loss: {avg_l1:.6f}")
    print(f"Average RMSE:    {avg_rmse:.6f}")
    print(f"Average SSIM:    {avg_ssim:.6f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cold Diffusion Model on Test Set")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--degradation",
        type=str,
        default="blur",
        help="Type of degradation to apply (i.e. 'blur' or 'pixelate')"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Specific path to a .pt weights file (defaults to the one in config)"
    )
    args = parser.parse_args()

    # Load YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Determine weights path
    weights_path = args.weights
    if weights_path is None:
        weights_path = os.path.join(config['training']['output_dir'], config['training']['save_name'])

    # Import degradation class
    if args.degradation == "blur":
        from src.degradations.blur import BlurDegradation as Degradation
    elif args.degradation == "pixelate":
        from src.degradations.pixelate import PixelateDegradation as Degradation
    else:
        raise ValueError("Unsupported degradation type. Choose 'blur' or 'pixelate'.")

    # Run testing
    test(config, weights_path)