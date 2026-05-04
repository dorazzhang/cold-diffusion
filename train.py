import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import os
import yaml
import argparse

from src.unet import UNet
from src.dataset import get_dataloader

def train(config):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing training on {device}...")

    os.makedirs(config['training']['output_dir'], exist_ok=True)

    # Model components
    model = UNet(
        in_channels=config['model']['in_channels'], 
        base_channels=config['model']['base_channels'], 
        time_emb_dim=config['model']['time_emb_dim']
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    degradation = Degradation(
        image_size=config['dataset']['image_size'],
        channels=config['model']['in_channels'],
        timesteps=config['degradation']['timesteps']
    ).to(device)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    # Data pipeline
    dataset_name = config['dataset']['name']
    print(f"Loading {dataset_name.upper()} dataset...")
    dataloader = get_dataloader(
        dataset_name=dataset_name,
        root=config['dataset']['root'],
        train=True,
        image_size=config['dataset']['image_size'],
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers']
    )

    # Train loop
    epochs = config['training']['epochs']
    
    model.train()
    for epoch in range(epochs):
        epoch_l1_loss = 0.0
        epoch_rmse = 0.0
        epoch_ssim = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_x0, _ in progress_bar:
            batch_x0 = batch_x0.to(device)
            current_batch_size = batch_x0.shape[0]

            optimizer.zero_grad()

            # Sample random timestamps
            t = torch.randint(1, config['degradation']['timesteps'], (current_batch_size,), device=device)

            # Apply degradation
            x_t = degradation(batch_x0, t)

            # Unet prediction
            predicted_x0 = model(x_t, t)

            # Compute loss
            loss = F.l1_loss(predicted_x0, batch_x0)

            # Backprop and optimization step
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(predicted_x0, batch_x0))
                ssim = ssim_metric(predicted_x0, batch_x0)

            epoch_l1_loss += loss.item()
            epoch_rmse += rmse.item()
            epoch_ssim += ssim.item()
            
            # Update the progress bar with all three metrics
            progress_bar.set_postfix({
                "L1": f"{loss.item():.4f}", 
                "RMSE": f"{rmse.item():.4f}", 
                "SSIM": f"{ssim.item():.4f}"
            })
        
        # Calculate averages for the entire epoch
        avg_l1 = epoch_l1_loss / len(dataloader)
        avg_rmse = epoch_rmse / len(dataloader)
        avg_ssim = epoch_ssim / len(dataloader)
        
        print(f"Epoch {epoch+1} Completed | L1 Loss: {avg_l1:.4f} | RMSE: {avg_rmse:.4f} | SSIM: {avg_ssim:.4f}\n")

    # Save trained model weights
    save_path = os.path.join(config['training']['output_dir'], config['training']['save_name'])
    torch.save(model.state_dict(), save_path)
    print(f"Training successfully completed. Weights saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cold Diffusion Model")
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
    args = parser.parse_args()

    # Load YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Import degradation class
    if args.degradation == "blur":
        from src.degradations.blur import BlurDegradation as Degradation
    elif args.degradation == "pixelate":
        from src.degradations.pixelate import PixelateDegradation as Degradation
    else:
        raise ValueError("Unsupported degradation type. Choose 'blur' or 'pixelate'.")

    # Run training
    train(config)