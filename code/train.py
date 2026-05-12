import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import os

from src.dataset import get_dataloader
import utils

def train(config, resume_path=None):
    device = utils.get_device()
    print(f"Initializing training on {device}...")

    os.makedirs(config['training']['output_dir'], exist_ok=True)

    model = utils.build_model(config, device)
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    start_epoch = 0
    if resume_path:
        print(f"Resuming training from checkpoint: {resume_path}")
        start_epoch = utils.load_weights(model, resume_path, device, optimizer)
        print(f"Successfully resumed at epoch {start_epoch}")

    degradation = utils.build_degradation(config, device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

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

    epochs = config['training']['epochs']
    model.train()
    
    for epoch in range(start_epoch, epochs):
        epoch_l1_loss, epoch_rmse, epoch_ssim = 0.0, 0.0, 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_x0, _ in progress_bar:
            batch_x0 = batch_x0.to(device)
            current_batch_size = batch_x0.shape[0]

            optimizer.zero_grad()
            t = torch.randint(1, config['degradation']['timesteps'] + 1, (current_batch_size,), device=device)

            x_t = degradation(batch_x0, t)
            predicted_x0 = model(x_t, t)

            loss = F.l1_loss(predicted_x0, batch_x0)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(predicted_x0, batch_x0))
                ssim = ssim_metric(predicted_x0, batch_x0)

            epoch_l1_loss += loss.item()
            epoch_rmse += rmse.item()
            epoch_ssim += ssim.item()
            
            progress_bar.set_postfix({
                "L1": f"{loss.item():.4f}", 
                "RMSE": f"{rmse.item():.4f}", 
                "SSIM": f"{ssim.item():.4f}"
            })
        
        avg_l1 = epoch_l1_loss / len(dataloader)
        avg_rmse = epoch_rmse / len(dataloader)
        avg_ssim = epoch_ssim / len(dataloader)
        
        print(f"Epoch {epoch+1} Completed | L1 Loss: {avg_l1:.4f} | RMSE: {avg_rmse:.4f} | SSIM: {avg_ssim:.4f}\n")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config["training"]["output_dir"], f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_l1,
                "config": config,
            }, checkpoint_path)

    save_path = os.path.join(config['training']['output_dir'], config['training']['save_name'])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Training successfully completed. Weights saved to {save_path}")

if __name__ == "__main__":
    parser = utils.setup_basic_parser("Train Cold Diffusion Model")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()

    config = utils.load_config(args.config)
    train(config, args.resume)