import torch
import torchvision
import yaml
import argparse
import os

from src.unet import UNet
# from src.sampler import Sampler # Algorithm 2
from src.adafusion import Sampler # Adafusion Algorithm
from src.dataset import get_dataloader  # Added to load real images

def generate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing Cold Diffusion Generation on {device}...")

    # Initialize model
    model = UNet(
        in_channels=config['model']['in_channels'], 
        base_channels=config['model']['base_channels'], 
        time_emb_dim=config['model']['time_emb_dim']
    ).to(device)
    
    # Load trained weights
    weights_path = os.path.join(config['training']['output_dir'], config['training']['save_name'])
    # weights_path = 'outputs/mnist/pixelate/checkpoint_epoch_300.pt'

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Cannot find weights at {weights_path}. Did you run train.py first?")
        
    # Load the file (set weights_only=False because checkpoints contain ints/floats like epoch and loss)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Check if it's a bundled checkpoint or just raw weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    print(f"Loaded weights from {weights_path}")

    # Initialize degradation
    timesteps = config['degradation']['timesteps']
    degradation = Degradation(
        image_size=config['dataset']['image_size'],
        channels=config['model']['in_channels'],
        timesteps=timesteps,
        # blur - mnist
        blur_size=11,
        blur_std=7.0,
        blur_routine='Constant',
        # blur - cifar10
        # blur_routine='Special_6_routine',
        # pixelate - mnist, cifar10
        # resolution_routine='Incremental_factor_2',
    ).to(device)

    # Initialize sampler
    sampler = Sampler(model, degradation, device)

    # Create starting state (x_T)
    batch_size = 16
    
    # True Cold Diffusion: Restoration from heavily degraded real images
    print("Loading test images for restoration...")
    test_loader = get_dataloader(
        dataset_name=config['dataset']['name'],
        root=config['dataset']['root'],
        train=False, # Use the test set so the model hasn't memorized them
        image_size=config['dataset']['image_size'],
        batch_size=batch_size,
        num_workers=1
    )
    
    # Grab one batch of real clean images
    real_clean_images, _ = next(iter(test_loader))
    real_clean_images = real_clean_images.to(device)
    
    print("Applying maximum degradation to create starting state (x_T)...")
    # Apply maximum degradation (t = timesteps)
    t_max = torch.full((real_clean_images.shape[0],), timesteps, device=device, dtype=torch.long)
    x_T = degradation(real_clean_images, t_max)

    # Run Algorithm 2
    print("Running Algorithm 2 Reverse Loop...")
    generated_images, xt_hist, x0_hat_hist = sampler.sample(x_T, timesteps, save_every=1)

    # Post-processing and saving
    os.makedirs("samples/xt_progression", exist_ok=True)
    os.makedirs("samples/x0_predictions", exist_ok=True)
    
    # Helper function to format and save a batch of images
    def save_image_grid(tensor_batch, filepath):
        # Convert from [-1, 1] to [0, 1]
        tensor_batch = (tensor_batch + 1) / 2
        tensor_batch = torch.clamp(tensor_batch, 0, 1)
        torchvision.utils.save_image(tensor_batch, filepath, nrow=4)

    print("Saving intermediate progression frames...")
    
    # Save the history of the degraded images (x_t)
    for t, xt_batch in xt_hist:
        filepath = f"samples/xt_progression/xt_step_{t:03d}.png"
        save_image_grid(xt_batch, filepath)

    # Save the history of what the model THOUGHT the clean image was at each step (x0_hat)
    for t, x0_hat_batch in x0_hat_hist:
        filepath = f"samples/x0_predictions/x0_hat_step_{t:03d}.png"
        save_image_grid(x0_hat_batch, filepath)

    # Save the final pristine output, along with the ground truth and starting state
    final_save_path = "samples/generated_grid_final.png"
    save_image_grid(generated_images, final_save_path)
    save_image_grid(real_clean_images, "samples/ground_truth.png")
    save_image_grid(x_T, "samples/starting_state_xT.png")
    
    print(f"\nSuccess! Final images saved to {final_save_path}")
    print("Intermediate frames saved in samples/xt_progression/ and samples/x0_predictions/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Images using Cold Diffusion")
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

    generate(config)