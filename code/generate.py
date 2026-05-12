import torch
import torchvision
import os

from src.dataset import get_dataloader
import utils

def save_image_grid(tensor_batch, filepath):
    """Helper to convert [-1, 1] to [0, 1] and save."""
    tensor_batch = torch.clamp((tensor_batch + 1) / 2, 0, 1)
    torchvision.utils.save_image(tensor_batch, filepath, nrow=4)

def generate(config, algorithm="sampler"):
    device = utils.get_device()
    print(f"Initializing Cold Diffusion Generation on {device}...")

    model = utils.build_model(config, device)
    weights_path = os.path.join(config['training']['output_dir'], config['training']['save_name'])
    utils.load_weights(model, weights_path, device)
    print(f"Loaded weights from {weights_path}")

    degradation = utils.build_degradation(config, device)
    timesteps = config['degradation']['timesteps']

    if algorithm == "direct":
        from src.direct import Sampler
    elif algorithm == "adafusion":
        from src.adafusion import Sampler
    elif algorithm == "projection":
        from src.projection import Sampler
    else:
        from src.sampler import Sampler
        
    sampler = Sampler(model, degradation, device)
    
    print("Loading test images for restoration...")
    test_loader = get_dataloader(
        dataset_name=config['dataset']['name'],
        root=config['dataset']['root'],
        train=False, 
        image_size=config['dataset']['image_size'],
        batch_size=16, # Fixed display batch size
        num_workers=1
    )
    
    real_clean_images, _ = next(iter(test_loader))
    real_clean_images = real_clean_images.to(device)
    
    print("Applying maximum degradation to create starting state (x_T)...")
    t_max = torch.full((real_clean_images.shape[0],), timesteps, device=device, dtype=torch.long)
    x_T = degradation(real_clean_images, t_max)

    print(f"Running {algorithm.capitalize()} Reverse Loop...")
    generated_images, xt_hist, x0_hat_hist = sampler.sample(x_T, timesteps, save_every=1)

    os.makedirs("samples/xt_progression", exist_ok=True)
    os.makedirs("samples/x0_predictions", exist_ok=True)
    
    print("Saving intermediate progression frames...")
    for t, xt_batch in xt_hist:
        save_image_grid(xt_batch, f"samples/xt_progression/xt_step_{t:03d}.png")

    for t, x0_hat_batch in x0_hat_hist:
        save_image_grid(x0_hat_batch, f"samples/x0_predictions/x0_hat_step_{t:03d}.png")

    final_save_path = "samples/generated_grid_final.png"
    save_image_grid(generated_images, final_save_path)
    save_image_grid(real_clean_images, "samples/ground_truth.png")
    save_image_grid(x_T, "samples/starting_state_xT.png")
    
    print(f"\nSuccess! Final images saved to {final_save_path}")

if __name__ == "__main__":
    parser = utils.setup_basic_parser("Generate Images using Cold Diffusion")
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="sampler", 
        choices=["sampler", "adafusion", "direct", "projection"],
        help="Which sampling algorithm to use"
    )
    args = parser.parse_args()

    config = utils.load_config(args.config)
    generate(config, args.algorithm)