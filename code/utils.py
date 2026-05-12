import os
import yaml
import argparse
import torch

from src.unet import UNet

def get_device():
    """Returns the device available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def build_model(config, device):
    """Initializes and returns the UNet model."""
    model = UNet(
        in_channels=config['model']['in_channels'], 
        base_channels=config['model']['base_channels'], 
        time_emb_dim=config['model']['time_emb_dim']
    ).to(device)
    return model

def build_degradation(config, device):
    """Initializes degradation based on YAML settings."""
    deg_config = config['degradation']
    deg_type = deg_config.get('type', 'blur')
    
    if deg_type == "blur":
        from src.degradations.blur import BlurDegradation as Degradation
    elif deg_type == "pixelate":
        from src.degradations.pixelate import PixelateDegradation as Degradation
    else:
        raise ValueError(f"Unsupported degradation type: {deg_type}. Choose 'blur' or 'pixelate'.")

    # Pass any extra kwargs defined in the YAML directly to the degradation class
    kwargs = deg_config.get('kwargs', {})
    
    return Degradation(
        image_size=config['dataset']['image_size'],
        channels=config['model']['in_channels'],
        timesteps=deg_config['timesteps'],
        **kwargs
    ).to(device)

def load_weights(model, weights_path, device, optimizer=None):
    """Weight loading supporting both raw state dicts and training checkpoints."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Cannot find weights at {weights_path}")
        
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    start_epoch = 0
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
    else:
        model.load_state_dict(checkpoint)
        
    return start_epoch

def setup_basic_parser(description):
    """Creates argument parser."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    return parser