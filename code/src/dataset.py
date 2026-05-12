"""
dataset.py defines the raw dataset-loading pipeline.

This file is responsible for downloading (if necessary), normalizing, and 
batching the pristine images. It does NOT handle physics or degradation.
That is handled natively on the GPU in the training loop for maximum throughput.
"""

import torch
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_raw_dataset(
    dataset_name: str,
    root: str = "./data",
    train: bool = True,
    image_size: int = 32,
    max_samples: Optional[int] = None,
):
    """
    Loads the raw, clean dataset.
    """
    dataset_name = dataset_name.lower()
    root = Path(root)

    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        dataset = datasets.MNIST(root=str(root), train=train, download=True, transform=transform)

    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR10(root=str(root), train=train, download=True, transform=transform)

    elif dataset_name == "celeba":
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        split = "train" if train else "test"
        dataset = datasets.CelebA(root=str(root), split=split, download=True, transform=transform)

    else:
        raise ValueError("Unknown dataset_name. Choose from 'mnist', 'cifar10', or 'celeba'.")

    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = Subset(dataset, range(max_samples))

    return dataset


def get_dataloader(
    dataset_name: str,
    root: str = "./data",
    train: bool = True,
    image_size: int = 32,
    batch_size: int = 128,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
):
    """
    Creates a PyTorch DataLoader that feeds clean images to the training loop.
    """
    dataset = get_raw_dataset(
        dataset_name=dataset_name,
        root=root,
        train=train,
        image_size=image_size,
        max_samples=max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True, # Keeps data in page-locked memory for faster GPU transfers
        drop_last=train,
    )

    return dataloader