"""
dataset.py defines the dataset-loading pipeline for cold diffusion training.

The raw datasets are stored as clean
images in the data folder. This file loads each clean image x0, samples a random timestep t, applies
the selected degradation D(x0, t), and returns both the clean and degraded images.

The training script should only call get_dataloader(...). That function
loads the requested dataset and returns a PyTorch DataLoader. When the training loop iterates over the DataLoader, 
the degraded image will be loadedon the fly.

Each batch returned to train.py contains:
    x0: the original clean image
    xt: the degraded image at timestep t
    t:  the sampled timestep

The model is then trained to predict:
    R_theta(xt, t) ≈ x0

Example usage in train.py:

    from dataset import get_dataloader
    from degradations.blur import BlurDegradation

    degradation = BlurDegradation(timesteps=100)

    train_loader = get_dataloader(
        dataset_name="cifar10",
        degradation=degradation,
        root="./data",
        train=True,
        image_size=32,
        timesteps=100,
        batch_size=128,
    )

    for batch in train_loader:
        x0 = batch["x0"]
        xt = batch["xt"]
        t = batch["t"]

        pred_x0 = model(xt, t)
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


class ColdDiffusionDataset(Dataset):
    """
    Dataset wrapper for cold diffusion training.

    Returns dict with:
        x0: clean image
        t: randomly sampled timestep
        xt: degraded image D(x0, t)
    """

    def __init__(
        self,
        dataset: Dataset,
        degradation: Callable,
        timesteps: int = 100,
    ):
        self.dataset = dataset
        self.degradation = degradation
        self.timesteps = timesteps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]

        if isinstance(item, tuple):
            x0 = item[0]
        else:
            x0 = item

        t = torch.randint(
            low=1,
            high=self.timesteps + 1,
            size=(1,),
            dtype=torch.long,
        ).squeeze(0)

        xt = self.degradation(x0, t)

        if torch.is_tensor(xt) and xt.dim() == 4:
            xt = xt[-1]

        sample = {
            "x0": x0,
            "xt": xt,
            "t": t,
        }

        return sample


def get_raw_dataset(
    dataset_name: str,
    root: str = "./data",
    train: bool = True,
    image_size: int = 32,
    max_samples: Optional[int] = None,
):
    """
    Loads the raw clean dataset.

    Args:
        dataset_name: "mnist", "cifar10", or "celeba"
        root: path to dataset folder
        train: whether to load train or test split
        image_size: output image size
        max_samples: optional subset size of dataset to load

    Returns:
        torchvision dataset or subset
    """

    dataset_name = dataset_name.lower()
    root = Path(root)

    if dataset_name == "mnist":
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5,),
                std=(0.5,),
            ),
        ])

        dataset = datasets.MNIST(
            root=str(root),
            train=train,
            download=True,
            transform=transforms.Compose(transform_list),
        )

    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ])

        dataset = datasets.CIFAR10(
            root=str(root),
            train=train,
            download=True,
            transform=transform,
        )

    elif dataset_name == "celeba":
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ])

        split = "train" if train else "test"

        dataset = datasets.CelebA(
            root=str(root),
            split=split,
            download=True,
            transform=transform,
        )

    else:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name}. "
            "Choose from 'mnist', 'cifar10', or 'celeba'."
        )

    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = Subset(dataset, range(max_samples))

    return dataset


def get_degraded_dataset(
    dataset_name: str,
    degradation: Callable,
    root: str = "./data",
    train: bool = True,
    image_size: int = 32,
    timesteps: int = 100,
    max_samples: Optional[int] = None,
):
    """
    Creates the full cold diffusion dataset:
        clean dataset + degradation function.
    """

    dataset = get_raw_dataset(
        dataset_name=dataset_name,
        root=root,
        train=train,
        image_size=image_size,
        max_samples=max_samples,
    )

    degraded_dataset = ColdDiffusionDataset(
        dataset=dataset,
        degradation=degradation,
        timesteps=timesteps,
    )

    return degraded_dataset


def get_dataloader(
    dataset_name: str,
    degradation: Callable,
    root: str = "./data",
    train: bool = True,
    image_size: int = 32,
    timesteps: int = 100,
    batch_size: int = 128,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
):
    """
    Creates a DataLoader for cold diffusion training or evaluation.
    """

    if shuffle is None:
        shuffle = train

    dataset = get_degraded_dataset(
        dataset_name=dataset_name,
        degradation=degradation,
        root=root,
        train=train,
        image_size=image_size,
        timesteps=timesteps,
        max_samples=max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )

    return dataloader