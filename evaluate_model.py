# utils/data_loader.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_root, transform, batch_size=32, num_workers=4, pin_memory=True):
    """
    Returns PyTorch DataLoaders for training, validation, and test sets.

    Args:
        data_root (str): Path to the directory containing 'train', 'val', and 'test' folders.
        transform (torchvision.transforms): Transformations to apply to images.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): Whether to use pinned memory (recommended for GPU training).

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=transform)
    val_dataset   = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=transform)
    test_dataset  = datasets.ImageFolder(os.path.join(data_root, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
