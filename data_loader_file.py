"""
Data loading utilities for RGFS
"""

import os
import random
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.folder import ImageFolder

from dataset import get_transforms, class_sorting, FewShotDataset


def prepare_data(config):
    """
    Prepare all dataloaders
    
    Args:
        config: Configuration object
    
    Returns:
        train_loader: Training dataloader
        test_loader: Testing dataloader  
        strict_test_loader: Strict testing dataloader (unseen classes only)
        train_list: List of training classes
        test_list: List of testing classes
        strict_test_list: List of strictly unseen classes
    """
    # Check if data exists
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {config.DATA_PATH}")
    
    # Load full dataset
    data_transform = get_transforms(
        image_size=config.IMAGE_SIZE,
        mean=config.IMAGE_MEAN,
        std=config.IMAGE_STD
    )
    
    fulldata = ImageFolder(root=config.DATA_PATH, transform=data_transform)
    
    # Split into train and test
    train_len = int(config.TRAIN_SPLIT * len(fulldata))
    test_len = len(fulldata) - train_len
    train_data_set, test_data_set = random_split(fulldata, [train_len, test_len])
    
    # Define class splits
    all_list = list(range(10))  # Assuming 10 classes
    train_list = random.sample(all_list, config.TRAIN_CLASS_LEN)
    test_list = list(range(10))
    strict_test_list = list(set(all_list) - set(train_list))
    
    print(f"Training classes: {train_list}")
    print(f"Testing classes: {test_list}")
    print(f"Strict testing classes (unseen): {strict_test_list}")
    
    # Filter datasets by class
    train_data = class_sorting(train_data_set, train_list)
    test_data = class_sorting(test_data_set, test_list)
    strict_test_data = class_sorting(test_data_set, strict_test_list)
    
    # Create few-shot datasets
    train_dataset = FewShotDataset(
        data=train_data,
        way=config.WAYS,
        shot=config.SHOTS,
        query=config.QUERIES,
        episode=config.TRAIN_EPISODES,
        patch_size=config.PATCH_SIZE,
        mask_ratio=config.MASK_RATIO
    )
    
    test_dataset = FewShotDataset(
        data=test_data,
        way=config.WAYS,
        shot=config.SHOTS,
        query=config.QUERIES,
        episode=config.TEST_EPISODES,
        patch_size=config.PATCH_SIZE,
        mask_ratio=config.MASK_RATIO
    )
    
    strict_test_dataset = FewShotDataset(
        data=strict_test_data,
        way=config.STRICT_WAYS,
        shot=config.SHOTS,
        query=config.QUERIES,
        episode=config.TEST_EPISODES,
        patch_size=config.PATCH_SIZE,
        mask_ratio=config.MASK_RATIO
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    strict_test_loader = DataLoader(
        strict_test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return (
        train_loader, 
        test_loader, 
        strict_test_loader,
        train_list,
        test_list,
        strict_test_list
    )
