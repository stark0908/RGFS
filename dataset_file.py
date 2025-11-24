"""
Dataset classes and utilities for RGFS
"""

import random
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder


def get_transforms(image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Get data transforms"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def class_sorting(dataset, class_list):
    """
    Filter dataset to only include specified classes
    
    Args:
        dataset: Original dataset (Subset)
        class_list: List of class indices to keep
    
    Returns:
        Filtered Subset
    """
    targets = dataset.dataset.targets
    indices = [i for i in dataset.indices if targets[i] in class_list]
    return Subset(dataset.dataset, indices)


class FewShotDataset(Dataset):
    """Dataset for episodic few-shot learning with masked autoencoding"""
    
    def __init__(self, data, way, shot, query, episode, patch_size=8, mask_ratio=0.1):
        super().__init__()
        self.data = data
        self.way = way
        self.shot = shot
        self.query = query
        self.episode = episode
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.class_to_indices = self._build_class_index()
        self.classes = list(self.class_to_indices.keys())
    
    @staticmethod
    def block_mask(img, patch_size=8, mask_ratio=0.1):
        """
        Apply block-wise masking to image
        
        Args:
            img: Input image tensor [C, H, W]
            patch_size: Size of each patch
            mask_ratio: Ratio of patches to mask
        
        Returns:
            masked_img: Image with masked patches
            mask: Binary mask indicating masked regions
        """
        C, H, W = img.shape
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        total_patches = num_patches_h * num_patches_w
        num_mask = int(mask_ratio * total_patches)

        patch_indices = [(i, j) for i in range(num_patches_h) for j in range(num_patches_w)]
        masked_indices = random.sample(patch_indices, num_mask)

        mask = torch.zeros((1, H, W))

        for i, j in masked_indices:
            h_start = i * patch_size
            w_start = j * patch_size
            mask[:, h_start:h_start+patch_size, w_start:w_start+patch_size] = 1.0

        masked_img = img.clone() * (1 - mask)
        return masked_img, mask

    def _build_class_index(self):
        """Build index mapping class labels to sample indices"""
        class_index = {}
        targets = self.data.dataset.targets

        for indexofsubset, indexoforiginal in enumerate(self.data.indices):
            label = targets[indexoforiginal]
            if label not in class_index:
                class_index[label] = []
            class_index[label].append(indexofsubset)

        return class_index
        
    def __len__(self):
        return self.episode
        
    def __getitem__(self, idx):
        """
        Generate one episode
        
        Returns:
            reconstruct_images: Masked support images
            mask: Binary masks
            support_images: Original support images
            support_labels: Support labels
            query_images: Query images
            query_labels: Query labels
        """
        selected_class = random.sample(self.classes, self.way)

        reconstruct_images, support_images, support_labels = [], [], []
        query_images, query_labels = [], []
        masks = []

        label_map = {class_name: i for i, class_name in enumerate(selected_class)}

        for class_name in selected_class:
            all_indices_for_class = self.class_to_indices[class_name]
            selected_index = random.sample(all_indices_for_class, self.shot + self.query)

            support_index = selected_index[:self.shot]
            query_index = selected_index[self.shot:]

            for i in support_index:
                image, _ = self.data[i]
                support_images.append(image)

                masked_image, mask = self.block_mask(
                    image, 
                    patch_size=self.patch_size, 
                    mask_ratio=self.mask_ratio
                )
                reconstruct_images.append(masked_image)
                masks.append(mask)
                support_labels.append(torch.tensor(label_map[class_name]))
                
            for i in query_index:
                image, _ = self.data[i]
                query_images.append(image)
                query_labels.append(torch.tensor(label_map[class_name]))
            
        return (
            torch.stack(reconstruct_images),
            torch.stack(masks),
            torch.stack(support_images),
            torch.stack(support_labels),
            torch.stack(query_images),
            torch.stack(query_labels)
        )
