"""
Utility functions for RGFS
"""

import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_prototypes(support_embeddings, support_labels, way):
    """
    Compute class prototypes from support embeddings
    
    Args:
        support_embeddings: Embeddings of support set
        support_labels: Labels of support set
        way: Number of classes
    
    Returns:
        prototypes: Mean embeddings for each class
    """
    embedding_dimensions = support_embeddings.size(-1)
    prototypes = torch.zeros(way, embedding_dimensions).to(support_embeddings.device)

    for c in range(way):
        class_mask = (support_labels == c)
        class_embeddings = support_embeddings[class_mask]
        prototypes[c] = class_embeddings.mean(dim=0)
    
    return prototypes


def classify_queries(prototypes, query_embeddings):
    """
    Classify query samples based on distance to prototypes
    
    Args:
        prototypes: Class prototypes
        query_embeddings: Query embeddings
    
    Returns:
        logits: Classification logits (negative distances)
    """
    n_query = query_embeddings.size(0)
    way = prototypes.size(0)

    query_exp = query_embeddings.unsqueeze(1).expand(n_query, way, -1)
    prototypes_exp = prototypes.unsqueeze(0).expand(n_query, way, -1)

    distances = torch.sum((query_exp - prototypes_exp) ** 2, dim=2)
    logits = -distances
    
    return logits


def compute_psnr(mse, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio"""
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr


def compute_rmse(img1, img2):
    """Compute Root Mean Square Error"""
    return torch.sqrt(torch.mean((img1 - img2) ** 2))


def masked_loss(recon, target, mask):
    """
    Compute loss only on masked regions
    
    Args:
        recon: Reconstructed image
        target: Target image
        mask: Binary mask (1 for masked regions)
    
    Returns:
        loss: Normalized MSE loss on masked regions
    """
    mask = mask.float()
    if mask.shape[1] == 1:
        mask = mask.expand_as(recon)
    
    loss = F.mse_loss(recon * mask, target * mask, reduction='sum')
    norm = mask.sum() + 1e-8
    return loss / norm


def identity_loss_fn(embedding_a, embedding_b):
    """L1 loss between embeddings"""
    return F.l1_loss(embedding_a, embedding_b)
