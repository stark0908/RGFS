"""
Model architectures for RGFS
"""

import torch
import torch.nn as nn
from torchvision import models


class DropBlock2D(nn.Module):
    """DropBlock regularization for CNNs"""
    
    def __init__(self, drop_prob=0.1, block_size=3):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        gamma = self._compute_gamma(x)
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) < gamma).float()
        mask = self._compute_block_mask(mask)
        countM = mask.numel()
        count_ones = mask.sum()
        return mask * x * (countM / count_ones)

    def _compute_block_mask(self, mask):
        block_mask = nn.functional.max_pool2d(
            input=mask,
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2
        )
        return 1 - block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class Encoder(nn.Module):
    """ResNet-50 based encoder with DropBlock regularization"""
    
    def __init__(self, drop_prob=0.3, block_size=3):
        super().__init__()
        resnet = models.resnet50(pretrained=True)

        # Freeze all layers except layer4
        for param in resnet.parameters():
            param.requires_grad = False
        
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        
        # Feature extractor with DropBlock
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            resnet.layer4,
            DropBlock2D(drop_prob=drop_prob, block_size=block_size)
        )

        # Bottleneck to reduce channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        for param in self.bottleneck.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bottleneck(x)  # Output: [B, 512, 7, 7]
        return x


class Decoder(nn.Module):
    """Transposed convolution decoder for image reconstruction"""
    
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)  # Output: [B, 3, 224, 224]


class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder for RGFS"""
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256)
        )
    
    def forward(self, masked_img, mask=None):
        """
        Forward pass
        
        Args:
            masked_img: Masked input image
            mask: Binary mask (optional, not used in forward)
        
        Returns:
            recon: Reconstructed image
            embedding: Feature embedding
        """
        latent = self.encoder(masked_img)
        recon = self.decoder(latent)
        embedding = self.embedding_head(latent)
        return recon, embedding


def build_model(drop_prob=0.3, block_size=3):
    """
    Build complete RGFS model
    
    Args:
        drop_prob: DropBlock probability
        block_size: DropBlock size
    
    Returns:
        model: Complete MaskedAutoencoder model
    """
    encoder = Encoder(drop_prob=drop_prob, block_size=block_size)
    decoder = Decoder()
    model = MaskedAutoencoder(encoder, decoder)
    return model
