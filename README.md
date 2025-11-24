# RGFS: Reconstruction-Guided Few-Shot Learning

A modular implementation of Reconstruction-Guided Few-Shot Learning with prototypical networks and masked autoencoding.

## Project Structure

```
├── config.py           # Configuration parameters
├── utils.py            # Utility functions (prototypes, losses, metrics)
├── dataset.py          # Dataset classes and data utilities
├── data_loader.py      # Data loading and preparation
├── model.py            # Model architectures (Encoder, Decoder, MAE)
├── trainer.py          # Training and evaluation logic
├── main.py             # Main training script
└── README.md           # This file
```

## Features

- **Modular Design**: Clean separation of concerns across multiple files
- **Few-Shot Learning**: Prototypical networks with N-way K-shot setup
- **Masked Autoencoding**: Self-supervised reconstruction task
- **Uncertainty Estimation**: Multiple forward passes with dropout
- **Variance Regularization**: Prediction stability loss
- **DropBlock Regularization**: Structured dropout for CNNs

## Requirements

```bash
torch
torchvision
numpy
tqdm
```

## Usage

### Basic Training

```python
python main.py
```

### Custom Configuration

Edit `config.py` to modify training parameters:

```python
class Config:
    WAYS = 5          # Number of classes per episode
    SHOTS = 5         # Number of support samples per class
    QUERIES = 5       # Number of query samples per class
    EPOCHS = 20       # Training epochs
    LEARNING_RATE = 1e-4
    # ... more parameters
```

### Using Individual Modules

```python
from config import Config
from data_loader import prepare_data
from model import build_model
from trainer import RGFSTrainer

# Load data
train_loader, test_loader, _, _, _, _ = prepare_data(Config)

# Build model
model = build_model()

# Create trainer
optimizer = torch.optim.Adam(model.parameters())
trainer = RGFSTrainer(model, optimizer, device, Config)

# Train
trainer.train_epoch(train_loader, epoch=0)
```

## Model Architecture

- **Encoder**: ResNet-50 with DropBlock regularization
- **Decoder**: Transposed convolution layers for reconstruction
- **Embedding Head**: Global average pooling + linear layer

## Loss Functions

1. **Cross-Entropy Loss**: Few-shot classification
2. **Reconstruction Loss**: Masked region reconstruction (MSE + L1)
3. **Variance Loss**: Prediction stability across multiple passes

Combined as:
```
Total Loss = recon_weight × Recon Loss + CE Loss + alpha × Variance Loss
```

## Key Parameters

- `N_TIMES = 15`: Number of forward passes for uncertainty
- `RECON_WEIGHT = 5`: Weight for reconstruction loss
- `ALPHA = 0.01`: Weight for variance loss
- `MASK_RATIO = 0.1`: Percentage of image to mask

## Testing

The code supports two testing modes:

1. **Standard Testing**: All classes (seen + unseen)
2. **Strict Testing**: Only unseen classes

## Notes
- This code was originally developed in Python 3.6
- Model automatically saves when achieving best accuracy
- Uses mixed training/eval mode for uncertainty estimation
- Implements episodic training for few-shot learning
- Supports multi-GPU training (specify GPU in config)

## Citation

If you use this code, please cite the original RGFS paper.
