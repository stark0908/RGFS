# RGFS: Reconstruction-Guided Few-Shot Learning

> **Note**: This codebase has been cleaned with AI assistance for improved readability, modularity, and maintainability. The implementation follows best practices with clear separation of concerns across multiple well-documented modules.

A modular implementation of Reconstruction-Guided Few-Shot Learning with prototypical networks and masked autoencoding.

You can get dataset from here: [**Download Dataset →**](https://www.kaggle.com/datasets/nilesh789/eurosat-rgb)

## Project Structure

```
├── config_file.py      # Configuration parameters
├── utils_file.py       # Utility functions (prototypes, losses, metrics)
├── dataset_file.py     # Dataset classes and data utilities
├── data_loader_file.py # Data loading and preparation
├── model_file.py       # Model architectures (Encoder, Decoder, MAE)
├── trainer_file.py     # Training and evaluation logic
├── main_file.py        # Main training script
└── readme_file.md      # This file
```

## Features

- **Modular Design**: Clean separation of concerns across multiple files
- **Few-Shot Learning**: Prototypical networks with 3-way 5-shot setup
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

## Configuration

The current configuration uses **3-way 5-shot** few-shot learning:

```python
class Config:
    WAYS = 3          # Number of classes per episode
    SHOTS = 5         # Number of support samples per class
    QUERIES = 5       # Number of query samples per class
    EPOCHS = 20       # Training epochs
    LEARNING_RATE = 1e-4
    # ... more parameters
```

## Usage

### Basic Training

```python
python main_file.py
```

### Custom Configuration

Edit `config_file.py` to modify training parameters:

```python
from config_file import Config

# Modify parameters as needed
Config.WAYS = 3
Config.SHOTS = 5
Config.QUERIES = 5
```

### Using Individual Modules

```python
from config_file import Config
from data_loader_file import prepare_data
from model_file import build_model
from trainer_file import RGFSTrainer
import torch

# Load data
train_loader, test_loader, _, _, _, _ = prepare_data(Config)

# Build model
model = build_model(drop_prob=Config.DROP_PROB, block_size=Config.BLOCK_SIZE)

# Create trainer
device = torch.device(f"cuda:{Config.GPU_NUM}" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
trainer = RGFSTrainer(model, optimizer, device, Config)

# Train
loss, acc, recon_loss, psnr = trainer.train_epoch(train_loader, epoch=0)
```

## Model Architecture

- **Encoder**: ResNet-50 with DropBlock regularization
  - Frozen layers: conv1, bn1, relu, maxpool, layer1, layer2, layer3
  - Trainable: layer4 and bottleneck
  - DropBlock applied after layer3 and layer4
- **Decoder**: Transposed convolution layers for reconstruction
  - 5 upsampling layers: 512→256→128→64→32→16→3 channels
  - Output: 224×224 RGB image
- **Embedding Head**: Global average pooling + linear layer (512→256)

## Loss Functions

1. **Cross-Entropy Loss**: Few-shot classification
2. **Reconstruction Loss**: Masked region reconstruction (MSE + L1)
3. **Variance Loss**: Prediction stability across multiple passes

Combined as:
```
Total Loss = RECON_WEIGHT × Recon Loss + CE Loss + ALPHA × Variance Loss
```

## Key Parameters

- `WAYS = 3`: Number of classes per episode (3-way classification)
- `SHOTS = 5`: Support examples per class
- `QUERIES = 5`: Query examples per class
- `N_TIMES = 15`: Number of forward passes for uncertainty estimation
- `RECON_WEIGHT = 5`: Weight for reconstruction loss
- `ALPHA = 0.01`: Weight for variance loss
- `MASK_RATIO = 0.1`: Percentage of image to mask (10%)
- `PATCH_SIZE = 8`: Size of masked patches
- `DROP_PROB = 0.3`: DropBlock probability
- `BLOCK_SIZE = 3`: DropBlock size

## Dataset

- **Path**: `/home/23dcs505/data/2750`
- **Split**: 80% train, 20% test
- **Classes**: 10 total classes
  - Training: 5 randomly selected classes
  - Testing (standard): All 10 classes
  - Testing (strict): 5 unseen classes only

## Testing Modes

The code supports two testing modes:

1. **Standard Testing**: All classes (seen + unseen)
   - Uses `test_list = [0,1,2,3,4,5,6,7,8,9]`
2. **Strict Testing**: Only unseen classes
   - Uses `strict_test_list` (classes not in training set)

## Training Process

1. **Episodic Training**: Each episode samples 3 classes
2. **Support Set**: 5 examples per class (masked for reconstruction)
3. **Query Set**: 5 examples per class (for classification)
4. **Multiple Passes**: 15 forward passes per episode for uncertainty
5. **Metrics**: Tracks accuracy, PSNR, and various loss components

## Expected Results

Based on the reference notebook (RGFS_3W5S.ipynb):

- **Training Accuracy**: ~99.9% by epoch 20
- **Test Accuracy**: ~87-92% (varies by epoch)
- **Reconstruction PSNR**: ~9.3-9.6 dB by epoch 20
- **Training Time**: ~10,000 seconds for 20 epochs (hardware dependent)

## Notes

- This codes and results has been tested on Python 3.6
- Model automatically saves when achieving best accuracy
- Uses mixed training/eval mode for uncertainty estimation during testing
- Implements episodic training for few-shot learning
- Supports multi-GPU training (specify GPU in config)
- Pin memory and multiple workers for faster data loading

## File Descriptions

- **config_file.py**: All hyperparameters and paths
- **dataset_file.py**: FewShotDataset class with block masking
- **data_loader_file.py**: Data preparation and loading utilities
- **model_file.py**: DropBlock2D, Encoder, Decoder, MaskedAutoencoder
- **trainer_file.py**: RGFSTrainer class with train/eval methods
- **utils_file.py**: Prototypes, classification, losses, metrics
- **main_file.py**: Main training loop and orchestration

## Citation

If you use this code, please cite the original RGFS paper and acknowledge the AI-assisted code refactoring.

