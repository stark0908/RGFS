"""
Configuration file for RGFS (Reconstruction-Guided Few-Shot) learning
"""

class Config:
    # Data configuration
    DATA_PATH = "/home/23dcs505/data/2750"
    TRAIN_SPLIT = 0.8
    
    # Few-shot learning parameters
    WAYS = 5
    SHOTS = 5
    QUERIES = 5
    STRICT_WAYS = 5
    TRAIN_CLASS_LEN = 5
    
    # Training parameters
    EPOCHS = 20
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 8
    N_TIMES = 15  # Number of forward passes for uncertainty estimation
    
    # Loss weights
    RECON_WEIGHT = 5
    ALPHA = 0.01  # Variance loss weight
    
    # Model parameters
    DROP_PROB = 0.3
    BLOCK_SIZE = 3
    MASK_RATIO = 0.1
    PATCH_SIZE = 8
    
    # Episode configuration
    TRAIN_EPISODES = 200
    TEST_EPISODES = 200
    
    # Device configuration
    GPU_NUM = 4
    SEED = 42
    
    # Model saving
    MODEL_SAVE_PATH = "/home/23dcs505/5w5s_resnet.pth"
    
    # Image normalization (ImageNet stats)
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = 224
