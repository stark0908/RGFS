"""
Main training script for RGFS
"""

import torch
import gc

from config import Config
from utils import set_seed
from data_loader import prepare_data
from model import build_model
from trainer import RGFSTrainer


def main():
    """Main training function"""
    # Set random seed
    set_seed(Config.SEED)
    
    # Setup device
    device = torch.device(f"cuda:{Config.GPU_NUM}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear GPU cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Prepare data
    print("Preparing data...")
    train_loader, test_loader, strict_test_loader, train_list, test_list, strict_test_list = prepare_data(Config)
    
    # Build model
    print("Building model...")
    model = build_model(drop_prob=Config.DROP_PROB, block_size=Config.BLOCK_SIZE)
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Create trainer
    trainer = RGFSTrainer(model, optimizer, device, Config)
    
    # Training loop
    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc, train_recon_loss, train_psnr = trainer.train_epoch(train_loader, epoch)
        print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, "
              f"Recon Loss: {train_recon_loss:.4f}, PSNR: {train_psnr:.4f}")
        
        # Evaluate on test set
        test_loss, test_acc, test_recon_loss, test_psnr = trainer.evaluate(test_loader, epoch, "Testing")
        print(f"Testing    - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, "
              f"Recon Loss: {test_recon_loss:.4f}, PSNR: {test_psnr:.4f}")
        
        # Save best model
        trainer.save_best_model(test_acc)
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    trainer.model = model
    
    # Test on all classes
    print(f"\nTesting on classes: {test_list}")
    test_loss, test_acc, test_recon_loss, test_psnr = trainer.evaluate(test_loader, Config.EPOCHS-1, "Testing (All)")
    print(f"Testing (Seen + Unseen) - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, "
          f"Recon Loss: {test_recon_loss:.4f}, PSNR: {test_psnr:.4f}")
    
    # Test on unseen classes only
    print(f"\nTesting on unseen classes: {strict_test_list}")
    strict_test_loss, strict_test_acc, strict_test_recon_loss, strict_test_psnr = trainer.evaluate(
        strict_test_loader, Config.EPOCHS-1, "Strict Testing"
    )
    print(f"Testing (Unseen Only)   - Loss: {strict_test_loss:.4f}, Accuracy: {strict_test_acc:.2f}%, "
          f"Recon Loss: {strict_test_recon_loss:.4f}, PSNR: {strict_test_psnr:.4f}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
