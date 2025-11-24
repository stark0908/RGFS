"""
Training and evaluation functions for RGFS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import compute_prototypes, classify_queries, masked_loss, compute_psnr


class RGFSTrainer:
    """Trainer class for RGFS model"""
    
    def __init__(self, model, optimizer, device, config):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.image_loss = nn.L1Loss()
        
        self.best_accuracy = 0.0
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss, total_correct, total_queries = 0, 0, 0
        total_final_psnr, total_final_recon_loss = 0, 0
        
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{self.config.EPOCHS}",
            leave=False
        )
        
        for episode in progress_bar:
            images, mask, support_images, support_labels, query_images, query_labels = episode
            
            # Move to device
            images = images.squeeze(0).to(self.device, non_blocking=True)
            mask = mask.squeeze(0).to(self.device, non_blocking=True)
            support_images = support_images.squeeze(0).to(self.device, non_blocking=True)
            query_images = query_images.squeeze(0).to(self.device, non_blocking=True)
            support_labels = support_labels.view(-1).to(self.device, non_blocking=True)
            query_labels = query_labels.view(-1).to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Multiple forward passes for uncertainty estimation
            all_ce_losses = []
            all_query_logits = []
            all_psnr = []
            all_reconstruct_loss = []
            
            for _ in range(self.config.N_TIMES):
                # Few-shot classification
                _, support_embeddings = self.model(support_images)
                _, query_embeddings = self.model(query_images)
                
                n_way = torch.unique(support_labels).size(0)
                prototypes = compute_prototypes(support_embeddings, support_labels, n_way)
                logits = classify_queries(prototypes, query_embeddings)
                ce_loss = self.loss_fn(logits, query_labels)
                
                all_ce_losses.append(ce_loss)
                all_query_logits.append(logits)
                
                # Reconstruction
                reconstructed_image, _ = self.model(images, mask)
                recon_loss = masked_loss(reconstructed_image, support_images, mask)
                img_loss = self.image_loss(reconstructed_image, support_images)
                recon_loss += img_loss
                
                mse_loss = F.mse_loss(reconstructed_image, support_images)
                psnr = compute_psnr(mse_loss, max_val=1.0)
                
                all_reconstruct_loss.append(recon_loss)
                all_psnr.append(psnr)
            
            # Aggregate losses
            total_ce_loss = torch.stack(all_ce_losses).mean()
            stacked_logits = torch.stack(all_query_logits)
            stacked_probs = torch.softmax(stacked_logits, dim=-1)
            
            true_class_probs = stacked_probs[
                torch.arange(self.config.N_TIMES)[:, None],
                torch.arange(len(query_labels)),
                query_labels
            ]
            
            total_recon_loss = torch.stack(all_reconstruct_loss).mean()
            total_psnr = torch.stack(all_psnr).mean()
            
            # Variance loss for prediction stability
            variance_loss = torch.std(true_class_probs, dim=0).sum()
            
            # Combined loss
            total_combined_loss = (
                self.config.RECON_WEIGHT * total_recon_loss + 
                total_ce_loss + 
                self.config.ALPHA * variance_loss
            )
            
            total_combined_loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_final_psnr += total_psnr.item()
            total_final_recon_loss += total_recon_loss.item()
            total_loss += total_combined_loss.item()
            
            mean_logits = stacked_logits.mean(dim=0)
            preds = torch.argmax(mean_logits, dim=1)
            total_correct += (preds == query_labels).sum().item()
            total_queries += query_labels.size(0)
            
            avg_acc_till = (total_correct / total_queries) * 100
            progress_bar.set_postfix(
                Phase="Training",
                Loss=f"{total_combined_loss.item():.4f}",
                Acc=f"{avg_acc_till:.2f}",
                PSNR=f"{total_psnr.item():.4f}",
                CE_Loss=f"{total_ce_loss.item():.4f}",
                Recon_Loss=f"{total_recon_loss.item():.4f}"
            )
        
        # Epoch statistics
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_final_recon_loss / len(dataloader)
        avg_psnr = total_final_psnr / len(dataloader)
        accuracy = (total_correct / total_queries) * 100
        
        return avg_loss, accuracy, avg_recon_loss, avg_psnr
    
    def evaluate(self, dataloader, epoch, phase="Testing"):
        """Evaluate model"""
        self.model.eval()
        total_loss, total_correct, total_queries = 0, 0, 0
        total_final_psnr, total_final_recon_loss = 0, 0
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{self.config.EPOCHS}",
            leave=False
        )
        
        with torch.no_grad():
            for episode in progress_bar:
                images, mask, support_images, support_labels, query_images, query_labels = episode
                
                images = images.squeeze(0).to(self.device, non_blocking=True)
                mask = mask.squeeze(0).to(self.device, non_blocking=True)
                support_images = support_images.squeeze(0).to(self.device, non_blocking=True)
                query_images = query_images.squeeze(0).to(self.device, non_blocking=True)
                support_labels = support_labels.view(-1).to(self.device, non_blocking=True)
                query_labels = query_labels.view(-1).to(self.device, non_blocking=True)
                
                all_ce_losses = []
                all_query_logits = []
                all_psnr = []
                all_reconstruct_loss = []
                
                # Enable dropout for uncertainty estimation
                self.model.train()
                for _ in range(self.config.N_TIMES):
                    _, support_embeddings = self.model(support_images)
                    _, query_embeddings = self.model(query_images)
                    
                    n_way = torch.unique(support_labels).size(0)
                    prototypes = compute_prototypes(support_embeddings, support_labels, n_way)
                    logits = classify_queries(prototypes, query_embeddings)
                    ce_loss = self.loss_fn(logits, query_labels)
                    
                    all_ce_losses.append(ce_loss)
                    all_query_logits.append(logits)
                    
                    reconstructed_image, _ = self.model(images, mask)
                    recon_loss = masked_loss(reconstructed_image, support_images, mask)
                    mse_loss = F.mse_loss(reconstructed_image, support_images)
                    psnr = compute_psnr(mse_loss, max_val=1.0)
                    
                    all_reconstruct_loss.append(recon_loss)
                    all_psnr.append(psnr)
                
                self.model.eval()
                
                total_ce_loss = torch.stack(all_ce_losses).mean()
                stacked_logits = torch.stack(all_query_logits)
                stacked_probs = torch.softmax(stacked_logits, dim=-1)
                
                true_class_probs = stacked_probs[
                    torch.arange(self.config.N_TIMES)[:, None],
                    torch.arange(len(query_labels)),
                    query_labels
                ]
                
                total_recon_loss = torch.stack(all_reconstruct_loss).mean()
                total_psnr = torch.stack(all_psnr).mean()
                variance_loss = torch.std(true_class_probs, dim=0).sum()
                
                total_combined_loss = (
                    self.config.RECON_WEIGHT * total_recon_loss + 
                    total_ce_loss + 
                    self.config.ALPHA * variance_loss
                )
                
                total_final_psnr += total_psnr.item()
                total_final_recon_loss += total_recon_loss.item()
                total_loss += total_combined_loss.item()
                
                mean_logits = stacked_logits.mean(dim=0)
                preds = torch.argmax(mean_logits, dim=1)
                total_correct += (preds == query_labels).sum().item()
                total_queries += query_labels.size(0)
                
                avg_acc_till = (total_correct / total_queries) * 100
                progress_bar.set_postfix(
                    Phase=phase,
                    Loss=f"{total_combined_loss.item():.4f}",
                    Acc=f"{avg_acc_till:.2f}",
                    PSNR=f"{total_psnr.item():.4f}"
                )
        
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_final_recon_loss / len(dataloader)
        avg_psnr = total_final_psnr / len(dataloader)
        accuracy = (total_correct / total_queries) * 100
        
        return avg_loss, accuracy, avg_recon_loss, avg_psnr
    
    def save_best_model(self, accuracy):
        """Save model if it achieves best accuracy"""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
            print(f"Model Saved with accuracy: {accuracy:.2f}%")
            return True
        return False
