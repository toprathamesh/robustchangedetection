#!/usr/bin/env python
"""
Training script for change detection model
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
import django
django.setup()

from change_detection.ml_models import (
    UNetChangeDetection, ChangeDetectionDataset, ChangeDetectionTrainer,
    get_default_transforms, calculate_metrics
)
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train change detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--model_dir', type=str, default='ml_models',
                       help='Directory to save trained models')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Setup data transforms
    train_transform = get_default_transforms(is_train=True)
    val_transform = get_default_transforms(is_train=False)
    
    # Create datasets
    train_dataset = ChangeDetectionDataset(
        args.data_dir, transform=train_transform, is_train=True
    )
    val_dataset = ChangeDetectionDataset(
        args.data_dir, transform=val_transform, is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = UNetChangeDetection(n_channels=6, n_classes=1)
    
    # Create trainer
    trainer = ChangeDetectionTrainer(model, device=device)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        print(f"Train Loss: {train_loss:.6f}")
        
        # Validate
        val_loss, val_accuracy = trainer.validate(val_loader)
        print(f"Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.model_dir, 'best_model.pth')
            trainer.save_model(model_path, epoch, val_loss)
            print(f"New best model saved with val_loss: {val_loss:.6f}")
        
        # Update learning rate
        trainer.scheduler.step(val_loss)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.model_dir, f'checkpoint_epoch_{epoch+1}.pth')
            trainer.save_model(checkpoint_path, epoch, val_loss)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main() 