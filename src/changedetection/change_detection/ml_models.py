import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetChangeDetection(nn.Module):
    """U-Net model for satellite image change detection"""
    
    def __init__(self, n_channels=6, n_classes=2, bilinear=False):
        """
        Args:
            n_channels: Number of input channels (6 for before+after RGB images)
            n_classes: Number of output classes (2 for change/no-change)
            bilinear: Use bilinear upsampling instead of transposed convolutions
        """
        super(UNetChangeDetection, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits


class ChangeDetectionDataset(Dataset):
    """Dataset for change detection training"""
    
    def __init__(self, data_dir: str, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # Get all image pairs
        self.image_pairs = self._load_image_pairs()
    
    def _load_image_pairs(self):
        """Load image pair file paths"""
        pairs = []
        pairs_file = os.path.join(self.data_dir, 'train_pairs.txt' if self.is_train else 'val_pairs.txt')
        
        if os.path.exists(pairs_file):
            with open(pairs_file, 'r') as f:
                for line in f:
                    before_path, after_path, mask_path = line.strip().split('\t')
                    pairs.append((before_path, after_path, mask_path))
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        before_path, after_path, mask_path = self.image_pairs[idx]
        
        # Load images
        before_img = Image.open(os.path.join(self.data_dir, before_path)).convert('RGB')
        after_img = Image.open(os.path.join(self.data_dir, after_path)).convert('RGB')
        mask = Image.open(os.path.join(self.data_dir, mask_path)).convert('L')
        
        if self.transform:
            # Apply same transform to all images
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            before_img = self.transform(before_img)
            
            torch.manual_seed(seed)
            after_img = self.transform(after_img)
            
            torch.manual_seed(seed)
            mask = self.transform(mask)
        
        # Concatenate before and after images
        input_tensor = torch.cat([before_img, after_img], dim=0)
        
        # Convert mask to binary (0/1)
        mask = (mask > 0.5).float()
        
        return input_tensor, mask.squeeze(0)


class ChangeDetectionTrainer:
    """Trainer class for change detection model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.squeeze(1), target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = self.criterion(output.squeeze(1), target)
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = torch.sigmoid(output.squeeze(1)) > 0.5
                total += target.numel()
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def save_model(self, path: str, epoch: int, loss: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        logger.info(f'Model saved to {path}')
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f'Model loaded from {path}')
        return checkpoint['epoch'], checkpoint['loss']


class ChangeDetectionInference:
    """Inference class for change detection"""
    
    def __init__(self, model_path: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = UNetChangeDetection(n_channels=6, n_classes=1)
        self.model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, before_image: np.ndarray, after_image: np.ndarray, 
                threshold: float = 0.5) -> Tuple[np.ndarray, float]:
        """
        Predict changes between two images
        
        Args:
            before_image: Before image (H, W, 3)
            after_image: After image (H, W, 3)
            threshold: Threshold for binary classification
            
        Returns:
            change_map: Binary change map (H, W)
            confidence: Average confidence score
        """
        # Convert to PIL Images
        before_pil = Image.fromarray(before_image)
        after_pil = Image.fromarray(after_image)
        
        # Preprocess
        before_tensor = self.preprocess(before_pil)
        after_tensor = self.preprocess(after_pil)
        
        # Concatenate and add batch dimension
        input_tensor = torch.cat([before_tensor, after_tensor], dim=0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.sigmoid(output.squeeze())
            change_map = (probabilities > threshold).cpu().numpy()
            confidence = probabilities.mean().item()
        
        return change_map, confidence
    
    def predict_batch(self, image_pairs, threshold: float = 0.5):
        """Predict changes for a batch of image pairs"""
        results = []
        
        for before_img, after_img in image_pairs:
            change_map, confidence = self.predict(before_img, after_img, threshold)
            results.append((change_map, confidence))
        
        return results


def get_default_transforms(is_train=True):
    """Get default data transformations"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    tp = ((predictions == 1) & (targets == 1)).sum()
    tn = ((predictions == 0) & (targets == 0)).sum()
    fp = ((predictions == 1) & (targets == 0)).sum()
    fn = ((predictions == 0) & (targets == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(targets)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'iou': iou
    } 