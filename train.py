#!/usr/bin/env python3
"""
Training script for LV Diastolic Diameter Point Detection using Swin Transformer
"""

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import SwinModel, SwinConfig
import torch.nn.functional as F

class LVPointDataset(Dataset):
    """Dataset for LV point detection"""
    
    def __init__(self, labels_file, images_dir, transform=None, image_size=224):
        self.image_size = image_size
        self.transform = transform
        self.images_dir = Path(images_dir)
        
        # Load labels
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        # Extract lv_coordinates from the nested structure
        if isinstance(list(self.labels.values())[0], dict) and 'lv_coordinates' in list(self.labels.values())[0]:
            # Convert from nested format to simple format
            simple_labels = {}
            for image_name, data in self.labels.items():
                simple_labels[image_name] = data['lv_coordinates']
            self.labels = simple_labels
        
        self.image_names = list(self.labels.keys())
        self.point_names = ['lv-ivs-top', 'lv-ivs-bottom', 'lv-pw-top', 'lv-pw-bottom']
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = self.images_dir / image_name
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Get keypoints and normalize to [0, 1]
        coordinates = self.labels[image_name]
        keypoints = []
        
        for point_name in self.point_names:
            if point_name in coordinates:
                x, y = coordinates[point_name]
                # Normalize coordinates to [0, 1] based on original image size
                x_norm = x / original_w
                y_norm = y / original_h
                keypoints.extend([x_norm, y_norm])
            else:
                # If point is missing, use center of image as fallback
                keypoints.extend([0.5, 0.5])
        
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        
        return image, keypoints, image_name

class SwinPointDetector(nn.Module):
    """Swin Transformer based point detector"""
    
    def __init__(self, num_points=4, pretrained=True):
        super().__init__()
        self.num_points = num_points
        
        # Load Swin Transformer backbone
        if pretrained:
            self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        else:
            config = SwinConfig(
                image_size=224,
                patch_size=4,
                num_channels=3,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.0,
                qkv_bias=True,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                drop_path_rate=0.1,
                hidden_act="gelu",
                use_absolute_embeddings=False,
                patch_norm=True,
                initializer_range=0.02,
                layer_norm_eps=1e-5,
            )
            self.swin = SwinModel(config)
        
        # Get the output dimension of Swin
        self.feature_dim = self.swin.config.hidden_size
        
        # Point regression head
        self.point_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_points * 2)  # x, y for each point
        )
        
    def forward(self, x):
        # Extract features using Swin Transformer
        outputs = self.swin(x)
        features = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Global average pooling and point regression
        features = features.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        points = self.point_head(features)
        
        # Reshape to (batch_size, num_points, 2)
        points = points.view(-1, self.num_points, 2)
        
        return points

def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    """Train the point detection model"""
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Train')
        
        for images, keypoints, _ in train_pbar:
            images = images.to(device)
            keypoints = keypoints.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_points = model(images)
            
            # Reshape keypoints to match prediction format
            target_points = keypoints.view(-1, 4, 2)
            
            loss = criterion(pred_points, target_points)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Val')
        
        with torch.no_grad():
            for images, keypoints, _ in val_pbar:
                images = images.to(device)
                keypoints = keypoints.to(device)
                
                pred_points = model(images)
                target_points = keypoints.view(-1, 4, 2)
                
                loss = criterion(pred_points, target_points)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.6f}')
        print(f'  Val Loss: {val_loss:.6f}')
        print(f'  LR: {scheduler.get_last_lr()[0]:.8f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
            print(f'  → New best model saved (val_loss: {val_loss:.6f})')
        
        print()
    
    return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(description='Train LV Point Detection Model')
    parser.add_argument('--data_dir', type=str, default='/icardio/lv_diast_data', help='Data directory')
    parser.add_argument('--labels_dir', type=str, default='/icardio/lv_diast_labels', help='Labels directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained Swin model')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    train_dataset = LVPointDataset(
        labels_file=os.path.join(args.labels_dir, 'train_labels.json'),
        images_dir=os.path.join(args.data_dir, 'images', 'train'),
        image_size=args.image_size
    )
    
    val_dataset = LVPointDataset(
        labels_file=os.path.join(args.labels_dir, 'val_labels.json'),
        images_dir=os.path.join(args.data_dir, 'images', 'val'),
        image_size=args.image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print(f'Train dataset: {len(train_dataset)} images')
    print(f'Val dataset: {len(val_dataset)} images')
    
    # Create model
    model = SwinPointDetector(num_points=4, pretrained=args.pretrained)
    model = model.to(device)
    
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=args.num_epochs, device=device
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print('Training completed!')

if __name__ == '__main__':
    main()
