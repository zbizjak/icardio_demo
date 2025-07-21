#!/usr/bin/env python3
"""
Test script for LV Diastolic Diameter Point Detection
Evaluates model on test dataset and visualizes predictions vs ground truth
"""

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from transformers import SwinModel, SwinConfig
import torch.nn.functional as F
from tqdm import tqdm

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
        
        # Keep original image for visualization
        original_image = image.copy()
        
        # Resize image for model
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
        
        return image, keypoints, image_name, original_image, (original_h, original_w)

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

def custom_collate_fn(batch):
    """Custom collate function to handle variable size original images"""
    images = torch.stack([item[0] for item in batch])
    keypoints = torch.stack([item[1] for item in batch])
    image_names = [item[2] for item in batch]
    original_images = [item[3] for item in batch]  # Keep as list, don't stack
    original_sizes = [item[4] for item in batch]
    
    return images, keypoints, image_names, original_images, original_sizes

def visualize_predictions(images, pred_points, gt_points, image_names, original_images, original_sizes, save_dir, num_samples=10):
    """Visualize predictions vs ground truth"""
    
    os.makedirs(save_dir, exist_ok=True)
    point_names = ['lv-ivs-top', 'lv-ivs-bottom', 'lv-pw-top', 'lv-pw-bottom']
    colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255)]  # Green, Yellow, Blue, Magenta
    
    for i in range(min(num_samples, len(original_images))):
        orig_img = original_images[i].copy()
        orig_h, orig_w = original_sizes[i]
        
        # Convert normalized coordinates back to pixel coordinates
        pred_pixels = pred_points[i] * np.array([orig_w, orig_h])
        gt_pixels = gt_points[i].reshape(4, 2) * np.array([orig_w, orig_h])
        
        # Draw ground truth points (filled circles)
        for j, (point_name, color) in enumerate(zip(point_names, colors)):
            gt_x, gt_y = int(gt_pixels[j, 0]), int(gt_pixels[j, 1])
            cv2.circle(orig_img, (gt_x, gt_y), 8, color, -1)
            cv2.putText(orig_img, f'GT_{point_name}', (gt_x + 15, gt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw predicted points (empty circles)
        for j, (point_name, color) in enumerate(zip(point_names, colors)):
            pred_x, pred_y = int(pred_pixels[j, 0]), int(pred_pixels[j, 1])
            cv2.circle(orig_img, (pred_x, pred_y), 8, color, 2)
            cv2.putText(orig_img, f'PRED_{point_name}', (pred_x + 15, pred_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw lines connecting GT and predictions
        for j in range(4):
            gt_x, gt_y = int(gt_pixels[j, 0]), int(gt_pixels[j, 1])
            pred_x, pred_y = int(pred_pixels[j, 0]), int(pred_pixels[j, 1])
            cv2.line(orig_img, (gt_x, gt_y), (pred_x, pred_y), (255, 255, 255), 2)
        
        # Draw measurement lines
        # Top diameter (ivs-top to pw-top)
        gt_ivs_top = (int(gt_pixels[0, 0]), int(gt_pixels[0, 1]))
        gt_pw_top = (int(gt_pixels[2, 0]), int(gt_pixels[2, 1]))
        pred_ivs_top = (int(pred_pixels[0, 0]), int(pred_pixels[0, 1]))
        pred_pw_top = (int(pred_pixels[2, 0]), int(pred_pixels[2, 1]))
        
        cv2.line(orig_img, gt_ivs_top, gt_pw_top, (0, 255, 0), 3)  # GT top diameter
        cv2.line(orig_img, pred_ivs_top, pred_pw_top, (0, 128, 0), 2)  # Pred top diameter
        
        # Bottom diameter (ivs-bottom to pw-bottom)
        gt_ivs_bottom = (int(gt_pixels[1, 0]), int(gt_pixels[1, 1]))
        gt_pw_bottom = (int(gt_pixels[3, 0]), int(gt_pixels[3, 1]))
        pred_ivs_bottom = (int(pred_pixels[1, 0]), int(pred_pixels[1, 1]))
        pred_pw_bottom = (int(pred_pixels[3, 0]), int(pred_pixels[3, 1]))
        
        cv2.line(orig_img, gt_ivs_bottom, gt_pw_bottom, (0, 255, 255), 3)  # GT bottom diameter
        cv2.line(orig_img, pred_ivs_bottom, pred_pw_bottom, (0, 128, 128), 2)  # Pred bottom diameter
        
        # Save image
        save_path = os.path.join(save_dir, f'test_{i:03d}_{image_names[i]}')
        cv2.imwrite(save_path, cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))

def calculate_measurements(points, original_sizes):
    """Calculate LV diameter measurements in pixels"""
    measurements = []
    
    for i in range(len(points)):
        orig_h, orig_w = original_sizes[i]
        # Convert normalized coordinates back to pixel coordinates
        pixels = points[i].reshape(4, 2) * np.array([orig_w, orig_h])
        
        # Calculate distances
        # Top diameter: ivs-top (0) to pw-top (2)
        top_dist = np.sqrt((pixels[0, 0] - pixels[2, 0])**2 + (pixels[0, 1] - pixels[2, 1])**2)
        
        # Bottom diameter: ivs-bottom (1) to pw-bottom (3)
        bottom_dist = np.sqrt((pixels[1, 0] - pixels[3, 0])**2 + (pixels[1, 1] - pixels[3, 1])**2)
        
        # Average diameter
        avg_dist = (top_dist + bottom_dist) / 2
        
        measurements.append([top_dist, bottom_dist, avg_dist])
    
    return np.array(measurements)

def test_model(model, test_loader, device='cuda', save_dir='test_results'):
    """Test the model and calculate metrics"""
    
    model.eval()
    all_pred_points = []
    all_gt_points = []
    all_image_names = []
    all_original_images = []
    all_original_sizes = []
    
    print("Running inference on test set...")
    
    with torch.no_grad():
        for images, keypoints, image_names, original_images, original_sizes in tqdm(test_loader):
            images = images.to(device)
            
            # Forward pass
            pred_points = model(images)
            
            # Convert to numpy
            pred_points_np = pred_points.cpu().numpy()
            gt_points_np = keypoints.numpy()
            
            all_pred_points.extend(pred_points_np)
            all_gt_points.extend(gt_points_np)
            all_image_names.extend(image_names)
            all_original_images.extend(original_images)
            all_original_sizes.extend(original_sizes)
    
    all_pred_points = np.array(all_pred_points)
    all_gt_points = np.array(all_gt_points)
    
    # Calculate point-wise MSE
    point_names = ['lv-ivs-top', 'lv-ivs-bottom', 'lv-pw-top', 'lv-pw-bottom']
    mse_per_point = []
    
    for i in range(4):
        pred_point = all_pred_points[:, i, :]  # Shape: (N, 2)
        gt_point = all_gt_points[:, i*2:(i+1)*2]  # Shape: (N, 2)
        
        mse = np.mean((pred_point - gt_point)**2, axis=0)  # MSE for x and y
        mse_total = np.mean((pred_point - gt_point)**2)  # Total MSE
        
        mse_per_point.append({'point': point_names[i], 'mse_x': mse[0], 'mse_y': mse[1], 'mse_total': mse_total})
    
    # Calculate measurement MSE
    pred_measurements = calculate_measurements(all_pred_points, all_original_sizes)
    gt_measurements = calculate_measurements(all_gt_points.reshape(-1, 4, 2), all_original_sizes)
    
    measurement_mse = np.mean((pred_measurements - gt_measurements)**2, axis=0)
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    print("\nPoint-wise MSE (in normalized coordinates):")
    for point_data in mse_per_point:
        print(f"{point_data['point']:15}: MSE_x={point_data['mse_x']:.6f}, MSE_y={point_data['mse_y']:.6f}, Total={point_data['mse_total']:.6f}")
    
    print(f"\nMeasurement MSE (in pixels):")
    print(f"Top diameter:    {measurement_mse[0]:.2f} pixels²")
    print(f"Bottom diameter: {measurement_mse[1]:.2f} pixels²")
    print(f"Average diameter: {measurement_mse[2]:.2f} pixels²")
    
    print(f"\nMeasurement RMSE (in pixels):")
    print(f"Top diameter:    {np.sqrt(measurement_mse[0]):.2f} pixels")
    print(f"Bottom diameter: {np.sqrt(measurement_mse[1]):.2f} pixels")
    print(f"Average diameter: {np.sqrt(measurement_mse[2]):.2f} pixels")
    
    # Visualize predictions
    print(f"\nGenerating visualizations...")
    visualize_predictions(
        None, all_pred_points, all_gt_points, all_image_names, 
        all_original_images, all_original_sizes, save_dir, num_samples=20
    )
    print(f"Visualizations saved to: {save_dir}")
    
    # Plot measurement comparison
    plt.figure(figsize=(15, 5))
    
    measurement_names = ['Top Diameter', 'Bottom Diameter', 'Average Diameter']
    
    for i, name in enumerate(measurement_names):
        plt.subplot(1, 3, i+1)
        plt.scatter(gt_measurements[:, i], pred_measurements[:, i], alpha=0.6)
        plt.plot([gt_measurements[:, i].min(), gt_measurements[:, i].max()], 
                [gt_measurements[:, i].min(), gt_measurements[:, i].max()], 'r--', lw=2)
        plt.xlabel(f'Ground Truth {name} (pixels)')
        plt.ylabel(f'Predicted {name} (pixels)')
        plt.title(f'{name}\nRMSE: {np.sqrt(measurement_mse[i]):.2f} pixels')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'measurement_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return mse_per_point, measurement_mse

def main():
    parser = argparse.ArgumentParser(description='Test LV Point Detection Model')
    parser.add_argument('--data_dir', type=str, default='/icardio/lv_diast_data', help='Data directory')
    parser.add_argument('--labels_dir', type=str, default='/icardio/lv_diast_labels', help='Labels directory')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='test_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create test dataset
    test_dataset = LVPointDataset(
        labels_file=os.path.join(args.labels_dir, 'test_labels.json'),
        images_dir=os.path.join(args.data_dir, 'images', 'test'),
        image_size=args.image_size
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    print(f'Test dataset: {len(test_dataset)} images')
    
    # Create model
    model = SwinPointDetector(num_points=4, pretrained=True)
    
    # Load trained weights
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from {args.model_path} (epoch {checkpoint["epoch"]}, val_loss: {checkpoint["val_loss"]:.6f})')
    else:
        print(f'Model file {args.model_path} not found!')
        return
    
    model = model.to(device)
    
    # Test model
    mse_per_point, measurement_mse = test_model(model, test_loader, device, args.save_dir)
    
    print('\nTesting completed!')

if __name__ == '__main__':
    main()
