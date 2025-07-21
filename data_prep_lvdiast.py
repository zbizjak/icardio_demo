#!/usr/bin/env python3
"""
Data preparation script for LV Diastolic Diameter measurements
Extracts images and labels that contain the required landmarks for LV diastolic diameter:
- lv-ivs-top, lv-ivs-bottom (Interventricular Septum)
- lv-pw-top, lv-pw-bottom (Posterior Wall)
"""

import json
import os
import shutil
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

def load_labels(labels_file: str) -> Dict:
    """Load labels from JSON file"""
    print(f"Loading labels from {labels_file}...")
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    print(f"Loaded {len(labels)} total images")
    return labels

def has_lv_diastolic_landmarks(image_labels: Dict) -> Tuple[bool, List[str]]:
    """
    Check if an image has all required landmarks for LV diastolic diameter measurement
    
    Required landmarks:
    - lv-ivs-top: top of interventricular septum
    - lv-ivs-bottom: bottom of interventricular septum  
    - lv-pw-top: top of posterior wall
    - lv-pw-bottom: bottom of posterior wall
    
    Returns:
        bool: True if all landmarks are present and valid
        List[str]: List of missing or invalid landmarks
    """
    required_landmarks = ['lv-ivs-top', 'lv-ivs-bottom', 'lv-pw-top', 'lv-pw-bottom']
    missing_landmarks = []
    
    if 'labels' not in image_labels:
        return False, ['No labels found']
    
    labels = image_labels['labels']
    
    for landmark in required_landmarks:
        if landmark not in labels:
            missing_landmarks.append(f"{landmark}: not found")
        else:
            landmark_data = labels[landmark]
            if landmark_data.get('type') != 'point':
                missing_landmarks.append(f"{landmark}: type is {landmark_data.get('type', 'unknown')}, not 'point'")
            elif not landmark_data.get('x') or not landmark_data.get('y'):
                missing_landmarks.append(f"{landmark}: missing x or y coordinates")
            else:
                try:
                    float(landmark_data['x'])
                    float(landmark_data['y'])
                except (ValueError, TypeError):
                    missing_landmarks.append(f"{landmark}: invalid coordinates")
    
    return len(missing_landmarks) == 0, missing_landmarks

def extract_lv_coordinates(image_labels: Dict) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Extract LV diastolic measurement coordinates from image labels
    
    Returns:
        Dict with landmark names as keys and (x, y) tuples as values
    """
    if 'labels' not in image_labels:
        return None
    
    labels = image_labels['labels']
    coordinates = {}
    
    landmarks = ['lv-ivs-top', 'lv-ivs-bottom', 'lv-pw-top', 'lv-pw-bottom']
    
    for landmark in landmarks:
        if landmark in labels and labels[landmark].get('type') == 'point':
            try:
                x = float(labels[landmark]['x'])
                y = float(labels[landmark]['y'])
                coordinates[landmark] = (x, y)
            except (ValueError, TypeError):
                return None
    
    return coordinates if len(coordinates) == 4 else None

def calculate_lv_diameter(coordinates: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """
    Calculate LV diastolic diameter measurements
    
    Returns:
        Dict with measurement names and pixel distances
    """
    measurements = {}
    
    # Distance between IVS top and PW top (septal-posterior wall distance at top level)
    ivs_top = coordinates['lv-ivs-top']
    pw_top = coordinates['lv-pw-top']
    top_distance = np.sqrt((ivs_top[0] - pw_top[0])**2 + (ivs_top[1] - pw_top[1])**2)
    measurements['lv_diam_top'] = top_distance
    
    # Distance between IVS bottom and PW bottom (septal-posterior wall distance at bottom level)
    ivs_bottom = coordinates['lv-ivs-bottom']
    pw_bottom = coordinates['lv-pw-bottom']
    bottom_distance = np.sqrt((ivs_bottom[0] - pw_bottom[0])**2 + (ivs_bottom[1] - pw_bottom[1])**2)
    measurements['lv_diam_bottom'] = bottom_distance
    
    # Average diameter
    measurements['lv_diam_avg'] = (top_distance + bottom_distance) / 2
    
    return measurements

def find_image_path(image_name: str, png_cache_dir: str) -> Optional[str]:
    """Find the full path to an image in the png-cache directory structure"""
    # Extract the hash from the filename (between first and second dash)
    parts = image_name.split('-')
    if len(parts) < 3:
        return None
    
    hash_part = parts[1]
    if len(hash_part) < 4:
        return None
    
    # Build the expected path: png-cache/01/hash[:2]/hash[2:4]/image_name
    subdir1 = hash_part[:2]
    subdir2 = hash_part[2:4]
    
    expected_path = os.path.join(png_cache_dir, '01', subdir1, subdir2, image_name)
    
    if os.path.exists(expected_path):
        return expected_path
    
    # If not found, search recursively (slower but more thorough)
    for root, dirs, files in os.walk(png_cache_dir):
        if image_name in files:
            return os.path.join(root, image_name)
    
    return None

def draw_landmarks_on_image(image_path: str, coordinates: Dict[str, Tuple[float, float]], output_path: str):
    """Draw landmarks on image and save to output path"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    # Colors for different landmarks
    colors = {
        'lv-ivs-top': (0, 255, 0),      # Green
        'lv-ivs-bottom': (0, 255, 255), # Yellow
        'lv-pw-top': (255, 0, 0),       # Blue
        'lv-pw-bottom': (255, 0, 255)   # Magenta
    }
    
    # Draw landmarks
    for landmark, (x, y) in coordinates.items():
        color = colors.get(landmark, (255, 255, 255))
        cv2.circle(image, (int(x), int(y)), 5, color, -1)
        cv2.putText(image, landmark, (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw lines between corresponding points
    if all(landmark in coordinates for landmark in ['lv-ivs-top', 'lv-pw-top']):
        pt1 = (int(coordinates['lv-ivs-top'][0]), int(coordinates['lv-ivs-top'][1]))
        pt2 = (int(coordinates['lv-pw-top'][0]), int(coordinates['lv-pw-top'][1]))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    
    if all(landmark in coordinates for landmark in ['lv-ivs-bottom', 'lv-pw-bottom']):
        pt1 = (int(coordinates['lv-ivs-bottom'][0]), int(coordinates['lv-ivs-bottom'][1]))
        pt2 = (int(coordinates['lv-pw-bottom'][0]), int(coordinates['lv-pw-bottom'][1]))
        cv2.line(image, pt1, pt2, (0, 255, 255), 2)
    
    # Save image
    cv2.imwrite(output_path, image)
    return True

def prepare_lv_diastolic_dataset(
    labels_file: str,
    png_cache_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    val_split: float = 0.1
):
    """
    Prepare dataset for LV diastolic diameter detection
    
    Args:
        labels_file: Path to labels JSON file
        png_cache_dir: Path to png-cache directory
        output_dir: Output directory for prepared dataset
        train_split: Fraction of data for training
        val_split: Fraction of data for validation (remainder goes to test)
    """
    
    # Create output directories
    output_path = Path(output_dir)
    
    dirs_to_create = [
        'lv_diast_data/images/train',
        'lv_diast_data/images/val', 
        'lv_diast_data/images/test',
        'lv_diast_data/images_with_labels',
        'lv_diast_labels'
    ]
    
    for dir_name in dirs_to_create:
        (output_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Load all labels
    all_labels = load_labels(labels_file)
    
    # Filter images with required landmarks
    valid_images = []
    invalid_count = 0
    
    print("\nFiltering images with LV diastolic landmarks...")
    
    for image_name, image_labels in all_labels.items():
        has_landmarks, missing = has_lv_diastolic_landmarks(image_labels)
        
        if has_landmarks:
            coordinates = extract_lv_coordinates(image_labels)
            if coordinates:
                measurements = calculate_lv_diameter(coordinates)
                valid_images.append({
                    'image_name': image_name,
                    'labels': image_labels,
                    'coordinates': coordinates,
                    'measurements': measurements
                })
            else:
                invalid_count += 1
        else:
            invalid_count += 1
    
    print(f"Found {len(valid_images)} images with valid LV diastolic landmarks")
    print(f"Filtered out {invalid_count} images without required landmarks")
    
    if len(valid_images) == 0:
        print("ERROR: No valid images found!")
        return
    
    # Shuffle and split data
    np.random.seed(42)  # For reproducible splits
    indices = np.random.permutation(len(valid_images))
    
    n_train = int(len(valid_images) * train_split)
    n_val = int(len(valid_images) * val_split)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    print(f"\nDataset splits:")
    print(f"Train: {len(train_indices)} images")
    print(f"Val: {len(val_indices)} images")
    print(f"Test: {len(test_indices)} images")
    
    # Copy images and prepare labels
    split_labels = {'train': {}, 'val': {}, 'test': {}}
    simple_labels = {}  # Just filename and coordinates
    copy_stats = {'train': 0, 'val': 0, 'test': 0}
    missing_images = []
    
    print(f"\nCopying images and preparing labels...")
    
    for split_name, split_indices in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        for idx in split_indices:
            image_data = valid_images[idx]
            image_name = image_data['image_name']
            
            # Find source image
            source_path = find_image_path(image_name, png_cache_dir)
            
            if source_path:
                # Copy image
                dest_path = output_path / 'lv_diast_data' / 'images' / split_name / image_name
                shutil.copy2(source_path, dest_path)
                copy_stats[split_name] += 1
                
                # Copy image with labels drawn
                labeled_dest_path = output_path / 'lv_diast_data' / 'images_with_labels' / image_name
                draw_landmarks_on_image(source_path, image_data['coordinates'], str(labeled_dest_path))
                
                # Add to split labels
                split_labels[split_name][image_name] = {
                    'original_labels': image_data['labels'],
                    'lv_coordinates': image_data['coordinates'],
                    'lv_measurements': image_data['measurements']
                }
                
                # Add to simple labels (just filename and coordinates)
                simple_labels[image_name] = image_data['coordinates']
                
            else:
                missing_images.append(image_name)
                print(f"WARNING: Could not find image {image_name}")
    
    print(f"\nCopy statistics:")
    for split_name, count in copy_stats.items():
        print(f"{split_name}: {count} images copied")
    
    if missing_images:
        print(f"\nMissing images: {len(missing_images)}")
        for img in missing_images[:10]:  # Show first 10
            print(f"  {img}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
    
    # Save split labels
    for split_name, labels in split_labels.items():
        labels_file = output_path / 'lv_diast_labels' / f'{split_name}_labels.json'
        with open(labels_file, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f"Saved {len(labels)} {split_name} labels to {labels_file}")
    
    # Save simple labels (just filename and coordinates)
    simple_labels_file = output_path / 'lv_diast_labels' / 'simple_labels.json'
    with open(simple_labels_file, 'w') as f:
        json.dump(simple_labels, f, indent=2)
    print(f"Saved {len(simple_labels)} simple labels to {simple_labels_file}")
    
    # Save dataset statistics
    stats = {
        'total_original_images': len(all_labels),
        'valid_lv_images': len(valid_images),
        'train_images': len(split_labels['train']),
        'val_images': len(split_labels['val']),
        'test_images': len(split_labels['test']),
        'missing_images': len(missing_images),
        'landmarks_required': ['lv-ivs-top', 'lv-ivs-bottom', 'lv-pw-top', 'lv-pw-bottom'],
        'measurements_calculated': ['lv_diam_top', 'lv_diam_bottom', 'lv_diam_avg']
    }
    
    stats_file = output_path / 'lv_diast_labels' / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset preparation complete!")
    print(f"Output directory: {output_path}")
    print(f"Dataset statistics saved to: {stats_file}")

def main():
    """Main function to run the data preparation"""
    
    # Configuration
    base_dir = "/icardio"
    labels_file = os.path.join(base_dir, "labels", "labels-all.json")
    png_cache_dir = os.path.join(base_dir, "png-cache")
    output_dir = base_dir
    
    # Check if files exist
    if not os.path.exists(labels_file):
        print(f"ERROR: Labels file not found: {labels_file}")
        return
    
    if not os.path.exists(png_cache_dir):
        print(f"ERROR: PNG cache directory not found: {png_cache_dir}")
        return
    
    print("LV Diastolic Diameter Dataset Preparation")
    print("=" * 50)
    print(f"Labels file: {labels_file}")
    print(f"Images directory: {png_cache_dir}")
    print(f"Output directory: {output_dir}")
    
    # Run data preparation
    prepare_lv_diastolic_dataset(
        labels_file=labels_file,
        png_cache_dir=png_cache_dir,
        output_dir=output_dir,
        train_split=0.8,
        val_split=0.1
    )

if __name__ == "__main__":
    main()
