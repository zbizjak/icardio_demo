# 15 Point Detection - LV Diastolic D## Train Model

```bash
# Basic training with increased shared memory
docker run --rm --gpus all --shm-size=50g -v $(pwd):/icardio point_detection_icardio python train.py --pretrained

# Custom parameters
docker run --rm --gpus all --shm-size=50g -v $(pwd):/icardio point_detection_icardio python3 train.py \
    --batch_size 32 \
    --num_epochs 150 \
    --image_size 224 \
    --pretrained
``` project implements a Swin Transformer-based model for detecting Left Ventricular (LV) diastolic diameter measurement points in cardiac ultrasound images.

## Model Architecture

The model uses a **Swin Transformer** backbone for feature extraction followed by a regression head for point detection:

- **Backbone**: Swin Transformer (microsoft/swin-tiny-patch4-window7-224)
- **Input**: 224x224 RGB images
- **Output**: 4 landmark points (8 coordinates)
  - `lv-ivs-top`: Top of interventricular septum
  - `lv-ivs-bottom`: Bottom of interventricular septum
  - `lv-pw-top`: Top of posterior wall
  - `lv-pw-bottom`: Bottom of posterior wall

### Architecture Details
- **Feature Extraction**: Swin Transformer with hierarchical feature maps
- **Regression Head**: 
  - Global Average Pooling
  - FC layers: 768 → 512 → 256 → 8 (4 points × 2 coordinates)
  - ReLU activation + Dropout for regularization
- **Loss Function**: MSE Loss on normalized coordinates
- **Optimizer**: AdamW with Cosine Annealing LR scheduler

## Build Docker Image

```bash
docker build -t point_detection_icardio .
```

## Run Data Preparation Script

```bash
docker run --rm -v $(pwd):/icardio point_detection_icardio python data_prep_lvdiast.py
```

## Train Model

```bash
# Basic training
docker run --rm --gpus all -v $(pwd):/icardio point_detection_icardio python train.py --pretrained

# Custom parameters
docker run --rm --gpus all -v $(pwd):/icardio point_detection_icardio python3 train.py \
    --batch_size 32 \
    --num_epochs 150 \
    --image_size 224 \
    --pretrained
```

### Training Parameters
- `--data_dir`: Data directory (default: /icardio/lv_diast_data)
- `--labels_dir`: Labels directory (default: /icardio/lv_diast_labels)
- `--batch_size`: Batch size (default: 16)
- `--num_epochs`: Number of epochs (default: 100)
- `--image_size`: Input image size (default: 224)
- `--pretrained`: Use pretrained Swin model
- `--device`: Device to use (default: cuda)

## Test Model

```bash
# Test trained model
docker run --rm --gpus all --shm-size=50g -v $(pwd):/icardio point_detection_icardio python3 test.py \
    --model_path best_model.pth \
    --save_dir test_results
```

### Test Parameters
- `--model_path`: Path to trained model (default: best_model.pth)
- `--save_dir`: Directory to save test results (default: test_results)