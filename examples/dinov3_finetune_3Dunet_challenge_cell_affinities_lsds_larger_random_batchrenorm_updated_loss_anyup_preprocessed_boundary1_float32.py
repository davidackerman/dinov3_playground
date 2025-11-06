"""
DINOv3 3D UNet Fine-tuning Pipeline with Preprocessed Features

This file demonstrates training a 3D UNet architecture using pre-extracted DINOv3 features
stored in TensorStore for fast, efficient training.

Key Strategy:
- Uses preprocessed DINOv3 features (no on-the-fly extraction)
- Fast loading from TensorStore with parallel I/O
- Supports float16 features for reduced memory and faster loading
- Compatible with original training API

Author: Updated for preprocessing
Date: 2025-10-24
"""

###
# python submit_preprocessing_jobs.py     --output-dir /nrs/cellmap/ackermand/to_delete/dinov3_volumes/cell/volumes/     --num-volumes 300     --start-index 0     --num-processors 12     --organelles cell     --inference-filter jrc_22ak351-leaf-2lb jrc_22ak351-leaf-3mb jrc_22ak351-leaf-3rb jrc_22ak351-leaf-3r jrc_c-elegans-op50-1 jrc_mus-liver-zon-1 jrc_mus-liver-zon-2     --crop-filter 1021 1063 1064 1065 1066 1067 510 514 515 1086 1087 1088 1089 1090 1076 1077 1078 1079 1080 129 156 508     --queue gpu_h200     --model-id facebook/dinov3-vitl16-pretrain-sat493m     --base-resolution 128     --min-resolution-for-raw 32     --output-image-dim 128     --min-label-fraction 0.01     --min-unique-ids 2     --min-ground-truth-fraction 0.05
# %%
# Add current directory to Python path for imports
import os
import sys
import torch
import numpy as np

# Import training function
from dinov3_playground.memory_efficient_training import (
    train_3d_unet_with_memory_efficient_loader,
)

# %%
# Configuration
print("=" * 60)
print("DINOV3 3D UNET TRAINING PIPELINE - PREPROCESSED MODE")
print("=" * 60)

# Model configuration
MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
base_resolution = 128  # Resolution for segmentation labels
min_resolution_for_raw = 32  # Raw data resolution

# Path to preprocessed volumes
PREPROCESSED_DIR = "/nrs/cellmap/ackermand/to_delete/dinov3_volumes/cell/volumes/"
print(f"Preprocessed directory: {PREPROCESSED_DIR}")

# Training configuration
DEBUG = False
OUTPUT_IMAGE_DIM = 128
BASE_CHANNELS = 128  # Matches original
DINOv3_SLICE_SIZE = (
    OUTPUT_IMAGE_DIM * base_resolution / min_resolution_for_raw
)  # Default DINOv3 slice size

# Export directory configuration
EXPORT_BASE_DIR = f"/nrs/cellmap/ackermand/dinov3_training/results/{os.path.splitext(os.path.basename(__file__))[0]}"
print(f"Export base directory: {EXPORT_BASE_DIR}")

# Create export directory if it doesn't exist
os.makedirs(EXPORT_BASE_DIR, exist_ok=True)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# Volume configuration
# Assuming you preprocessed volumes with indices 0-299
NUM_VOLUMES = 230
val_volume_pool_size = 5
train_volume_pool_size = NUM_VOLUMES - val_volume_pool_size

print(f"Total preprocessed volumes: {NUM_VOLUMES}")
print(f"Training volumes: {train_volume_pool_size}")
print(f"Validation volumes: {val_volume_pool_size}")

# %%
# Define affinity offsets (must match what was used during preprocessing)
affinity_offsets = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (3, 0, 0),
    (0, 3, 0),
    (0, 0, 3),
    (9, 0, 0),
    (0, 9, 0),
    (0, 0, 9),
]

# Calculate total output channels: 10 LSDs + len(affinity_offsets) affinities
num_output_channels = 10 + len(affinity_offsets)
print(
    f"Output channels: {num_output_channels} = 10 LSDs + {len(affinity_offsets)} affinities"
)

# %%
# Training with preprocessed data
print("=" * 60)
print("Training 3D UNet with Preprocessed Features")
print("=" * 60)
DEBUG = False
volumes_per_batch = 1 if DEBUG else 3
training_results = x

print(f"\nTraining completed!")
print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
print(f"Epochs trained: {training_results['epochs_trained']}")
print(f"Checkpoints saved to: {training_results['checkpoint_dir']}")
