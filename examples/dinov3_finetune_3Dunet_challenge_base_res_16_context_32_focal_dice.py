"""
DINOv3 3D UNet Fine-tuning Pipeline with Multi-Scale Context Fusion

This file demonstrates training a 3D UNet architecture that processes DINOv3 features
with multi-scale context fusion for improved volumetric segmentation.

Key Strategy:
- Uses highest available resolution for raw data (better DINOv3 features)
- Uses context resolution (64nm) for broader spatial awareness
- Uses base_resolution for segmentation labels (memory efficiency)
- Multi-scale fusion at skip connections for spatial context guidance

Author: GitHub Copilot
Date: 2025-10-04
"""

# %%
# Add current directory to Python path for imports
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import all modularized functions
from dinov3_playground import (
    dinov3_core,
    data_processing,
    models,
    model_training,
    visualization,
    memory_efficient_training,
)
from importlib import reload

reload(dinov3_core)
reload(data_processing)
reload(models)
reload(model_training)
reload(visualization)
reload(memory_efficient_training)

from dinov3_playground.memory_efficient_training import (
    MemoryEfficientDataLoader3D,
    train_3d_unet_with_memory_efficient_loader,  # Updated import
)

from dinov3_playground.dinov3_core import enable_amp_inference, process, output_channels

from dinov3_playground.data_processing import (
    load_random_3d_training_data,
    generate_multi_organelle_dataset_pairs,
    get_class_names_from_dataset_pairs,
)

from dinov3_playground.models import (
    DINOv3UNet3D,
    DINOv3UNet3DPipeline,
    print_model_summary,
)


from dinov3_playground import (
    initialize_dinov3,
    get_current_model_info,  # Add these imports
)

# %%
# Configuration
print("=" * 60)
print("DINOV3 3D UNET TRAINING WITH FOCAL+DICE LOSS")
print("=" * 60)

# Model configuration
# MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"

base_resolution = 16  # Resolution for segmentation labels
min_resolution_for_raw = 4  # Don't use scales finer than 4nm for raw data
context_resolution = 32  # Context resolution for multi-scale fusion

# For multi-resolution processing, DINOv3 should process at native resolution
# The raw data is 512×512 slices, so we want DINOV3_SLICE_SIZE = 512
# This avoids unnecessary upsampling
DINOV3_SLICE_SIZE = (
    128 * base_resolution / min_resolution_for_raw
)  # Native resolution of raw data slices

# For DINOv3 model initialization, we can use the same size
IMAGE_SIZE = DINOV3_SLICE_SIZE

# 3D UNet specific configuration
VOLUME_SIZE = (128, 128, 128)  # Target size for training (matches GT)
BASE_CHANNELS = 128  # Lower for 3D due to memory constraints

# LOSS FUNCTION CONFIGURATION
# Options: 'ce', 'weighted_ce', 'focal', 'dice', 'focal_dice', 'tversky'
LOSS_TYPE = "focal_dice"  # Use combined Focal + Dice loss for best results
FOCAL_GAMMA = (
    2.0  # Focusing parameter (2.0 is standard, higher = more focus on hard examples)
)
FOCAL_WEIGHT = 0.5  # Weight for focal component in combined loss
DICE_WEIGHT = 0.5  # Weight for dice component in combined loss
DICE_SMOOTH = 1.0  # Smoothing constant for Dice loss

print(f"Multi-resolution configuration with context fusion:")
print(f"  Raw data resolution: {min_resolution_for_raw}nm (actual data)")
print(f"  GT data resolution: {base_resolution}nm")
print(f"  Context resolution: {context_resolution}nm (broader spatial coverage)")
print(f"  DINOv3 slice size: {DINOV3_SLICE_SIZE}×{DINOV3_SLICE_SIZE} (native)")
print(f"  Training volume size: {VOLUME_SIZE} (matches GT)")
print(f"\nLoss function configuration:")
print(f"  Loss type: {LOSS_TYPE}")
if LOSS_TYPE in ["focal", "focal_dice"]:
    print(f"  Focal gamma: {FOCAL_GAMMA}")
if LOSS_TYPE == "focal_dice":
    print(f"  Focal weight: {FOCAL_WEIGHT}")
    print(f"  Dice weight: {DICE_WEIGHT}")
    print(f"  Dice smooth: {DICE_SMOOTH}")
print(
    f"  Strategy: Multi-scale context fusion with DINOv3 features from both raw and context"
)

# Export directory configuration
filename = os.path.basename(__file__).replace(".py", "")
EXPORT_BASE_DIR = f"/nrs/cellmap/ackermand/dinov3_training/results/{filename}"
print(f"Export base directory: {EXPORT_BASE_DIR}")

# Create export directory if it doesn't exist
os.makedirs(EXPORT_BASE_DIR, exist_ok=True)

# Initialize DINOv3 with our specific model and size
processor, model, output_channels = initialize_dinov3(
    model_id=MODEL_ID, image_size=IMAGE_SIZE
)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get model info
model_info = get_current_model_info()
print(f"DINOv3 Model ID: {model_info['model_id']}")
print(f"Output channels: {model_info['output_channels']}")
print(f"Model device: {model_info['device']}")

# %%

# Load data
print("Loading datasets...")
import dinov3_playground.zarr_util

reload(dinov3_playground.zarr_util)
from dinov3_playground.zarr_util import update_datapaths_with_target_scales


# Strategy: Use highest available resolution for raw data (better DINOv3 features)
# but use base_resolution for segmentation labels (memory efficiency)

# Define your dataset pairs here - NEW MULTI-CLASS FORMAT with CONTEXT
dataset_pairs = generate_multi_organelle_dataset_pairs(
    organelle_list=["nuc", "mito", "er"],
    min_resolution_for_raw=min_resolution_for_raw,
    base_resolution=base_resolution,
    apply_scale_updates=True,
    require_all_organelles=True,  # Only include datasets that have ALL specified organelles
    context_scale=context_resolution,  # NEW: Add context data at specified resolution
)
# %%
NUM_VOLUMES = 300
# Load 3D training data with CONTEXT SUPPORT
print("Loading 3D volumetric training data with context...")

# Extract class names from dataset_pairs for better metric visualization
class_names = get_class_names_from_dataset_pairs(dataset_pairs)
print(f"Detected class names: {class_names}")

# Use the existing context-aware loading function (now with detailed ROI logging)
print("🎯 Loading 3D volumetric training data with detailed ROI coordinate logging...")

raw, gt, gt_masks, context_volumes, dataset_sources, num_classes = (
    load_random_3d_training_data(
        dataset_pairs=dataset_pairs,
        volume_shape=VOLUME_SIZE,  # (128, 128, 128)
        base_resolution=base_resolution,
        min_label_fraction=0.01,
        num_volumes=NUM_VOLUMES,  # Number of 3D volumes to sample
        seed=42,
        min_resolution_for_raw=min_resolution_for_raw,
        allow_gt_extension=True,  # Extend GT to avoid boundary issues
        context_scale=context_resolution,  # Context at specified resolution
    )
)

print(f"Loaded 3D volumetric training data with context:")
print(f"  Raw data shape: {raw.shape}")
print(f"  Ground truth shape: {gt.shape}")
print(f"  GT masks shape: {gt_masks.shape}")
print(f"  Context volumes shape: {context_volumes.shape}")
print(f"  Number of classes: {num_classes}")
print(f"  Unique GT values: {np.unique(gt)}")
print(f"  Overall label fraction: {np.sum(gt) / gt.size:.3f}")

# Verify each volume has the correct shape
for i in range(min(5, len(raw))):
    vol_label_fraction = np.sum(gt[i]) / gt[i].size
    print(
        f"  Volume {i}: raw {raw[i].shape}, gt {gt[i].shape}, context {context_volumes[i].shape}, label fraction {vol_label_fraction:.3f}"
    )

# Print ROI information for spatial context understanding
print(f"\nSpatial ROI Information:")
print(f"  GT volume size: {VOLUME_SIZE} voxels at {base_resolution}nm resolution")
print(
    f"  GT spatial coverage: {VOLUME_SIZE[0] * base_resolution}×{VOLUME_SIZE[1] * base_resolution}×{VOLUME_SIZE[2] * base_resolution} nm"
)

# Raw ROI (higher resolution, same spatial coverage as GT)
raw_voxel_size = (
    VOLUME_SIZE[0] * base_resolution // min_resolution_for_raw,
    VOLUME_SIZE[1] * base_resolution // min_resolution_for_raw,
    VOLUME_SIZE[2] * base_resolution // min_resolution_for_raw,
)
print(
    f"  Raw volume size: {raw_voxel_size} voxels at {min_resolution_for_raw}nm resolution"
)
print(
    f"  Raw spatial coverage: {raw_voxel_size[0] * min_resolution_for_raw}×{raw_voxel_size[1] * min_resolution_for_raw}×{raw_voxel_size[2] * min_resolution_for_raw} nm (same as GT)"
)

# Context ROI (lower resolution, much larger spatial coverage)
context_spatial_coverage = (
    raw_voxel_size[0] * context_resolution,
    raw_voxel_size[1] * context_resolution,
    raw_voxel_size[2] * context_resolution,
)
print(
    f"  Context volume size: {raw_voxel_size} voxels at {context_resolution}nm resolution"
)
print(
    f"  Context spatial coverage: {context_spatial_coverage[0]}×{context_spatial_coverage[1]}×{context_spatial_coverage[2]} nm"
)
print(
    f"  Context vs Raw spatial ratio: {context_resolution // min_resolution_for_raw}x larger in each dimension"
)
print(
    f"  Context vs Raw total volume ratio: {(context_resolution // min_resolution_for_raw)**3}x larger"
)

# Create dataset name mapping for plotting
dataset_names = []
for i, pair in enumerate(dataset_pairs):
    if isinstance(pair, dict):
        # New dictionary format - extract a descriptive name from the raw path
        raw_path = pair.get("raw", "unknown")
        # Extract dataset name from path (e.g., jrc_mus-liver-zon-1)
        if isinstance(raw_path, str):
            path_parts = raw_path.split("/")
            for part in path_parts:
                if part.startswith("jrc_") or part.startswith("cellmap"):
                    dataset_names.append(part)
                    break
            else:
                dataset_names.append(f"Dataset-{i+1}")
        else:
            dataset_names.append(f"Dataset-{i+1}")
    else:
        # Legacy tuple format
        dataset_names.append(f"Dataset-{i+1}")

print(f"  Dataset names: {dataset_names}")

# Visualize 3D data (show percentile slices from multiple volumes)
fig, axes = plt.subplots(
    20, 5, figsize=(20, 48)
)  # 20 rows (4 types × 5 slices), 5 columns (volumes)

print("Processing sample volumes through DINOv3...")

# Define 5 physical depths to visualize (in nm from volume center)
# We'll plot at the center and ±25%, ±50% of the GT volume depth
gt_depth_nm = (
    VOLUME_SIZE[2] * base_resolution
)  # Total GT depth in nm (e.g., 128 * 16 = 2048nm)
physical_depths_nm = [
    -gt_depth_nm // 2,  # Bottom of GT volume (relative to center)
    -gt_depth_nm // 4,  # 25% below center
    0,  # Center
    gt_depth_nm // 4,  # 25% above center
    gt_depth_nm // 2
    - base_resolution,  # Top of GT volume (minus one voxel to stay in bounds)
]
depth_labels = ["-50%", "-25%", "0% (center)", "+25%", "+50%"]

for vol_idx in range(5):  # 5 volumes (columns)
    if vol_idx < raw.shape[0]:
        # Get dataset source for this volume
        source_idx = dataset_sources[vol_idx] if vol_idx < len(dataset_sources) else 0
        source_name = (
            dataset_names[source_idx]
            if source_idx < len(dataset_names)
            else f"Dataset-{source_idx}"
        )

        for slice_idx, (depth_nm, depth_label) in enumerate(
            zip(physical_depths_nm, depth_labels)
        ):  # 5 slices per volume
            # Calculate z-indices for each volume based on physical depth from center
            # All volumes are centered, so:
            # - Center voxel: volume_depth // 2
            # - Physical depth from center: depth_nm
            # - Z-index: center_voxel + (depth_nm / resolution)

            gt_center = VOLUME_SIZE[2] // 2  # e.g., 128 // 2 = 64
            gt_z = int(
                gt_center + depth_nm / base_resolution
            )  # Convert nm to GT voxel index

            raw_center = raw.shape[1] // 2  # e.g., 512 // 2 = 256
            raw_z = int(
                raw_center + depth_nm / min_resolution_for_raw
            )  # Convert nm to raw voxel index

            context_center = context_volumes.shape[1] // 2  # e.g., 512 // 2 = 256
            context_z = int(
                context_center + depth_nm / context_resolution
            )  # Convert nm to context voxel index

            # Clamp indices to valid ranges
            gt_z = max(0, min(gt_z, VOLUME_SIZE[2] - 1))
            raw_z = max(0, min(raw_z, raw.shape[1] - 1))
            context_z = max(0, min(context_z, context_volumes.shape[1] - 1))

            raw_slice = raw[vol_idx, raw_z]  # Shape: (512, 512) at 4nm resolution
            gt_slice = gt[vol_idx, gt_z]  # Shape: (128, 128) at 16nm resolution
            mask_slice = gt_masks[vol_idx, gt_z]  # Shape: (128, 128) at 16nm resolution
            context_slice = context_volumes[
                vol_idx, context_z
            ]  # Shape: (512, 512) at 64nm resolution

            # Process slices through DINOv3
            slice_features = process(raw_slice[np.newaxis, ...])  # Add batch dim
            context_features = process(
                context_slice[np.newaxis, ...]
            )  # Context DINOv3 features

            # Calculate row indices for the 4 image types
            raw_row = slice_idx * 4  # 0, 4, 8, 12, 16
            feat_row = slice_idx * 4 + 1  # 1, 5, 9, 13, 17
            context_row = slice_idx * 4 + 2  # 2, 6, 10, 14, 18
            context_feat_row = slice_idx * 4 + 3  # 3, 7, 11, 15, 19
            # Raw slice with GT overlay and mask boundaries
            axes[raw_row, vol_idx].imshow(raw_slice, cmap="gray")
            # Resize GT to match raw slice dimensions for proper overlay
            from scipy.ndimage import zoom
            from matplotlib.patches import Rectangle
            import cv2

            scale_factor = raw_slice.shape[0] / gt_slice.shape[0]  # e.g., 512/128 = 4
            gt_resized = zoom(
                gt_slice, scale_factor, order=0
            )  # Nearest neighbor for labels
            mask_resized = zoom(
                mask_slice, scale_factor, order=0
            )  # Nearest neighbor for mask

            # Overlay GT segmentation with transparency (only show non-background)
            gt_overlay = np.ma.masked_where(gt_resized == 0, gt_resized)
            axes[raw_row, vol_idx].imshow(
                gt_overlay, cmap="tab10", alpha=0.4, vmin=0, vmax=num_classes - 1
            )

            # Add red boxes around regions with valid GT (where mask = 1)
            # Find contours of valid GT regions
            mask_uint8 = (mask_resized > 0).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # Get bounding rectangle for each valid GT region
                x, y, w, h = cv2.boundingRect(contour)
                if w > 10 and h > 10:  # Only show boxes for reasonably large regions
                    rect = Rectangle(
                        (x, y),
                        w,
                        h,
                        linewidth=1,
                        edgecolor="red",
                        facecolor="none",
                        alpha=0.8,
                    )
                    axes[raw_row, vol_idx].add_patch(rect)

            if slice_idx == 0:  # Only show volume info on first slice
                axes[raw_row, vol_idx].set_title(
                    f"Vol {vol_idx+1} ({source_name})\nRaw + GT + Valid ({depth_label})\nGT_z={gt_z}, Raw_z={raw_z}",
                    fontsize=8,
                )
            else:
                axes[raw_row, vol_idx].set_title(
                    f"Raw + GT + Valid ({depth_label})\nGT_z={gt_z}, Raw_z={raw_z}",
                    fontsize=8,
                )
            axes[raw_row, vol_idx].axis("off")

            # DINOv3 feature channel (from raw data)
            axes[feat_row, vol_idx].imshow(slice_features[200, 0], cmap="viridis")
            axes[feat_row, vol_idx].set_title(
                f"Raw Features ({depth_label})", fontsize=8
            )
            axes[feat_row, vol_idx].axis("off")

            # Context slice (16x broader spatial coverage at 64nm resolution)
            axes[context_row, vol_idx].imshow(context_slice, cmap="gray")

            # Add red square to show where raw data corresponds within context
            # Context: 64nm resolution, Raw: 4nm resolution → 64/4 = 16x larger spatial coverage
            # Same pixels (512x512) but context covers 16x larger area
            context_center = context_slice.shape[0] // 2
            resolution_ratio = context_resolution // min_resolution_for_raw  # 64/4 = 16
            raw_coverage_in_context = (
                context_slice.shape[0] // resolution_ratio
            )  # 512/16 = 32 pixels

            # Calculate red square boundaries (centered)
            half_raw = raw_coverage_in_context // 2  # 16 pixels
            x1, x2 = context_center - half_raw, context_center + half_raw  # 256±16
            y1, y2 = context_center - half_raw, context_center + half_raw

            # Draw red rectangle outline
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            axes[context_row, vol_idx].add_patch(rect)

            axes[context_row, vol_idx].set_title(
                f"Context (16x coverage) ({depth_label})\nCtx_z={context_z}", fontsize=8
            )
            axes[context_row, vol_idx].axis("off")

            # Context DINOv3 features
            axes[context_feat_row, vol_idx].imshow(
                context_features[200, 0], cmap="plasma"
            )
            axes[context_feat_row, vol_idx].set_title(
                f"Context Features ({depth_label})", fontsize=8
            )
            axes[context_feat_row, vol_idx].axis("off")

plt.suptitle(
    "Multi-Scale Context Fusion Training Data (Same Physical Depth)\n(Raw+GT+ValidRegions, RawFeatures, Context+RawField, ContextFeatures - 5 Physical Depths)",
    fontsize=16,
)
plt.tight_layout()
plt.show()


# %%
# # Example 1: 3D UNet Architecture Testing
# print("=" * 60)
# print("EXAMPLE 1: 3D UNet Architecture Testing")
# print("=" * 60)

# # Get current model info to use correct input channels
# model_info = get_current_model_info()
# current_output_channels = model_info["output_channels"]

# print("Creating DINOv3 3D UNet model with context fusion...")
# unet3d = DINOv3UNet3D(
#     input_channels=current_output_channels,
#     num_classes=num_classes,
#     base_channels=BASE_CHANNELS,
#     input_size=VOLUME_SIZE,
#     use_context_fusion=True,  # Enable context fusion
#     context_channels=current_output_channels,  # Same as raw features
# ).to(device)

# # Print model summary
# print_model_summary(unet3d, input_shape=(current_output_channels, *VOLUME_SIZE))

# # Test forward pass
# print("Testing 3D forward pass...")

# # Check device status first
# print(f"Target device: {device}")
# print(f"Model device: {next(unet3d.parameters()).device}")
# print(f"CUDA available: {torch.cuda.is_available()}")

# # Create test input and explicitly move to device
# if torch.cuda.is_available() and device.type == "cuda":
#     test_input = torch.randn(1, current_output_channels, *VOLUME_SIZE, device=device)
# else:
#     test_input = torch.randn(1, current_output_channels, *VOLUME_SIZE)
#     unet3d = unet3d.cpu()  # Move model to CPU if CUDA not available
#     device = torch.device("cpu")

# print(f"Input device after creation: {test_input.device}")
# print(f"Model device after setup: {next(unet3d.parameters()).device}")

# try:
#     with torch.no_grad():
#         test_output = unet3d(test_input)

#     print(f"✅ Forward pass successful!")
#     print(f"Input shape: {test_input.shape}")
#     print(f"Output shape: {test_output.shape}")
#     print(f"Output contains logits for {test_output.shape[1]} classes")

# except Exception as e:
#     print(f"❌ Forward pass failed: {e}")
#     print("Trying with CPU fallback...")

#     # Fallback: move everything to CPU
#     unet3d = unet3d.cpu()
#     test_input = test_input.cpu()

#     with torch.no_grad():
#         test_output = unet3d(test_input)

#     print(f"✅ CPU forward pass successful!")
#     print(f"Input shape: {test_input.shape}")
#     print(f"Output shape: {test_output.shape}")

# # Show memory usage
# memory_info = unet3d.get_memory_usage(batch_size=1)
# print(f"\nMemory usage estimates:")
# for key, value in memory_info.items():
#     if isinstance(value, (int, float)):
#         print(f"  {key}: {value:.2f} MB")

# %%
# Example 3: Memory-Efficient 3D UNet Training
print("=" * 60)
print("EXAMPLE 3: Memory-Efficient 3D UNet Training")
print("=" * 60)

# Get current model info to use correct input channels
model_info = get_current_model_info()
current_output_channels = model_info["output_channels"]

# Use the updated memory-efficient training function
print("Training 3D UNet with memory-efficient loader and context fusion...")
# Choose your loss function
LOSS_TYPE = "focal_dice"  # Options: 'ce', 'weighted_ce', 'focal', 'dice', 'focal_dice', 'tversky'

# Configure loss parameters
FOCAL_GAMMA = 2.0  # For focal loss (higher = more focus on hard examples)
FOCAL_WEIGHT = 0.5  # Weight for focal component in combined loss
DICE_WEIGHT = 0.5  # Weight for dice component in combined loss
DICE_SMOOTH = 1.0  # Smoothing for Dice loss

training_results = train_3d_unet_with_memory_efficient_loader(
    raw_data=raw,
    gt_data=gt,
    gt_masks=gt_masks,  # GT extension masks for proper loss calculation
    context_data=context_volumes,  # Context volumes for multi-scale fusion (auto-enables fusion)
    context_scale=context_resolution,  # Context resolution for feature processing
    train_volume_pool_size=NUM_VOLUMES - 5,  # Use 15 volumes for training pool
    val_volume_pool_size=5,  # Use 5 volumes for validation
    num_classes=num_classes,  # Auto-detected from data
    class_names=class_names,  # Class names for better metric visualization
    target_volume_size=VOLUME_SIZE,  # Target 3D volume size
    dinov3_slice_size=DINOV3_SLICE_SIZE,  # DINOv3 slice size
    volumes_per_batch=1,  # Process 1 volume per batch (memory efficient)
    batches_per_epoch=20,  # 20 batches per epoch
    epochs=1000,  # Training epochs
    learning_rate=1e-3,  # Learning rate
    weight_decay=1e-4,  # Weight decay for regularization
    patience=50,  # Early stopping patience
    min_delta=0.001,  # Minimum improvement threshold
    base_channels=BASE_CHANNELS,  # Base channels for 3D UNet
    device=device,  # Training device
    seed=42,  # Random seed for reproducibility
    model_id=MODEL_ID,  # DINOv3 model identifier
    export_base_dir=EXPORT_BASE_DIR,  # Directory for saving results
    save_checkpoints=True,  # Enable checkpoint saving
    checkpoint_every_n_epochs=10,  # Save model checkpoint every 10 epochs (in addition to best)
    use_class_weighting=True,  # Use class-balanced loss
    use_orthogonal_planes=True,  # Use orthogonal planes for training (processes context too)
    # LOSS FUNCTION OPTIONS
    loss_type=LOSS_TYPE,  # Type of loss function
    focal_gamma=FOCAL_GAMMA,  # Focusing parameter for Focal Loss
    focal_weight=FOCAL_WEIGHT,  # Weight for focal component in combined loss
    dice_weight=DICE_WEIGHT,  # Weight for dice component in combined loss
    dice_smooth=DICE_SMOOTH,  # Smoothing for Dice loss
    # MEMORY EFFICIENCY OPTIONS
    use_mixed_precision=True,  # 30-50% memory reduction
    use_half_precision=False,  # Additional 50% reduction (but potential numerical issues)
    memory_efficient_mode="auto",  # Let system decide based on GPU memory
    enable_detailed_timing=False,
    verbose=False,  # Suppress verbose DINOv3 processing messages
    # DATA RESOLUTION PARAMETERS
    min_resolution_for_raw=min_resolution_for_raw,  # Raw data resolution (4nm)
    base_resolution=base_resolution,  # Ground truth resolution (16nm)
)

print(f"Training completed!")
print(f"Best validation accuracy: {training_results['best_val_acc']:.4f}")
print(f"Epochs trained: {training_results['epochs_trained']}")
print(f"Checkpoints saved to: {training_results['checkpoint_dir']}")
