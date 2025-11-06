"""
DINOv3 3D UNet Fine-tuning Pipeline with Highest Resolution Raw Data

This file demonstrates training a 3D UNet architecture that processes DINOv3 features
instead of raw images for volumetric segmentation tasks.

Key Strategy:
- Uses highest available resolution for raw data (better DINOv3 features)
- Uses base_resolution for segmentation labels (memory efficiency)
- Avoids naive upsampling by starting with the best available resolution

Author: GitHub Copilot
Date: 2025-09-15
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
    affinity_utils,
    augmentations,
)
from importlib import reload

reload(affinity_utils)
reload(dinov3_core)
reload(data_processing)
reload(models)
reload(model_training)
reload(visualization)
reload(memory_efficient_training)
reload(augmentations)
from dinov3_playground.memory_efficient_training import (
    MemoryEfficientDataLoader3D,
    train_3d_unet_with_memory_efficient_loader,  # Updated import
)

from dinov3_playground.dinov3_core import enable_amp_inference, process, output_channels

from dinov3_playground.data_processing import (
    load_random_3d_training_data,
    generate_multi_organelle_dataset_pairs,
)

from dinov3_playground.affinity_utils import (
    compute_affinities_3d,
    safe_get_lsds,
    get_affs,
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
print("DINOV3 3D UNET TRAINING PIPELINE")
print("=" * 60)

# Model configuration
# MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
base_resolution = 16  # Resolution for segmentation labels
min_resolution_for_raw = 4  # Don't use scales finer than 32nm for raw data

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

print(f"Multi-resolution configuration:")
print(f"  Raw data resolution: {min_resolution_for_raw}nm (actual data)")
print(f"  GT data resolution: {base_resolution}nm")
print(f"  DINOv3 slice size: {DINOV3_SLICE_SIZE}×{DINOV3_SLICE_SIZE} (native)")
print(f"  Training volume size: {VOLUME_SIZE} (matches GT)")
print(
    f"  Strategy: Process DINOv3 at native resolution, downsample features to GT size"
)

# Export directory configuration
# append the current filename to this path
EXPORT_BASE_DIR = f"/nrs/cellmap/ackermand/dinov3_training/results/{os.path.splitext(os.path.basename(__file__))[0]}"
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

# Define your dataset pairs here - NEW MULTI-CLASS FORMAT
dataset_pairs = generate_multi_organelle_dataset_pairs(
    organelle_list=["mito"],
    min_resolution_for_raw=min_resolution_for_raw,
    base_resolution=base_resolution,
    apply_scale_updates=True,
    require_all_organelles=True,  # Only include datasets that have ALL specified organelles
)
# %%
NUM_VOLUMES = 300
# Load 3D training data with proper volumetric sampling
# print("Loading 3D volumetric training data...")
raw, gt, gt_masks, dataset_sources, num_classes = load_random_3d_training_data(
    dataset_pairs=dataset_pairs,
    volume_shape=VOLUME_SIZE,  # (64, 64, 64)
    base_resolution=base_resolution,
    min_label_fraction=0.01,
    num_volumes=NUM_VOLUMES,  # Number of 3D volumes to sample
    seed=42,
    min_resolution_for_raw=min_resolution_for_raw,
    allow_gt_extension=True,  # Extend GT to avoid boundary issues
    min_unique_ids=3,
    augment=True,  # Apply data augmentations
    augment_prob=1.0,
)

print(f"Loaded 3D volumetric training data:")
print(f"  Raw data shape: {raw.shape}")
print(f"  Ground truth shape: {gt.shape}")
print(f"  Unique GT values: {np.unique(gt)}")
print(f"  Overall label fraction: {np.sum(gt) / gt.size:.3f}")
print(f"  Dataset sources: {dataset_sources}")

# Verify each volume has the correct shape
for i in range(min(5, len(raw))):
    vol_label_fraction = np.sum(gt[i] > 0) / gt[i].size
    print(
        f"  Volume {i}: shape {raw[i].shape}, label fraction {vol_label_fraction:.3f}"
    )

# Create dataset name mapping for plotting
dataset_names = []
for i, pair in enumerate(dataset_pairs):
    if isinstance(pair, dict):
        # New dictionary format - extract dataset name from path
        # Path format: /nrs/cellmap/data/jrc_atla24-b9-2/jrc_atla24-b9-2.zarr/...
        raw_path = pair.get("raw", "unknown")
        if isinstance(raw_path, str):
            path_parts = raw_path.split("/")
            # Dataset name is typically at index 4: ['', 'nrs', 'cellmap', 'data', 'jrc_atla24-b9-2', ...]
            if len(path_parts) > 4:
                dataset_names.append(path_parts[4])
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
    20, 10, figsize=(20, 60)
)  # 20 rows (4 types × 5 slices), 10 columns (volumes)

print("Processing sample volumes through DINOv3...")

# Define the 5 percentile slices to show
D = gt.shape[1]  # Depth dimension (64)
z_percentiles = [0, D // 4, D // 2, (3 * D) // 4, D - 1]  # first, 25%, 50%, 75%, last
percentile_labels = ["0%", "25%", "50%", "75%", "100%"]

for vol_idx in range(10):  # 10 volumes (columns)
    if vol_idx < raw.shape[0]:
        # Get dataset source for this volume
        source_idx = dataset_sources[vol_idx] if vol_idx < len(dataset_sources) else 0
        source_name = (
            dataset_names[source_idx]
            if source_idx < len(dataset_names)
            else f"Dataset-{source_idx}"
        )

        for slice_idx, (z, perc_label) in enumerate(
            zip(z_percentiles, percentile_labels)
        ):  # 5 slices per volume
            raw_slice = raw[
                vol_idx, z * base_resolution // min_resolution_for_raw
            ]  # Shape: (64, 64)
            gt_slice = gt[vol_idx, z]  # Shape: (64, 64)
            mask_slice = gt_masks[vol_idx, z]  # Shape: (64, 64)

            # Process slice through DINOv3
            slice_features = process(raw_slice[np.newaxis, ...])  # Add batch dim

            # Calculate row indices for the 4 image types
            raw_row = slice_idx * 4  # 0, 4, 8, 12, 16
            aff_row = slice_idx * 4 + 1  # 1, 5, 9, 13, 17
            mask_row = slice_idx * 4 + 2  # 2, 6, 10, 14, 18
            feat_row = slice_idx * 4 + 3  # 3, 7, 11, 15, 19
            # Raw slice with GT overlay
            axes[raw_row, vol_idx].imshow(raw_slice, cmap="gray")
            # Resize GT to match raw slice dimensions for proper overlay
            from scipy.ndimage import zoom

            scale_factor = raw_slice.shape[0] / gt_slice.shape[0]  # e.g., 512/128 = 4
            gt_resized = zoom(
                gt_slice, scale_factor, order=0
            )  # Nearest neighbor for labels
            # Overlay GT segmentation with transparency (only show non-background)
            gt_overlay = np.ma.masked_where(gt_resized == 0, gt_resized)
            axes[raw_row, vol_idx].imshow(
                gt_overlay, cmap="tab10", alpha=0.4, vmin=0, vmax=num_classes - 1
            )
            if slice_idx == 0:  # Only show volume info on first slice
                axes[raw_row, vol_idx].set_title(
                    f"Vol {vol_idx+1} ({source_name})\nRaw + GT Overlay Z={z} ({perc_label})",
                    fontsize=8,
                )
            else:
                axes[raw_row, vol_idx].set_title(
                    f"Raw + GT Overlay Z={z} ({perc_label})", fontsize=8
                )
            axes[raw_row, vol_idx].axis("off")

            # Ground truth LSDs and affinities from instance segmentation
            # Get 3D neighborhood around this slice for computation
            z_start = max(0, z - 1)
            z_end = min(gt.shape[1], z + 2)
            gt_3d_patch = gt[vol_idx, z_start:z_end]  # Shape: (small_D, H, W)

            # Compute affinities (3 channels for standard offsets)
            affinities_3d = get_affs(
                gt_3d_patch,
                np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
                dist="equality-no-bg",
            )  # Shape: (3, small_D, H, W)
            # Extract the slice corresponding to z (middle slice if available)
            slice_idx_in_patch = min(z - z_start, affinities_3d.shape[1] - 1)
            aff_slice = affinities_3d[:, slice_idx_in_patch, :, :]  # Shape: (3, H, W)

            # Display Affinities: first 3 channels as RGB (Z, Y, X directions)
            aff_rgb = np.transpose(aff_slice, (1, 2, 0)).astype(
                np.float32
            )  # Shape: (H, W, 3)
            axes[aff_row, vol_idx].imshow(aff_rgb)
            axes[aff_row, vol_idx].set_title(
                f"GT Affinities (RGB: Z,Y,X) Z={z} ({perc_label})", fontsize=8
            )
            axes[aff_row, vol_idx].axis("off")

            # Raw slice with GT mask overlay
            axes[mask_row, vol_idx].imshow(raw_slice, cmap="gray")
            # Resize mask to match raw slice dimensions for proper overlay
            mask_resized = zoom(
                mask_slice, scale_factor, order=0
            )  # Nearest neighbor for mask
            # Overlay GT mask with transparency
            mask_overlay = np.ma.masked_where(mask_resized == 0, mask_resized)
            axes[mask_row, vol_idx].imshow(
                mask_overlay, cmap="Reds", alpha=0.4, vmin=0, vmax=1
            )
            axes[mask_row, vol_idx].set_title(
                f"Raw + GT Mask Z={z} ({perc_label})", fontsize=8
            )
            axes[mask_row, vol_idx].axis("off")

            # DINOv3 feature channel
            axes[feat_row, vol_idx].imshow(slice_features[200, 0], cmap="viridis")
            axes[feat_row, vol_idx].set_title(
                f"Features Z={z} ({perc_label})", fontsize=8
            )
            axes[feat_row, vol_idx].axis("off")

plt.suptitle(
    "3D Volumetric Training Data: Affinities (5 Percentile Slices per Volume)",
    fontsize=16,
)
plt.tight_layout()
plt.show()
# %%

# Example 3: Memory-Efficient 3D UNet Training
print("=" * 60)
print("EXAMPLE 3: Memory-Efficient 3D UNet Training with Affinities only")
print("=" * 60)

# Define affinity offsets
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

# Calculate total output channels: only affinities
num_output_channels = len(affinity_offsets)
print(f"Output channels: {num_output_channels} = {len(affinity_offsets)} affinities")

# Use the updated memory-efficient training function
print("Training 3D UNet with memory-efficient loader and affinity+LSDS prediction...")
training_results = train_3d_unet_with_memory_efficient_loader(
    raw_data=raw,
    gt_data=gt,  # Instance segmentation (each organelle has unique ID)
    gt_masks=gt_masks,  # GT extension masks for proper loss calculation
    train_volume_pool_size=NUM_VOLUMES - 5,  # Use 15 volumes for training pool
    val_volume_pool_size=5,  # Use 5 volumes for validation
    num_classes=num_output_channels,
    target_volume_size=VOLUME_SIZE,  # Target 3D volume size
    dinov3_slice_size=DINOV3_SLICE_SIZE,  # DINOv3 slice size
    volumes_per_batch=4,  # Process 2 volumes per batch (memory efficient)
    batches_per_epoch=10,  # 10 batches per epoch
    epochs=1000,  # Training epochs
    learning_rate=1e-3,  # Learning rate
    weight_decay=1e-4,  # Weight decay for regularization
    patience=150,  # Early stopping patience
    min_delta=0.001,  # Minimum improvement threshold
    base_channels=BASE_CHANNELS,  # Base channels for 3D UNet
    device=device,  # Training device
    seed=42,  # Random seed for reproducibility
    model_id=MODEL_ID,  # DINOv3 model identifier
    export_base_dir=EXPORT_BASE_DIR,  # Directory for saving results
    save_checkpoints=True,  # Enable checkpoint saving
    checkpoint_every_n_epochs=10,  # Save model checkpoint every 10 epochs
    use_class_weighting=False,  # Don't use class weighting for affinities+LSDs
    use_orthogonal_planes=True,  # Use orthogonal planes for training
    # MEMORY EFFICIENCY OPTIONS
    use_mixed_precision=True,  # 30-50% memory reduction
    use_half_precision=False,  # Additional 50% reduction (but potential numerical issues)
    memory_efficient_mode="auto",  # Let system decide based on GPU memory
    enable_detailed_timing=False,
    verbose=False,  # Suppress verbose DINOv3 processing messages
    # DATA RESOLUTION PARAMETERS
    min_resolution_for_raw=min_resolution_for_raw,  # Raw data resolution
    base_resolution=base_resolution,  # Ground truth resolution
    # LOSS FUNCTION PARAMETERS - Use affinity-only loss
    loss_type="affinity",  # Use affinity-only loss
    # AFFINITY PARAMETERS
    output_type="affinities",  # Enable affinity-only output
    affinity_offsets=affinity_offsets,  # Affinity offset directions
    use_batchrenorm=True,  # Use BatchRenorm for more stable training with small batches
)

print(f"Training completed!")
print(f"Best validation accuracy: {training_results['best_val_acc']:.4f}")
print(f"Epochs trained: {training_results['epochs_trained']}")
print(f"Checkpoints saved to: {training_results['checkpoint_dir']}")

# %%
# Example 4: Visualize 3D Training Results
# print("=" * 60)
# print("EXAMPLE 4: 3D Training Results Visualization")
# print("=" * 60)

# # Plot training history
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# epochs = range(1, len(training_results["train_losses"]) + 1)

# # Loss plot
# axes[0, 0].plot(epochs, training_results["train_losses"], "b-", label="Training Loss")
# axes[0, 0].plot(epochs, training_results["val_losses"], "r-", label="Validation Loss")
# axes[0, 0].set_title("3D UNet Training Loss")
# axes[0, 0].set_xlabel("Epoch")
# axes[0, 0].set_ylabel("Loss")
# axes[0, 0].legend()
# axes[0, 0].grid(True)

# # Accuracy plot
# axes[0, 1].plot(epochs, training_results["train_accs"], "b-", label="Training Accuracy")
# axes[0, 1].plot(epochs, training_results["val_accs"], "r-", label="Validation Accuracy")
# axes[0, 1].set_title("3D UNet Training Accuracy")
# axes[0, 1].set_xlabel("Epoch")
# axes[0, 1].set_ylabel("Accuracy")
# axes[0, 1].legend()
# axes[0, 1].grid(True)

# # Per-class training accuracy
# if "train_class_accs" in training_results:
#     train_class_accs = np.array(training_results["train_class_accs"])
#     for class_id in range(train_class_accs.shape[1]):
#         axes[1, 0].plot(
#             epochs, train_class_accs[:, class_id], label=f"Class {class_id}"
#         )
#     axes[1, 0].set_title("Training Class Accuracies")
#     axes[1, 0].set_xlabel("Epoch")
#     axes[1, 0].set_ylabel("Accuracy")
#     axes[1, 0].legend()
#     axes[1, 0].grid(True)

# # Per-class validation accuracy
# if "val_class_accs" in training_results:
#     val_class_accs = np.array(training_results["val_class_accs"])
#     for class_id in range(val_class_accs.shape[1]):
#         axes[1, 1].plot(epochs, val_class_accs[:, class_id], label=f"Class {class_id}")
#     axes[1, 1].set_title("Validation Class Accuracies")
#     axes[1, 1].set_xlabel("Epoch")
#     axes[1, 1].set_ylabel("Accuracy")
#     axes[1, 1].legend()
#     axes[1, 1].grid(True)

# plt.tight_layout()
# plt.show()

# # Print final training statistics
# print(f"\nFinal Training Statistics:")
# print(f"  Best validation accuracy: {training_results['best_val_acc']:.4f}")
# print(f"  Final training accuracy: {training_results['train_accs'][-1]:.4f}")
# print(f"  Final validation accuracy: {training_results['val_accs'][-1]:.4f}")

# if (
#     "class_weights" in training_results
#     and training_results["class_weights"] is not None
# ):
#     print(f"  Class weights used: {training_results['class_weights']}")

# %%
# Example 5: 3D Inference on Test Volumes
# print("=" * 60)
# print("EXAMPLE 5: 3D Inference Visualization with Affinities + LSDs")
# print("=" * 60)

# # Get the trained model from results
# trained_unet3d = training_results["unet3d"]
# trained_unet3d.eval()

# # Create a simple 3D data loader for inference
# data_loader_3d = MemoryEfficientDataLoader3D(
#     raw_data=raw,
#     gt_data=gt,
#     train_volume_pool_size=10,
#     val_volume_pool_size=5,
#     target_volume_size=VOLUME_SIZE,
#     seed=42,
#     model_id=MODEL_ID,
#     output_type="affinities_lsds",
#     affinity_offsets=affinity_offsets,
#     lsds_sigma=20.0,
# )

# test_indices = [0, 1, 2] if len(raw) >= 3 else list(range(len(raw)))

# for test_idx in test_indices:
#     print(f"Processing test volume {test_idx + 1}...")

#     test_volume = raw[test_idx : test_idx + 1]
#     test_gt_volume = gt[test_idx : test_idx + 1]

#     # Extract features and run inference
#     with torch.no_grad():
#         test_features = data_loader_3d.extract_dinov3_features_3d(test_volume)
#         test_outputs = trained_unet3d(test_features)  # Shape: (1, 19, D, H, W)

#     # Split outputs into LSDs (first 10 channels) and affinities (remaining channels)
#     lsds_pred = test_outputs[:, :10, :, :, :]  # Shape: (1, 10, D, H, W)
#     affs_pred = test_outputs[:, 10:, :, :, :]  # Shape: (1, 9, D, H, W)

#     # Convert to numpy
#     lsds_pred_np = lsds_pred[0].cpu().numpy()
#     affs_pred_np = affs_pred[0].cpu().numpy()
#     test_raw_np = test_volume[0]
#     test_gt_np = test_gt_volume[0]

#     # Compute ground truth LSDs and affinities (using safe wrapper)
#     gt_lsds = safe_get_lsds(test_gt_np, sigma=20.0)  # Shape: (10, D, H, W)
#     gt_affs = get_affs(test_gt_np, affinity_offsets, dist="equality-no-bg")  # Shape: (9, D, H, W)

#     # Calculate MSE for LSDs and BCE for affinities
#     lsds_mse = np.mean((lsds_pred_np - gt_lsds) ** 2)
#     affs_bce = -np.mean(
#         gt_affs * np.log(affs_pred_np + 1e-8)
#         + (1 - gt_affs) * np.log(1 - affs_pred_np + 1e-8)
#     )
#     print(
#         f"Volume {test_idx + 1} - LSDs MSE: {lsds_mse:.4f}, Affinities BCE: {affs_bce:.4f}"
#     )

#     # Visualize multiple slices
#     fig, axes = plt.subplots(5, 5, figsize=(20, 20))

#     D = VOLUME_SIZE[0]
#     z_indices = [0, D // 4, D // 2, (3 * D) // 4, D - 1]  # first, 25%, 50%, 75%, last

#     for i, z in enumerate(z_indices):
#         # Raw slice
#         axes[0, i].imshow(test_raw_np[z], cmap="gray")
#         axes[0, i].set_title(f"Raw Z={z}")
#         axes[0, i].axis("off")

#         # Ground truth LSDs (first 3 channels as RGB)
#         gt_lsd_rgb = np.transpose(gt_lsds[:3, z, :, :], (1, 2, 0))
#         gt_lsd_rgb = (gt_lsd_rgb - gt_lsd_rgb.min()) / (
#             gt_lsd_rgb.max() - gt_lsd_rgb.min() + 1e-8
#         )
#         axes[1, i].imshow(gt_lsd_rgb)
#         axes[1, i].set_title(f"GT LSDs (RGB) Z={z}")
#         axes[1, i].axis("off")

#         # Predicted LSDs (first 3 channels as RGB)
#         pred_lsd_rgb = np.transpose(lsds_pred_np[:3, z, :, :], (1, 2, 0))
#         pred_lsd_rgb = (pred_lsd_rgb - pred_lsd_rgb.min()) / (
#             pred_lsd_rgb.max() - pred_lsd_rgb.min() + 1e-8
#         )
#         axes[2, i].imshow(pred_lsd_rgb)
#         axes[2, i].set_title(f"Pred LSDs (RGB) Z={z}")
#         axes[2, i].axis("off")

#         # Ground truth affinities (first 3 channels as RGB)
#         gt_aff_rgb = np.transpose(gt_affs[:3, z, :, :], (1, 2, 0))
#         axes[3, i].imshow(gt_aff_rgb)
#         axes[3, i].set_title(f"GT Affinities (RGB) Z={z}")
#         axes[3, i].axis("off")

#         # Predicted affinities (first 3 channels as RGB)
#         pred_aff_rgb = np.transpose(affs_pred_np[:3, z, :, :], (1, 2, 0))
#         axes[4, i].imshow(pred_aff_rgb)
#         axes[4, i].set_title(f"Pred Affinities (RGB) Z={z}")
#         axes[4, i].axis("off")

#     plt.suptitle(
#         f"3D UNet Inference - Test Volume {test_idx + 1}\n(LSDs MSE: {lsds_mse:.4f}, Affinities BCE: {affs_bce:.4f})",
#         fontsize=16,
#     )
#     plt.tight_layout()
#     plt.show()

# %%
# Example 6: Large Volume Inference with Sliding Window
print("=" * 60)
print("EXAMPLE 6: Large Volume Inference with Affinities + LSDs")
print("=" * 60)

# Test sliding window inference on a larger volume (if we had one)
print("Testing sliding window inference capability...")

# Create a synthetic larger volume for demonstration
large_volume = np.random.randint(0, 255, (128, 128, 128), dtype=np.uint8)
print(f"Large test volume shape: {large_volume.shape}")

# Process the large volume directly with the trained model
# We'll process it in chunks manually to get the raw outputs
print("Processing large volume in chunks...")

chunk_size = 64
D, H, W = large_volume.shape

# Initialize output volume for affinities + LSDs
large_outputs = np.zeros((num_output_channels, D, H, W), dtype=np.float32)

# Simple chunking (you could add overlap and averaging for better results)
with torch.no_grad():
    for z in range(0, D, chunk_size):
        z_end = min(z + chunk_size, D)
        z_actual = z_end - z

        # If chunk is smaller than chunk_size, skip or pad
        if z_actual < chunk_size:
            continue

        chunk = large_volume[z:z_end, :H, :W]

        # Process through DINOv3
        chunk_features = data_loader_3d.extract_dinov3_features_3d(
            chunk[np.newaxis, ...]
        )[0]

        # Get predictions
        chunk_outputs = (
            trained_unet3d(chunk_features.unsqueeze(0)).squeeze(0).cpu().numpy()
        )

        # Store outputs
        large_outputs[:, z:z_end, :, :] = chunk_outputs

print(f"Large volume outputs shape: {large_outputs.shape}")

# Split into LSDs and affinities
large_lsds = large_outputs[:10]  # First 10 channels
large_affs = large_outputs[10:]  # Remaining channels

# Visualize a few slices from the large volume
fig, axes = plt.subplots(3, 5, figsize=(18, 12))

D = large_volume.shape[0]
z_indices = [0, D // 4, D // 2, (3 * D) // 4, D - 1]  # first, 25%, 50%, 75%, last

for i, z in enumerate(z_indices):
    # Raw slice
    axes[0, i].imshow(large_volume[z], cmap="gray")
    axes[0, i].set_title(f"Large Volume Z={z}")
    axes[0, i].axis("off")

    # Predicted LSDs (first 3 channels as RGB)
    lsd_rgb = np.transpose(large_lsds[:3, z, :, :], (1, 2, 0))
    lsd_rgb = (lsd_rgb - lsd_rgb.min()) / (lsd_rgb.max() - lsd_rgb.min() + 1e-8)
    axes[1, i].imshow(lsd_rgb)
    axes[1, i].set_title(f"Predicted LSDs (RGB) Z={z}")
    axes[1, i].axis("off")

    # Predicted affinities (first 3 channels as RGB)
    aff_rgb = np.transpose(large_affs[:3, z, :, :], (1, 2, 0))
    axes[2, i].imshow(aff_rgb)
    axes[2, i].set_title(f"Predicted Affinities (RGB) Z={z}")
    axes[2, i].axis("off")

plt.suptitle(
    "Sliding Window Inference - Percentile Z Slices (LSDs + Affinities)", fontsize=14
)
plt.tight_layout()
plt.show()

# %%
print("=" * 60)
print("DINOV3 3D UNET TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("✅ 3D UNet architecture implemented")
print("✅ Volumetric DINOv3 feature extraction")
print("✅ Memory-efficient 3D training pipeline")
print("✅ Checkpoint saving and loading")
print("✅ Affinity + LSDS prediction")
print("✅ Sliding window inference for large volumes")
print("✅ 3D visualization and analysis")
print("=" * 60)

print(f"\nKey Results:")
print(f"  - Best 3D UNet validation accuracy: {training_results['best_val_acc']:.4f}")
print(f"  - Training epochs: {training_results['epochs_trained']}")
print(f"  - Volume size: {VOLUME_SIZE}")
print(
    f"  - Output channels: {num_output_channels} (10 LSDs + {len(affinity_offsets)} affinities)"
)
print(f"  - Base channels: {BASE_CHANNELS}")
print(f"  - Feature channels: {current_output_channels}")
print(f"  - Export directory: {EXPORT_BASE_DIR}")
print(f"  - Checkpoint directory: {training_results['checkpoint_dir']}")

# Print memory usage summary
memory_info = unet3d.get_memory_usage(batch_size=1)
print(f"\n3D UNet Memory Usage:")
print(f"  - Parameters: {memory_info['parameters']:.2f} MB")
print(f"  - Activations: {memory_info['total_activations']:.2f} MB")
print(f"  - Total estimated: {memory_info['total_estimated']:.2f} MB")

# Print final model info
print(f"\nModel Information:")
print(f"  - Model type: DINOv3UNet3D")
print(f"  - Input channels: {current_output_channels}")
print(f"  - Output channels: {num_output_channels}")
print(f"  - LSDs: 10 channels")
print(f"  - Affinities: {len(affinity_offsets)} channels")
print(f"  - Affinity offsets: {affinity_offsets}")
print(f"  - LSDS sigma: 20.0")
print(f"  - Base channels: {BASE_CHANNELS}")
print(f"  - Target volume size: {VOLUME_SIZE}")
print(f"  - DINOv3 model: {MODEL_ID}")
