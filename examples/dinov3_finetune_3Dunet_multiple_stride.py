"""
DINOv3 3D UNet Fine-tuning Pipeline

This file demonstrates tradinoVOLUME_SIZE = (128, 128, 128)  # Restored full size (memory issue was validation caching)
DINOV3_SLICE_SIZE = IMAGE_SIZE  # Size for processing each 2D slice
BASE_CHANNELS = 64  # Restored original sizestride = 8  # Restored to 8 for better resolution (memory issue was validation caching)ning a 3D UNet architecture that processes DINOv3 features
instead of raw images for volumetric segmentation tasks.

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
MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
# MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
dinov3_stride = 4  # Use stride of 8 for memory efficiency (was 8, causing OOM)
IMAGE_SIZE = 128

# 3D UNet specific configuration
VOLUME_SIZE = (
    128,
    128,
    128,
)  # Reduced depth for memory efficiency with stride=4examples/dinov3_finetune_3Dunet_multiple_stride.py
DINOV3_SLICE_SIZE = IMAGE_SIZE  # Size for processing each 2D slice (match DINOv3 init)
BASE_CHANNELS = 64  # Further reduced for memory efficiency
learn_upsampling = False
# Export directory configuration
EXPORT_BASE_DIR = (
    "/nrs/cellmap/ackermand/dinov3_training/results/multiple_3d_hybrid_stride_4"
)
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

base_resolution = 64

# Define your dataset pairs here - NEW MULTI-CLASS FORMAT
dataset_pairs = [
    # Example 1: Multi-class dataset with nuclei and mitochondria
    {
        "raw": "/nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/",
        "nuc": "/nrs/cellmap/zouinkhim/predictions/salivary/jrc_mus-salivary-1.zarr/postprocess/nuc_filled",
        "mito": "/nrs/cellmap/zouinkhim/predictions/salivary/jrc_mus-salivary-1.zarr/postprocess/mito_filled",
    },
    {
        "raw": "/nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/em/fibsem-int16/",
        "nuc": "/nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/labels/inference/segmentations/nuc",
        "mito": "/nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/labels/inference/segmentations/mito",
    },
    {
        "raw": "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8",
        "nuc": "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/labels/inference/segmentations/nuc",
        "mito": "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/labels/inference/segmentations/mito",
    },
    # {
    #     "raw": "/nrs/cellmap/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/em/fibsem-int16/",
    #     "nuc": "/nrs/cellmap/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/labels/inference/segmentations/nuc",
    #     "mito": "/nrs/cellmap/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/labels/inference/segmentations/mito",
    # },
]

# Legacy format (will be automatically converted)
# dataset_pairs = [
#     (
#         "/nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/",
#         "/nrs/cellmap/zouinkhim/predictions/salivary/jrc_mus-salivary-1.zarr/postprocess/nuc_filled",
#     ),
#     (
#         "/nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/em/fibsem-int16/",
#         "/nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/labels/inference/segmentations/nuc",
#     ),
# ]
dataset_pairs = update_datapaths_with_target_scales(dataset_pairs, base_resolution)


# Load 3D training data with proper volumetric sampling
print("Loading 3D volumetric training data...")
num_volumes = (
    100  # Increased back up (memory issue was validation caching, not data loading)
)
raw, gt, dataset_sources, num_classes = load_random_3d_training_data(
    dataset_pairs=dataset_pairs,
    volume_shape=VOLUME_SIZE,  # (128, 128, 128)
    base_resolution=base_resolution,
    min_label_fraction=0.1,
    num_volumes=num_volumes,  # Number of 3D volumes to sample
    seed=42,
    dinov3_stride=dinov3_stride,  # ✅ Enable ROI-level padding for sliding window inference!
)

print(f"Loaded 3D volumetric training data:")
print(f"  Raw data shape: {raw.shape}")  # Should be (20, 64, 64, 64)
print(f"  Ground truth shape: {gt.shape}")  # Should be (20, 64, 64, 64)
print(f"  Unique GT values: {np.unique(gt)}")
print(f"  Overall label fraction: {np.sum(gt) / gt.size:.3f}")
print(f"  Dataset sources: {dataset_sources}")

# Verify each volume has the correct shape
for i in range(min(5, len(raw))):
    vol_label_fraction = np.sum(gt[i]) / gt[i].size
    print(
        f"  Volume {i}: shape {raw[i].shape}, label fraction {vol_label_fraction:.3f}"
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
    15, 5, figsize=(20, 36)
)  # 15 rows (3 types × 5 slices), 5 columns (volumes)

print("Processing sample volumes through DINOv3...")

# Define the 5 percentile slices to show
D = raw.shape[1]  # Depth dimension (64)
z_percentiles = [0, D // 4, D // 2, (3 * D) // 4, D - 1]  # first, 25%, 50%, 75%, last
percentile_labels = ["0%", "25%", "50%", "75%", "100%"]

for vol_idx in range(5):  # 5 volumes (columns)
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
            raw_slice = raw[vol_idx, z]  # Shape: (64, 64)
            gt_slice = gt[vol_idx, z]  # Shape: (64, 64)

            # Process slice through DINOv3
            slice_features = process(
                raw_slice[np.newaxis, ...], stride=dinov3_stride
            )  # Add batch dim

            # Calculate row indices for the 3 image types
            raw_row = slice_idx * 3  # 0, 3, 6, 9, 12
            gt_row = slice_idx * 3 + 1  # 1, 4, 7, 10, 13
            feat_row = slice_idx * 3 + 2  # 2, 5, 8, 11, 14

            # Raw slice
            axes[raw_row, vol_idx].imshow(raw_slice, cmap="gray")
            if slice_idx == 0:  # Only show volume info on first slice
                axes[raw_row, vol_idx].set_title(
                    f"Vol {vol_idx+1} ({source_name})\nRaw Z={z} ({perc_label})",
                    fontsize=8,
                )
            else:
                axes[raw_row, vol_idx].set_title(
                    f"Raw Z={z} ({perc_label})", fontsize=8
                )
            axes[raw_row, vol_idx].axis("off")

            # Ground truth slice (with proper class visualization)
            axes[gt_row, vol_idx].imshow(
                gt_slice, cmap="tab10", vmin=0, vmax=num_classes - 1
            )
            axes[gt_row, vol_idx].set_title(f"GT Z={z} ({perc_label})", fontsize=8)
            axes[gt_row, vol_idx].axis("off")

            # DINOv3 feature channel
            axes[feat_row, vol_idx].imshow(slice_features[200, 0], cmap="viridis")
            axes[feat_row, vol_idx].set_title(
                f"Features Z={z} ({perc_label})", fontsize=8
            )
            axes[feat_row, vol_idx].axis("off")

plt.suptitle(
    "3D Volumetric Training Data (5 Percentile Slices per Volume)", fontsize=16
)
plt.tight_layout()
plt.show()

# %%
# Show volume structure for one example
if len(raw) > 0:
    print("Visualizing 3D volume structure...")

    # Show different Z-slices from the first volume
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    volume_idx = 0
    D = VOLUME_SIZE[0]
    z_indices = [0, D // 4, D // 2, (3 * D) // 4, D - 1]

    for i, z in enumerate(z_indices):
        # Raw slice at different depths
        axes[0, i].imshow(raw[volume_idx, z], cmap="gray")
        axes[0, i].set_title(f"Raw Z={z}")
        axes[0, i].axis("off")

        # GT slice at different depths
        axes[1, i].imshow(gt[volume_idx, z], cmap="tab10", vmin=0, vmax=num_classes - 1)
        axes[1, i].set_title(f"GT Z={z}")
        axes[1, i].axis("off")

    plt.suptitle(
        f"3D Volume Structure - Volume {volume_idx+1} (Shape: {raw[volume_idx].shape})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

# %%
# Example 1: 3D UNet Architecture Testing
print("=" * 60)
print("EXAMPLE 1: 3D UNet Architecture Testing")
print("=" * 60)

# Get current model info to use correct input channels
model_info = get_current_model_info()
current_output_channels = model_info["output_channels"]

print("Creating DINOv3 3D UNet model...")
unet3d = DINOv3UNet3D(
    input_channels=current_output_channels,
    num_classes=num_classes,
    base_channels=BASE_CHANNELS,
    input_size=VOLUME_SIZE,
).to(device)

# Print model summary
print_model_summary(unet3d, input_shape=(current_output_channels, *VOLUME_SIZE))

# Test forward pass
print("Testing 3D forward pass...")

# Check device status first
print(f"Target device: {device}")
print(f"Model device: {next(unet3d.parameters()).device}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Create test input and explicitly move to device
if torch.cuda.is_available() and device.type == "cuda":
    test_input = torch.randn(1, current_output_channels, *VOLUME_SIZE, device=device)
else:
    test_input = torch.randn(1, current_output_channels, *VOLUME_SIZE)
    unet3d = unet3d.cpu()  # Move model to CPU if CUDA not available
    device = torch.device("cpu")

print(f"Input device after creation: {test_input.device}")
print(f"Model device after setup: {next(unet3d.parameters()).device}")

try:
    with torch.no_grad():
        test_output = unet3d(test_input)

    print(f"✅ Forward pass successful!")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output contains logits for {test_output.shape[1]} classes")

except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    print("Trying with CPU fallback...")

    # Fallback: move everything to CPU
    unet3d = unet3d.cpu()
    test_input = test_input.cpu()

    with torch.no_grad():
        test_output = unet3d(test_input)

    print(f"✅ CPU forward pass successful!")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")

# Show memory usage
memory_info = unet3d.get_memory_usage(batch_size=1)
print(f"\nMemory usage estimates:")
for key, value in memory_info.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.2f} MB")

# %%
# Example 2: 3D UNet Pipeline Testing
# print("=" * 60)
# print("EXAMPLE 2: 3D UNet Pipeline Testing")
# print("=" * 60)

# print("Creating DINOv3 3D UNet Pipeline...")
# pipeline3d = DINOv3UNet3DPipeline(
#     num_classes=2,
#     input_size=VOLUME_SIZE,
#     dinov3_slice_size=DINOV3_SLICE_SIZE,
#     base_channels=BASE_CHANNELS,
#     device=device,
# ).to(device)

# # Test the complete pipeline
# print("Testing complete 3D pipeline...")
# test_volume = raw[0:1]  # Take first volume
# print(f"Test volume shape: {test_volume.shape}")

# # Run through pipeline
# with torch.no_grad():
#     pipeline_output = pipeline3d(test_volume)
#     pipeline_predictions = pipeline3d.predict(test_volume)

# print(f"Pipeline output shape: {pipeline_output.shape}")
# print(f"Pipeline predictions shape: {pipeline_predictions.shape}")

# # Visualize pipeline results
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# # Show percentile Z-slices
# D = VOLUME_SIZE[0]
# z_indices = [0, D // 2, D - 1]  # first, middle, last

# # Raw volume slice
# axes[0, 0].imshow(test_volume[0, z_indices[0]], cmap="gray")
# axes[0, 0].set_title(f"Raw Z={z_indices[0]}")
# axes[0, 0].axis("off")

# axes[0, 1].imshow(test_volume[0, z_indices[1]], cmap="gray")
# axes[0, 1].set_title(f"Raw Z={z_indices[1]}")
# axes[0, 1].axis("off")

# axes[0, 2].imshow(test_volume[0, z_indices[2]], cmap="gray")
# axes[0, 2].set_title(f"Raw Z={z_indices[2]}")
# axes[0, 2].axis("off")

# # Predictions at the same slices
# for i, z in enumerate(z_indices):
#     axes[1, i].imshow(pipeline_predictions[0, z].cpu().numpy(), cmap="tab10")
#     axes[1, i].set_title(f"Prediction Z={z}")
#     axes[1, i].axis("off")

# plt.suptitle("3D UNet Pipeline Results (First/Middle/Last)", fontsize=14)
# plt.tight_layout()
# plt.show()

# %%
# Example 3: Memory-Efficient 3D UNet Training
print("=" * 60)
print("EXAMPLE 3: Memory-Efficient 3D UNet Training")
print("=" * 60)

# Use the updated memory-efficient training function
print("Training 3D UNet with memory-efficient loader...")
training_results = train_3d_unet_with_memory_efficient_loader(
    raw_data=raw,
    gt_data=gt,
    train_volume_pool_size=num_volumes - 5,  # Use all but 20 volumes for training
    val_volume_pool_size=5,  # Use 20 volumes for validation
    num_classes=num_classes,  # Auto-detected from data
    target_volume_size=VOLUME_SIZE,  # Target 3D volume size
    dinov3_slice_size=DINOV3_SLICE_SIZE,  # DINOv3 slice size
    volumes_per_batch=2,  # Restored to 2 volumes per batch
    batches_per_epoch=10,  # Restored to 10 batches per epoch
    epochs=500,  # Training epochs
    learning_rate=1e-3,  # Learning rate
    weight_decay=1e-4,  # Weight decay for regularization
    patience=15,  # Early stopping patience
    min_delta=0.001,  # Minimum improvement threshold
    base_channels=BASE_CHANNELS,  # Base channels for 3D UNet
    device=device,  # Training device
    seed=42,  # Random seed for reproducibility
    model_id=MODEL_ID,  # DINOv3 model identifier
    export_base_dir=EXPORT_BASE_DIR,  # Directory for saving results
    save_checkpoints=True,  # Enable checkpoint saving
    use_class_weighting=True,  # Use class-balanced loss
    # NEW MEMORY EFFICIENCY OPTIONS
    use_mixed_precision=True,  # 30-50% memory reduction
    use_half_precision=False,  # Additional 50% reduction (but potential numerical issues)
    memory_efficient_mode="auto",  # Let system decide based on GPU memory
    learn_upsampling=learn_upsampling,  # Use learned upsampling layers
    dinov3_stride=dinov3_stride,  # NEW: Sliding window inference (2x resolution, 4x slower)
)

print(f"Training completed!")
print(f"Best mean IoU: {training_results['best_mean_iou']:.4f}")
print(f"Epochs trained: {training_results['epochs_trained']}")
print(f"Checkpoints saved to: {training_results['checkpoint_dir']}")

# %%
# Example 4: Visualize 3D Training Results
print("=" * 60)
print("EXAMPLE 4: 3D Training Results Visualization")
print("=" * 60)

# Plot comprehensive training history
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

epochs = range(1, len(training_results["train_losses"]) + 1)

# Loss plot
axes[0, 0].plot(epochs, training_results["train_losses"], "b-", label="Training Loss")
axes[0, 0].plot(epochs, training_results["val_losses"], "r-", label="Validation Loss")
axes[0, 0].set_title("3D UNet Training Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy plot
axes[0, 1].plot(epochs, training_results["train_accs"], "b-", label="Training Accuracy")
axes[0, 1].plot(epochs, training_results["val_accs"], "r-", label="Validation Accuracy")
axes[0, 1].set_title("3D UNet Training Accuracy")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Per-class training accuracy
if "train_class_accs" in training_results:
    train_class_accs = np.array(training_results["train_class_accs"])
    for class_id in range(train_class_accs.shape[1]):
        axes[1, 0].plot(
            epochs, train_class_accs[:, class_id], label=f"Class {class_id}"
        )
    axes[1, 0].set_title("Training Class Accuracies")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

# Per-class validation recall
if "val_class_accs" in training_results:
    val_class_accs = np.array(training_results["val_class_accs"])
    for class_id in range(val_class_accs.shape[1]):
        axes[1, 1].plot(epochs, val_class_accs[:, class_id], label=f"Class {class_id}")
    axes[1, 1].set_title("Validation Class Recall")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

# Per-class validation precision
if "val_class_precisions" in training_results:
    val_class_precisions = np.array(training_results["val_class_precisions"])
    for class_id in range(val_class_precisions.shape[1]):
        axes[2, 0].plot(
            epochs, val_class_precisions[:, class_id], label=f"Class {class_id}"
        )
    axes[2, 0].set_title("Validation Class Precision")
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("Precision")
    axes[2, 0].legend()
    axes[2, 0].grid(True)

# Per-class validation F1-Score and IoU
if "val_class_f1s" in training_results and "val_class_ious" in training_results:
    val_class_f1s = np.array(training_results["val_class_f1s"])
    val_class_ious = np.array(training_results["val_class_ious"])

    # F1 scores (solid lines)
    for class_id in range(val_class_f1s.shape[1]):
        axes[2, 1].plot(
            epochs, val_class_f1s[:, class_id], "-", label=f"F1 Class {class_id}"
        )

    # IoU scores (dashed lines)
    for class_id in range(val_class_ious.shape[1]):
        axes[2, 1].plot(
            epochs, val_class_ious[:, class_id], "--", label=f"IoU Class {class_id}"
        )

    axes[2, 1].set_title("Validation F1-Score & IoU")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Score")
    axes[2, 1].legend()
    axes[2, 1].grid(True)

plt.tight_layout()
plt.show()

# Print final training statistics
print(f"\nFinal Training Statistics:")
print(f"  Best mean IoU: {training_results['best_mean_iou']:.4f}")
print(f"  Final training accuracy: {training_results['train_accs'][-1]:.4f}")
print(f"  Final validation accuracy: {training_results['val_accs'][-1]:.4f}")

if (
    "class_weights" in training_results
    and training_results["class_weights"] is not None
):
    print(f"  Class weights used: {training_results['class_weights']}")

# %%
# Example 5: 3D Inference on Test Volumes
print("=" * 60)
print("EXAMPLE 5: 3D Inference Visualization")
print("=" * 60)

# Get the trained model from results
trained_unet3d = training_results["unet3d"]
trained_unet3d.eval()

# Create a simple 3D data loader for inference
data_loader_3d = MemoryEfficientDataLoader3D(
    raw_data=raw,
    gt_data=gt,
    train_volume_pool_size=10,
    val_volume_pool_size=5,
    target_volume_size=VOLUME_SIZE,
    dinov3_slice_size=DINOV3_SLICE_SIZE,  # Add slice size for consistency
    dinov3_stride=dinov3_stride,  # ✅ Enable ROI-level padding optimization!
    seed=42,
    model_id=MODEL_ID,
    learn_upsampling=learn_upsampling,  # Match training configuration
)

test_indices = [0, 1, 2] if len(raw) >= 3 else list(range(len(raw)))

for test_idx in test_indices:
    print(f"Processing test volume {test_idx + 1}...")

    test_volume = raw[test_idx : test_idx + 1]
    test_gt_volume = gt[test_idx : test_idx + 1]

    # Extract features and run inference
    with torch.no_grad():
        test_features = data_loader_3d.extract_dinov3_features_3d(test_volume)
        test_logits = trained_unet3d(test_features)
        test_probabilities = torch.softmax(test_logits, dim=1)
        test_predictions = torch.argmax(test_logits, dim=1)

    # Convert to numpy
    pred_np = test_predictions[0].cpu().numpy()
    prob_np = test_probabilities[0].cpu().numpy()
    test_raw_np = test_volume[0]
    test_gt_np = test_gt_volume[0]

    # Calculate volume-wise accuracy
    volume_acc = (pred_np == test_gt_np).mean()
    print(f"Volume {test_idx + 1} accuracy: {volume_acc:.4f}")

    # Visualize multiple slices
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    D = VOLUME_SIZE[0]
    z_indices = [0, D // 4, D // 2, (3 * D) // 4, D - 1]  # first, 25%, 50%, 75%, last

    for i, z in enumerate(z_indices):
        # Raw slice
        axes[0, i].imshow(test_raw_np[z], cmap="gray")
        axes[0, i].set_title(f"Raw Z={z}")
        axes[0, i].axis("off")

        # Ground truth slice
        axes[1, i].imshow(test_gt_np[z], cmap="tab10", vmin=0, vmax=num_classes - 1)
        axes[1, i].set_title(f"Ground Truth Z={z}")
        axes[1, i].axis("off")

        # Prediction slice
        axes[2, i].imshow(pred_np[z], cmap="tab10", vmin=0, vmax=num_classes - 1)
        axes[2, i].set_title(f"Predictions Z={z}")
        axes[2, i].axis("off")

        # Probability slice (show highest non-background class if available)
        prob_class_idx = min(
            1, num_classes - 1
        )  # Use class 1 or highest available class
        im = axes[3, i].imshow(
            prob_np[prob_class_idx, z], cmap="viridis", vmin=0, vmax=1
        )
        axes[3, i].set_title(f"Class {prob_class_idx} Prob Z={z}")
        axes[3, i].axis("off")

    # Add colorbar for probabilities
    plt.colorbar(im, ax=axes[3, :], orientation="horizontal", pad=0.1, shrink=0.8)

    plt.suptitle(
        f"3D UNet Inference - Test Volume {test_idx + 1} (Acc: {volume_acc:.3f})",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()

# %%
# Example 6: Large Volume Inference with Sliding Window
print("=" * 60)
print("EXAMPLE 6: Large Volume Inference")
print("=" * 60)

# Test sliding window inference on a larger volume (if we had one)
print("Testing sliding window inference capability...")

# Create a synthetic larger volume for demonstration
large_volume = np.random.randint(0, 255, (128, 128, 128), dtype=np.uint8)
print(f"Large test volume shape: {large_volume.shape}")

# Create a simple pipeline for sliding window inference
temp_pipeline = DINOv3UNet3DPipeline(
    num_classes=num_classes,
    input_size=VOLUME_SIZE,
    dinov3_slice_size=DINOV3_SLICE_SIZE,
    base_channels=BASE_CHANNELS,
    device=device,
)

# Replace the UNet with our trained one
temp_pipeline.unet3d = trained_unet3d

# Use sliding window inference
print("Running sliding window inference...")
large_predictions = temp_pipeline.predict_large_volume(
    large_volume, chunk_size=64, overlap=16
)

print(f"Large volume predictions shape: {large_predictions.shape}")

# Visualize a few slices from the large volume
fig, axes = plt.subplots(2, 5, figsize=(18, 8))

D = large_volume.shape[0]
z_indices = [0, D // 4, D // 2, (3 * D) // 4, D - 1]  # first, 25%, 50%, 75%, last

for i, z in enumerate(z_indices):
    # Raw slice
    axes[0, i].imshow(large_volume[z], cmap="gray")
    axes[0, i].set_title(f"Large Volume Z={z}")
    axes[0, i].axis("off")

    # Prediction slice
    axes[1, i].imshow(large_predictions[z], cmap="tab10", vmin=0, vmax=num_classes - 1)
    axes[1, i].set_title(f"Predictions Z={z}")
    axes[1, i].axis("off")

plt.suptitle("Sliding Window Inference - Percentile Z Slices", fontsize=14)
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
print("✅ Class-balanced training with weights")
print("✅ Sliding window inference for large volumes")
print("✅ 3D visualization and analysis")
print("=" * 60)

print(f"\nKey Results:")
print(f"  - Best 3D UNet mean IoU: {training_results['best_mean_iou']:.4f}")
print(f"  - Training epochs: {training_results['epochs_trained']}")
print(f"  - Volume size: {VOLUME_SIZE}")
print(f"  - Number of classes: {num_classes}")
print(f"  - Classes in final data: {np.unique(gt)}")
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
print(f"  - Output classes: {num_classes}")
print(f"  - Base channels: {BASE_CHANNELS}")
print(f"  - Target volume size: {VOLUME_SIZE}")
print(f"  - DINOv3 model: {MODEL_ID}")

# Print class distribution summary
print(f"\nClass Distribution Summary:")
unique_classes, class_counts = np.unique(gt, return_counts=True)
total_voxels = gt.size
for cls, count in zip(unique_classes, class_counts):
    percentage = (count / total_voxels) * 100
    print(f"  - Class {cls}: {count:,} voxels ({percentage:.2f}%)")
