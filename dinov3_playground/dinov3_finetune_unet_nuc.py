"""
DINOv3 UNet Fine-tuning Pipeline

This file demonstrates training a UNet architecture that processes DINOv3 features
instead of raw images for pixel-level segmentation tasks.

Author: GitHub Copilot
Date: 2025-09-13
"""
# %%
# Add current directory to Python path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from funlib.geometry import Coordinate, Roi
from cellmap_flow.image_data_interface import ImageDataInterface

# Import all modularized functions
from dinov3_playground import dinov3_core, data_processing, models, model_training, visualization, memory_efficient_training
from importlib import reload
reload(dinov3_core)
reload(data_processing)
reload(models)
reload(model_training)
reload(visualization)
reload(memory_efficient_training)

from dinov3_playground.memory_efficient_training import (
    MemoryEfficientDataLoader,
    train_unet_memory_efficient,
    train_with_unet_memory_efficient_loader,
    load_checkpoint,
    list_checkpoints
)

from dinov3_playground.dinov3_core import (
    enable_amp_inference,
    process,
    output_channels
)

from dinov3_playground.data_processing import (
    load_random_training_data,
    get_example_dataset_pairs,
    sample_training_data
)

from dinov3_playground.models import (
    DINOv3UNet,
    create_model,
    print_model_summary
)

from dinov3_playground.visualization import (
    plot_training_history
)

from dinov3_playground import (
    initialize_dinov3, get_current_model_info,  # Add these imports
)

# %%
# Configuration
print("=" * 60)
print("DINOV3 UNET TRAINING PIPELINE")
print("=" * 60)

# Model configuration
MODEL_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
IMAGE_SIZE = 896

# Export directory configuration
EXPORT_BASE_DIR = "/nrs/cellmap/ackermand/dinov3_training/results"
print(f"Export base directory: {EXPORT_BASE_DIR}")

# Create export directory if it doesn't exist
os.makedirs(EXPORT_BASE_DIR, exist_ok=True)

# Initialize DINOv3 with our specific model and size
processor, model, output_channels = initialize_dinov3(
    model_id=MODEL_ID,
    image_size=IMAGE_SIZE
)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get model info
model_info = get_current_model_info()
print(f"DINOv3 Model ID: {model_info['model_id']}")
print(f"Output channels: {model_info['output_channels']}")
print(f"Model device: {model_info['device']}")

print("Loading datasets...")
# %%
# Load data



# Define your dataset pairs here
dataset_pairs = [
    # Add your actual dataset pairs
    ("/nrs/cellmap/data/jrc_22ak351-leaf-2lb/jrc_22ak351-leaf-2lb.zarr/recon-1/em/fibsem-uint8/s5",
     "/groups/cellmap/cellmap/parkg/for Aubrey/2lb_s3.zarr/jrc_22ak351-leaf-2lb_nuc/s0"),
     ("/nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/em/fibsem-int16/s5",
    "/nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/labels/inference/segmentations/nuc/s3"),

     ("/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8/s4",
     "/nrs/cellmap/ackermand/cellmap/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/nuc/s0"),

    ("/nrs/cellmap/data/jrc_mus-heart-1/jrc_mus-heart-1.zarr/recon-1/em/fibsem-uint8/s4",
     "/nrs/cellmap/data/jrc_mus-heart-1/jrc_mus-heart-1.zarr/recon-1/labels/inference/segmentations/nuc/s0"),
    ("/nrs/cellmap/data/jrc_mus-kidney/jrc_mus-kidney.zarr/recon-1/em/fibsem-uint8/s4",
     "/nrs/cellmap/data/jrc_mus-kidney/jrc_mus-kidney.zarr/recon-1/labels/inference/segmentations/nuc/s0"),
      # Add more datasets here:
    # (
    #     "/path/to/dataset2/raw.zarr",
    #     "/path/to/dataset2/gt.zarr"
    # ),
    # (
    #     "/path/to/dataset3/raw.zarr", 
    #     "/path/to/dataset3/gt.zarr"
    # ),
]

# Load random training data from multiple datasets
# try:
raw, gt, dataset_sources = load_random_training_data(
    dataset_pairs=dataset_pairs,
    crop_shape=(250, 224, 224),  # 250 slices in Z, adjust as needed
    base_resolution=128,
    min_label_fraction=0.05,
    seed=42,
    random_orientations=True  # Enable random orientations
)

print(f"Loaded training data:")
print(f"  Raw data shape: {raw.shape}")
print(f"  Ground truth shape: {gt.shape}")
print(f"  Unique GT values: {np.unique(gt)}")
print(f"  Label fraction: {np.sum(gt) / gt.size:.3f}")
print(f"  Dataset sources: {dataset_sources}")

# Create dataset name mapping for plotting
dataset_names = []
for i, (raw_path, gt_path) in enumerate(dataset_pairs):
    # Extract a short name from the path
    if "jrc_mus-heart-1" in raw_path:
        dataset_names.append("Heart-1")
    elif "jrc_mus-kidney" in raw_path:
        dataset_names.append("Kidney")
    elif "jrc_mus-liver-zon-1" in raw_path:
        dataset_names.append("Liver")
    else:
        dataset_names.append(f"Dataset-{i+1}")

print(f"  Dataset names: {dataset_names}")

# except Exception as e:
#     print(f"Error loading random training data: {e}")
#     print("Falling back to single dataset...")
    
#     # Fallback to original single dataset approach
#     raw = ImageDataInterface("/nrs/cellmap/data/jrc_22ak351-leaf-3mb/jrc_22ak351-leaf-3mb.zarr/recon-1/em/fibsem-uint8/s4").to_ndarray_ts()
#     gt = ImageDataInterface("/groups/cellmap/cellmap/parkg/for Aubrey/3mb_s3.zarr/jrc_22ak351-leaf-3mb_nuc/s1").to_ndarray_ts() > 0
    
#     print(f"Raw data shape: {raw.shape}")
#     print(f"Ground truth shape: {gt.shape}")
#     print(f"Unique GT values: {np.unique(gt)}")
#     dataset_sources = [0] * raw.shape[0]  # All from dataset 0
#     dataset_names = ["Leaf-3mb"]

fig, axes = plt.subplots(3, 10, figsize=(16, 12))

print("Processing through DINOv3...")
features = process(raw)

for i in range(10):
    # Get dataset source for this slice
    source_idx = dataset_sources[i] if i < len(dataset_sources) else 0
    source_name = dataset_names[source_idx] if source_idx < len(dataset_names) else f"Dataset-{source_idx}"
    
    # Raw image
    axes[0, i].imshow(raw[i], cmap='gray')
    axes[0, i].set_title(f'Raw {i+1}\n({source_name})', fontsize=8)
    axes[0, i].axis('off')
    
    # Ground truth
    axes[1, i].imshow(gt[i], cmap='tab10')
    axes[1, i].set_title(f'GT {i+1}\n({source_name})', fontsize=8)
    axes[1, i].axis('off')
    
    # First DINOv3 feature channel
    axes[2, i].imshow(features[200, i], cmap='viridis')
    axes[2, i].set_title(f'Features {i+1}\n({source_name})', fontsize=8)
    axes[2, i].axis('off')

plt.suptitle('Multi-Dataset Training Data with Random Orientations', fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Example 1: UNet Architecture Testing
print("=" * 60)
print("EXAMPLE 1: UNet Architecture Testing")
print("=" * 60)

# Get current model info to use correct input channels
model_info = get_current_model_info()
current_output_channels = model_info['output_channels']

print("Creating DINOv3 UNet model...")
unet = DINOv3UNet(
    input_channels=current_output_channels,  # Use current model's output channels
    num_classes=2, 
    base_channels=64
).to(device)

# Print model summary
print_model_summary(unet, input_shape=(current_output_channels, 224, 224))

# Test forward pass
print("Testing forward pass...")
test_input = torch.randn(2, current_output_channels, 224, 224).to(device)
with torch.no_grad():
    test_output = unet(test_input)

print(f"Input shape: {test_input.shape}")
print(f"Output shape: {test_output.shape}")
print(f"Output contains logits for {test_output.shape[1]} classes")


# %%
# # Example 2: Sample and visualize data processing
# print("=" * 60)
# print("EXAMPLE 2: Data Processing Visualization")
# print("=" * 60)

# # Sample training data
# print("Sampling training data...")
# sample_images, sample_gt = sample_training_data(
#     raw, gt, 
#     target_size=224, 
#     num_samples=4, 
#     method="flexible",
#     seed=42,
#     use_augmentation=False
# )

# print(f"Sample images shape: {sample_images.shape}")
# print(f"Sample GT shape: {sample_gt.shape}")

# # Process through DINOv3 to get features
# print("Processing through DINOv3...")
# sample_features = process(sample_images)
# print(f"DINOv3 features shape: {sample_features.shape}")

# # Visualize the data pipeline
# print("Creating visualization...")
# fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# for i in range(4):
#     # Raw image
#     axes[0, i].imshow(sample_images[i], cmap='gray')
#     axes[0, i].set_title(f'Raw Image {i+1}')
#     axes[0, i].axis('off')
    
#     # Ground truth
#     axes[1, i].imshow(sample_gt[i], cmap='tab10')
#     axes[1, i].set_title(f'Ground Truth {i+1}')
#     axes[1, i].axis('off')
    
#     # First DINOv3 feature channel
#     axes[2, i].imshow(sample_features[0, i], cmap='viridis')
#     axes[2, i].set_title(f'DINOv3 Features {i+1}')
#     axes[2, i].axis('off')

# plt.tight_layout()
# plt.show()

# %%
# Example 3: Memory-efficient UNet training
print("=" * 60)
print("EXAMPLE 3: Memory-Efficient UNet Training")
print("=" * 60)

# Train UNet using memory-efficient data loader
print("Starting UNet training with memory-efficient loader...")
unet_results = train_with_unet_memory_efficient_loader(
    raw_data=raw, 
    gt_data=gt,
    train_pool_size=30,     # Reduced for UNet (more memory intensive)
    val_pool_size=8,        # Reduced for UNet
    images_per_batch=2,     # Reduced batch size for UNet
    batches_per_epoch=15,   # More batches to compensate for smaller batch size
    num_classes=2,
    epochs=1000,             # Fewer epochs for demo
    target_size=224,
    base_channels=64,       # Base channels for UNet
    device=device,
    model_id=MODEL_ID,      # Pass the model ID
    export_base_dir=EXPORT_BASE_DIR  # Pass the export directory
)

print(f"UNet training completed!")
print(f"Best validation accuracy: {unet_results['best_val_acc']:.4f}")

# Plot training history
print("Plotting training history...")
plot_training_history(unet_results)

# %%
# Example 4: UNet inference and visualization
print("=" * 60)
print("EXAMPLE 4: UNet Inference and Visualization")
print("=" * 60)

# Sample some test images
print("Sampling test images...")
test_images, test_gt = sample_training_data(
    raw, gt, 
    target_size=224, 
    num_samples=3, 
    method="flexible",
    seed=999,
    use_augmentation=False
)

# Create data loader to extract features
data_loader = MemoryEfficientDataLoader(
    raw_data=raw,
    gt_data=gt,
    train_image_pool_size=10,
    val_image_pool_size=5,
    target_size=224,
    seed=42
)

# Extract features for inference
print("Extracting DINOv3 features for inference...")
test_features = data_loader.extract_dinov3_features_for_unet(test_images)
test_labels = torch.tensor(test_gt, dtype=torch.long).to(device)

print(f"Test features shape: {test_features.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Run inference
print("Running UNet inference...")
trained_unet = unet_results['unet']
trained_unet.eval()

with torch.no_grad():
    pred_logits = trained_unet(test_features)
    pred_probs = torch.softmax(pred_logits, dim=1)
    predictions = torch.argmax(pred_logits, dim=1)
    confidence = torch.max(pred_probs, dim=1)[0]

# Convert to numpy for visualization
predictions_np = predictions.cpu().numpy()
confidence_np = confidence.cpu().numpy()
test_gt_np = test_gt

print(f"Predictions shape: {predictions_np.shape}")
print(f"Confidence shape: {confidence_np.shape}")

# Calculate accuracy
correct_pixels = (predictions_np == test_gt_np).sum()
total_pixels = test_gt_np.size
accuracy = correct_pixels / total_pixels
print(f"Test accuracy: {accuracy:.4f}")

# Visualize results
print("Creating inference visualization...")
fig, axes = plt.subplots(3, 5, figsize=(20, 12))

for i in range(3):
    # Raw image
    axes[i, 0].imshow(test_images[i], cmap='gray')
    axes[i, 0].set_title(f'Raw Image {i+1}')
    axes[i, 0].axis('off')
    
    # Ground truth
    axes[i, 1].imshow(test_gt_np[i], cmap='tab10', vmin=0, vmax=1)
    axes[i, 1].set_title(f'Ground Truth')
    axes[i, 1].axis('off')
    
    # Predictions
    axes[i, 2].imshow(predictions_np[i], cmap='tab10', vmin=0, vmax=1)
    axes[i, 2].set_title(f'UNet Predictions')
    axes[i, 2].axis('off')
    
    # Confidence
    im_conf = axes[i, 3].imshow(confidence_np[i], cmap='viridis', vmin=0, vmax=1)
    axes[i, 3].set_title(f'Confidence')
    axes[i, 3].axis('off')
    
    # Overlay: Class 1 predictions on raw image
    class1_mask = (predictions_np[i] == 1).astype(float)
    axes[i, 4].imshow(test_images[i], cmap='gray')
    overlay = np.zeros((*class1_mask.shape, 4))  # RGBA
    overlay[class1_mask == 1] = [1, 0, 0, 0.5]  # Red with 50% transparency
    axes[i, 4].imshow(overlay)
    axes[i, 4].set_title(f'Class 1 Overlay')
    axes[i, 4].axis('off')

# Add colorbar for confidence
plt.colorbar(im_conf, ax=axes[:, 3], orientation='horizontal', pad=0.1, shrink=0.8, label='Confidence Score')

plt.tight_layout()
plt.show()

# %%
# Example 5: Compare UNet with MLP classifier
print("=" * 60)
print("EXAMPLE 5: Architecture Comparison")
print("=" * 60)

# Show model size comparison
print("Model Architecture Comparison:")
print("-" * 40)

# UNet
unet_params = sum(p.numel() for p in trained_unet.parameters())
unet_size_mb = unet_params * 4 / 1024 / 1024

print(f"DINOv3 UNet:")
print(f"  Parameters: {unet_params:,}")
print(f"  Size: {unet_size_mb:.2f} MB")
print(f"  Input: Feature maps (B, {output_channels}, H, W)")
print(f"  Output: Segmentation maps (B, {2}, H, W)")
print(f"  Architecture: Encoder-Decoder with skip connections")

# Compare with MLP classifier (for reference)
from dinov3_playground.models import ImprovedClassifier
mlp_classifier = ImprovedClassifier(input_dim=output_channels, num_classes=2)
mlp_params = sum(p.numel() for p in mlp_classifier.parameters())
mlp_size_mb = mlp_params * 4 / 1024 / 1024

print(f"\nImproved MLP Classifier (for comparison):")
print(f"  Parameters: {mlp_params:,}")
print(f"  Size: {mlp_size_mb:.2f} MB")
print(f"  Input: Flattened features (B, {output_channels})")
print(f"  Output: Class logits (B, {2})")
print(f"  Architecture: Multi-layer perceptron with regularization")

print(f"\nUNet is {unet_params/mlp_params:.1f}x larger than MLP but preserves spatial structure")

# %%
# Example 6: Load test data and run inference
print("=" * 60)
print("EXAMPLE 6: Inference on New Test Data")
print("=" * 60)

# First, check if we have a trained model from previous examples, otherwise load from checkpoint
if 'unet_results' in locals() and 'data_loader' in locals():
    print("Using trained UNet from previous examples...")
    trained_unet = unet_results['unet']
    # Get checkpoint directory for saving figures
    checkpoint_dir = unet_results.get('checkpoint_dir', None)
    # data_loader already exists
else:
    print("Loading UNet from checkpoint...")
    
    # Get current model info to construct the correct checkpoint path
    model_info = get_current_model_info()
    current_model_id = model_info['model_id']
    current_output_channels = model_info['output_channels']
    
    # Clean model ID for filesystem compatibility
    model_id_clean = current_model_id.replace("/", "_").replace("-", "_")
    model_type_with_id = f"dinov3_unet_{model_id_clean}"
    
    print(f"Looking for checkpoints with model type: {model_type_with_id}")
    
    # List available UNet checkpoints for the current model
    checkpoints = list_checkpoints(
        base_results_dir=EXPORT_BASE_DIR,  # Use the configured export directory
        model_type=model_type_with_id
    )
    
    if not checkpoints:
        print(f"No UNet checkpoints found for model {current_model_id}.")
        print("Available checkpoint directories:")
        if os.path.exists(EXPORT_BASE_DIR):
            for item in os.listdir(EXPORT_BASE_DIR):
                if "dinov3_unet" in item:
                    print(f"  - {item}")
        print("Please run Example 3 first to train a UNet.")
        print("Skipping Example 6...")
        checkpoint_dir = None
    else:
        # Find the best checkpoint
        best_checkpoint = None
        for checkpoint in checkpoints:
            if 'best.pkl' in checkpoint:
                best_checkpoint = checkpoint
                break
        
        if best_checkpoint is None:
            best_checkpoint = checkpoints[0]  # Use latest if no best found
            
        print(f"Loading checkpoint: {os.path.basename(best_checkpoint)}")
        print(f"Full path: {best_checkpoint}")
        
        # Get checkpoint directory for saving figures
        checkpoint_dir = os.path.dirname(best_checkpoint)
        
        # Load checkpoint
        checkpoint_data = load_checkpoint(best_checkpoint, device)
        config = checkpoint_data['training_config']
        
        # Get input channels from checkpoint config (preferred) or current model
        if 'input_channels' in config:
            input_channels = config['input_channels']
            print(f"Using input channels from checkpoint: {input_channels}")
        else:
            input_channels = current_output_channels
            print(f"Using input channels from current model: {input_channels}")
        
        # Recreate UNet with correct architecture
        trained_unet = DINOv3UNet(
            input_channels=input_channels,
            num_classes=config['num_classes'],
            base_channels=config['base_channels']
        ).to(device)
        
        trained_unet.load_state_dict(checkpoint_data['unet_state_dict'])
        trained_unet.eval()
        
        print(f"UNet loaded successfully!")
        print(f"  Epoch: {checkpoint_data['epoch']}")
        print(f"  Best validation accuracy: {checkpoint_data['best_val_acc']:.4f}")
        print(f"  Model architecture:")
        print(f"    - Input channels: {input_channels}")
        print(f"    - Output classes: {config['num_classes']}")
        print(f"    - Base channels: {config['base_channels']}")
        print(f"  Training was done with model: {config.get('model_id', 'Unknown')}")
        
        # Verify model compatibility
        if config.get('model_id') != current_model_id:
            print(f"WARNING: Checkpoint was trained with {config.get('model_id', 'Unknown')}")
            print(f"         but current model is {current_model_id}")
            print(f"         This may cause issues if the models have different output dimensions!")
        
        # Create data loader for inference (we need this for feature extraction)
        print("Creating data loader for inference...")
        data_loader = MemoryEfficientDataLoader(
            raw_data=raw if 'raw' in locals() else np.zeros((1,224,224), dtype=np.uint8),
            gt_data=gt if 'gt' in locals() else np.zeros((1,224,224), dtype=np.uint8),
            train_image_pool_size=10,  # Small pool for inference
            val_image_pool_size=5,
            target_size=224,
            seed=42,
            model_id=current_model_id  # Use current model ID
        )

if 'trained_unet' in locals() and 'data_loader' in locals():
    # Define test datasets to loop over
    test_datasets = [
        {
            'name': 'Leaf-3m',
            'path': "/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/recon-1/em/fibsem-uint8/s4",
            'roi': Roi(np.array((0, 0, 0))*128, [256*128,256*128,1000*128])
        },
        {
            'name': 'Liver-Zon-1',
            'path': "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8/s4",
            'roi': Roi(np.array((450, 311, 220))*128, [750*128,750*128,750*128])
        }
    ]
    
    print(f"Testing on {len(test_datasets)} different datasets...")
    
    all_results = {}
    
    # Loop over test datasets
    for dataset_idx, dataset_info in enumerate(test_datasets):
        dataset_name = dataset_info['name']
        dataset_path = dataset_info['path']
        dataset_roi = dataset_info['roi']
        
        print(f"\n{'-'*50}")
        print(f"Processing Dataset {dataset_idx + 1}: {dataset_name}")
        print(f"Path: {dataset_path}")
        print(f"ROI: {dataset_roi}")
        print(f"{'-'*50}")
        
        try:
            # Load test data
            print(f"Loading test data from {dataset_name}...")
            raw_test = ImageDataInterface(dataset_path).to_ndarray_ts(dataset_roi)
            print(f"Test data shape: {raw_test.shape}")
            
            # Sample patches from test data
            print(f"Sampling patches from {dataset_name}...")
            test_patches, _ = sample_training_data(
                raw_test, raw_test,  # Use same as dummy GT
                target_size=224,
                num_samples=10,
                method="flexible",
                seed=123 + dataset_idx,  # Different seed for each dataset
                use_augmentation=False
            )
            
            print(f"Test patches shape: {test_patches.shape}")
            
            # Extract features and run inference
            print(f"Extracting DINOv3 features for {dataset_name}...")
            test_features_new = data_loader.extract_dinov3_features_for_unet(test_patches)
            
            print(f"Test features shape: {test_features_new.shape}")
            
            print(f"Running UNet inference on {dataset_name}...")
            with torch.no_grad():
                new_pred_logits = trained_unet(test_features_new)
                new_predictions = torch.argmax(new_pred_logits, dim=1).cpu().numpy()
                new_confidence = torch.max(torch.softmax(new_pred_logits, dim=1), dim=1)[0].cpu().numpy()
            
            print(f"Predictions shape: {new_predictions.shape}")
            print(f"Confidence shape: {new_confidence.shape}")
            
            # Store results for comparison
            all_results[dataset_name] = {
                'patches': test_patches,
                'predictions': new_predictions,
                'confidence': new_confidence,
                'dataset_path': dataset_path
            }
            
            # Show prediction statistics for this dataset
            unique_preds, pred_counts = np.unique(new_predictions, return_counts=True)
            print(f"\nPrediction distribution for {dataset_name}:")
            for class_id, count in zip(unique_preds, pred_counts):
                percentage = count / new_predictions.size * 100
                print(f"  Class {class_id}: {count:,} pixels ({percentage:.1f}%)")
                
            # Calculate confidence statistics
            print(f"\nConfidence statistics for {dataset_name}:")
            print(f"  Mean confidence: {new_confidence.mean():.4f}")
            print(f"  Min confidence: {new_confidence.min():.4f}")
            print(f"  Max confidence: {new_confidence.max():.4f}")
            print(f"  Std confidence: {new_confidence.std():.4f}")
            
            # Visualize results for this dataset
            print(f"Visualizing results for {dataset_name}...")
            fig, axes = plt.subplots(2, 6, figsize=(24, 8))
            
            for i in range(6):
                # Raw image
                axes[0, i].imshow(test_patches[i], cmap='gray')
                axes[0, i].set_title(f'{dataset_name} Image {i+1}')
                axes[0, i].axis('off')
                
                # Predictions with overlay
                axes[1, i].imshow(test_patches[i], cmap='gray')
                class1_mask = (new_predictions[i] == 1).astype(float)
                overlay = np.zeros((*class1_mask.shape, 4))
                overlay[class1_mask == 1] = [1, 0, 0, 0.6]  # Red overlay
                axes[1, i].imshow(overlay)
                axes[1, i].set_title(f'Predictions (Class 1)')
                axes[1, i].axis('off')
            
            plt.suptitle(f'UNet Inference on {dataset_name}', fontsize=16, y=0.98)
            plt.tight_layout()
            
            # Save figure to checkpoint directory
            if checkpoint_dir:
                figure_path = os.path.join(checkpoint_dir, f"inference_{dataset_name.lower().replace('-', '_')}.png")
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved to: {figure_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            print(f"Skipping {dataset_name}...")
            continue
    
    # Create comparison visualization if we have results from multiple datasets
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON ACROSS DATASETS")
        print(f"{'='*60}")
        
        # Create comparison figure
        dataset_names = list(all_results.keys())
        num_datasets = len(dataset_names)
        
        fig, axes = plt.subplots(2 * num_datasets, 6, figsize=(24, 4 * num_datasets))
        
        for dataset_idx, dataset_name in enumerate(dataset_names):
            results = all_results[dataset_name]
            patches = results['patches']
            predictions = results['predictions']
            
            for i in range(6):
                row_raw = dataset_idx * 2
                row_pred = dataset_idx * 2 + 1
                
                # Raw image
                axes[row_raw, i].imshow(patches[i], cmap='gray')
                if i == 0:
                    axes[row_raw, i].set_ylabel(f'{dataset_name}\nRaw', fontsize=12, rotation=0, ha='right', va='center')
                axes[row_raw, i].set_title(f'Image {i+1}' if dataset_idx == 0 else '')
                axes[row_raw, i].axis('off')
                
                # Predictions with overlay
                axes[row_pred, i].imshow(patches[i], cmap='gray')
                class1_mask = (predictions[i] == 1).astype(float)
                overlay = np.zeros((*class1_mask.shape, 4))
                overlay[class1_mask == 1] = [1, 0, 0, 0.6]  # Red overlay
                axes[row_pred, i].imshow(overlay)
                if i == 0:
                    axes[row_pred, i].set_ylabel(f'Predictions', fontsize=12, rotation=0, ha='right', va='center')
                axes[row_pred, i].axis('off')
        
        plt.suptitle('UNet Inference Comparison Across Datasets', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save comparison figure
        if checkpoint_dir:
            comparison_path = os.path.join(checkpoint_dir, "inference_comparison_all_datasets.png")
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            print(f"Comparison figure saved to: {comparison_path}")
        
        plt.show()
        
        # Print comparison statistics
        print(f"\nComparison Statistics:")
        print(f"{'Dataset':<15} {'Class 0 %':<10} {'Class 1 %':<10} {'Mean Conf':<12} {'Std Conf':<10}")
        print("-" * 60)
        
        for dataset_name, results in all_results.items():
            predictions = results['predictions']
            confidence = results['confidence']
            
            unique_preds, pred_counts = np.unique(predictions, return_counts=True)
            total_pixels = predictions.size
            
            class0_pct = 0.0
            class1_pct = 0.0
            for class_id, count in zip(unique_preds, pred_counts):
                if class_id == 0:
                    class0_pct = (count / total_pixels) * 100
                elif class_id == 1:
                    class1_pct = (count / total_pixels) * 100
            
            print(f"{dataset_name:<15} {class0_pct:<10.1f} {class1_pct:<10.1f} {confidence.mean():<12.4f} {confidence.std():<10.4f}")
    
    # Save comprehensive statistics to text file
    if checkpoint_dir and all_results:
        stats_path = os.path.join(checkpoint_dir, "inference_statistics_all_datasets.txt")
        with open(stats_path, 'w') as f:
            f.write("UNet Inference Statistics - Multi-Dataset Comparison\n")
            f.write("=" * 60 + "\n\n")
            
            for dataset_name, results in all_results.items():
                patches = results['patches']
                predictions = results['predictions']
                confidence = results['confidence']
                dataset_path = results['dataset_path']
                
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Path: {dataset_path}\n")
                f.write(f"Test patches shape: {patches.shape}\n")
                f.write(f"Predictions shape: {predictions.shape}\n")
                f.write(f"Confidence shape: {confidence.shape}\n\n")
                
                unique_preds, pred_counts = np.unique(predictions, return_counts=True)
                f.write("Prediction distribution:\n")
                for class_id, count in zip(unique_preds, pred_counts):
                    percentage = count / predictions.size * 100
                    f.write(f"  Class {class_id}: {count:,} pixels ({percentage:.1f}%)\n")
                
                f.write(f"\nConfidence statistics:\n")
                f.write(f"  Mean confidence: {confidence.mean():.4f}\n")
                f.write(f"  Min confidence: {confidence.min():.4f}\n")
                f.write(f"  Max confidence: {confidence.max():.4f}\n")
                f.write(f"  Std confidence: {confidence.std():.4f}\n")
                f.write("\n" + "-" * 50 + "\n\n")
        
        print(f"Comprehensive statistics saved to: {stats_path}")

else:
    print("No trained UNet available and no checkpoints found.")
    print("Please run Example 3 first to train a UNet model.")

# %%
# # Example 7: Save and load UNet model
# print("=" * 60)
# print("EXAMPLE 7: Model Checkpointing")
# print("=" * 60)

# # List available checkpoints - now they'll be organized by model ID
# print("Available UNet checkpoints:")
# checkpoints = list_checkpoints(
#     base_results_dir=EXPORT_BASE_DIR,  # Use the configured export directory
#     model_type="dinov3_unet_facebook_dinov3_vits16_pretrain_lvd1689m"  # Include model ID in search
# )
# for i, checkpoint in enumerate(checkpoints):
#     print(f"  {i+1}. {os.path.basename(checkpoint)}")

# if checkpoints:
#     # Load the latest checkpoint
#     latest_checkpoint = checkpoints[0]
#     print(f"\nLoading checkpoint: {os.path.basename(latest_checkpoint)}")
    
#     checkpoint_data = load_checkpoint(latest_checkpoint, device)
#     config = checkpoint_data['training_config']
    
#     print(f"Checkpoint info:")
#     print(f"  Epoch: {checkpoint_data['epoch']}")
#     print(f"  Best validation accuracy: {checkpoint_data['best_val_acc']:.4f}")
#     print(f"  Model type: {config['model_type']}")
#     print(f"  Base channels: {config['base_channels']}")
    
#     # Recreate and load model
#     loaded_unet = DINOv3UNet(
#         input_channels=output_channels,
#         num_classes=config['num_classes'],
#         base_channels=config['base_channels']
#     ).to(device)
    
#     loaded_unet.load_state_dict(checkpoint_data['unet_state_dict'])
#     loaded_unet.eval()
    
#     print("Model loaded successfully from checkpoint!")
    
# else:
#     print("No UNet checkpoints found.")

# %%
print("=" * 60)
print("DINOV3 UNET TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("✅ UNet architecture implemented")
print("✅ Memory-efficient training pipeline")
print("✅ DINOv3 feature integration")
print("✅ Spatial structure preservation")
print("✅ Image-level segmentation capability")
print("✅ Model checkpointing and loading")
print("✅ Comprehensive visualization")
print("=" * 60)

print(f"\nKey Results:")
print(f"  - Best UNet validation accuracy: {unet_results['best_val_acc']:.4f}")
print(f"  - Training epochs: {unet_results['epochs_trained']}")
print(f"  - Model parameters: {unet_params:,}")
print(f"  - Model size: {unet_size_mb:.2f} MB")
print(f"  - Input resolution: 224x224")
print(f"  - Feature channels: {output_channels}")
print(f"  - Export directory: {EXPORT_BASE_DIR}")