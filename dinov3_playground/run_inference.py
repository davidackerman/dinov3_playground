
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
    print_unet_summary
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
# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
processor, model, output_channels = initialize_dinov3(
    model_id=MODEL_ID,
    image_size=IMAGE_SIZE
)
# Get model info
model_info = get_current_model_info()
print(f"DINOv3 Model ID: {model_info['model_id']}")
print(f"Output channels: {model_info['output_channels']}")
print(f"Model device: {model_info['device']}")

print("Loading datasets...")
EXPORT_BASE_DIR = "/nrs/cellmap/ackermand/dinov3_training/results/"  # Updated export directory
# First, check if we have a trained model from previous examples, otherwise load from checkpoint
if 'unet_results' in locals() and 'data_loader' in locals():
    print("Using trained UNet from previous examples...")
    trained_unet = unet_results['unet']
    # Get checkpoint directory for saving figures
    checkpoint_dir = unet_results.get('checkpoint_dir', None)
    # data_loader already exists
else:
    print("Loading UNet from checkpoint...")
    # # Get current model info to construct the correct checkpoint path
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
    # Load different data for testing
    print("Loading test data from different region...")
    try:
        raw_test = ImageDataInterface("/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/recon-1/em/fibsem-uint8/s4").to_ndarray_ts(Roi(np.array((0, 0, 0))*128, [256*128,256*128,1000*128]))
        # raw_test = ImageDataInterface("/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8/s4").to_ndarray_ts(Roi(np.array((450, 311, 220))*128, [750*128,750*128,750*128]))
        print(f"Test data shape: {raw_test.shape}")
        
        # Sample patches from test data
        test_patches, _ = sample_training_data(
            raw_test, raw_test,  # Use same as dummy GT
            target_size=224,
            num_samples=2,
            method="flexible",
            seed=123,
            use_augmentation=False
        )
        
        print(f"Test patches shape: {test_patches.shape}")
        
        # Extract features and run inference
        print("Extracting DINOv3 features for inference...")
        test_features_new = data_loader.extract_dinov3_features_for_unet(test_patches)
        
        print(f"Test features shape: {test_features_new.shape}")
        
        print("Running UNet inference...")
        with torch.no_grad():
            new_pred_logits = trained_unet(test_features_new)
            new_predictions = torch.argmax(new_pred_logits, dim=1).cpu().numpy()
            new_confidence = torch.max(torch.softmax(new_pred_logits, dim=1), dim=1)[0].cpu().numpy()
        
        print(f"Predictions shape: {new_predictions.shape}")
        print(f"Confidence shape: {new_confidence.shape}")
        
        # Visualize new test results
        print("Visualizing results on new test data...")
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        
        for i in range(6):
            # Raw image
            axes[0, i].imshow(test_patches[i], cmap='gray')
            axes[0, i].set_title(f'Test Image {i+1}')
            axes[0, i].axis('off')
            
            # Predictions with overlay
            axes[1, i].imshow(test_patches[i], cmap='gray')
            class1_mask = (new_predictions[i] == 1).astype(float)
            overlay = np.zeros((*class1_mask.shape, 4))
            overlay[class1_mask == 1] = [1, 0, 0, 0.6]  # Red overlay
            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f'Predictions (Class 1)')
            axes[1, i].axis('off')
        
        plt.suptitle('UNet Inference on New Test Data', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save figure to checkpoint directory
        if checkpoint_dir:
            figure_path = os.path.join(checkpoint_dir, "inference_new_test_data.png")
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {figure_path}")
        
        plt.show()
        
        # Show prediction statistics
        unique_preds, pred_counts = np.unique(new_predictions, return_counts=True)
        print(f"\nPrediction distribution on new test data:")
        for class_id, count in zip(unique_preds, pred_counts):
            percentage = count / new_predictions.size * 100
            print(f"  Class {class_id}: {count} pixels ({percentage:.1f}%)")
            
        # Calculate confidence statistics
        print(f"\nConfidence statistics:")
        print(f"  Mean confidence: {new_confidence.mean():.4f}")
        print(f"  Min confidence: {new_confidence.min():.4f}")
        print(f"  Max confidence: {new_confidence.max():.4f}")
        print(f"  Std confidence: {new_confidence.std():.4f}")
        
        # Save statistics to text file
        if checkpoint_dir:
            stats_path = os.path.join(checkpoint_dir, "inference_statistics.txt")
            with open(stats_path, 'w') as f:
                f.write("UNet Inference Statistics on New Test Data\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Test patches shape: {test_patches.shape}\n")
                f.write(f"Predictions shape: {new_predictions.shape}\n")
                f.write(f"Confidence shape: {new_confidence.shape}\n\n")
                
                f.write("Prediction distribution:\n")
                for class_id, count in zip(unique_preds, pred_counts):
                    percentage = count / new_predictions.size * 100
                    f.write(f"  Class {class_id}: {count} pixels ({percentage:.1f}%)\n")
                
                f.write(f"\nConfidence statistics:\n")
                f.write(f"  Mean confidence: {new_confidence.mean():.4f}\n")
                f.write(f"  Min confidence: {new_confidence.min():.4f}\n")
                f.write(f"  Max confidence: {new_confidence.max():.4f}\n")
                f.write(f"  Std confidence: {new_confidence.std():.4f}\n")
            
            print(f"Statistics saved to: {stats_path}")
        
    except Exception as e:
        print(f"Could not load test data: {e}")
        print("Using original training data for inference demonstration...")
        
        # Fallback: use original data
        test_patches, test_gt = sample_training_data(
            raw, gt, 
            target_size=224, 
            num_samples=6, 
            method="flexible",
            seed=999,
            use_augmentation=False
        )
        
        print("Extracting DINOv3 features for inference...")
        test_features_new = data_loader.extract_dinov3_features_for_unet(test_patches)
        
        print("Running UNet inference...")
        with torch.no_grad():
            new_pred_logits = trained_unet(test_features_new)
            new_predictions = torch.argmax(new_pred_logits, dim=1).cpu().numpy()
            new_confidence = torch.max(torch.softmax(new_pred_logits, dim=1), dim=1)[0].cpu().numpy()
        
        # Calculate accuracy on ground truth
        correct_pixels = (new_predictions == test_gt).sum()
        total_pixels = test_gt.size
        accuracy = correct_pixels / total_pixels
        print(f"Test accuracy on original data: {accuracy:.4f}")
        
        # Visualize results
        print("Visualizing results...")
        fig, axes = plt.subplots(3, 6, figsize=(24, 12))
        
        for i in range(6):
            # Raw image
            axes[0, i].imshow(test_patches[i], cmap='gray')
            axes[0, i].set_title(f'Test Image {i+1}')
            axes[0, i].axis('off')
            
            # Ground truth
            axes[1, i].imshow(test_gt[i], cmap='tab10', vmin=0, vmax=1)
            axes[1, i].set_title(f'Ground Truth')
            axes[1, i].axis('off')
            
            # Predictions with overlay
            axes[2, i].imshow(test_patches[i], cmap='gray')
            class1_mask = (new_predictions[i] == 1).astype(float)
            overlay = np.zeros((*class1_mask.shape, 4))
            overlay[class1_mask == 1] = [1, 0, 0, 0.6]  # Red overlay
            axes[2, i].imshow(overlay)
            axes[2, i].set_title(f'Predictions (Class 1)')
            axes[2, i].axis('off')
        
        plt.suptitle('UNet Inference on Original Training Data', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save figure to checkpoint directory
        if checkpoint_dir:
            figure_path = os.path.join(checkpoint_dir, "inference_original_data.png")
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {figure_path}")
        
        plt.show()
        
        # Show prediction statistics
        unique_preds, pred_counts = np.unique(new_predictions, return_counts=True)
        print(f"\nPrediction distribution:")
        for class_id, count in zip(unique_preds, pred_counts):
            percentage = count / new_predictions.size * 100
            print(f"  Class {class_id}: {count} pixels ({percentage:.1f}%)")
        
        # Save statistics to text file
        if checkpoint_dir:
            stats_path = os.path.join(checkpoint_dir, "inference_statistics.txt")
            with open(stats_path, 'w') as f:
                f.write("UNet Inference Statistics on Original Training Data\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Test accuracy: {accuracy:.4f}\n")
                f.write(f"Test patches shape: {test_patches.shape}\n")
                f.write(f"Predictions shape: {new_predictions.shape}\n")
                f.write(f"Confidence shape: {new_confidence.shape}\n\n")
                
                f.write("Prediction distribution:\n")
                for class_id, count in zip(unique_preds, pred_counts):
                    percentage = count / new_predictions.size * 100
                    f.write(f"  Class {class_id}: {count} pixels ({percentage:.1f}%)\n")
            
            print(f"Statistics saved to: {stats_path}")

else:
    print("No trained UNet available and no checkpoints found.")
    print("Please run Example 3 first to train a UNet model.")

# %%
