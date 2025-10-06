"""
DINOv3 Playground Package

This package contains modularized functions for DINOv3 feature extraction and training.

Modules:
- dinov3_core: Core DINOv3 processing functions
- data_processing: Data sampling and augmentation functions
- models: Neural network model classes
- model_training: Training and class balancing functions
- visualization: Plotting and visualization functions
- memory_efficient_training: Memory-efficient training system

Author: GitHub Copilot
Date: 2025-09-11
"""

# Core functionality
from .dinov3_core import (
    enable_amp_inference,
    process,
    normalize_features,
    apply_normalization_stats,
    initialize_dinov3,
    get_current_model_info,
    ensure_initialized,
    output_channels,
    DEVICE,
    processor,
    model,
)

# Data processing
from .data_processing import (
    sample_training_data,
    apply_intensity_augmentation,
    get_class_names_from_dataset_pairs,
)

# Models
from .models import (
    SimpleClassifier,
    ImprovedClassifier,
    DINOv3UNet,  # 2D UNet
    DINOv3UNet3D,  # 3D UNet - This was missing
    DINOv3UNet3DPipeline,  # 3D Pipeline - This was missing
    create_model,
    print_model_summary,
)

# Training functions
from .model_training import balance_classes

# Loss functions
from .losses import (
    FocalLoss,
    DiceLoss,
    FocalDiceLoss,
    TverskyLoss,
    get_loss_function,
)

# Visualization
try:
    from .visualization import (
        plot_training_history,
        plot_class_distribution,
        plot_sample_predictions,
    )
except ImportError:
    # If visualization module doesn't exist or functions don't exist, define placeholders
    def plot_training_history(*args, **kwargs):
        print("Visualization function plot_training_history not implemented yet")

    def plot_class_distribution(*args, **kwargs):
        print("Visualization function plot_class_distribution not implemented yet")

    def plot_sample_predictions(*args, **kwargs):
        print("Visualization function plot_sample_predictions not implemented yet")


# Memory-efficient training
from .memory_efficient_training import (
    MemoryEfficientDataLoader,
    MemoryEfficientDataLoader3D,
    train_classifier_memory_efficient,
    train_unet_memory_efficient,
    train_with_memory_efficient_loader,
    train_with_unet_memory_efficient_loader,
    train_3d_unet_memory_efficient_v2,
    train_3d_unet_with_memory_efficient_loader,
    load_checkpoint,
    list_checkpoints,
    restore_model_from_checkpoint,
)

# Inference
from .inference import (
    DINOv3UNetInference,
    DINOv3UNet3DInference,
    load_inference_model,
    demo_2d_inference,
    demo_3d_inference,
)

__version__ = "0.1.0"
__author__ = "GitHub Copilot"

# Package metadata
__all__ = [
    # Core
    "enable_amp_inference",
    "process",
    "normalize_features",
    "apply_normalization_stats",
    "initialize_dinov3",
    "get_current_model_info",
    "setup_deterministic_training",
    "ensure_initialized",
    "output_channels",
    "DEVICE",
    "processor",
    "model",
    # Data processing
    "sample_training_data",
    "apply_intensity_augmentation",
    "get_class_names_from_dataset_pairs",
    # Models
    "SimpleClassifier",
    "ImprovedClassifier",
    "DINOv3UNet",
    "DINOv3UNet3D",
    "UNetPipeline",
    "DINOv3UNet3DPipeline",
    "create_model",
    "print_model_summary",
    # Training
    "balance_classes",
    # Loss functions
    "FocalLoss",
    "DiceLoss",
    "FocalDiceLoss",
    "TverskyLoss",
    "get_loss_function",
    # Visualization
    "plot_training_history",
    "plot_class_distribution",
    "plot_sample_predictions",
    # Memory-efficient training
    "MemoryEfficientDataLoader",
    "MemoryEfficientDataLoader3D",
    "train_classifier_memory_efficient",
    "train_unet_memory_efficient",
    "train_with_memory_efficient_loader",
    "train_with_unet_memory_efficient_loader",
    "train_3d_unet_memory_efficient_v2",
    "train_3d_unet_with_memory_efficient_loader",
    "load_checkpoint",
    "list_checkpoints",
    "restore_model_from_checkpoint",
    # Inference
    "DINOv3UNetInference",
    "DINOv3UNet3DInference",
    "load_inference_model",
    "demo_2d_inference",
    "demo_3d_inference",
]
