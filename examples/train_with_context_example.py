"""
Example: Training 3D UNet with Multi-Scale Context

This example demonstrates how to use the new multi-scale context functionality
to provide spatial context information during training.

Author: GitHub Copilot
Date: 2025-10-03
"""

import os
import torch
import numpy as np

# Import the updated functions
from dinov3_playground.data_processing import (
    generate_multi_organelle_dataset_pairs,
    load_random_3d_training_data,
)
from dinov3_playground.memory_efficient_training import (
    train_3d_unet_with_memory_efficient_loader,
)
from dinov3_playground import initialize_dinov3, get_current_model_info


def train_with_context_example():
    """
    Example of training with multi-scale context.
    """

    # Configuration
    MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
    base_resolution = 16  # Resolution for segmentation labels (nm)
    min_resolution_for_raw = 4  # High-resolution raw data (nm)
    context_scale = 32  # Context resolution (nm) - 8x lower res than local

    VOLUME_SIZE = (64, 64, 64)  # Target size for training
    DINOV3_SLICE_SIZE = 256
    BASE_CHANNELS = 64
    NUM_VOLUMES = 50

    print("=" * 60)
    print("MULTI-SCALE CONTEXT TRAINING EXAMPLE")
    print("=" * 60)

    print(f"Configuration:")
    print(f"  Raw data resolution: {min_resolution_for_raw}nm (high-res local)")
    print(f"  GT data resolution: {base_resolution}nm")
    print(f"  Context resolution: {context_scale}nm (spatial context)")
    print(
        f"  Context scale factor: {context_scale/min_resolution_for_raw}x lower resolution"
    )
    print(f"  Volume size: {VOLUME_SIZE}")

    # Export directory
    EXPORT_BASE_DIR = "/nrs/cellmap/ackermand/dinov3_training/results/context_example"
    os.makedirs(EXPORT_BASE_DIR, exist_ok=True)

    # Initialize DINOv3
    processor, model, output_channels = initialize_dinov3(
        model_id=MODEL_ID, image_size=DINOV3_SLICE_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate dataset pairs WITH context
    print("\nGenerating dataset pairs with context data...")
    dataset_pairs = generate_multi_organelle_dataset_pairs(
        organelle_list=["mito", "er", "nuc"],
        min_resolution_for_raw=min_resolution_for_raw,
        base_resolution=base_resolution,
        context_scale=context_scale,  # NEW: Add context data at 32nm
        apply_scale_updates=True,
        require_all_organelles=True,
    )

    print(f"Generated {len(dataset_pairs)} dataset pairs")
    if dataset_pairs:
        example_keys = list(dataset_pairs[0].keys())
        print(f"Dataset pair keys: {example_keys}")
        # Should now include: 'raw', 'raw_32nm', 'mito', 'er', 'nuc'

    # Load 3D training data WITH context
    print(f"\nLoading {NUM_VOLUMES} training volumes with context...")
    result = load_random_3d_training_data(
        dataset_pairs=dataset_pairs,
        volume_shape=VOLUME_SIZE,
        base_resolution=base_resolution,
        min_label_fraction=0.01,
        num_volumes=NUM_VOLUMES,
        seed=42,
        min_resolution_for_raw=min_resolution_for_raw,
        allow_gt_extension=True,
        context_scale=context_scale,  # NEW: Load context data
    )

    # Unpack results (context_volumes will be included if context_scale was provided)
    if len(result) == 6:  # With context and GT extension
        raw, gt, gt_masks, context_volumes, dataset_sources, num_classes = result
        print(f"Loaded data with context:")
        print(f"  Raw volumes: {raw.shape}")
        print(f"  GT volumes: {gt.shape}")
        print(f"  GT masks: {gt_masks.shape}")
        print(f"  Context volumes: {context_volumes.shape}")
        print(f"  Context scale: {context_scale}nm")
    elif len(result) == 5:  # Without context but with GT extension
        raw, gt, gt_masks, dataset_sources, num_classes = result
        context_volumes = None
        print(f"Loaded data without context:")
        print(f"  Raw volumes: {raw.shape}")
        print(f"  GT volumes: {gt.shape}")
        print(f"  GT masks: {gt_masks.shape}")
    else:
        raise ValueError(f"Unexpected number of return values: {len(result)}")

    print(f"  Number of classes: {num_classes}")
    print(f"  Unique GT values: {np.unique(gt)}")

    # Train with context
    print(f"\nStarting training with multi-scale context...")
    training_results = train_3d_unet_with_memory_efficient_loader(
        raw_data=raw,
        gt_data=gt,
        gt_masks=gt_masks,
        context_data=context_volumes,  # NEW: Pass context data
        context_scale=context_scale,  # NEW: Pass context scale
        train_volume_pool_size=NUM_VOLUMES - 5,
        val_volume_pool_size=5,
        num_classes=num_classes,
        target_volume_size=VOLUME_SIZE,
        dinov3_slice_size=DINOV3_SLICE_SIZE,
        volumes_per_batch=1,  # Start small for context experiments
        batches_per_epoch=10,
        epochs=20,  # Short run for testing
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=10,
        min_delta=0.001,
        base_channels=BASE_CHANNELS,
        device=device,
        seed=42,
        model_id=MODEL_ID,
        export_base_dir=EXPORT_BASE_DIR,
        save_checkpoints=True,
        checkpoint_every_n_epochs=5,
        use_class_weighting=True,
        use_orthogonal_planes=True,
        use_mixed_precision=True,
        enable_detailed_timing=False,
        verbose=False,
    )

    print(f"\nTraining completed!")
    print(f"  Best validation accuracy: {training_results['best_val_acc']:.4f}")
    print(f"  Epochs trained: {training_results['epochs_trained']}")
    print(f"  Used multi-scale context: {context_volumes is not None}")

    return training_results


def compare_with_without_context():
    """
    Example comparing training with and without context.
    """

    print("=" * 60)
    print("CONTEXT vs NO-CONTEXT COMPARISON")
    print("=" * 60)

    # Same configuration for both
    base_config = {
        "model_id": "facebook/dinov3-vitl16-pretrain-sat493m",
        "base_resolution": 16,
        "min_resolution_for_raw": 4,
        "volume_size": (64, 64, 64),
        "num_volumes": 30,
    }

    # Train without context (baseline)
    print("\n1. Training WITHOUT context (baseline)...")
    dataset_pairs_no_context = generate_multi_organelle_dataset_pairs(
        organelle_list=["mito"],
        min_resolution_for_raw=base_config["min_resolution_for_raw"],
        base_resolution=base_config["base_resolution"],
        # No context_scale parameter
        apply_scale_updates=True,
    )

    (
        raw_baseline,
        gt_baseline,
        gt_masks_baseline,
        dataset_sources_baseline,
        num_classes,
    ) = load_random_3d_training_data(
        dataset_pairs=dataset_pairs_no_context,
        volume_shape=base_config["volume_size"],
        base_resolution=base_config["base_resolution"],
        num_volumes=base_config["num_volumes"],
        min_resolution_for_raw=base_config["min_resolution_for_raw"],
        allow_gt_extension=True,
        # No context_scale parameter
    )

    # Train with context
    print("\n2. Training WITH context...")
    dataset_pairs_with_context = generate_multi_organelle_dataset_pairs(
        organelle_list=["mito"],
        min_resolution_for_raw=base_config["min_resolution_for_raw"],
        base_resolution=base_config["base_resolution"],
        context_scale=32,  # 8x lower resolution context
        apply_scale_updates=True,
    )

    (
        raw_context,
        gt_context,
        gt_masks_context,
        context_volumes,
        dataset_sources_context,
        num_classes,
    ) = load_random_3d_training_data(
        dataset_pairs=dataset_pairs_with_context,
        volume_shape=base_config["volume_size"],
        base_resolution=base_config["base_resolution"],
        num_volumes=base_config["num_volumes"],
        min_resolution_for_raw=base_config["min_resolution_for_raw"],
        allow_gt_extension=True,
        context_scale=32,
    )

    print(f"\nComparison summary:")
    print(f"  Baseline input channels: {384}")  # Standard DINOv3 channels
    print(f"  Context input channels: {384 * 2}")  # Double for local + context
    print(f"  Context provides {context_volumes.shape} additional spatial information")

    return {
        "baseline": (raw_baseline, gt_baseline, gt_masks_baseline),
        "context": (raw_context, gt_context, gt_masks_context, context_volumes),
    }


if __name__ == "__main__":
    # Run the context training example
    try:
        results = train_with_context_example()
        print("\n✅ Context training example completed successfully!")

    except Exception as e:
        print(f"\n❌ Error in context training: {e}")
        print("This might be due to missing data paths or configuration issues.")

    # Optionally run the comparison
    # comparison_data = compare_with_without_context()
