#!/usr/bin/env python3
"""
Example showing how to use learned upsampling in existing training scripts.

This demonstrates how to modify existing training calls to use the new
learned upsampling feature.
"""

# EXAMPLE 1: Basic usage in training function
from dinov3_playground.memory_efficient_training import (
    train_3d_unet_with_memory_efficient_loader,
)


def example_learned_upsampling_training():
    """Example showing learned upsampling in training."""

    # Your existing training call (traditional approach)
    results_traditional = train_3d_unet_with_memory_efficient_loader(
        raw_data=raw_data,
        gt_data=gt_data,
        num_classes=2,
        target_volume_size=(128, 128, 128),
        dinov3_slice_size=896,
        epochs=100,
        base_channels=32,
        # All your existing parameters...
        learn_upsampling=False,  # Traditional (default)
    )

    # NEW: Enable learned upsampling (just add one parameter!)
    results_learned = train_3d_unet_with_memory_efficient_loader(
        raw_data=raw_data,
        gt_data=gt_data,
        num_classes=2,
        target_volume_size=(128, 128, 128),
        dinov3_slice_size=896,
        epochs=100,
        base_channels=32,
        # All your existing parameters...
        learn_upsampling=True,  # NEW: Enable learned upsampling
    )

    return results_traditional, results_learned


# EXAMPLE 2: Memory-efficient settings with learned upsampling
def example_memory_efficient_setup():
    """Example combining learned upsampling with other memory optimizations."""

    results = train_3d_unet_with_memory_efficient_loader(
        raw_data=raw_data,
        gt_data=gt_data,
        num_classes=2,
        target_volume_size=(128, 128, 128),
        # Memory optimization settings
        learn_upsampling=True,  # Reduce feature memory usage
        use_mixed_precision=True,  # Reduce model memory usage
        use_half_precision=False,  # Keep full precision for stability
        base_channels=32,  # Can increase due to memory savings
        volumes_per_batch=2,  # Can increase batch size
        # Training settings
        epochs=120,  # May need more epochs for convergence
        patience=30,  # Increase patience
        learning_rate=1e-3,  # Standard learning rate
        # Other parameters...
        dinov3_slice_size=896,
        save_checkpoints=True,
        export_base_dir="/path/to/results",
    )

    return results


# EXAMPLE 3: Comparing both approaches
def compare_upsampling_approaches(raw_data, gt_data):
    """Compare traditional vs learned upsampling on the same data."""

    print("Training with traditional interpolated upsampling...")
    results_traditional = train_3d_unet_with_memory_efficient_loader(
        raw_data=raw_data,
        gt_data=gt_data,
        num_classes=2,
        target_volume_size=(64, 64, 64),
        epochs=50,
        learn_upsampling=False,
        export_base_dir="/tmp/traditional",
    )

    print("Training with learned upsampling...")
    results_learned = train_3d_unet_with_memory_efficient_loader(
        raw_data=raw_data,
        gt_data=gt_data,
        num_classes=2,
        target_volume_size=(64, 64, 64),
        epochs=50,
        learn_upsampling=True,
        export_base_dir="/tmp/learned",
    )

    # Compare results
    print("\nComparison Results:")
    print(f"Traditional approach: {results_traditional['best_val_acc']:.4f}")
    print(f"Learned upsampling:  {results_learned['best_val_acc']:.4f}")

    return results_traditional, results_learned


# EXAMPLE 4: Usage in your existing training scripts
def modify_existing_script():
    """
    Example of how to modify your existing training scripts.

    In your existing script, you probably have something like:

    OLD CODE:
    --------
    results = train_3d_unet_with_memory_efficient_loader(
        raw_data=dataset_pairs['raw'],
        gt_data=dataset_pairs['mito'],
        num_classes=2,
        epochs=100,
        # ... other parameters
    )

    NEW CODE (just add one line):
    ----------------------------
    results = train_3d_unet_with_memory_efficient_loader(
        raw_data=dataset_pairs['raw'],
        gt_data=dataset_pairs['mito'],
        num_classes=2,
        epochs=100,
        learn_upsampling=True,    # <-- ADD THIS LINE
        # ... other parameters
    )
    """
    pass


if __name__ == "__main__":
    print("Learned Upsampling Usage Examples")
    print("=================================")
    print()
    print("To use learned upsampling in your training:")
    print("1. Add 'learn_upsampling=True' to your training function call")
    print("2. Consider increasing epochs/patience for convergence")
    print("3. Monitor memory usage - should be lower")
    print("4. Compare results with traditional approach")
    print()
    print("See the examples above for detailed usage patterns.")
