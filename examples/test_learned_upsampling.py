#!/usr/bin/env python3
"""
Test script to demonstrate learned upsampling vs interpolated upsampling in 3D UNet.

This script shows the difference between:
1. Traditional approach: DINOv3 features interpolated to full resolution, then fed to UNet
2. Learned upsampling: DINOv3 features at native resolution, UNet learns upsampling

Usage:
    python examples/test_learned_upsampling.py
"""

import numpy as np
import torch
from dinov3_playground.memory_efficient_training import (
    train_3d_unet_with_memory_efficient_loader,
)


def create_test_data(num_volumes=10, volume_size=(64, 64, 64)):
    """Create synthetic test data for demonstration."""
    raw_data = np.random.rand(num_volumes, *volume_size).astype(np.float32)
    # Create simple binary segmentation (nucleus/background)
    gt_data = (np.random.rand(num_volumes, *volume_size) > 0.7).astype(np.int32)
    return raw_data, gt_data


def test_traditional_upsampling():
    """Test traditional interpolated upsampling approach."""
    print("=" * 60)
    print("TESTING TRADITIONAL INTERPOLATED UPSAMPLING")
    print("=" * 60)

    raw_data, gt_data = create_test_data(num_volumes=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = train_3d_unet_with_memory_efficient_loader(
        raw_data=raw_data,
        gt_data=gt_data,
        train_volume_pool_size=4,
        val_volume_pool_size=2,
        num_classes=2,
        target_volume_size=(32, 32, 32),  # Smaller for demo
        dinov3_slice_size=256,  # Smaller for demo
        volumes_per_batch=1,
        batches_per_epoch=3,
        epochs=5,
        base_channels=16,  # Smaller for demo
        device=device,
        learn_upsampling=False,  # Traditional approach
        use_mixed_precision=False,  # Disable for stability in demo
    )

    print(f"\nTraditional Results:")
    print(f"  Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"  Final train accuracy: {results['train_accs'][-1]:.4f}")

    return results


def test_learned_upsampling():
    """Test learned upsampling approach."""
    print("\n" + "=" * 60)
    print("TESTING LEARNED UPSAMPLING")
    print("=" * 60)

    raw_data, gt_data = create_test_data(num_volumes=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = train_3d_unet_with_memory_efficient_loader(
        raw_data=raw_data,
        gt_data=gt_data,
        train_volume_pool_size=4,
        val_volume_pool_size=2,
        num_classes=2,
        target_volume_size=(32, 32, 32),  # Smaller for demo
        dinov3_slice_size=256,  # Smaller for demo
        volumes_per_batch=1,
        batches_per_epoch=3,
        epochs=5,
        base_channels=16,  # Smaller for demo
        device=device,
        learn_upsampling=True,  # NEW: Let UNet learn upsampling
        use_mixed_precision=False,  # Disable for stability in demo
    )

    print(f"\nLearned Upsampling Results:")
    print(f"  Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"  Final train accuracy: {results['train_accs'][-1]:.4f}")

    return results


def compare_approaches():
    """Compare both approaches."""
    print("\n" + "=" * 80)
    print("COMPARISON OF UPSAMPLING APPROACHES")
    print("=" * 80)

    print("\nKey Differences:")
    print("\n1. TRADITIONAL INTERPOLATED UPSAMPLING:")
    print("   - DINOv3 features: 256x256 → interpolated to 32x32x32")
    print("   - UNet input: Full resolution features")
    print("   - UNet task: Segmentation only")
    print("   - Memory: Higher (stores full-res features)")
    print("   - Training: Faster convergence (pre-upsampled features)")

    print("\n2. LEARNED UPSAMPLING:")
    print("   - DINOv3 features: 256x256 → kept at native 16x16 resolution")
    print("   - UNet input: Low resolution features")
    print("   - UNet task: Upsampling + Segmentation")
    print("   - Memory: Lower (stores low-res features)")
    print("   - Training: May need more epochs (learns upsampling)")

    print("\n3. ADVANTAGES OF LEARNED UPSAMPLING:")
    print("   ✓ Reduced memory usage (lower resolution features)")
    print("   ✓ UNet learns task-specific upsampling")
    print("   ✓ Potentially better feature adaptation")
    print("   ✓ More end-to-end learning")

    print("\n4. WHEN TO USE LEARNED UPSAMPLING:")
    print("   ✓ Limited GPU memory")
    print("   ✓ Large volume sizes")
    print("   ✓ When interpolation artifacts are problematic")
    print("   ✓ For optimal end-to-end optimization")


if __name__ == "__main__":
    print("3D UNet Upsampling Comparison Demo")
    print("This demo compares traditional interpolation vs learned upsampling")

    # Test both approaches
    traditional_results = test_traditional_upsampling()
    learned_results = test_learned_upsampling()

    # Compare results
    compare_approaches()

    print(f"\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(
        f"Traditional approach - Best Val Acc: {traditional_results['best_val_acc']:.4f}"
    )
    print(f"Learned upsampling  - Best Val Acc: {learned_results['best_val_acc']:.4f}")

    if learned_results["best_val_acc"] > traditional_results["best_val_acc"]:
        print("🎉 Learned upsampling achieved better accuracy!")
    elif traditional_results["best_val_acc"] > learned_results["best_val_acc"]:
        print("📈 Traditional interpolation achieved better accuracy!")
    else:
        print("🤝 Both approaches achieved similar accuracy!")

    print(f"\nNote: This is a small demo with limited training.")
    print(f"For real applications, train for more epochs and larger datasets.")
