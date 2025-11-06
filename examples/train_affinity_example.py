"""
Example: Training DINOv3 UNet3D for Affinity Prediction

This script demonstrates how to train a model to predict affinities instead of class labels.
Affinities indicate the probability that neighboring pixels belong to the same instance.

Author: GitHub Copilot
Date: 2025-10-06
"""

import sys

sys.path.append("/groups/cellmap/cellmap/ackermand/Programming/dinov3_playground")

import numpy as np
import torch
from dinov3_playground.memory_efficient_training import (
    MemoryEfficientDataLoader3D,
    train_dinov3_unet3d_memory_efficient,
)
from dinov3_playground.affinity_utils import AffinityLoss


# Example: Load your instance segmentation data
# Replace this with your actual data loading code
def load_instance_data():
    """
    Load instance segmentation data.

    Returns:
    --------
    raw_data : numpy.ndarray
        Shape (num_volumes, D, H, W) - raw EM images
    instance_data : numpy.ndarray
        Shape (num_volumes, D, H, W) - instance segmentations
        Each unique non-zero value represents a different instance
    gt_masks : numpy.ndarray
        Shape (num_volumes, D, H, W) - valid GT regions (1=valid, 0=invalid)
    """
    # This is a placeholder - replace with your actual data loading
    num_volumes = 25
    D, H, W = 64, 64, 64

    raw_data = np.random.randint(0, 255, (num_volumes, D, H, W), dtype=np.uint8)

    # Create synthetic instance segmentation
    instance_data = np.zeros((num_volumes, D, H, W), dtype=np.int32)
    for v in range(num_volumes):
        # Create a few random instances per volume
        for i in range(1, 5):  # 4 instances
            # Random blob
            center_z = np.random.randint(10, D - 10)
            center_y = np.random.randint(10, H - 10)
            center_x = np.random.randint(10, W - 10)
            size = np.random.randint(5, 15)

            z, y, x = np.ogrid[0:D, 0:H, 0:W]
            mask = (
                (z - center_z) ** 2 + (y - center_y) ** 2 + (x - center_x) ** 2
            ) < size**2
            instance_data[v][mask] = i

    # Full valid masks (all GT is valid)
    gt_masks = np.ones_like(instance_data, dtype=np.uint8)

    return raw_data, instance_data, gt_masks


def main():
    """Main training function."""

    print("Loading instance segmentation data...")
    raw_data, instance_data, gt_masks = load_instance_data()

    print(f"Data shapes:")
    print(f"  Raw: {raw_data.shape}")
    print(f"  Instances: {instance_data.shape}")
    print(f"  Masks: {gt_masks.shape}")
    print(
        f"  Num instances per volume (avg): {[len(np.unique(instance_data[i])) - 1 for i in range(3)]}"
    )

    # Create data loader with affinity output
    print("\nCreating data loader with affinity output...")
    data_loader = MemoryEfficientDataLoader3D(
        raw_data=raw_data,
        gt_data=instance_data,  # Instance segmentations will be converted to affinities
        gt_masks=gt_masks,
        train_volume_pool_size=20,
        val_volume_pool_size=5,
        target_volume_size=(64, 64, 64),
        dinov3_slice_size=512,
        seed=42,
        model_id="facebook/dinov3-vitl16-pretrain-sat493m",
        output_type="affinities",  # KEY: Enable affinity output
        affinity_offsets=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],  # +z, +y, +x
    )

    # Test sampling
    print("\nTesting batch sampling...")
    batch_volumes, batch_gt, batch_masks, batch_context = (
        data_loader.sample_training_batch(2)
    )

    print(f"Batch shapes:")
    print(f"  Volumes: {batch_volumes.shape}")
    print(f"  GT (affinities): {batch_gt.shape}")  # Should be (2, 3, 64, 64, 64)
    print(f"  Masks: {batch_masks.shape}")

    # Check affinity values
    print(f"\nAffinity statistics:")
    for i, offset in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        aff = batch_gt[0, i]  # First volume, i-th offset
        print(
            f"  Offset {offset}: mean={aff.mean():.3f}, min={aff.min()}, max={aff.max()}"
        )

    # Training configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION FOR AFFINITY PREDICTION")
    print("=" * 60)

    training_config = {
        # Model architecture
        "num_classes": 3,  # 3 affinity channels (not classes!)
        "base_channels": 64,
        "input_size": (64, 64, 64),
        # Training
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "volumes_per_batch": 2,
        # Loss function - use AffinityLoss instead of CrossEntropyLoss
        "loss_type": "affinity",  # You'll need to add this option
        # Export
        "export_dir": "/nrs/cellmap/ackermand/dinov3_training/results/affinity_test",
        "experiment_name": "dinov3_unet3d_affinity_mito",
    }

    print("\nKey differences for affinity training:")
    print("  1. Output type: 'affinities' (not 'labels')")
    print("  2. GT shape: (batch, num_offsets, D, H, W) not (batch, D, H, W)")
    print("  3. Model output: 3 channels (affinities) not num_classes")
    print("  4. Loss: AffinityLoss (BCE-based) not CrossEntropyLoss")
    print("  5. GT values: Instance IDs converted to binary affinities (0 or 1)")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("\n1. Modify DINOv3UNet3D to output affinities:")
    print("   - Change final conv to output 3 channels (not num_classes)")
    print("   - Remove softmax (use sigmoid for affinities)")

    print("\n2. Update training loop to use AffinityLoss:")
    print("   - Import: from dinov3_playground.affinity_utils import AffinityLoss")
    print("   - Replace criterion with: criterion = AffinityLoss()")

    print("\n3. Handle GT tensor conversion:")
    print("   - For labels: torch.tensor(batch_gt, dtype=torch.long)")
    print("   - For affinities: torch.tensor(batch_gt, dtype=torch.float32)")

    print("\n4. Post-process predictions:")
    print("   - Apply sigmoid to get probabilities")
    print("   - Use affinities_to_instances() for instance segmentation")
    print("   - Or use mutex watershed for better results")

    print("\n" + "=" * 60)

    # Show sample code for training loop modification
    print("\nSAMPLE CODE - Training loop for affinities:")
    print("-" * 60)
    print(
        """
    # In training loop:
    from dinov3_playground.affinity_utils import AffinityLoss
    
    # Create affinity loss
    criterion = AffinityLoss(use_class_weights=True)
    
    # Convert GT to float (not long)
    train_labels = torch.tensor(train_gt_volumes, dtype=torch.float32).to(device)
    # Shape: (batch, 3, D, H, W)
    
    # Forward pass - logits are affinity predictions
    logits = unet3d(train_features, context_features=train_context_features)
    # Shape: (batch, 3, D, H, W)
    
    # Compute loss
    loss = criterion(logits, train_labels, mask=train_masks_tensor)
    
    # Get probabilities for visualization
    affinity_probs = torch.sigmoid(logits)
    """
    )
    print("-" * 60)

    print("\n✓ Affinity training example setup complete!")
    print("\nTo actually train, you need to:")
    print("  1. Create affinity-compatible training function")
    print("  2. Or modify existing train_dinov3_unet3d_memory_efficient()")
    print("  3. Change model output channels to len(affinity_offsets)")
    print("  4. Use AffinityLoss instead of CrossEntropyLoss/DiceLoss")


if __name__ == "__main__":
    main()
