#!/usr/bin/env python3
"""
GT Extension Feature Summary

This document summarizes the GT extension functionality added to the DINOv3 3D UNet training pipeline.

MOTIVATION:
- Ground truth annotations often only cover subvolumes of the full raw data
- Training with larger raw context can improve predictions
- Need to mask losses to only consider predictions within valid GT regions

IMPLEMENTATION:

1. DATA LOADING (data_processing.py):
   - Added `allow_gt_extension=False` parameter to `load_random_3d_training_data()`
   - When enabled, allows sampling raw volumes that extend beyond GT bounds
   - Uses funlib.geometry.Roi.intersect() to calculate valid GT regions
   - Returns GT masks indicating where predictions should be evaluated

2. MEMORY EFFICIENT LOADER (memory_efficient_training.py):
   - Updated MemoryEfficientDataLoader3D to handle GT masks
   - Added `gt_masks` parameter to constructor
   - Modified sample_training_batch() and get_validation_data() to return masks
   - Creates full masks (all 1s) when masks not provided for backward compatibility

3. USAGE EXAMPLE:

```python
# Enable GT extension during data loading
raw, gt, gt_masks, dataset_sources, num_classes = load_random_3d_training_data(
    dataset_pairs=dataset_pairs,
    volume_shape=(128, 128, 128),
    base_resolution=base_resolution,
    allow_gt_extension=True,  # <-- NEW PARAMETER
    # ... other parameters
)

# Pass masks to data loader
data_loader_3d = MemoryEfficientDataLoader3D(
    raw_data=raw,
    gt_data=gt,
    gt_masks=gt_masks,  # <-- NEW PARAMETER
    # ... other parameters
)

# During training, use masks to weight losses
train_volumes, train_gt, train_masks = data_loader_3d.sample_training_batch(batch_size)

# Apply mask to loss calculation
loss = criterion(predictions, train_gt)
masked_loss = loss * train_masks.float()  # Zero out loss outside GT regions
final_loss = masked_loss.sum() / train_masks.sum()  # Normalize by valid voxels
```

4. MASK PROPERTIES:
   - Shape: (num_volumes, D, H, W) - same as GT data
   - Values: 1 where GT is valid, 0 where extended (no GT available)
   - Use for: Loss masking, metric calculation, visualization

5. BENEFITS:
   - Larger contextual information for better predictions
   - No penalty for predictions in regions without ground truth
   - Maintains training stability while using extended raw data
   - Backward compatible (works without masks)

TESTING:
- Run `python test_gt_extension.py` to verify functionality
- Check that masks are binary (0/1) and properly sized
- Verify both modes work (with/without GT extension)
"""

if __name__ == "__main__":
    print(__doc__)
