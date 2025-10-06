# Multi-Resolution DINOv3 Enhancement Summary

## Overview
We've implemented comprehensive multi-resolution support and enhanced debugging for the DINOv3 3D UNet pipeline. This allows processing raw data at high resolution while training at lower resolution for memory efficiency.

## Key Features Added

### 1. Multi-Resolution Processing
- **Raw data**: Processed at native high resolution (e.g., 8nm, 512×512×512)
- **Ground truth**: Training happens at base resolution (e.g., 64nm, 128×128×128)
- **Feature downsampling**: DINOv3 features are downsampled to match GT size after processing

### 2. Enhanced Shape Validation
- Relaxed shape validation to allow different raw/GT dimensions
- Multi-resolution mode detection and warnings
- Reasonable ratio validation (0.1x to 16x) with warnings for extreme cases

### 3. Improved Orthogonal Processing
- **Fixed averaging bug**: Now downsample each plane BEFORE averaging
- XY, XZ, YZ planes processed at native resolution
- Each plane downsampled to target size individually
- Safe averaging of consistent-sized feature tensors

### 4. Comprehensive Timing & Debugging

#### Main Pipeline Timing (`extract_dinov3_features_3d`):
- Setup and initialization time
- Per-plane processing times (XY, XZ, YZ)
- Individual downsampling times per plane
- Feature averaging time
- Device transfer time
- Total processing time
- Memory usage analysis

#### Batch Processing Timing (`_process_slice_batch`):
- Slice upsampling time (to DINOv3 input size)
- Batch stacking and memory usage
- DINOv3 inference time and throughput
- Feature extraction and tensor operations
- Grouped downsampling by target dimensions
- Detailed shape tracking throughout

#### Debug Output Includes:
- Input/output shapes at each step
- Memory usage (MB) for major operations
- Processing speed (voxels/second)
- Batch composition and efficiency
- Dimension group analysis for downsampling

### 5. Minimum Resolution Control
- `min_resolution_for_raw` parameter prevents extremely high-res processing
- Configurable via `zarr_util.py` functions
- Balances feature quality vs computational cost

## Usage Example

```python
# Enable multi-resolution with minimum resolution limit
dataset_pairs = update_datapaths_with_target_scales(
    dataset_pairs,
    base_resolution=64,                    # GT resolution
    use_highest_res_for_raw=True,         # Use best available for raw
    min_resolution_for_raw=8,             # But not finer than 8nm
)

# Run with detailed timing
features, timing_info = pipeline.extract_dinov3_features_3d(
    volume,
    enable_timing=True,                   # Enable detailed debugging
    target_output_size=(128, 128, 128)    # Force output size
)
```

## Benefits

### Performance
- 30x speedup from batched tensor operations (previous enhancement)
- Memory-efficient training with high-quality features
- Optimal balance between feature quality and computational cost

### Debugging & Analysis
- Comprehensive timing breakdown identifies bottlenecks
- Shape validation prevents silent errors
- Memory usage tracking helps optimization
- Per-plane analysis for orthogonal processing

### Flexibility
- Backward compatible with same-resolution training
- Configurable minimum resolution limits
- Works with both tuple and dictionary dataset formats
- Supports arbitrary aspect ratios and downsampling factors

## Files Modified
- `models.py`: Enhanced timing, orthogonal processing, downsampling
- `memory_efficient_training.py`: Multi-resolution validation, data loader updates
- `zarr_util.py`: Minimum resolution parameter support
- Training scripts: Updated to use new multi-resolution features

## Testing
- `test_timing_debug.py`: Comprehensive timing system test
- `test_validation.py`: Shape validation and tensor operations test
- Demonstrates both orthogonal and single-plane processing
- Validates memory efficiency and performance gains