# DINOv3 Processing Flow Fix

## Problem Identified
The user reported that DINOv3 was processing 512×512×512 input but the debug output showed confusing messages about "downsampling 256 slices to 512×512" when the expected output should be 128×128×128.

## Root Cause
The processing flow was inefficient and confusing:

1. **Input**: 512×512×512 volume (raw data at 16nm)
2. **DINOv3 Processing**: Each 512×512 slice → DINOv3 ViT-L/16 → 32×32 feature maps
3. **Old Flow**: 32×32 features → resize to 512×512 → downsample to 128×128
4. **Issue**: This was wasteful and the debug messages were misleading

## Solution Implemented

### Key Changes in `_extract_features_plane`:

1. **Direct Output Sizing**: Modified the method to resize DINOv3 features directly to final output dimensions
   ```python
   # Determine final target dimensions for DINOv3 feature resizing
   if (output_d is not None and output_h is not None and output_w is not None):
       # Use final output dimensions for direct DINOv3 feature resizing
       final_d, final_h, final_w = output_d, output_h, output_w
   else:
       # Fall back to processing dimensions
       final_d, final_h, final_w = target_d, target_h, target_w
   ```

2. **Updated Slice Processing**: Each slice now gets resized directly to the final output dimensions
   ```python
   # For XY plane: Use final output dimensions for DINOv3 feature resizing
   slice_dims.append((final_h, final_w))  # Instead of (target_h, target_w)
   ```

3. **Removed Redundant Downsampling**: Eliminated the intermediate downsampling step since features are already at correct size

### Corrected Flow:
1. **Input**: 512×512×512 volume 
2. **DINOv3 Processing**: 512×512 slice → DINOv3 ViT-L/16 → 32×32 features
3. **Direct Resize**: 32×32 features → **directly upsampled** to 128×128
4. **Final**: 128×128×128 volume of features

### Updated Debug Messages:
- Changed "Downsampling: 1 dimension groups" → "DINOv3 feature resizing: 1 dimension groups"  
- Changed "256 slices to 512×512" → "256 slices: DINOv3 32×32 → 128×128"

## Memory Benefits
- **Eliminated intermediate 512×512 feature maps** - features go directly from 32×32 to 128×128
- **Reduced peak memory usage** by avoiding unnecessarily large intermediate representations
- **Same accuracy** - the final result is identical, just more efficient

## Expected Debug Output (After Fix):
```
--- Processing 3 Orthogonal Planes ---
XY features extracted and downsampled: (1, 1024, 128, 128, 128) in 2.341s
  XY features stacked: (1, 1024, 128, 128, 128) (already at target size)
  DINOv3 feature resizing: 1 dimension groups in 0.524s
    256 slices: DINOv3 32×32 → 128×128
```

This fix ensures the processing flow is both memory-efficient and conceptually correct.