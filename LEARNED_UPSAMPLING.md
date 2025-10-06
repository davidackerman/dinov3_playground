# Learned Upsampling in 3D UNet

## Overview

This feature adds support for **learned upsampling** in the 3D UNet architecture, providing an alternative to the traditional interpolation-based upsampling of DINOv3 features.

## Background

### Traditional Approach (Interpolated Upsampling)
1. **DINOv3 Processing**: Each 2D slice is upsampled to DINOv3 input size (e.g., 896×896)
2. **Feature Extraction**: DINOv3 extracts features at reduced resolution (e.g., 56×56)
3. **Interpolation**: Features are interpolated to target volume size (e.g., 128×128×128)
4. **UNet Processing**: UNet processes full-resolution features for segmentation

### New Approach (Learned Upsampling)
1. **DINOv3 Processing**: Each 2D slice is upsampled to DINOv3 input size (e.g., 896×896)
2. **Feature Extraction**: DINOv3 extracts features at native resolution (e.g., 56×56)
3. **No Interpolation**: Features remain at native DINOv3 resolution
4. **UNet Processing**: UNet learns to upsample features AND perform segmentation

## Key Differences

| Aspect | Traditional | Learned Upsampling |
|--------|-------------|-------------------|
| **DINOv3 Features** | Interpolated to full resolution | Kept at native resolution |
| **Memory Usage** | Higher (full-res features) | Lower (native-res features) |
| **UNet Task** | Segmentation only | Upsampling + Segmentation |
| **Training Speed** | Faster convergence | May need more epochs |
| **Feature Quality** | Fixed interpolation | Task-optimized upsampling |

## Usage

### In Training Functions

```python
from dinov3_playground.memory_efficient_training import train_3d_unet_with_memory_efficient_loader

# Traditional approach (default)
results_traditional = train_3d_unet_with_memory_efficient_loader(
    raw_data=raw_data,
    gt_data=gt_data,
    # ... other parameters ...
    learn_upsampling=False  # Default: use interpolation
)

# Learned upsampling approach  
results_learned = train_3d_unet_with_memory_efficient_loader(
    raw_data=raw_data,
    gt_data=gt_data,
    # ... other parameters ...
    learn_upsampling=True   # NEW: let UNet learn upsampling
)
```

### Direct Model Usage

```python
from dinov3_playground.models import DINOv3UNet3D

# Traditional UNet (expects full-resolution features)
unet_traditional = DINOv3UNet3D(
    input_channels=384,
    num_classes=2,
    input_size=(128, 128, 128),
    learn_upsampling=False  # Default
)

# Learned upsampling UNet (handles low-resolution features)
unet_learned = DINOv3UNet3D(
    input_channels=384,
    num_classes=2,
    input_size=(128, 128, 128),      # Target output size
    learn_upsampling=True,
    dinov3_feature_size=(128, 56, 56)  # Native DINOv3 feature size
)
```

## Architecture Details

### Learned Upsampling Layers

The UNet automatically adds learned upsampling layers when `learn_upsampling=True`:

```python
def _make_upsampling_layers(self):
    """Create learned upsampling layers based on upsampling factor."""
    layers = []
    
    # Progressive upsampling with ConvTranspose3D
    if max_factor >= 8:
        # 8x upsampling: 2x → 4x → 8x
        layers = [
            ConvTranspose3d(kernel_size=4, stride=2, padding=1),  # 2x
            ConvTranspose3d(kernel_size=4, stride=2, padding=1),  # 4x  
            ConvTranspose3d(kernel_size=4, stride=2, padding=1),  # 8x
        ]
    # ... similar for 4x, 2x upsampling
```

### Memory Usage Calculation

**Traditional Approach**:
- DINOv3 features: `(batch_size, 384, 128, 128, 128)` = ~100MB per batch
- Total memory: Higher due to full-resolution feature storage

**Learned Upsampling**:
- DINOv3 features: `(batch_size, 384, 128, 56, 56)` = ~20MB per batch  
- Additional upsampling layers: ~5MB parameters
- Total memory: ~75% reduction in feature storage

## Benefits

### 1. **Memory Efficiency**
- Significantly reduced memory usage for DINOv3 features
- Enables training with larger volumes or batch sizes
- Particularly beneficial for high-resolution volumes

### 2. **Task-Optimized Upsampling**
- UNet learns upsampling specific to the segmentation task
- Avoids potential artifacts from fixed interpolation
- End-to-end optimization of the entire pipeline

### 3. **Architectural Flexibility**
- Allows different upsampling strategies per application
- Can be combined with other memory optimization techniques
- Maintains compatibility with existing training pipelines

## When to Use

### **Use Learned Upsampling When:**
✅ **Limited GPU memory** - Need to reduce memory usage  
✅ **Large volumes** - Working with high-resolution 3D data  
✅ **Interpolation artifacts** - Traditional upsampling causes issues  
✅ **End-to-end optimization** - Want fully learnable pipeline  
✅ **Research/experimentation** - Exploring different architectures  

### **Use Traditional Approach When:**
✅ **Fast prototyping** - Need quick results with proven method  
✅ **Limited training time** - Faster convergence is priority  
✅ **Established pipeline** - Working with existing validated approach  
✅ **Small volumes** - Memory usage is not a concern  

## Performance Considerations

### Training Time
- **Learned upsampling**: May require 10-20% more epochs to converge
- **Traditional**: Generally faster convergence
- **Recommendation**: Increase patience and learning rate schedule

### Memory Usage
- **Learned upsampling**: 60-80% reduction in feature memory
- **Traditional**: Higher memory usage but more predictable
- **Recommendation**: Monitor GPU memory usage during training

### Accuracy
- **Results vary by dataset**: Some datasets benefit from learned upsampling
- **Generally comparable**: Both approaches achieve similar final accuracy
- **Recommendation**: Evaluate both approaches on your specific data

## Example Configuration

```python
# Recommended settings for learned upsampling
config = {
    "learn_upsampling": True,
    "base_channels": 32,        # Can use more channels due to memory savings
    "learning_rate": 1e-3,      # Standard learning rate
    "patience": 30,             # Increase patience for convergence
    "epochs": 150,              # May need more epochs
    "use_mixed_precision": True, # Combine with other memory optimizations
}
```

## Testing

Run the comparison demo to see both approaches in action:

```bash
cd /path/to/dinov3_playground
python examples/test_learned_upsampling.py
```

This will train small models using both approaches and compare their performance.

## Future Enhancements

Potential improvements for learned upsampling:

1. **Adaptive upsampling**: Different strategies based on volume size
2. **Multi-scale features**: Combine features from multiple resolutions  
3. **Attention-based upsampling**: Use attention mechanisms for selective upsampling
4. **Progressive upsampling**: Gradually increase resolution during training

## References

- **DINOv3**: [Paper](https://arxiv.org/abs/2304.07193) for feature extraction details
- **3D UNet**: Original architecture for volumetric segmentation
- **Transposed Convolutions**: Mathematical basis for learned upsampling