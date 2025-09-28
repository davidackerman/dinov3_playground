# Multi-Class Training Support in DINOv3 Playground

## Ove### 4. Multi-Class Ground Truth Generation

The system creates proper multi-class ground truth volumes where:
- **Class 0**: Background (default)
- **Class 1, 2, 3, ...**: Your labeled classes (assigned alphabetically by class name)

**Class Assignment Rules:**
- Classes are sorted **alphabetically** by key name
- Higher-numbered classes override lower ones in overlapping regions
- Example: {"nuc": ..., "mito": ..., "vesicles": ...} → mito=1, nuc=2, vesicles=3The DINOv3 playground now supports multi-class training with an improved dataset format that allows for multiple semantic classes per dataset. This update provides backward compatibility with the legacy tuple format while introducing a more flexible dictionary-based format.

## New Features

### 1. Dictionary-Based Dataset Format

Instead of simple tuples `(raw_path, gt_path)`, you can now use dictionaries with multiple class labels:

```python
dataset_pairs = [
    {
        "raw": "/path/to/raw_image.zarr",
        "class_1": "/path/to/nuclei_labels.zarr",
        "class_2": "/path/to/mitochondria_labels.zarr", 
        "class_3": "/path/to/vesicles_labels.zarr",
    },
    # Add more datasets...
]
```

### 2. Flexible Class Naming

You can use any descriptive names for your classes - no need to follow a specific format:

```python
dataset_pairs = [
    {
        "raw": "/path/to/em_data.zarr",
        "nuc": "/path/to/nuclei.zarr",           # ✅ Descriptive names
        "mito": "/path/to/mitochondria.zarr",    # ✅ Any name works
        "vesicles": "/path/to/vesicles.zarr",    # ✅ No "class_N" required
        "organelles": "/path/to/other.zarr",     # ✅ Multiple words OK
    }
]
```

**Key Points:**
- ✅ **Any key name works** (except "raw" which is reserved)
- ✅ **Descriptive names encouraged**: "nuc", "mito", "vesicles" vs "class_1", "class_2"
- ✅ **Alphabetical ordering**: Class numbers assigned alphabetically (mito=1, nuc=2, vesicles=3)
- ✅ **Mixed naming OK**: Can combine "class_1" with "nuclei" in same dataset

### 3. Automatic Class Detection

The system automatically detects the number of classes from your dataset:

```python
# Number of classes is automatically determined
raw, gt, dataset_sources, num_classes = load_random_3d_training_data(
    dataset_pairs=dataset_pairs,
    # ... other parameters
)
print(f"Detected {num_classes} classes")  # Includes background class (class 0)
```

### 3. Multi-Class Ground Truth Generation

The system creates proper multi-class ground truth volumes where:
- **Class 0**: Background (default)
- **Class 1, 2, 3, ...**: Your labeled classes

Higher-numbered classes take priority when regions overlap.

### 4. Backward Compatibility

Legacy tuple format is still supported and automatically converted:

```python
# Legacy format (still works)
dataset_pairs = [
    ("/path/to/raw1.zarr", "/path/to/gt1.zarr"),
    ("/path/to/raw2.zarr", "/path/to/gt2.zarr"),
]
# Automatically converted to {"raw": ..., "class_1": ...} format
```

## Usage Examples

### Basic Multi-Class Setup

```python
# Define datasets with multiple classes using descriptive names
dataset_pairs = [
    {
        "raw": "/nrs/cellmap/data/sample1/em.zarr",
        "nuc": "/nrs/cellmap/data/sample1/nuclei.zarr",
        "mito": "/nrs/cellmap/data/sample1/mitochondria.zarr",
    },
    {
        "raw": "/nrs/cellmap/data/sample2/em.zarr", 
        "nuc": "/nrs/cellmap/data/sample2/nuclei.zarr",
        "mito": "/nrs/cellmap/data/sample2/mitochondria.zarr",
        "vesicles": "/nrs/cellmap/data/sample2/vesicles.zarr",
    },
]

# Load data (automatically detects 4 classes: 0=background, 1=mito, 2=nuc, 3=vesicles)
# Class numbers are assigned alphabetically by key name
raw, gt, sources, num_classes = load_random_3d_training_data(
    dataset_pairs=dataset_pairs,
    volume_shape=(64, 64, 64),
    base_resolution=64,
    num_volumes=10,
)

# Train model (num_classes is auto-detected)
training_results = train_3d_unet_with_memory_efficient_loader(
    raw_data=raw,
    gt_data=gt,
    num_classes=num_classes,  # Or omit - will be auto-detected
    # ... other parameters
)
```

### Mixed Format Support

```python
# You can mix legacy and new formats, and use any descriptive class names
dataset_pairs = [
    # Legacy format
    ("/path/to/raw1.zarr", "/path/to/binary_labels1.zarr"),
    
    # New multi-class format with descriptive names
    {
        "raw": "/path/to/raw2.zarr",
        "nuclei": "/path/to/nuclei2.zarr", 
        "mitochondria": "/path/to/mito2.zarr",
        "vesicles": "/path/to/vesicles2.zarr",
    },
    
    # Mixed naming conventions also work
    {
        "raw": "/path/to/raw3.zarr",
        "class_1": "/path/to/labels3.zarr",
        "organelles": "/path/to/organelles3.zarr",
    },
]
```

## Key Functions

### `convert_dataset_pairs_format(dataset_pairs)`
Converts any dataset format to standardized dictionary format.

### `get_num_classes_from_dataset_pairs(dataset_pairs)`
Determines the total number of classes (including background) from dataset pairs.

### `load_random_3d_training_data(...)`
Updated to return `(raw, gt, sources, num_classes)` tuple with multi-class support.

## Class Assignment Logic

When creating multi-class ground truth:

1. **Initialize** with background (class 0)
2. **Process classes in order** (class_1, class_2, class_3, ...)
3. **Higher classes override lower classes** in overlapping regions
4. **Final result**: Each voxel assigned to exactly one class

Example:
```python
# If a voxel is labeled in both class_1 and class_2:
# Final assignment = class_2 (higher priority)

gt_volume = np.zeros(shape, dtype=np.uint8)  # Start with background
gt_volume[class_1_mask] = 1  # Assign class 1
gt_volume[class_2_mask] = 2  # Class 2 overrides class 1 where they overlap
```

## Training Benefits

### Automatic Class Weighting
The training system automatically computes class weights based on the actual distribution:

```python
# Automatically handles imbalanced classes
training_results = train_3d_unet_with_memory_efficient_loader(
    raw_data=raw,
    gt_data=gt,
    use_class_weighting=True,  # Recommended for multi-class
    # ...
)
```

### Per-Class Metrics
Training tracks accuracy for each class individually:

```python
# Access per-class training metrics
train_class_accs = training_results["train_class_accs"]  # Shape: (epochs, num_classes)
val_class_accs = training_results["val_class_accs"]      # Shape: (epochs, num_classes)
```

## Visualization Updates

All visualization functions now properly handle multiple classes:

- **Ground truth**: Uses `cmap="tab10"` with proper `vmin=0, vmax=num_classes-1`
- **Predictions**: Shows all classes with consistent color mapping
- **Probabilities**: Can visualize any class probability channel

## Migration Guide

### From Legacy Format

**Before:**
```python
dataset_pairs = [
    ("/path/to/raw.zarr", "/path/to/labels.zarr"),
]

raw, gt, sources = load_random_3d_training_data(dataset_pairs, ...)
training_results = train_3d_unet_with_memory_efficient_loader(
    raw_data=raw, gt_data=gt, num_classes=2, ...
)
```

**After:**
```python
# Option 1: Keep legacy format (automatic conversion)
dataset_pairs = [
    ("/path/to/raw.zarr", "/path/to/labels.zarr"),
]

# Option 2: Use new format for clarity
dataset_pairs = [
    {
        "raw": "/path/to/raw.zarr",
        "class_1": "/path/to/labels.zarr",
    }
]

raw, gt, sources, num_classes = load_random_3d_training_data(dataset_pairs, ...)
training_results = train_3d_unet_with_memory_efficient_loader(
    raw_data=raw, gt_data=gt, num_classes=num_classes, ...  # Auto-detected
)
```

## Error Handling

The system provides clear error messages for common issues:

- **Missing 'raw' key**: All datasets must have a 'raw' field
- **No class keys**: Must have at least one 'class_N' field  
- **Shape mismatches**: All class volumes must match in size
- **Invalid formats**: Clear messages for unsupported dataset formats

## Testing

Run the test suite to verify multi-class functionality:

```bash
cd /groups/cellmap/cellmap/ackermand/Programming/dinov3_playground
python3 test_multiclass_support.py
```

## Troubleshooting

### TensorStore/Zarr Compatibility Issues

Some Zarr files may contain metadata that is incompatible with TensorStore, causing errors like:
```
FAILED_PRECONDITION: Error opening 'zarr' driver... Object includes extra members: 'checksum'
```

#### Diagnostic Tools

Use the diagnostic utility to check your datasets:

```python
# Check a single Zarr file
python3 zarr_diagnostics.py /path/to/your/zarr/array

# Check all files in your dataset pairs
from zarr_diagnostics import diagnose_dataset_pairs
diagnose_dataset_pairs(your_dataset_pairs)
```

#### Fixing Compatibility Issues

Option 1: Fix in place (modifies original files):
```bash
python3 zarr_fix.py /path/to/problematic/zarr --fix
```

Option 2: Create compatible copies:
```bash
python3 zarr_fix.py /path/to/problematic/zarr --copy --output /path/to/fixed/zarr
```

#### Automatic Handling

The training system automatically skips problematic datasets and continues with available ones. You'll see messages like:
```
Dataset 5: TensorStore compatibility issue - 'checksum' field in compressor not supported
Skipping dataset 5 and continuing with remaining datasets...
```

### Common Issues and Solutions

1. **"Dataset pairs format not recognized"**
   - Ensure your dataset pairs follow the dictionary format with 'raw' key
   - Use `convert_dataset_pairs_format()` to check format compatibility

2. **"No valid datasets found"**
   - Check that all paths in your dataset pairs exist
   - Verify Zarr arrays are valid using the diagnostic tool

3. **"Class names must be strings"**
   - Ensure all non-'raw' keys in your dataset dictionaries are strings

4. **"TensorStore Error: FAILED_PRECONDITION"**
   - Use the zarr diagnostic and fix utilities provided
   - Consider using alternative data loaders for problematic files

This comprehensive update maintains full backward compatibility while enabling powerful multi-class semantic segmentation workflows.