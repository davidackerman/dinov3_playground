# DINOv3 Playground

A modular package for DINOv3 feature extraction and fine-tuning on cell image analysis tasks.

## Description

This package contains modularized functions for DINOv3 feature extraction and training, including:

- **dinov3_core**: Core DINOv3 processing functions
- **data_processing**: Data sampling and augmentation functions with **multi-class support**
- **models**: Neural network model classes (2D/3D UNet, classifiers)
- **model_training**: Training and class balancing functions
- **visualization**: Plotting and visualization functions
- **memory_efficient_training**: Memory-efficient training system with **automatic class detection**
- **zarr_util**: Utilities for working with Zarr datasets
- **inference**: Automated model loading and inference system

## Key Features

✅ **Multi-class segmentation support** - Train models with multiple semantic classes  
✅ **Automatic class detection** - No need to manually specify number of classes  
✅ **Dictionary-based dataset format** - Flexible specification of multiple ground truth labels  
✅ **Backward compatibility** - Legacy tuple format still supported  
✅ **Memory-efficient 3D training** - Optimized for large volumetric data  
✅ **Automated inference system** - Load trained models with a single function call

## Installation

### From Source (Editable Mode)

To install the package in editable mode for development:

```bash
# Clone the repository
git clone https://github.com/davidackerman/dinov3_playground.git
cd dinov3_playground

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install Dependencies Only

If you just want to install the required dependencies:

```bash
pip install -r requirements.txt
```

For development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Usage

### Basic Multi-Class Training

```python
from dinov3_playground.data_processing import load_random_3d_training_data
from dinov3_playground.memory_efficient_training import train_3d_unet_with_memory_efficient_loader

# Define multi-class datasets with descriptive names
dataset_pairs = [
    {
        "raw": "/path/to/raw_em.zarr",
        "nuc": "/path/to/nuclei_labels.zarr",
        "mito": "/path/to/mitochondria_labels.zarr",
        "vesicles": "/path/to/vesicles_labels.zarr",
    },
    # Add more datasets...
]

# Load data with automatic class detection
raw, gt, sources, num_classes = load_random_3d_training_data(
    dataset_pairs=dataset_pairs,
    volume_shape=(64, 64, 64),
    num_volumes=20,
)

# Train with automatic class detection
results = train_3d_unet_with_memory_efficient_loader(
    raw_data=raw,
    gt_data=gt,
    num_classes=num_classes,  # Auto-detected
    export_base_dir="/path/to/results",
)
```

### Automated Inference

```python
from dinov3_playground.inference import load_inference_model

# Load trained model automatically
model = load_inference_model("/path/to/export/directory")

# Run inference
prediction = model.predict(your_volume)
```

See [MULTICLASS_SUPPORT.md](MULTICLASS_SUPPORT.md) for detailed documentation.

## Troubleshooting

### Zarr/TensorStore Compatibility Issues

If you encounter errors like `FAILED_PRECONDITION: Error opening 'zarr' driver... Object includes extra members: 'checksum'`, use the provided diagnostic tools:

```bash
# Check for compatibility issues
python3 zarr_diagnostics.py /path/to/your/zarr/array

# Fix problematic Zarr files
python3 zarr_fix.py /path/to/problematic/zarr --fix
```

The training system automatically skips problematic datasets and continues with available ones.

## Dependencies

- torch>=1.9.0
- numpy
- pillow  
- transformers>=4.20.0
- matplotlib
- scikit-image
- scipy
- zarr
- funlib.geometry
- cellmap_flow

## Development

To set up for development:

```bash
git clone https://github.com/davidackerman/dinov3_playground.git
cd dinov3_playground
pip install -e ".[dev]"
```

This will install the package in editable mode with all development dependencies.

## License

See LICENSE file for details.