# DINOv3 Playground

A modular package for DINOv3 feature extraction and fine-tuning on cell image analysis tasks.

## Description

This package contains modularized functions for DINOv3 feature extraction and training, including:

- **dinov3_core**: Core DINOv3 processing functions
- **data_processing**: Data sampling and augmentation functions  
- **models**: Neural network model classes (2D/3D UNet, classifiers)
- **model_training**: Training and class balancing functions
- **visualization**: Plotting and visualization functions
- **memory_efficient_training**: Memory-efficient training system
- **zarr_util**: Utilities for working with Zarr datasets

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

```python
import dinov3_playground

# Initialize DINOv3 model
dinov3_playground.initialize_dinov3()

# Process images with DINOv3
features = dinov3_playground.process(image_data)

# Create and train models
model = dinov3_playground.create_model('unet', num_classes=2)
```

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