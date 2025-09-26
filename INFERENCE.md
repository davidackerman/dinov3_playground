# DINOv3 UNet Inference

This module provides easy-to-use classes for loading trained DINOv3 UNet models and running inference on new images or volumes.

## Features

- **Automatic Model Loading**: Automatically loads the best model weights and DINOv3 configuration from training export directories
- **2D and 3D Support**: Separate classes for 2D image and 3D volume inference
- **Batch Processing**: Support for processing multiple images/volumes at once
- **Large Volume Inference**: Sliding window approach for volumes larger than training size
- **Flexible Output**: Option to return predictions only or predictions + probabilities
- **Auto-Detection**: Automatically detect whether a model is 2D or 3D

## Quick Start

### Basic Usage

```python
from dinov3_playground.inference import load_inference_model
import numpy as np

# Load model (automatically detects 2D vs 3D)
model = load_inference_model("/path/to/export/directory")

# For 2D images
image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
predictions = model.predict(image)

# For 3D volumes  
volume = np.random.randint(0, 255, (128, 128, 128), dtype=np.uint8)
predictions = model.predict(volume)
```

### Command Line Usage

```bash
# Run inference on a single image/volume
python examples/quick_inference.py /path/to/export/directory /path/to/input.png

# Save probabilities as well
python examples/quick_inference.py /path/to/export/directory /path/to/input.png --probabilities

# Specify output path
python examples/quick_inference.py /path/to/export/directory /path/to/input.png -o /path/to/output.png
```

## Detailed Usage

### 2D Image Inference

```python
from dinov3_playground.inference import DINOv3UNetInference
from PIL import Image
import numpy as np

# Load the inference model
model = DINOv3UNetInference("/path/to/export/directory")

# Load an image
image = np.array(Image.open("image.png").convert('L'))

# Run inference
predictions = model.predict(image)

# Get predictions and probabilities
predictions, probabilities = model.predict(image, return_probabilities=True)

# Batch inference
batch_images = np.stack([image1, image2, image3])
batch_predictions = model.predict_batch(batch_images)

# Get model information
info = model.get_model_info()
print(f"Model classes: {info['num_classes']}")
```

### 3D Volume Inference

```python
from dinov3_playground.inference import DINOv3UNet3DInference
import zarr

# Load the 3D inference model
model = DINOv3UNet3DInference("/path/to/export/directory")

# Load a volume
volume = zarr.open("volume.zarr", mode='r')[:]

# Run inference on standard volume
predictions = model.predict(volume)

# For large volumes, use sliding window
large_volume = zarr.open("large_volume.zarr", mode='r')[:]
predictions = model.predict_large_volume(
    large_volume, 
    chunk_size=64, 
    overlap=16
)

# Batch inference
batch_volumes = np.stack([volume1, volume2])
batch_predictions = model.predict_batch(batch_volumes)
```

### Auto-Detection

```python
from dinov3_playground.inference import load_inference_model

# Automatically detect model type and load
model = load_inference_model("/path/to/export/directory", model_type='auto')

# The model will be either DINOv3UNetInference or DINOv3UNet3DInference
model_info = model.get_model_info()
print(f"Detected model type: {model_info['model_type']}")
```

## API Reference

### DINOv3UNetInference (2D)

**Constructor:**
- `__init__(export_dir, device=None)`: Load 2D model from export directory

**Methods:**
- `predict(image, return_probabilities=False)`: Run inference on single image
- `predict_batch(images, return_probabilities=False)`: Run inference on batch of images
- `get_model_info()`: Get model configuration information

### DINOv3UNet3DInference (3D)

**Constructor:**
- `__init__(export_dir, device=None)`: Load 3D model from export directory

**Methods:**
- `predict(volume, return_probabilities=False)`: Run inference on single volume
- `predict_large_volume(volume, chunk_size=64, overlap=16, return_probabilities=False)`: Sliding window inference
- `predict_batch(volumes, return_probabilities=False)`: Run inference on batch of volumes
- `get_model_info()`: Get model configuration information

### Utility Functions

- `load_inference_model(export_dir, model_type='auto', device=None)`: Auto-load appropriate model
- `demo_2d_inference(export_dir)`: Demo function for 2D inference
- `demo_3d_inference(export_dir)`: Demo function for 3D inference

## Directory Structure

The inference classes expect the export directory to have this structure:

```
export_directory/
├── timestamp_checkpoints/
│   ├── training_config.json      # Training configuration
│   ├── best_model.pth           # Best model weights
│   └── *.pth                    # Other checkpoints
└── other_files...
```

## Model Configuration

The inference classes automatically reconstruct the model configuration from:

1. **Checkpoint metadata**: `model_config` in the .pth file
2. **Training config**: `training_config.json` file
3. **Fallback defaults**: If neither is available

Key configuration parameters:
- `num_classes`: Number of output classes
- `base_channels`: Base number of channels in UNet
- `input_size`: Expected input dimensions
- `model_id`: DINOv3 model identifier
- `dinov3_slice_size`: DINOv3 processing size (3D only)

## Memory Considerations

### 2D Models
- Input images are processed at full resolution
- Memory usage scales with image size and model depth

### 3D Models
- Standard inference: Loads entire volume into memory
- Large volume inference: Uses sliding window to reduce memory usage
- Batch inference: Memory scales linearly with batch size

### Memory-Saving Tips

```python
# For large 3D volumes, use sliding window
predictions = model.predict_large_volume(
    volume, 
    chunk_size=64,    # Smaller chunks = less memory
    overlap=16        # Ensure smooth boundaries
)

# Process volumes one at a time instead of batches
for volume in volume_list:
    pred = model.predict(volume)
    # Process pred immediately
```

## Error Handling

Common issues and solutions:

1. **Export directory not found**: Check the path to your training results
2. **No checkpoints found**: Ensure training completed and saved checkpoints
3. **CUDA out of memory**: Use smaller chunk sizes or switch to CPU
4. **Import errors**: Ensure dinov3_playground is properly installed

## Examples

See the `examples/` directory for complete working examples:

- `inference_examples.py`: Comprehensive examples showing all features
- `quick_inference.py`: Command-line tool for quick inference

## Integration with Training

The inference classes are designed to work seamlessly with models trained using the memory-efficient training functions:

```python
# After training
training_results = train_3d_unet_with_memory_efficient_loader(...)
export_dir = training_results['checkpoint_dir']

# Load for inference
model = DINOv3UNet3DInference(export_dir)
predictions = model.predict(new_volume)
```