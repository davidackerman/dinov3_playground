# Random Crop Feature Extraction

Simple tool to extract DINOv3 features from random crops of a dataset.

## Overview

This is a **much simpler** alternative to the standard preprocessing pipeline. Instead of dealing with organelles, ground truth, filtering, etc., this just:
1. Takes a dataset path (raw data)
2. Randomly samples a crop from it
3. Extracts DINOv3 features
4. Saves features to zarr

**No ground truth, no organelles, no filters - just raw → features.**

## Files

- **`dinov3_preprocessing_random_crops.py`**: Core functions
  - `extract_dinov3_features_3d_standalone()`: Extract features without a dataloader
  - `preprocess_and_save_random_crop()`: Main function
- **`preprocess_dataset_random_crops.py`**: CLI script for single crop
- **`submit_random_crop_jobs.py`**: Batch job submission for parallel processing

## Usage

### Single Crop

```bash
python preprocess_dataset_random_crops.py \
    --crop-index 0 \
    --output-dir /path/to/output \
    --dataset-path /path/to/raw.zarr \
    --output-image-dim 128 \
    --use-anyup
```

### Batch Processing (LSF)

```bash
# Dry run first
python submit_random_crop_jobs.py \
    --output-dir /path/to/output \
    --dataset-path /path/to/raw.zarr \
    --num-crops 300 \
    --output-image-dim 128 \
    --use-anyup \
    --dry-run

# Actually submit
python submit_random_crop_jobs.py \
    --output-dir /path/to/output \
    --dataset-path /path/to/raw.zarr \
    --num-crops 300 \
    --output-image-dim 128 \
    --use-anyup
```

## Key Parameters

### Required
- `--crop-index`: Crop index (used as random seed)
- `--output-dir`: Where to save output
- `--dataset-path`: Path to raw dataset
- `--num-crops`: (batch only) Number of crops to process

### Model
- `--model-id`: DINOv3 model (default: `facebook/dinov3-vitl16-pretrain-sat493m`)
- `--input-resolution`: Resolution of input data in nm (default: 32)
- `--output-resolution`: Target resolution in nm (default: 128)
- `--output-image-dim`: Crop size (default: 128, makes 128³ cube)

### Feature Extraction
- `--use-anyup`: Use AnyUp (default: True)
- `--no-anyup`: Disable AnyUp

### Storage
- `--compression`: Use LZ4 compression
- `--no-save-raw`: Only save features, not raw data
- `--num-threads`: TensorStore threads (default: 8)

### LSF (batch only)
- `--num-processors`: CPUs per job (default: 16)
- `--memory-gb`: Memory per job (default: 64)
- `--no-gpu`: Don't request GPU
- `--queue`: LSF queue name
- `--walltime`: Time limit (default: "2:00")

## Output

Each crop generates:

**Zarr file**: `crop_{index:06d}.zarr/`
- `features`: DINOv3 features (C, D, H, W) as float16
- `raw`: (optional) Raw data (D, H, W) as uint8

**Metadata file**: `crop_{index:06d}_metadata.json`
- Source dataset path and ROI
- Configuration (model, resolutions, etc.)
- Timing and storage stats

## Example

```bash
# Extract features from 500 random crops
python submit_random_crop_jobs.py \
    --output-dir /scratch/my_features \
    --dataset-path /data/em/sample1/raw.zarr \
    --num-crops 500 \
    --output-image-dim 128 \
    --input-resolution 32 \
    --output-resolution 128 \
    --use-anyup \
    --compression

# Monitor
bjobs -P cellmap | grep dinov3_crop

# Check completion
ls /scratch/my_features/*_metadata.json | wc -l
```

## How It Works

1. **Random Sampling**: Uses `ImageDataInterface` to load the dataset and randomly sample a crop based on `crop_index` as seed
2. **Feature Extraction**: Creates a `DINOv3UNet3DPipeline` to extract features (no dataloader needed!)
3. **Storage**: Writes features (and optionally raw) to TensorStore zarr

## Differences from Standard Preprocessing

| Standard Preprocessing | This Tool |
|------------------------|-----------|
| Organelle pairs, filtering | Just a dataset path |
| Ground truth, masks, affinities | Only raw data |
| Complex data loading | Simple ImageDataInterface |
| Many parameters | Minimal parameters |
| `volume_{index}.zarr` | `crop_{index}.zarr` |

Use this when you just want features from raw data without all the training infrastructure!
