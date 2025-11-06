# DINOv3 Feature Preprocessing with TensorStore

This system preprocesses 3D volumes by extracting DINOv3 features and caching them using TensorStore for fast parallel I/O during training.

## Overview

**Problem**: Extracting DINOv3 features during training is slow (~2-5 seconds per volume), creating a bottleneck.

**Solution**: Pre-extract features once and cache them in TensorStore format, enabling:
- **~100-1000x faster loading** during training (~10-50ms vs 2-5s)
- **Parallel I/O** with configurable thread count
- **Optional compression** to save disk space
- **Parallel preprocessing** across volumes using LSF

## Files

- `dinov3_preprocessing.py` - Core preprocessing functions
- `preprocess_volume.py` - CLI script to preprocess a single volume
- `submit_preprocessing_jobs.py` - Submit batch jobs to LSF
- `check_preprocessing.py` - Monitor progress and verify outputs
- `preprocessed_dataloader.py` - PyTorch DataLoader for preprocessed volumes

## Quick Start

### 1. Preprocess volumes (parallel on LSF)

```bash
# Submit 100 jobs to preprocess volumes 0-99
python submit_preprocessing_jobs.py \
    --output-dir /path/to/preprocessed_volumes \
    --num-volumes 100 \
    --num-processors 16 \
    --memory-gb 64 \
    --organelles cell \
    --walltime 2:00

# Monitor jobs
bjobs -P cellmap | grep dinov3_preprocess

# Check progress
python check_preprocessing.py --output-dir /path/to/preprocessed_volumes
```

### 2. Use preprocessed volumes in training

```python
from preprocessed_dataloader import create_preprocessed_dataloader

# Create DataLoader
train_loader = create_preprocessed_dataloader(
    preprocessed_dir="/path/to/preprocessed_volumes",
    batch_size=4,
    shuffle=True,
    num_threads=8,  # TensorStore parallelism
    num_workers=0,  # Use TensorStore's internal parallelism
)

# Training loop
for features, gt, mask in train_loader:
    # features: (B, C, D, H, W) - already extracted DINOv3 features
    # gt: (B, D, H, W) - ground truth labels
    # mask: (B, D, H, W) - valid GT regions (or None)
    
    # Your training code here...
    pass
```

## Detailed Usage

### Single Volume Preprocessing

Process a single volume (useful for testing):

```bash
python preprocess_volume.py \
    --volume-index 0 \
    --output-dir /path/to/output \
    --organelles cell \
    --base-resolution 128 \
    --min-resolution-for-raw 32 \
    --output-image-dim 128 \
    --num-threads 8
```

**Arguments:**
- `--volume-index`: Volume index (also used as random seed)
- `--output-dir`: Where to save preprocessed data
- `--organelles`: List of organelles (default: cell)
- `--inference-filter`: Dataset filters (e.g., `jrc_mus-liver-zon-1`)
- `--compression`: Enable LZ4 compression (slower but smaller)
- `--num-threads`: TensorStore parallelism (default: 8)

### Batch Job Submission

Submit many preprocessing jobs in parallel:

```bash
python submit_preprocessing_jobs.py \
    --output-dir /nrs/cellmap/ackermand/to_delete/jasdafadf/preprocessed_volumes \
    --num-volumes 10 \
    --start-index 0 \
    --num-processors 16 \
    --memory-gb 64 \
    --walltime 2:00 \
    --organelles cell nucleus \
    --gpu False
    --inference-filter jrc_mus-liver-zon-1 jrc_mus-kidney-1
```

**LSF Configuration:**
- `--num-processors`: CPUs per job (default: 16)
- `--memory-gb`: Memory per job (default: 64)
- `--walltime`: Time limit (default: 2:00 = 2 hours)
- `--queue`: Queue name (optional)
- `--no-gpu`: Disable GPU request
- `--dry-run`: Print commands without submitting

**Data Configuration:**
- `--organelles`: List of organelles to process
- `--inference-filter`: Dataset filters
- `--min-label-fraction`: Minimum label fraction (default: 0.01)
- `--min-unique-ids`: Minimum unique IDs (default: 2)
- `--min-ground-truth-fraction`: Minimum GT fraction (default: 0.05)
- `--compression`: Enable compression (saves ~60% disk space but slower reads)

### Monitoring Progress

Check preprocessing status:

```bash
# Full report
python check_preprocessing.py --output-dir /path/to/preprocessed_volumes

# Also verify volumes by loading them
python check_preprocessing.py \
    --output-dir /path/to/preprocessed_volumes \
    --verify \
    --sample-size 5

# Verify specific volumes
python check_preprocessing.py \
    --output-dir /path/to/preprocessed_volumes \
    --verify \
    --verify-indices 0 1 2 3 4
```

The report shows:
- Number of completed volumes
- Missing volume indices (gaps)
- Timing statistics (extraction, write, total)
- Storage usage (total, per volume, breakdown)
- Dataset sources
- Configuration used
- Job logs and errors

### Loading Preprocessed Volumes

#### Basic Usage

```python
from preprocessed_dataloader import PreprocessedDINOv3Dataset

# Create dataset
dataset = PreprocessedDINOv3Dataset(
    preprocessed_dir="/path/to/preprocessed_volumes",
    num_threads=8,
)

# Access a volume
features, gt, mask = dataset[0]
print(f"Features: {features.shape}")  # (C, D, H, W)
print(f"GT: {gt.shape}")              # (D, H, W)
print(f"Mask: {mask.shape}")          # (D, H, W) or None
```

#### DataLoader Usage

```python
from preprocessed_dataloader import create_preprocessed_dataloader

# Create DataLoader
loader = create_preprocessed_dataloader(
    preprocessed_dir="/path/to/preprocessed_volumes",
    batch_size=4,
    shuffle=True,
    num_threads=8,
)

# Iterate
for batch_idx, (features, gt, mask) in enumerate(loader):
    # features: (B, C, D, H, W)
    # gt: (B, D, H, W)
    # mask: (B, D, H, W) or None
    
    # Training code...
    pass
```

#### Subset of Volumes

```python
# Use only specific volumes
loader = create_preprocessed_dataloader(
    preprocessed_dir="/path/to/preprocessed_volumes",
    volume_indices=[0, 1, 2, 5, 10],  # Only use these volumes
    batch_size=2,
    shuffle=True,
)
```

#### With Metadata

```python
dataset = PreprocessedDINOv3Dataset(
    preprocessed_dir="/path/to/preprocessed_volumes",
    return_metadata=True,
)

features, gt, mask, metadata = dataset[0]
print(f"Source dataset: {metadata['source_dataset']['paths']}")
print(f"Volume index: {metadata['volume_index']}")
print(f"Feature extraction time: {metadata['timing']['feature_extraction_seconds']:.2f}s")
```

### Benchmark Loading Speed

```bash
python preprocessed_dataloader.py \
    --preprocessed-dir /path/to/preprocessed_volumes \
    --batch-size 4 \
    --num-batches 20 \
    --num-threads 8
```

Expected output:
```
First batch (cold load):
  Features shape: torch.Size([4, 1024, 128, 128, 128])
  GT shape: torch.Size([4, 128, 128, 128])
  Mask shape: torch.Size([4, 128, 128, 128])
  Load time: 1523.4ms

Batch 2: 45.2ms
Batch 3: 42.8ms
...

Benchmark Results:
Cold load (first batch): 1523.4ms
Warm loads (cached):
  Mean: 43.5ms
  Std: 2.1ms
  Min: 40.2ms
  Max: 48.3ms

Speedup vs cold: 35.0x
```

## Output Structure

Each preprocessed volume creates 4 files:

```
preprocessed_volumes/
├── volume_000000_features.zarr/    # DINOv3 features (largest file)
├── volume_000000_gt.zarr/          # Ground truth labels
├── volume_000000_mask.zarr/        # Valid GT mask (if GT extension enabled)
├── volume_000000_metadata.json     # Metadata and provenance
├── volume_000001_features.zarr/
├── volume_000001_gt.zarr/
├── volume_000001_mask.zarr/
├── volume_000001_metadata.json
├── ...
└── logs/
    ├── volume_000000.out           # Job stdout
    ├── volume_000000.err           # Job stderr
    └── ...
```

### Metadata Format

`volume_XXXXXX_metadata.json` contains:

```json
{
  "volume_index": 0,
  "volume_name": "volume_000000",
  "timestamp": "2025-01-15T10:30:00",
  "paths": {
    "features": "/.../volume_000000_features.zarr",
    "gt": "/.../volume_000000_gt.zarr",
    "mask": "/.../volume_000000_mask.zarr"
  },
  "source_dataset": {
    "index": 42,
    "paths": {"raw": "s3://...", "cell": "s3://..."}
  },
  "shapes": {
    "raw": [512, 128, 128],
    "features": [1024, 128, 128, 128],
    "gt": [128, 128, 128],
    "mask": [128, 128, 128]
  },
  "configuration": {
    "model_id": "facebook/dinov3-vitl16-pretrain-sat493m",
    "base_resolution": 128,
    "compression": "none",
    "chunks": [256, 64, 64, 64]
  },
  "timing": {
    "feature_extraction_seconds": 2.34,
    "tensorstore_write_seconds": 0.15,
    "total_seconds": 2.49
  },
  "storage": {
    "features_gb": 4.295,
    "gt_gb": 0.002,
    "mask_gb": 0.002,
    "total_gb": 4.299
  },
  "statistics": {
    "num_classes": 2,
    "unique_classes": [0, 1],
    "valid_gt_fraction": 0.876
  }
}
```

## Performance

### Without Compression (Recommended for Training)

- **Write**: ~0.15s per volume
- **Read (cold)**: ~1.5s per volume
- **Read (warm/cached)**: ~10-50ms per volume
- **Size**: ~4.3 GB per volume (128³ at 1024 channels)
- **Speedup**: ~50-200x vs on-the-fly extraction

### With Compression (LZ4 level 3)

- **Write**: ~2.5s per volume
- **Read (cold)**: ~1.6-2s per volume
- **Read (warm/cached)**: ~40-100ms per volume
- **Size**: ~1.7 GB per volume (~60% compression)
- **Speedup**: ~30-100x vs on-the-fly extraction

**Recommendation**: Use no compression for maximum speed during training. Storage is cheap, training time is expensive.

## Storage Requirements

For 1000 volumes:
- **Without compression**: ~4.3 TB
- **With compression**: ~1.7 TB

Plan accordingly based on available storage.

## Troubleshooting

### Issue: "No preprocessed volumes found"

Check that preprocessing completed successfully:
```bash
python check_preprocessing.py --output-dir /path/to/preprocessed_volumes
ls /path/to/preprocessed_volumes/*_metadata.json | wc -l
```

### Issue: Jobs failing on LSF

Check error logs:
```bash
cat /path/to/preprocessed_volumes/logs/volume_000000.err
```

Common issues:
- **Out of memory**: Increase `--memory-gb`
- **Timeout**: Increase `--walltime`
- **Dataset incompatibility**: Check preprocessing script output

### Issue: Slow loading during training

1. Increase TensorStore threads: `num_threads=16`
2. Check disk I/O: Run `iostat` to see if disk is bottleneck
3. Use local SSD instead of network storage if possible
4. Reduce batch size if memory constrained

### Issue: Missing volumes

Find gaps:
```bash
python check_preprocessing.py --output-dir /path/to/preprocessed_volumes
```

Resubmit missing volumes:
```bash
# If volumes 10, 15, 23 are missing
python submit_preprocessing_jobs.py \
    --output-dir /path/to/preprocessed_volumes \
    --num-volumes 3 \
    --start-index 10
    # Then manually change indices to 10, 15, 23
```

Or resubmit individually:
```bash
python preprocess_volume.py --volume-index 10 --output-dir /path/to/preprocessed_volumes
python preprocess_volume.py --volume-index 15 --output-dir /path/to/preprocessed_volumes
python preprocess_volume.py --volume-index 23 --output-dir /path/to/preprocessed_volumes
```

## Tips

1. **Test first**: Process 1-5 volumes before submitting 1000 jobs
2. **No compression**: Faster is better for training
3. **Use 8-16 threads**: Diminishing returns beyond this
4. **Monitor disk space**: ~4.3 GB per volume adds up quickly
5. **Verify samples**: Run with `--verify` to ensure data integrity
6. **Save metadata**: Metadata files are small but contain crucial provenance info

## Integration with Existing Training

Replace your existing data loading:

```python
# OLD (slow)
for epoch in range(num_epochs):
    for batch in train_loader:
        raw, gt, mask = batch
        features = extract_dinov3_features(raw)  # SLOW!
        loss = model(features, gt, mask)
        ...

# NEW (fast)
from preprocessed_dataloader import create_preprocessed_dataloader

train_loader = create_preprocessed_dataloader(
    preprocessed_dir="/path/to/preprocessed_volumes",
    batch_size=4,
    shuffle=True,
    num_threads=8,
)

for epoch in range(num_epochs):
    for features, gt, mask in train_loader:  # FAST!
        loss = model(features, gt, mask)
        ...
```

The features are already extracted, so you skip the bottleneck entirely!
