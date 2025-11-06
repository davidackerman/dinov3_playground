# DINOv3 Feature Preprocessing System - Quick Start

## What This Does

Speeds up your DINOv3 training by **50-200x** by preprocessing features once instead of extracting them every iteration.

**Before**: 2-5 seconds per volume to extract features during training  
**After**: 10-50ms per volume to load preprocessed features

## Files You Got

1. **dinov3_preprocessing.py** - Core preprocessing functions
2. **preprocess_volume.py** - CLI to preprocess one volume
3. **submit_preprocessing_jobs.py** - Submit batch jobs to LSF
4. **check_preprocessing.py** - Monitor progress
5. **preprocessed_dataloader.py** - PyTorch DataLoader for cached features
6. **tensorstore_dinov3_benchmark.py** - Benchmark script (optional)
7. **README_preprocessing.md** - Full documentation

## Usage in 3 Steps

### Step 1: Test with One Volume

```bash
python preprocess_volume.py \
    --volume-index 0 \
    --output-dir /path/to/preprocessed_volumes \
    --organelles cell \
    --num-threads 8
```

Check the output:
```bash
ls /path/to/preprocessed_volumes/
# Should see: volume_000000_features.zarr, volume_000000_gt.zarr, 
#             volume_000000_mask.zarr, volume_000000_metadata.json
```

### Step 2: Preprocess Many Volumes in Parallel

```bash
# Submit 100 jobs to LSF (each processes 1 volume)
python submit_preprocessing_jobs.py \
    --output-dir /path/to/preprocessed_volumes \
    --num-volumes 100 \
    --num-processors 16 \
    --memory-gb 64 \
    --organelles cell \
    --walltime 2:00

# Monitor
bjobs -P cellmap | grep dinov3_preprocess

# Check progress
python check_preprocessing.py --output-dir /path/to/preprocessed_volumes
```

### Step 3: Use in Training

```python
from preprocessed_dataloader import create_preprocessed_dataloader

# Replace your old DataLoader with this
train_loader = create_preprocessed_dataloader(
    preprocessed_dir="/path/to/preprocessed_volumes",
    batch_size=4,
    shuffle=True,
    num_threads=8,
)

# Training loop - features are already extracted!
for features, gt, mask in train_loader:
    # features: (B, 1024, 128, 128, 128) - DINOv3 features
    # gt: (B, 128, 128, 128) - ground truth
    # mask: (B, 128, 128, 128) - valid GT regions
    
    loss = model(features, gt, mask)
    loss.backward()
    optimizer.step()
```

## Key Parameters

### When submitting jobs:

- `--num-volumes`: How many volumes to preprocess (e.g., 1000)
- `--output-dir`: Where to save (needs ~4.3 GB per volume)
- `--num-processors`: CPUs per job (16 works well)
- `--memory-gb`: RAM per job (64 GB is safe)
- `--organelles`: Which organelles (e.g., `cell nucleus mitochondria`)
- `--inference-filter`: Dataset filters (e.g., `jrc_mus-liver-zon-1`)

### When loading:

- `batch_size`: Batch size (start with 2-4 for 128³ volumes)
- `num_threads`: TensorStore parallelism (8-16 works well)
- `volume_indices`: Specific volumes to use (optional)

## Storage

Each 128³ volume with 1024 DINOv3 features:
- **Without compression**: ~4.3 GB (FAST - recommended)
- **With compression**: ~1.7 GB (slower, saves 60% space)

For 1000 volumes: **4.3 TB** (no compression) or **1.7 TB** (compressed)

## Compression: Yes or No?

**Recommended: NO compression** for training

| Metric | No Compression | LZ4 Compression |
|--------|---------------|-----------------|
| Write time | 0.15s | 2.5s |
| Cold read | 1.5s | 1.6-2s |
| Warm read | 10-50ms | 40-100ms |
| Size | 4.3 GB | 1.7 GB |
| **Training speed** | **Fastest** | Slower |

Storage is cheap. Your training time is expensive. Use no compression.

## Common Commands

```bash
# Check what you've preprocessed
python check_preprocessing.py --output-dir /path/to/preprocessed_volumes

# Verify volumes are valid
python check_preprocessing.py --output-dir /path/to/preprocessed_volumes --verify

# Benchmark loading speed
python preprocessed_dataloader.py \
    --preprocessed-dir /path/to/preprocessed_volumes \
    --batch-size 4 \
    --num-batches 20

# Resubmit a single missing volume
python preprocess_volume.py --volume-index 42 --output-dir /path/to/preprocessed_volumes
```

## Troubleshooting

**"No preprocessed volumes found"**
- Check: `ls /path/to/preprocessed_volumes/*_metadata.json`
- Run: `python check_preprocessing.py --output-dir /path/to/preprocessed_volumes`

**Jobs failing**
- Check logs: `cat /path/to/preprocessed_volumes/logs/volume_000000.err`
- Increase memory: `--memory-gb 128`
- Increase time: `--walltime 4:00`

**Slow loading**
- Increase threads: `num_threads=16`
- Check disk I/O with `iostat`
- Consider local SSD vs network storage

**Missing volumes**
- Find gaps: `python check_preprocessing.py --output-dir /path/to/preprocessed_volumes`
- Resubmit individually or in small batches

## What Gets Saved

For each volume, you get:
1. **Features** (volume_XXXXXX_features.zarr) - DINOv3 embeddings
2. **Ground truth** (volume_XXXXXX_gt.zarr) - Labels
3. **Mask** (volume_XXXXXX_mask.zarr) - Valid GT regions
4. **Metadata** (volume_XXXXXX_metadata.json) - Provenance, timing, config

Metadata includes:
- Which raw dataset it came from
- Random seed used
- Timing statistics
- Storage size
- Configuration used

This lets you track exactly where each training volume came from!

## Expected Performance

With 8 threads, no compression, loading from SSD:
- **First load (cold)**: ~1.5s per volume
- **Subsequent loads (cached)**: ~10-50ms per volume
- **Speedup vs extraction**: ~50-200x

Your training will fly! 🚀

## Tips

1. ✅ Test with 1-5 volumes before submitting 1000 jobs
2. ✅ Use no compression for training (faster is better)
3. ✅ Use 8-16 threads (diminishing returns beyond)
4. ✅ Plan disk space: ~4.3 GB per volume × number of volumes
5. ✅ Verify a few random samples after preprocessing
6. ✅ Keep metadata files - they're small but crucial for provenance

## Questions?

Read the full documentation: **README_preprocessing.md**

Good luck with your training! 🎉
