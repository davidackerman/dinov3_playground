# Context Fusion Memory Optimization Guide

## Why Memory Increased with Context Fusion

### Old Architecture (Concatenation)
```python
# Extract features from raw
raw_features = dinov3(raw_slices)  # 1x DINOv3 call

# Concatenate with context (happens at data level, no DINOv3 processing)
features = concat([raw_features, raw_context_data], dim=1)  # Simple concat

# UNet forward pass
logits = unet(features)  # 2048 input channels
```
**Memory Cost:** 1x DINOv3 extraction + 2048-channel UNet

### New Architecture (Attention Fusion)
```python
# Extract features from raw
raw_features = dinov3(raw_slices)  # 1x DINOv3 call

# Extract features from context (NEW!)
context_features = dinov3(context_slices)  # 2x DINOv3 call (DOUBLES this step)

# UNet forward pass with attention fusion
logits = unet(raw_features, context_features=context_features)  # 1024 input channels
```
**Memory Cost:** 2x DINOv3 extraction + 1024-channel UNet with attention modules

## Where Memory Increased

### 1. **DINOv3 Feature Extraction** (MAJOR INCREASE)
- **Old:** 1 DINOv3 call per batch
- **New:** 2 DINOv3 calls per batch (raw + context)
- **Impact:** ~2x memory during feature extraction phase
- **Duration:** Temporary (only during feature extraction)

### 2. **Feature Storage** (MODERATE INCREASE)
- **Old:** 1 feature tensor in GPU memory
- **New:** 2 feature tensors in GPU memory (raw + context)
- **Impact:** 2x feature memory until UNet forward pass
- **Duration:** Temporary (cleared after forward pass)

### 3. **UNet Forward Pass** (ACTUALLY DECREASED!)
- **Old:** 2048-channel encoder (concatenated features)
- **New:** 1024-channel encoder + 4 attention modules
- **Impact:** Lower memory in UNet, but attention modules add some overhead
- **Net effect:** Slight decrease

## Memory Optimization Strategies

### Level 1: Essential (For 24GB GPUs)
```python
# Reduce volumes per batch
volumes_per_batch=1,  # Down from 2
batches_per_epoch=20,  # Up from 10 to maintain training

# Reduce UNet capacity
BASE_CHANNELS = 64,  # Down from 128
```
**Memory Savings:** ~40-50%
**Impact on Quality:** Minimal (64 channels is still quite good for 3D)

### Level 2: Moderate (For 16GB GPUs)
```python
# Level 1 settings plus:

# Use smaller DINOv3 model
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"  # Already using this

# Reduce volume size
VOLUME_SIZE = (96, 96, 96)  # Down from (128, 128, 128)
```
**Memory Savings:** ~60-70% total
**Impact on Quality:** Moderate (smaller receptive field)

### Level 3: Aggressive (For 12GB GPUs)
```python
# Level 1 + Level 2 settings plus:

# Enable half precision for UNet
use_half_precision=True,

# Further reduce channels
BASE_CHANNELS = 32,

# Smaller volumes
VOLUME_SIZE = (64, 64, 64)
```
**Memory Savings:** ~75-80% total
**Impact on Quality:** Significant but potentially acceptable

## Monitoring Memory Usage

### During Training
Watch for these messages:
```
DINOv3 extraction: X.XXs (XX.X%)  ← Should be ~50% of batch time now (was ~30%)
UNet training:      X.XXs (XX.X%)  ← Should be ~50% of batch time (was ~70%)
```

### GPU Memory Tracking
```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Look for:
# - Memory spikes during DINOv3 extraction (normal)
# - Steady memory during UNet forward pass (good)
# - OOM errors during feature extraction (reduce volumes_per_batch)
# - OOM errors during UNet forward (reduce BASE_CHANNELS)
```

## Expected Memory Usage (24GB GPU)

| Configuration | DINOv3 Peak | UNet Peak | Total Peak | Status |
|---------------|-------------|-----------|------------|--------|
| **Old (concat)** | ~8GB | ~12GB | ~15GB | ✅ Comfortable |
| **New (default)** | ~14GB | ~10GB | ~22GB | ⚠️ Tight |
| **New (optimized)** | ~10GB | ~8GB | ~16GB | ✅ Comfortable |

### Optimized Configuration (Recommended)
```python
volumes_per_batch=1,
batches_per_epoch=20,
BASE_CHANNELS=64,
use_mixed_precision=True,
```

## Why the Tradeoff is Worth It

Despite higher memory during feature extraction:

1. **Better Architecture:** Attention fusion is semantically correct
2. **Better Performance:** Context guides features rather than diluting them
3. **Manageable:** Simple parameter adjustments solve the issue
4. **Temporary:** Memory spike only during feature extraction, not whole training

## Quick Troubleshooting

**Symptom:** OOM during "DINOv3 extraction"
- **Solution:** Reduce `volumes_per_batch` to 1
- **Why:** Too many volumes being processed simultaneously

**Symptom:** OOM during "UNet training" or "UNet inference"
- **Solution:** Reduce `BASE_CHANNELS` from 128 to 64
- **Why:** UNet is too large

**Symptom:** OOM during validation
- **Solution:** Code already processes validation volume-by-volume
- **Why:** Should not happen; if it does, reduce `VOLUME_SIZE`

**Symptom:** Training is very slow
- **Expected:** Context fusion processes 2x through DINOv3
- **Normal:** 40-60% slower than concatenation approach
- **Benefit:** Better convergence should compensate

## Memory Budgeting Example

For a **24GB GPU** with **Context Fusion**:

```
DINOv3 raw features:      ~5GB  (1 volume, 1024 channels, 128³)
DINOv3 context features:  ~5GB  (1 volume, 1024 channels, 128³)
UNet activations:         ~6GB  (64 base channels, gradient checkpointing)
Optimizer states:         ~2GB  (Adam optimizer)
Model weights:            ~1GB  (DINOv3 + UNet)
System overhead:          ~2GB  (PyTorch, CUDA)
---------------------------------------------------
Total:                   ~21GB  ✅ Fits in 24GB
```

With `volumes_per_batch=2`:
```
DINOv3 features:         ~20GB  (doubled!)  ❌ OOM
```

## Conclusion

**The memory increase is real but manageable:**
- Primary cause: 2x DINOv3 calls (raw + context)
- Simple fix: `volumes_per_batch=1`, `BASE_CHANNELS=64`
- Better architecture justifies the tradeoff
- Performance improvements will compensate for slower training
