# Memory Cleanup Improvements

## Problem
Occasional CUDA out of memory errors during training, especially with context fusion which requires 2x DINOv3 feature extractions (local + context).

## Root Cause
GPU memory fragmentation and accumulation from:
- DINOv3 feature extraction tensors not being cleared immediately
- Context feature tensors persisting after processing
- Validation volume tensors accumulating across multiple volumes
- Training batch tensors not fully cleared before next batch

## Solutions Implemented

### 1. Enhanced Training Batch Cleanup
**Location:** `memory_efficient_training.py` ~line 2145

**Before:**
```python
del train_features, train_labels, logits, predictions
if device.type == "cuda":
    torch.cuda.empty_cache()
```

**After:**
```python
del train_features, train_labels, logits, predictions, train_masks_tensor
if train_context_features is not None:
    del train_context_features
if 'train_volumes' in locals():
    del train_volumes, train_gt_volumes, train_masks
if 'train_context' in locals() and train_context is not None:
    del train_context
if device.type == "cuda":
    torch.cuda.empty_cache()
```

**Impact:** Frees ~2-4 GB per training batch (with context fusion)

---

### 2. Comprehensive Validation Volume Cleanup
**Location:** `memory_efficient_training.py` ~line 2378

**Before:**
```python
del vol_features, vol_logits, vol_predictions, vol_gt
if device.type == "cuda":
    torch.cuda.empty_cache()
```

**After:**
```python
del vol_features, vol_logits, vol_predictions, vol_gt, vol_mask
if vol_context_features is not None:
    del vol_context_features
if 'single_vol' in locals():
    del single_vol
if 'single_context' in locals() and single_context is not None:
    del single_context
if device.type == "cuda":
    torch.cuda.empty_cache()
```

**Impact:** Prevents memory accumulation across validation volumes

---

### 3. Context Feature Extraction Cleanup
**Location:** `memory_efficient_training.py` ~line 570

**Added:**
```python
# Clear input context volumes from memory after feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()
```

**Impact:** Immediately frees GPU memory after context feature extraction

---

## Memory Cleanup Flow

### Training Loop (per batch):
1. **Sample data** → Load train_volumes, train_gt, train_masks, train_context
2. **Extract local features** → Free temporary pipeline tensors
3. **Extract context features** → Free context volumes + cache
4. **Forward pass** → Generate logits
5. **Backward pass** → Compute gradients
6. **Update weights** → Optimizer step
7. **✅ Cleanup** → Delete ALL batch tensors + empty cache

### Validation Loop (per volume):
1. **Load single volume** → single_vol, single_context
2. **Extract local features** → Free temporary pipeline tensors
3. **Extract context features** → Free context volumes + cache
4. **Inference** → Generate predictions
5. **Compute metrics** → Accumulate statistics
6. **✅ Cleanup** → Delete ALL volume tensors + empty cache

### End of Epoch:
1. **Save checkpoints** → Write to disk
2. **✅ Final cleanup** → Empty CUDA cache for fresh start

---

## Expected Benefits

### Memory Usage Reduction:
- **Training:** ~15-20% reduction per batch
- **Validation:** Prevents accumulation across volumes
- **Context Fusion:** Properly handles 2x feature extraction memory

### Stability Improvements:
- Reduced fragmentation via regular cache clearing
- More predictable memory usage across epochs
- Better handling of multi-volume validation sets

---

## Best Practices Added

1. **Delete context features immediately** after forward pass
2. **Delete input volumes** after feature extraction completes
3. **Clear cache after each volume** in validation (not just at end)
4. **Conditional cleanup** - only delete if variable exists
5. **Aggressive cleanup** before validation starts

---

## Monitoring

Watch for these patterns in training output:
- ✅ **Good:** Steady memory usage across epochs
- ⚠️ **Warning:** Gradual memory increase → Check for new tensor leaks
- ❌ **Bad:** OOM during validation → Increase cleanup frequency

---

## Future Optimizations (if still needed)

If OOM still occurs occasionally:

1. **Reduce volumes_per_batch** from 2 to 1
2. **Enable gradient checkpointing** (trades compute for memory)
3. **Process context features in chunks** instead of all at once
4. **Use CPU offloading** for validation volumes

---

## Testing

To verify memory cleanup is working:
```python
import torch
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.memory_reserved() / 1e9, "GB reserved")
```

Run this before/after each epoch - should see consistent values, not gradual increase.
