# Context Fusion Quick Reference

## The Fix in One Picture

### BEFORE (Wrong - Naive Concatenation)
```
Raw (4nm)      Context (64nm)
   |                |
   v                v
DINOv3          DINOv3
1024 ch         1024 ch
   |                |
   +----------------+
          |
    CONCATENATE <-- PROBLEM: Mixes semantics!
          |
      2048 ch
          |
          v
    First Conv (applies SAME kernels to all 2048 channels)
          |
          v
       UNet...
```

### AFTER (Correct - Attention Fusion)
```
Raw (4nm)          Context (64nm)
   |                    |
   v                    v
DINOv3              DINOv3
1024 ch             1024 ch
   |                    |
   v                    v
Project              Project
128 ch               128 ch
   |                    |
   v                    |
Encoder 1            Context 1
   |                    |
   +----Attention ←-----+  <-- Context GUIDES raw, doesn't mix
   |     Fusion
   v
Encoder 2            Context 2
   |                    |
   +----Attention ←-----+
   |     Fusion
   v
  ...

Result: Context modulates raw features via learned attention
```

---

## Code Changes Summary

### 1. Feature Extraction
```python
# OLD:
features = extract_multi_scale_dinov3_features_3d(raw, context)
# Returns: (batch, 2048, D, H, W) concatenated

# NEW:
raw_features, context_features = extract_multi_scale_dinov3_features_3d(raw, context)
# Returns: Two separate tensors (batch, 1024, D, H, W) each
```

### 2. UNet Forward
```python
# OLD:
logits = unet3d(features)  # Single concatenated input

# NEW:
logits = unet3d(raw_features, context_features=context_features)  # Two separate!
```

### 3. Model Initialization
```python
# Now automatically enabled:
unet3d = DINOv3UNet3D(
    input_channels=1024,           # Raw only
    use_context_fusion=True,       # Auto-enabled when context provided
    context_channels=1024,          # Context separate
)
```

---

## What You'll See

### Startup Messages
```
✓ Context fusion ENABLED: Using attention-based multi-scale fusion
  - Raw features: 1024 channels from facebook/dinov3-vitl16-pretrain-sat493m
  - Context features: 1024 channels at 64nm
  - Fusion: Context guides raw features via attention at skip connections

Using DINOv3UNet3D with 128 base channels and 1024 input channels
  - Context fusion layers: 4 attention modules at encoder skip connections
```

### Debug Output (First Batch)
```
DEBUG - Tensor shapes:
  train_features: torch.Size([2, 1024, 512, 512, 512])
  train_context_features: torch.Size([2, 1024, 512, 512, 512])
  train_labels: torch.Size([2, 128, 128, 128])
```

---

## Why This Matters

| Aspect | Concatenation (Wrong) | Attention Fusion (Correct) |
|--------|---------------------|---------------------------|
| **Semantic separation** | Lost immediately | Preserved throughout |
| **Context role** | Dilutes raw features | Guides raw features |
| **Kernel operations** | Same on all channels | Specialized per stream |
| **Memory** | Higher (2048→128 conv) | Lower (1024→128 each) |
| **Disambiguation** | Limited | Strong (spatial context) |

### Example: Nuclear Pore vs Mitochondrion
Both appear as small dark blobs locally.

**With concatenation:**
- UNet sees mixed 2048-channel mess
- Hard to separate "what" from "where"
- Confusion persists

**With attention fusion:**
- Raw: "Small dark blob with specific texture"
- Context: "Inside large dark nucleus region"
- Attention: "Focus on nuclear-texture features, ignore mito-texture features"
- Correct prediction: **Nuclear pore!**

---

## No Script Changes Needed!

Your existing training script already works correctly:

```python
# This automatically uses proper context fusion:
training_results = train_3d_unet_with_memory_efficient_loader(
    raw_data=raw,
    gt_data=gt,
    context_data=context_volumes,  # Just provide this
    context_scale=64,
    ...
)
```

The fix is entirely internal to the training functions!

---

## Technical Implementation

### Attention Fusion Module
```python
class ContextAttentionFusion(nn.Module):
    def forward(self, raw_features, context_features):
        # Generate attention weights from context
        attention = sigmoid(
            expand(
                bottleneck(context_features)
            )
        )  # Shape: (B, C, D, H, W), values ∈ [0,1]
        
        # Apply to raw features
        attended = raw_features * attention
        
        # Residual connection
        return raw_features + attended
```

### Applied at Each Skip Connection
```python
# At each encoder level (1, 2, 3, 4):
enc_fused = context_fusion(encoder_out, context_at_same_scale)

# Used in decoder:
decoder_input = concat(upsampled_from_below, enc_fused)
```

---

## Verification Checklist

- [ ] See "Context fusion ENABLED" message at startup
- [ ] Debug output shows separate feature tensors
- [ ] Training runs without errors
- [ ] GPU memory similar or lower than before
- [ ] Validation metrics improve over time
- [ ] No "shape mismatch" errors

If all checked, context fusion is working correctly! 🎉
