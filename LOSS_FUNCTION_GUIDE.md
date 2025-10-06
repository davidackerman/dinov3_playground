# Loss Function Quick Reference Guide

## Available Loss Functions

### 1. Cross Entropy (`'ce'`)
**Use when:** Balanced dataset, baseline comparison
```python
loss_type='ce',
```
- Standard multi-class classification loss
- No special parameters needed
- **Pros:** Simple, well-understood, fast
- **Cons:** Poor for class imbalance

---

### 2. Weighted Cross Entropy (`'weighted_ce'`)
**Use when:** Moderate class imbalance, baseline with class weighting
```python
loss_type='weighted_ce',
use_class_weighting=True,  # Enables automatic class weight calculation
```
- Applies inverse frequency weights to each class
- **Pros:** Simple fix for imbalance
- **Cons:** Can cause over-prediction, not optimal for severe imbalance

---

### 3. Focal Loss (`'focal'`) ⭐
**Use when:** High recall but low precision (over-prediction problem)
```python
loss_type='focal',
focal_gamma=2.0,  # Standard value, increase for more focus on hard examples
use_class_weighting=True,  # Optional: combine with class weights
```
- Down-weights easy examples (well-classified background)
- Focuses on hard examples (rare classes, boundaries)
- **Pros:** Excellent for class imbalance, reduces over-prediction
- **Cons:** One more hyperparameter (gamma) to tune
- **Recommended gamma values:**
  - `gamma=0.0`: Equivalent to CrossEntropyLoss
  - `gamma=2.0`: Standard (from paper), good starting point
  - `gamma=3.0`: More aggressive focusing
  - `gamma=5.0`: Very aggressive (use if 2.0 insufficient)

---

### 4. Dice Loss (`'dice'`)
**Use when:** Want to directly optimize IoU/overlap
```python
loss_type='dice',
dice_smooth=1.0,  # Smoothing constant (Laplace smoothing)
```
- Directly optimizes Dice coefficient (similar to IoU)
- Treats all classes equally (implicit balance)
- **Pros:** Directly optimizes target metric, handles imbalance well
- **Cons:** Can be unstable early in training, may struggle with very small objects
- **Recommended smooth values:**
  - `smooth=1.0`: Standard Laplace smoothing
  - `smooth=0.1`: Less smoothing (sharper gradients)
  - `smooth=10.0`: More smoothing (stabler training)

---

### 5. Focal + Dice Combined (`'focal_dice'`) ⭐⭐ RECOMMENDED
**Use when:** Severe class imbalance + want to optimize IoU
```python
loss_type='focal_dice',
focal_gamma=2.0,        # Focusing parameter
focal_weight=0.5,       # Weight for focal component
dice_weight=0.5,        # Weight for dice component
dice_smooth=1.0,        # Dice smoothing
use_class_weighting=True,  # Apply class weights to focal loss
```
- Combines benefits of both Focal and Dice
- **Focal:** Handles class imbalance, focuses on hard examples
- **Dice:** Directly optimizes IoU
- **Pros:** Best of both worlds, excellent for medical imaging
- **Cons:** More hyperparameters to tune
- **Recommended configurations:**
  - **Balanced:** `focal_weight=0.5, dice_weight=0.5`
  - **More IoU focus:** `focal_weight=0.3, dice_weight=0.7`
  - **More imbalance handling:** `focal_weight=0.7, dice_weight=0.3`

---

### 6. Tversky Loss (`'tversky'`)
**Use when:** Need fine control over FP/FN tradeoff
```python
loss_type='tversky',
tversky_alpha=0.7,  # Weight for False Positives
tversky_beta=0.3,   # Weight for False Negatives
dice_smooth=1.0,
```
- Generalization of Dice with adjustable FP/FN balance
- **Pros:** Fine-grained control, can emphasize precision or recall
- **Cons:** Requires understanding of FP/FN tradeoff
- **Recommended configurations:**
  - **Dice equivalent:** `alpha=0.5, beta=0.5`
  - **Reduce over-prediction (more precision):** `alpha=0.7, beta=0.3`
  - **Reduce under-prediction (more recall):** `alpha=0.3, beta=0.7`
  - **IoU equivalent:** `alpha=1.0, beta=1.0`

---

## Decision Tree

```
Do you have class imbalance? (e.g., 80% background, 3% rare class)
│
├─ NO → Use 'ce' (standard Cross Entropy)
│
└─ YES → What's your main problem?
    │
    ├─ High recall, low IoU (over-predicting)?
    │   └─ Use 'focal' with gamma=2.0 or 'focal_dice'
    │
    ├─ Want to directly optimize IoU?
    │   └─ Use 'focal_dice' (best overall)
    │
    ├─ Need fine FP/FN control?
    │   └─ Use 'tversky'
    │
    └─ Just starting / baseline?
        └─ Use 'weighted_ce' (simplest)
```

---

## Example Configurations

### For Your Current Problem (High Recall, Low IoU)

**Start with Focal Loss:**
```python
loss_type='focal',
focal_gamma=2.0,
use_class_weighting=True,
```

**If you want better IoU, upgrade to Focal+Dice:**
```python
loss_type='focal_dice',
focal_gamma=2.0,
focal_weight=0.5,
dice_weight=0.5,
dice_smooth=1.0,
use_class_weighting=True,
```

**If still over-predicting, use Tversky:**
```python
loss_type='tversky',
tversky_alpha=0.7,  # Penalize FP more
tversky_beta=0.3,   # Penalize FN less
dice_smooth=1.0,
```

---

## Expected Training Output

When you run training, you'll see:

```
Loss Configuration:
  - Loss type: focal_dice
  - Focal gamma: 2.0
  - Focal weight: 0.5
  - Dice weight: 0.5
  - Dice smooth: 1.0
  - Class weights: [1.25 10.0 14.3 33.3]
```

This confirms your loss function is configured correctly.

---

## Monitoring Results

### What to watch during training:

1. **Loss decreasing?**
   - ✅ Good: Loss goes down steadily
   - ❌ Bad: Loss plateaus early or increases

2. **Mean IoU improving?**
   - ✅ Good: IoU increases over time
   - 🎯 Target: Mean IoU > 0.5 is decent, > 0.7 is great

3. **Per-class metrics balanced?**
   ```
   Class    Val Recall  Val Precision  Val IoU
   ------------------------------------------------
   bg            0.95        0.92        0.88  ✅ Good balance
   er            0.82        0.79        0.68  ✅ Good balance
   mito          0.91        0.45        0.42  ❌ Over-predicting (high recall, low precision)
   nuc           0.67        0.71        0.52  ✅ Reasonable balance
   ```

4. **Focal Loss specific:**
   - Should see **lower training loss** than weighted CE (easy examples down-weighted)
   - Should see **better IoU** on rare classes
   - May train slightly slower (more complex loss computation)

5. **Dice Loss specific:**
   - Loss values will be different scale (0-1 range instead of 0-∞)
   - IoU should correlate closely with (1 - Dice Loss)
   - May see initial instability (first few epochs)

---

## Troubleshooting

### Problem: Loss is NaN or exploding
**Solution:**
- Reduce learning rate: `learning_rate=1e-4` instead of `1e-3`
- Increase dice_smooth: `dice_smooth=10.0`
- Use gradient clipping (already enabled in code)

### Problem: Focal loss not helping with over-prediction
**Solution:**
- Increase gamma: Try `focal_gamma=3.0` or `focal_gamma=5.0`
- Switch to Tversky with `alpha=0.7, beta=0.3`
- Reduce class weights (they might be too aggressive)

### Problem: Dice loss unstable early
**Solution:**
- Use combined Focal+Dice instead of pure Dice
- Increase dice_smooth: `dice_smooth=10.0`
- Start with pretrained weights if available

### Problem: Still getting low IoU despite Focal+Dice
**Solution:**
- Check data quality (are labels accurate?)
- Increase model capacity: `BASE_CHANNELS=256`
- Train longer: `epochs=2000`
- Adjust loss weights: `focal_weight=0.3, dice_weight=0.7`

---

## Performance Comparison (Expected)

Based on your data (80% background, 10% ER, 7% mito, 3% nuc):

| Loss Type | Training Speed | Mean IoU | Over-prediction | Stability |
|-----------|---------------|----------|-----------------|-----------|
| CE | ⚡ Fast | 0.35 | ❌ High | ✅ Excellent |
| Weighted CE | ⚡ Fast | 0.42 | ⚠️ Moderate | ✅ Excellent |
| Focal | 🐢 Moderate | 0.55 | ✅ Low | ✅ Good |
| Dice | 🐢 Moderate | 0.52 | ✅ Low | ⚠️ Can be unstable |
| **Focal+Dice** | 🐢 Moderate | **0.62** | ✅ Low | ✅ Good |
| Tversky | 🐢 Moderate | 0.58 | ✅ Adjustable | ✅ Good |

*Values are estimates based on typical performance*

---

## Summary

**Quick Recommendation for Your Use Case:**

1. **Start:** `loss_type='focal'` with `gamma=2.0`
2. **Upgrade:** `loss_type='focal_dice'` with `focal_weight=0.5, dice_weight=0.5`
3. **Fine-tune:** Adjust weights based on which metric needs improvement

The Focal+Dice combination is the **state-of-the-art** for imbalanced medical/microscopy segmentation and should give you the best results! 🎯
