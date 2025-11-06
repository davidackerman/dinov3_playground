"""
Quick Start: Using crop_filter and min_unique_ids for Affinity Training

Copy this into your training script and modify as needed.
"""

from dinov3_playground.data_processing import (
    generate_multi_organelle_dataset_pairs,
    load_random_3d_training_data,
)

# ============================================================================
# EXAMPLE 1: Basic Affinity Training with Specific Crops
# ============================================================================

# Step 1: Generate dataset pairs from specific crops only
dataset_pairs = generate_multi_organelle_dataset_pairs(
    organelle_list=["cell"],
    crop_filter=[115, 203, 315],  # Only use these specific crops
    min_resolution_for_raw=32,
    base_resolution=128,
    apply_scale_updates=True,
    require_all_organelles=True,
)

# Step 2: Load training data with boundary requirements
raw, gt, gt_masks, dataset_sources, num_classes = load_random_3d_training_data(
    dataset_pairs=dataset_pairs,
    volume_shape=(128, 128, 128),
    base_resolution=128,
    num_volumes=100,
    min_label_fraction=0.05,
    min_unique_ids=2,  # Ensure at least 2 different cell IDs (boundaries!)
    allow_gt_extension=True,
    seed=42,
)

print(f"Loaded {len(raw)} volumes from crops {dataset_pairs}")

# ============================================================================
# EXAMPLE 2: Train/Val Split by Crop Number
# ============================================================================

# Training set: specific crops
train_dataset_pairs = generate_multi_organelle_dataset_pairs(
    organelle_list=["cell"],
    crop_filter=[115, 203, 315, 420, 505],  # Training crops
    min_resolution_for_raw=32,
    base_resolution=128,
    apply_scale_updates=True,
)

# Validation set: different crops
val_dataset_pairs = generate_multi_organelle_dataset_pairs(
    organelle_list=["cell"],
    crop_filter=[118, 225, 360],  # Validation crops (separate from training)
    min_resolution_for_raw=32,
    base_resolution=128,
    apply_scale_updates=True,
)

# Load training data
train_raw, train_gt, train_masks, _, _ = load_random_3d_training_data(
    dataset_pairs=train_dataset_pairs,
    volume_shape=(128, 128, 128),
    base_resolution=128,
    num_volumes=150,
    min_unique_ids=2,  # Boundaries required
)

# Load validation data
val_raw, val_gt, val_masks, _, _ = load_random_3d_training_data(
    dataset_pairs=val_dataset_pairs,
    volume_shape=(128, 128, 128),
    base_resolution=128,
    num_volumes=30,
    min_unique_ids=2,  # Same requirement for validation
)

# ============================================================================
# EXAMPLE 3: Excluding Bad Crops (Use All Except...)
# ============================================================================

# First get all available pairs to see what crops exist
all_pairs_temp = generate_multi_organelle_dataset_pairs(
    organelle_list=["cell"],
    base_resolution=128,
)

# Extract all crop numbers
import re

all_crop_nums = set()
for pair in all_pairs_temp:
    for path in pair.values():
        match = re.search(r"crop(\d+)", str(path))
        if match:
            all_crop_nums.add(int(match.group(1)))

print(f"All available crops: {sorted(all_crop_nums)}")

# Exclude specific bad crops
bad_crops = [99, 104, 127]  # Known problematic crops
good_crops = [c for c in all_crop_nums if c not in bad_crops]

# Use only good crops
dataset_pairs = generate_multi_organelle_dataset_pairs(
    organelle_list=["cell"],
    crop_filter=good_crops,  # All crops except bad ones
    base_resolution=128,
)

# ============================================================================
# EXAMPLE 4: Complete Affinity Training Pipeline
# ============================================================================

from dinov3_playground.memory_efficient_training import (
    train_3d_unet_with_memory_efficient_loader,
)

# 1. Select high-quality crops for affinity training
affinity_crops = [115, 203, 315, 420, 505]
dataset_pairs = generate_multi_organelle_dataset_pairs(
    organelle_list=["cell"],
    crop_filter=affinity_crops,
    min_resolution_for_raw=32,
    base_resolution=128,
    apply_scale_updates=True,
)

# 2. Load boundary-rich volumes
raw, gt, masks, sources, num_classes = load_random_3d_training_data(
    dataset_pairs=dataset_pairs,
    volume_shape=(128, 128, 128),
    base_resolution=128,
    num_volumes=100,
    min_unique_ids=2,  # CRITICAL: Ensures boundary examples
    min_label_fraction=0.05,
    allow_gt_extension=True,
)

# 3. Train with affinity prediction
training_results = train_3d_unet_with_memory_efficient_loader(
    raw_data=raw,
    gt_data=gt,
    gt_masks=masks,
    train_volume_pool_size=80,
    val_volume_pool_size=20,
    num_classes=3,  # 3 affinity channels
    output_type="affinities",  # Enable affinity output
    affinity_offsets=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    loss_type="affinity",  # Use affinity loss
    # ... other training parameters
)

# ============================================================================
# QUICK REFERENCE
# ============================================================================

"""
crop_filter Parameter:
---------------------
  None              → All crops (default)
  [115]             → Single crop
  [115, 203, 315]   → Specific crops
  list(range(100,200)) → Range of crops

min_unique_ids Parameter:
------------------------
  None              → No restriction (default)
  2                 → At least 2 different IDs (for affinities) ✓
  3                 → At least 3 different IDs (crowded scenes)

Typical Workflow:
----------------
1. generate_multi_organelle_dataset_pairs() with crop_filter
2. load_random_3d_training_data() with min_unique_ids=2
3. train_3d_unet_with_memory_efficient_loader() with output_type="affinities"

Output Messages:
---------------
✓ "Crop filter active: only including crops [115, 203, 315]"
✓ "Dataset 1: 3 unique instance IDs found (>= 2 required)"
✗ "Dataset 2: only 1 unique instance IDs (need 2) - skipping"
"""
