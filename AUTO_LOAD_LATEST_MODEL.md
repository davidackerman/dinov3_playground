# Auto-Loading Most Recent Model - Quick Guide

## Feature: Automatic Timestamp Folder Selection

The `load_inference_model()` function now automatically finds and loads the most recent model checkpoint when you provide a parent directory path instead of a full timestamp path.

## How It Works

### Before (Manual Timestamp):
```python
# Had to specify exact timestamp folder
path = "/nrs/cellmap/.../dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/run_20251003_110551"
inference = load_inference_model(path)
```

### After (Automatic Selection):
```python
# Just provide parent folder - auto-selects most recent
path = "/nrs/cellmap/.../dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"
inference = load_inference_model(path)
```

**Output:**
```
Path does not end with timestamp folder: .../dinov3_unet3d_dinov3_vitl16_pretrain_sat493m
Looking for most recent timestamp folder...
Found 3 timestamp folder(s)
Using most recent: run_20251003_110551
Loading 3D model from: .../run_20251003_110551/model/best.pkl
```

## Timestamp Pattern Recognition

The function detects timestamp folders with these patterns:
- `run_YYYYMMDD_HHMMSS` (e.g., `run_20251003_110551`)
- `YYYYMMDD_HHMMSS` (e.g., `20251003_110551`)

**Examples of valid timestamp folders:**
- ✅ `run_20251003_110551`
- ✅ `20251003_110551`
- ✅ `run_20250930_020133`

**Examples that trigger auto-selection:**
- ❌ `dinov3_unet3d_dinov3_vitl16_pretrain_sat493m` → Searches for timestamps
- ❌ `results` → Searches for timestamps
- ❌ `my_model_v2` → Searches for timestamps

## Usage Examples

### Example 1: Multiple Training Runs (Most Common)

If you have multiple training runs:
```
/results/my_model/
  ├── run_20251001_120000/  # Older run
  ├── run_20251002_140000/  # Older run
  └── run_20251003_110551/  # Latest run ← This will be used
```

Just provide the parent path:
```python
path = "/results/my_model"
inference = load_inference_model(path)
# Automatically uses run_20251003_110551
```

### Example 2: Single Training Run

If you only have one timestamp folder:
```
/results/my_model/
  └── run_20251003_110551/  # Only run
```

Both ways work:
```python
# Option 1: Parent path (auto-finds the only timestamp)
path = "/results/my_model"
inference = load_inference_model(path)

# Option 2: Full path (explicit)
path = "/results/my_model/run_20251003_110551"
inference = load_inference_model(path)
```

### Example 3: Direct Timestamp Path (Explicit)

If you want a specific run (not the most recent):
```python
# Explicitly specify the timestamp - no auto-selection
path = "/results/my_model/run_20251001_120000"  # Use this specific older run
inference = load_inference_model(path)
```

## Benefits

1. **Simplified Paths**: No need to copy/paste long timestamp strings
2. **Always Latest**: Automatically uses your most recent training
3. **Backward Compatible**: Old explicit paths still work
4. **Clear Feedback**: Prints which folder was selected

## Updated Inference Script

The example script now shows both options:

```python
# Option 1: Full path with timestamp (explicit - for specific runs)
# path = ".../run_20251003_110551"

# Option 2: Parent path (auto-selects most recent) - RECOMMENDED
path = ".../dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"

inference = load_inference_model(path)
```

## Troubleshooting

### Issue: "No timestamp folders found"
**Cause:** The provided path contains no timestamp subdirectories
**Solution:** Provide the correct parent directory or use explicit path

### Issue: "Wrong model loaded"
**Cause:** Auto-selection picked a different run than expected
**Solution:** Use explicit timestamp path for specific runs:
```python
path = "/path/to/model/run_20251001_120000"  # Specific run
```

### Issue: "Multiple models, want second-latest"
**Cause:** Auto-selection always picks the most recent
**Solution:** Use explicit path or rename folders to change sort order

## Implementation Details

```python
import re

# Timestamp regex pattern
timestamp_pattern = r'(run_)?\d{8}_\d{6}'

# Check if last folder matches pattern
last_folder = Path(export_dir).name
if not re.match(timestamp_pattern, last_folder):
    # Not a timestamp - search for timestamp subdirectories
    timestamp_dirs = [d for d in path.iterdir() 
                     if d.is_dir() and re.match(timestamp_pattern, d.name)]
    
    if timestamp_dirs:
        # Sort by name (YYYYMMDD_HHMMSS naturally sorts chronologically)
        most_recent = sorted(timestamp_dirs, key=lambda x: x.name)[-1]
        export_path = most_recent
```

The sorting works because timestamp format `YYYYMMDD_HHMMSS` naturally sorts chronologically:
- `20251001_120000` < `20251002_140000` < `20251003_110551`

## Migration Guide

### Old Scripts:
```python
# Had to update this every time you retrained
path = "/nrs/cellmap/.../run_20251003_110551"
inference = load_inference_model(path)
```

### New Scripts (Recommended):
```python
# Set once, always uses latest
path = "/nrs/cellmap/.../dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"
inference = load_inference_model(path)
```

### For Specific Runs:
```python
# Explicitly specify when you need a particular run
path = "/nrs/cellmap/.../run_20251001_120000"
inference = load_inference_model(path)
```

## Performance Impact

**None** - The auto-selection only scans one directory level and is instantaneous even with dozens of timestamp folders.
