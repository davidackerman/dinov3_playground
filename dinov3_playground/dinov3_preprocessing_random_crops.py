"""
Simple preprocessing for extracting DINOv3 features from random crops.

This module provides standalone functionality to:
1. Randomly sample a crop from a dataset's raw data
2. Extract DINOv3 features (with or without AnyUp)
3. Save features to TensorStore zarr format

No ground truth, organelles, or complex filtering - just raw → features.
"""

import os
import json
import numpy as np
import torch
import tensorstore as ts
from pathlib import Path
from datetime import datetime
from cellmap_flow.image_data_interface import ImageDataInterface
from funlib.geometry import Roi, Coordinate
import random
from dinov3_playground.zarr_util import find_target_scale


def extract_dinov3_features_3d_standalone(
    raw_volume,
    model_id="facebook/dinov3-vits16-pretrain-lvd493m",
    output_image_dim=128,
    input_resolution=32,
    output_resolution=128,
    use_anyup=True,
    use_orthogonal_planes=True,
):
    """
    Extract DINOv3 features from a 3D volume without needing a dataloader.

    Parameters:
    -----------
    raw_volume : np.ndarray
        Raw volume of shape (D, H, W)
    model_id : str
        DINOv3 model identifier
    input_resolution : int
        Resolution of input raw data in nm
    output_resolution : int
        Target resolution for features in nm
    use_anyup : bool
        Use AnyUp for feature extraction

    Returns:
    --------
    torch.Tensor : Features of shape (C, D, H, W)
    """
    from dinov3_playground import initialize_dinov3
    from dinov3_playground.models import DINOv3UNet3DPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Redundant calculation since we are calculating volume shape based on input and output resolutions
    dinov3_slice_size = output_image_dim * output_resolution / input_resolution

    print(f"  Volume size: {raw_volume.shape}")
    print(f"  DINOv3 slice size: {dinov3_slice_size}")
    print(f"  Use AnyUp: {use_anyup}")

    # Initialize DINOv3 model
    print("  Initializing DINOv3...")
    # Initialize model (stores globally, we don't need the return values)
    initialize_dinov3(model_id=model_id, image_size=dinov3_slice_size)
    volume_size = raw_volume.shape
    # Create pipeline for feature extraction
    pipeline = DINOv3UNet3DPipeline(
        input_size=output_image_dim,  # for the overall unet model
        dinov3_slice_size=dinov3_slice_size,
        device=device,
        use_orthogonal_planes=use_orthogonal_planes,
        use_anyup=use_anyup,
    )

    # Convert to tensor and add batch dimension
    # Force conversion to a proper numpy array first
    # if not isinstance(raw_volume, torch.Tensor):
    #     # Use torch.tensor() instead of from_numpy() to avoid type checking issues
    #     # torch.tensor() always creates a copy, which bypasses wrapper type problems
    #     raw_volume_np = np.ascontiguousarray(raw_volume, dtype=np.float32)
    #     raw_tensor = torch.tensor(raw_volume_np, dtype=torch.float32).unsqueeze(0)
    # else:
    #     raw_tensor = raw_volume.float().unsqueeze(0)

    # #raw_tensor = raw_tensor.to(device)

    # Extract features
    print("  Extracting features...")
    with torch.no_grad():
        features, timing = pipeline.extract_dinov3_features_3d(
            raw_volume,
            use_orthogonal_planes=use_orthogonal_planes,
            target_output_size=(output_image_dim, output_image_dim, output_image_dim),
            enable_timing=True,
        )

    # Clean up
    del pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return features


def write_features_to_tensorstore(
    output_path,
    features,
    raw=None,
    use_compression=False,
    num_threads=None,
):
    """
    Write extracted features to TensorStore zarr format.

    Parameters:
    -----------
    output_path : str or Path
        Path to the output Zarr file
    features : np.ndarray or torch.Tensor
        DINOv3 features (C, D, H, W)
    raw : np.ndarray, optional
        Raw data (D, H, W), will be saved as uint8
    use_compression : bool
        Use LZ4 compression if True
    num_threads : int, optional
        Number of threads for parallel I/O

    Returns:
    --------
    tuple : (sizes_dict, write_time)
    """
    # Determine number of threads
    if num_threads is None:
        bsub_threads = os.environ.get("LSB_DJOB_NUMPROC")
        if bsub_threads is not None:
            num_threads = int(bsub_threads) * 2
        else:
            num_threads = 16

    output_path = Path(output_path)

    # Delete existing zarr file if it exists
    if output_path.exists():
        print(f"  Removing existing file: {output_path}")
        import shutil

        shutil.rmtree(output_path)

    # Prepare compression settings
    if use_compression:
        compressor = {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 3,
            "shuffle": 2,
        }
        compression_str = "lz4_level3"
    else:
        compressor = None
        compression_str = "none"

    # Set up TensorStore context
    context = ts.Context(
        {
            "data_copy_concurrency": {"limit": num_threads},
        }
    )

    print(f"\nWriting to TensorStore (compression: {compression_str})...")
    print(f"Output path: {output_path}")

    import time

    t0 = time.time()

    # Convert features to numpy and float16 for storage
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    else:
        features_np = features

    if features_np.ndim == 5:  # (B, C, D, H, W)
        features_np = features_np[0]  # Remove batch dimension

    features_np = features_np.astype("float16")

    sizes = {}

    # Write features
    print("  Writing features...")
    features_chunks = (256, 64, 64, 64)
    features_spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": str(output_path),
        },
        "path": "features",
        "metadata": {
            "shape": list(features_np.shape),
            "chunks": list(features_chunks),
            "dtype": "<f2",  # float16
            "compressor": compressor,
        },
        "recheck_cached_data": False,
        "recheck_cached_metadata": False,
    }
    features_dataset = ts.open(
        features_spec, create=True, delete_existing=True, context=context
    ).result()
    features_dataset[:] = features_np
    sizes["features_gb"] = (
        sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e9
    )

    # Write raw data if provided
    if raw is not None:
        print("  Writing raw data...")
        # Ensure raw is uint8
        if raw.dtype != np.uint8:
            raw_min, raw_max = raw.min(), raw.max()
            if raw_max > raw_min:
                raw = ((raw - raw_min) / (raw_max - raw_min) * 255).astype(np.uint8)
            else:
                raw = raw.astype(np.uint8)

        raw_spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(output_path),
            },
            "path": "raw",
            "metadata": {
                "shape": list(raw.shape),
                "chunks": list(raw.shape),
                "dtype": "|u1",  # uint8
                "compressor": compressor,
            },
        }
        raw_dataset = ts.open(raw_spec, create=True, context=context).result()
        raw_dataset[:] = raw
        prev_size = sizes["features_gb"]
        current_size = (
            sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e9
        )
        sizes["raw_gb"] = current_size - prev_size
    else:
        sizes["raw_gb"] = 0

    write_time = time.time() - t0
    sizes["total_gb"] = sum(sizes.values())

    print(f"\nTensorStore write completed in {write_time:.2f}s")
    print(f"  Features: {sizes['features_gb']:.3f} GB")
    if raw is not None:
        print(f"  Raw: {sizes['raw_gb']:.3f} GB")
    print(f"  Total: {sizes['total_gb']:.3f} GB")

    return sizes, write_time


def preprocess_and_save_random_crop(
    crop_index,
    output_dir,
    dataset_path,
    model_id="facebook/dinov3-vitl16-pretrain-sat493m",
    input_resolution=32,
    output_resolution=128,
    output_image_dim=128,
    use_compression=False,
    use_anyup=True,
    num_threads=None,
    save_raw=True,
    use_orthogonal_planes=True,
):
    """
    Sample a random crop from a dataset and extract DINOv3 features.

    Parameters:
    -----------
    crop_index : int
        Index of crop (used as random seed for sampling location)
    output_dir : str
        Directory to save preprocessed crops
    dataset_path : str
        Path to the dataset (e.g., zarr path to raw data)
    model_id : str
        DINOv3 model identifier
    input_resolution : int
        Resolution of input raw data in nm
    output_resolution : int
        Target resolution for features in nm
    output_image_dim : int
        Size of crop in each dimension
    use_compression : bool
        Use LZ4 compression if True
    use_anyup : bool
        Use AnyUp for feature extraction
    num_threads : int, optional
        Number of threads for TensorStore operations
    save_raw : bool
        Save raw data alongside features

    Returns:
    --------
    dict : Metadata about the processed crop
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed based on crop index
    np.random.seed(crop_index)
    random.seed(crop_index)

    print(f"\n{'='*60}")
    print(f"Processing Random Crop {crop_index}")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset path: {dataset_path}")
    print(f"Random seed: {crop_index}")
    print(f"Output image dim: {output_image_dim}")
    print(f"Input resolution: {input_resolution} nm")
    print(f"Output resolution: {output_resolution} nm")

    # Find the target scale path for the desired resolution
    print("\nFinding target scale in dataset...")
    target_path, target_scale, target_offset, target_shape = find_target_scale(
        dataset_path, input_resolution
    )
    print(f"  Using scale: {target_scale}")
    print(f"  Resolution: {input_resolution} nm")

    # Load dataset using ImageDataInterface with target resolution
    print("\nLoading dataset...")
    idi = ImageDataInterface(target_path, output_voxel_size=input_resolution)

    # Get dataset bounds
    dataset_roi = idi.roi
    dataset_shape = dataset_roi.shape  # in world units (nm)
    dataset_offset = dataset_roi.offset  # in world units (nm)

    print(f"  Dataset shape: {dataset_shape}")
    print(f"  Dataset offset: {dataset_offset}")
    print(f"  Dataset resolution: {idi.voxel_size}")

    # Calculate crop size in world units (nm)
    crop_size_nm = output_image_dim * output_resolution
    crop_size_nm_coord = Coordinate([crop_size_nm] * 3)

    print(f"  Crop size (nm): {crop_size_nm_coord}")

    # Calculate maximum offset in world units
    max_offset_nm = [dataset_shape[i] - crop_size_nm for i in range(3)]

    if any(m < 0 for m in max_offset_nm):
        raise ValueError(
            f"Dataset too small for requested crop size. "
            f"Dataset shape: {dataset_shape}, Crop size: {crop_size_nm_coord}"
        )

    # Randomly sample offset in world units
    # Generate random offset in voxels first, then convert to world units
    max_offset_voxels = [int(max_offset_nm[i] // idi.voxel_size[i]) for i in range(3)]
    random_offset_voxels = [
        np.random.randint(0, max_offset_voxels[i] + 1) for i in range(3)
    ]
    random_offset_nm = Coordinate(
        [
            dataset_offset[i] + random_offset_voxels[i] * idi.voxel_size[i]
            for i in range(3)
        ]
    )

    # Create ROI for the crop (in world units)
    crop_roi = Roi(random_offset_nm, crop_size_nm_coord)

    print(f"  Crop ROI offset: {crop_roi.offset}")
    print(f"  Crop ROI shape: {crop_roi.shape}")

    # Load the crop
    print("\nLoading crop from dataset...")
    import time

    t0 = time.time()
    raw_volume = idi.to_ndarray_ts(crop_roi)
    load_time = time.time() - t0

    print(f"  Loaded crop in {load_time:.2f}s")
    print(f"  Raw volume shape: {raw_volume.shape}")
    print(f"  Raw volume dtype: {raw_volume.dtype}")

    # Extract DINOv3 features
    print("\nExtracting DINOv3 features...")
    t0 = time.time()

    features = extract_dinov3_features_3d_standalone(
        raw_volume=raw_volume,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        model_id=model_id,
        output_image_dim=output_image_dim,
        use_anyup=use_anyup,
        use_orthogonal_planes=use_orthogonal_planes,
    )

    extraction_time = time.time() - t0
    print(f"  Feature extraction completed in {extraction_time:.2f}s")
    print(f"  Features shape: {features.shape}")

    # Write to TensorStore
    crop_name = f"crop_{crop_index:06d}"
    crop_path = output_dir / f"{crop_name}.zarr"

    sizes, write_time = write_features_to_tensorstore(
        output_path=crop_path,
        features=features,
        raw=raw_volume if save_raw else None,
        use_compression=use_compression,
        num_threads=num_threads,
    )

    # Create metadata
    metadata = {
        "crop_index": crop_index,
        "crop_name": crop_name,
        "timestamp": datetime.now().isoformat(),
        "paths": {
            "crop": str(crop_path),
            "source_dataset": dataset_path,
            "source_dataset_target_path": target_path,
            "source_dataset_target_scale": target_scale,
        },
        "crop_roi": {
            "offset": list(crop_roi.offset),
            "shape": list(crop_roi.shape),
        },
        "dataset_info": {
            "roi_offset": list(dataset_offset),
            "roi_shape": list(dataset_shape),
            "scale": list(idi.voxel_size),
        },
        "shapes": {
            "raw": list(raw_volume.shape),
            "features": list(
                features.shape if isinstance(features, torch.Tensor) else features.shape
            ),
        },
        "configuration": {
            "model_id": model_id,
            "input_resolution": input_resolution,
            "output_resolution": output_resolution,
            "output_image_dim": output_image_dim,
            "use_anyup": use_anyup,
            "compression": "lz4_level3" if use_compression else "none",
            "save_raw": save_raw,
        },
        "timing": {
            "data_loading_seconds": load_time,
            "feature_extraction_seconds": extraction_time,
            "tensorstore_write_seconds": write_time,
            "total_seconds": load_time + extraction_time + write_time,
        },
        "storage": sizes,
    }

    # Save metadata
    metadata_path = output_dir / f"{crop_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")
    print(f"\n{'='*60}")
    print(f"Crop {crop_index} preprocessing complete!")
    print(f"{'='*60}")

    return metadata
