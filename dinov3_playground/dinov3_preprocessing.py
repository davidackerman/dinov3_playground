# %%
"""
Preprocessing functions for extracting and caching DINOv3 features to TensorStore.
Updated to include downsampled raw data.
"""

import os
import json
import numpy as np
import torch
import tensorstore as ts
from pathlib import Path
from datetime import datetime
from dinov3_playground.data_processing import (
    load_random_3d_training_data,
    generate_multi_organelle_dataset_pairs,
)
from dinov3_playground import initialize_dinov3
from dinov3_playground.affinity_utils import (
    compute_affinities_and_lsds_3d,
    compute_affinities_3d,
    compute_boundary_weights,
)
from scipy.ndimage import zoom


def write_volume_to_tensorstore(
    output_path,
    features,
    gt,
    target,
    raw=None,
    boundary_weights=None,
    mask=None,
    use_compression=False,
    num_threads=None,
):
    """
    Write a preprocessed volume to TensorStore as a single Zarr with multiple datasets.

    Parameters:
    -----------
    output_path : str or Path
        Path to the output Zarr file
    features : np.ndarray
        DINOv3 features (C, D, H, W)
    gt : np.ndarray
        Ground truth segmentation (D, H, W)
    target : np.ndarray
        Target affinities/LSDs for training
    raw : np.ndarray, optional
        Raw data downsampled to GT resolution (D, H, W), will be saved as uint8
    boundary_weights : np.ndarray, optional
        Boundary weights for loss computation
    mask : np.ndarray, optional
        Valid GT mask (D, H, W)
    use_compression : bool
        Use LZ4 compression if True
    num_threads : int, optional
        Number of threads for parallel I/O. If None, uses 2x LSB_DJOB_NUMPROC or 16 as default

    Returns:
    --------
    tuple : (sizes_dict, write_time)
        Dictionary with size information for each dataset and total write time
    """
    # Determine number of threads: 2x bsub allocation or default to 16
    if num_threads is None:
        bsub_threads = os.environ.get("LSB_DJOB_NUMPROC")
        if bsub_threads is not None:
            num_threads = int(bsub_threads) * 2
            print(f"Using {num_threads} threads (2x LSB_DJOB_NUMPROC={bsub_threads})")
        else:
            num_threads = 16
            print(f"Using default {num_threads} threads (LSB_DJOB_NUMPROC not set)")

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

    # Convert features to float16 for storage
    features_np = features.astype("float16")

    sizes = {}

    # Write features
    print("  Writing features...")
    features_chunks = (256, 64, 64, 64)  # Optimal for parallel I/O
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

    # Write GT
    print("  Writing ground truth...")
    gt_spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": str(output_path),
        },
        "path": "gt",
        "metadata": {
            "shape": list(gt.shape),
            "chunks": list(gt.shape),  # Single chunk for GT
            "dtype": "<u4",  # uint32
            "compressor": compressor,
        },
    }
    gt_dataset = ts.open(gt_spec, create=True, context=context).result()
    gt_dataset[:] = gt
    current_size = (
        sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e9
    )
    sizes["gt_gb"] = current_size - sizes["features_gb"]

    # Write raw data if provided
    if raw is not None:
        print("  Writing raw data...")
        # Ensure raw is uint8
        if raw.dtype != np.uint8:
            # Normalize to 0-255 range if needed
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
                "chunks": list(raw.shape),  # Single chunk for raw
                "dtype": "|u1",  # uint8
                "compressor": compressor,
            },
        }
        raw_dataset = ts.open(raw_spec, create=True, context=context).result()
        raw_dataset[:] = raw
        prev_size = current_size
        current_size = (
            sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e9
        )
        sizes["raw_gb"] = current_size - prev_size
    else:
        sizes["raw_gb"] = 0

    # Write target (affinities/LSDs)
    print("  Writing target...")
    # Make chunks evenly divide dimensions for neuroglancer compatibility
    target_chunks = (target.shape[0], 64, 64, 64)  # Use full channel dimension
    target_spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": str(output_path),
        },
        "path": "target",
        "metadata": {
            "shape": list(target.shape),
            "chunks": list(target_chunks),
            "dtype": "<f4",  # float32
            "compressor": compressor,
        },
    }
    target_dataset = ts.open(target_spec, create=True, context=context).result()
    target_dataset[:] = target
    prev_size = current_size
    current_size = (
        sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e9
    )
    sizes["target_gb"] = current_size - prev_size

    # Write boundary weights if present
    if boundary_weights is not None:
        print("  Writing boundary weights...")
        # Match chunking to actual array dimensions
        if len(boundary_weights.shape) == 4:
            weights_chunks = (min(boundary_weights.shape[0], 16), 64, 64, 64)
        else:  # 3D array
            weights_chunks = tuple(min(s, 64) for s in boundary_weights.shape)

        weights_spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(output_path),
            },
            "path": "boundary_weights",
            "metadata": {
                "shape": list(boundary_weights.shape),
                "chunks": list(weights_chunks),
                "dtype": "<f4",  # float32
                "compressor": compressor,
            },
        }
        weights_dataset = ts.open(weights_spec, create=True, context=context).result()
        weights_dataset[:] = boundary_weights
        prev_size = current_size
        current_size = (
            sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e9
        )
        sizes["boundary_weights_gb"] = current_size - prev_size
    else:
        sizes["boundary_weights_gb"] = 0

    # Write mask if present
    if mask is not None:
        print("  Writing mask...")
        mask_spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(output_path),
            },
            "path": "mask",
            "metadata": {
                "shape": list(mask.shape),
                "chunks": list(mask.shape),  # Single chunk for mask
                "dtype": "|u1",  # uint8
                "compressor": compressor,
            },
        }
        mask_dataset = ts.open(mask_spec, create=True, context=context).result()
        mask_dataset[:] = mask
        prev_size = current_size
        current_size = (
            sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e9
        )
        sizes["mask_gb"] = current_size - prev_size
    else:
        sizes["mask_gb"] = 0

    write_time = time.time() - t0
    sizes["total_gb"] = sum(sizes.values())

    print(f"\nTensorStore write completed in {write_time:.2f}s")
    print(f"  Features: {sizes['features_gb']:.3f} GB")
    print(f"  GT: {sizes['gt_gb']:.3f} GB")
    if raw is not None:
        print(f"  Raw: {sizes['raw_gb']:.3f} GB")
    print(f"  Target: {sizes['target_gb']:.3f} GB")
    if boundary_weights is not None:
        print(f"  Boundary weights: {sizes['boundary_weights_gb']:.3f} GB")
    if mask is not None:
        print(f"  Mask: {sizes['mask_gb']:.3f} GB")
    print(f"  Total: {sizes['total_gb']:.3f} GB")

    return sizes, write_time


def downsample_raw_to_gt(raw_volume, gt_shape):
    """
    Downsample raw volume to match GT shape.

    Parameters:
    -----------
    raw_volume : np.ndarray
        Raw data at high resolution (D, H, W)
    gt_shape : tuple
        Target shape (D, H, W) to match GT

    Returns:
    --------
    np.ndarray : Downsampled raw volume matching GT shape
    """
    return raw_volume
    if raw_volume.shape == gt_shape:
        return raw_volume

    # Calculate zoom factors
    zoom_factors = [gt_shape[i] / raw_volume.shape[i] for i in range(3)]

    print(f"  Downsampling raw from {raw_volume.shape} to {gt_shape}")
    print(f"  Zoom factors: {zoom_factors}")

    # Use order=1 (bilinear) for smooth downsampling
    downsampled = zoom(raw_volume, zoom_factors, order=1)

    # Ensure exact shape match (zoom can be slightly off due to rounding)
    if downsampled.shape != gt_shape:
        # Crop or pad to exact size
        slices = tuple(
            slice(0, min(downsampled.shape[i], gt_shape[i])) for i in range(3)
        )
        result = np.zeros(gt_shape, dtype=downsampled.dtype)
        result[slices] = downsampled[slices]
        downsampled = result

    return downsampled


def preprocess_and_save_volume(
    volume_index,
    output_dir,
    model_id="facebook/dinov3-vitl16-pretrain-sat493m",
    base_resolution=128,
    min_resolution_for_raw=32,
    output_image_dim=128,
    volume_size=None,
    organelle_list=None,
    crop_filter=None,
    inference_filter=None,
    min_label_fraction=0.01,
    min_unique_ids=2,
    min_ground_truth_fraction=0.05,
    boundary_weight=5,
    allow_gt_extension=True,
    use_compression=False,
    lsds_sigma=20.0,
    affinity_offsets=[
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
        (9, 0, 0),
        (0, 9, 0),
        (0, 0, 9),
    ],
    loss_type="boundary_affinity_focal_lsds",  # Use combined affinity+LSDS loss
    output_type="affinities_lsds",  # Enable affinity+LSDS output
    num_threads=None,
):
    """
    Preprocess a single volume: extract DINOv3 features and save to TensorStore.

    Parameters:
    -----------
    volume_index : int
        Index of volume to process (also used as random seed)
    output_dir : str
        Directory to save preprocessed volumes
    model_id : str
        DINOv3 model identifier
    base_resolution : int
        Base resolution in nm
    min_resolution_for_raw : int
        Minimum resolution for raw data
    output_image_dim : int
        Output image dimension
    volume_size : tuple
        Volume size (D, H, W), defaults to (output_image_dim,)*3
    organelle_list : list
        List of organelles to load, defaults to ["cell"]
    inference_filter : list
        Dataset filters, defaults to None
    min_label_fraction : float
        Minimum label fraction
    min_unique_ids : int
        Minimum unique IDs required
    min_ground_truth_fraction : float
        Minimum GT fraction
    allow_gt_extension : bool
        Allow GT extension
    use_compression : bool
        Use compression (LZ4 level 3) if True, no compression if False
    num_threads : int, optional
        Number of threads for TensorStore operations. If None, uses 2x LSB_DJOB_NUMPROC or 16 as default
    lsds_sigma : float
        Sigma for LSDS computation
    affinity_offsets : list
        Affinity offsets for output
    loss_type : str
        Loss type for training
    output_type : str
        Output type for feature extraction

    Returns:
    --------
    dict : Metadata about the processed volume
    """
    # Set defaults
    if volume_size is None:
        volume_size = (output_image_dim, output_image_dim, output_image_dim)
    if organelle_list is None:
        organelle_list = ["cell"]

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate DINOv3 slice size
    dinov3_slice_size = output_image_dim * base_resolution / min_resolution_for_raw

    print(f"\n{'='*60}")
    print(f"Processing Volume {volume_index}")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {volume_index}")
    print(f"Volume size: {volume_size}")
    print(f"DINOv3 slice size: {dinov3_slice_size}")

    # Initialize DINOv3 model
    print("\nInitializing DINOv3 model...")
    from dinov3_playground.memory_efficient_training import MemoryEfficientDataLoader3D

    for i in range(3):
        try:
            processor, model, output_channels = initialize_dinov3(
                model_id=model_id, image_size=dinov3_slice_size
            )
            break
        except Exception as e:
            import time

            time.sleep(0.5)
            if i == 2:  # Last attempt failed
                raise RuntimeError(
                    "Failed to initialize DINOv3 after 3 attempts"
                ) from e

    # Generate dataset pairs
    print(f"\nGenerating dataset pairs for organelles {organelle_list}...")
    dataset_pairs = generate_multi_organelle_dataset_pairs(
        organelle_list=organelle_list,
        min_resolution_for_raw=min_resolution_for_raw,
        base_resolution=base_resolution,
        crop_filter=crop_filter,
        apply_scale_updates=True,
        require_all_organelles=True,
        inference_filter=inference_filter,
    )
    print(f"Found {len(dataset_pairs)} dataset pairs")

    # Load the volume
    print(f"\nLoading volume with seed={volume_index}...")
    result = load_random_3d_training_data(
        dataset_pairs=dataset_pairs,
        volume_shape=volume_size,
        base_resolution=base_resolution,
        min_label_fraction=min_label_fraction,
        num_volumes=1,  # Process one at a time
        seed=volume_index,  # Use volume index as seed
        min_resolution_for_raw=min_resolution_for_raw,
        allow_gt_extension=allow_gt_extension,
        min_unique_ids=min_unique_ids,
        augment=True,
        augment_prob=1.0,
        min_ground_truth_fraction=min_ground_truth_fraction,
        max_attempts=3000,
    )

    # Unpack results
    raw, gt, gt_masks, dataset_sources, num_classes = result

    # Get the single volume
    raw_volume = raw[0]
    gt_volume = gt[0]
    gt_mask = gt_masks[0] if gt_masks is not None else None
    dataset_source = dataset_sources[0]

    # Compute boundary weights if needed
    boundary_weights = None
    if "boundary" in loss_type:
        print("\nComputing boundary weights...")
        boundary_weights = compute_boundary_weights(
            instance_segmentation=gt_volume,
            mask=gt_mask,
            boundary_weight=boundary_weight,
        )
        print(f"  Boundary weights shape: {boundary_weights.shape}")

    # Convert to affinities or affinities+lsds
    print(f"\nComputing targets (type: {output_type})...")
    if output_type == "affinities":
        target = compute_affinities_3d(gt_volume, offsets=affinity_offsets)
    elif output_type == "affinities_lsds":
        target = compute_affinities_and_lsds_3d(
            gt_volume, offsets=affinity_offsets, lsds_sigma=lsds_sigma
        )
    else:
        raise ValueError(f"Unknown output_type: {output_type}")

    print(f"  Target shape: {target.shape}")

    print(f"\nVolume loaded successfully:")
    print(f"  Raw shape: {raw_volume.shape}")
    print(f"  GT shape: {gt_volume.shape}")
    if gt_mask is not None:
        print(f"  GT mask shape: {gt_mask.shape}")
        print(f"  Valid GT fraction: {np.mean(gt_mask):.3f}")
    print(f"  Dataset source index: {dataset_source}")
    print(f"  Dataset path: {dataset_pairs[dataset_source]}")

    # Extract DINOv3 features
    print("\nExtracting DINOv3 features...")
    import time

    t0 = time.time()

    # Create a temporary data loader just for feature extraction
    data_loader_3d = MemoryEfficientDataLoader3D(
        dinov3_slice_size=dinov3_slice_size,
        raw_data=raw,
        gt_data=gt,
        train_volume_pool_size=1,
        val_volume_pool_size=0,
        target_volume_size=volume_size,
        seed=volume_index,
        model_id=model_id,
        output_type=output_type,
        affinity_offsets=affinity_offsets,
        lsds_sigma=lsds_sigma,
        use_anyup=True,
    )

    features = data_loader_3d.extract_dinov3_features_3d(raw_volume[np.newaxis, ...])
    extraction_time = time.time() - t0

    print(f"Feature extraction completed in {extraction_time:.2f}s")
    print(f"Features shape: {features.shape}")

    # Convert features to numpy
    features_np = features[0].cpu().numpy()

    # Write everything to TensorStore
    volume_name = f"volume_{volume_index:06d}"
    volume_path = output_dir / f"{volume_name}.zarr"
    raw_downsampled = downsample_raw_to_gt(raw_volume, gt_volume.shape)

    sizes, write_time = write_volume_to_tensorstore(
        output_path=volume_path,
        features=features_np,
        gt=gt_volume,
        target=target,
        raw=raw_downsampled,
        boundary_weights=boundary_weights,
        mask=gt_mask,
        use_compression=use_compression,
        num_threads=num_threads,
    )

    # Create metadata
    metadata = {
        "volume_index": volume_index,
        "volume_name": volume_name,
        "timestamp": datetime.now().isoformat(),
        "paths": {
            "volume": str(volume_path),
        },
        "source_dataset": {
            "index": int(dataset_source),
            "paths": dataset_pairs[dataset_source],
        },
        "shapes": {
            "raw": list(raw_volume.shape),
            "features": list(features_np.shape),
            "gt": list(gt_volume.shape),
            "target": list(target.shape),
            "boundary_weights": (
                list(boundary_weights.shape) if boundary_weights is not None else None
            ),
            "mask": list(gt_mask.shape) if gt_mask is not None else None,
        },
        "configuration": {
            "model_id": model_id,
            "base_resolution": base_resolution,
            "min_resolution_for_raw": min_resolution_for_raw,
            "output_image_dim": output_image_dim,
            "dinov3_slice_size": float(dinov3_slice_size),
            "organelle_list": organelle_list,
            "inference_filter": inference_filter,
            "compression": "lz4_level3" if use_compression else "none",
            "loss_type": loss_type,
            "output_type": output_type,
            "affinity_offsets": affinity_offsets,
            "lsds_sigma": lsds_sigma,
        },
        "timing": {
            "feature_extraction_seconds": extraction_time,
            "tensorstore_write_seconds": write_time,
            "total_seconds": extraction_time + write_time,
        },
        "storage": sizes,
        "statistics": {
            "num_classes": int(num_classes),
            "unique_classes": [int(x) for x in np.unique(gt_volume)],
            "valid_gt_fraction": (
                float(np.mean(gt_mask)) if gt_mask is not None else 1.0
            ),
        },
    }

    # Save metadata
    metadata_path = output_dir / f"{volume_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")
    print(f"\n{'='*60}")
    print(f"Volume {volume_index} preprocessing complete!")
    print(f"{'='*60}")

    return metadata


def load_preprocessed_volume(volume_path, num_threads=None, load_datasets=None):
    """
    Load a preprocessed volume from TensorStore.

    Parameters:
    -----------
    volume_path : str or Path
        Path to volume Zarr file or metadata JSON file
    num_threads : int, optional
        Number of threads for parallel loading. If None, uses 2x LSB_DJOB_NUMPROC or 16 as default
    load_datasets : list of str, optional
        Specific datasets to load. If None, loads all available datasets.
        Options: ['features', 'gt', 'target', 'boundary_weights', 'mask']

    Returns:
    --------
    dict : Dictionary containing loaded datasets and metadata
        Keys: 'features', 'gt', 'target', 'boundary_weights', 'mask', 'metadata'
        (only requested datasets will be present)
    """
    # Determine number of threads: 2x bsub allocation or default to 16
    if num_threads is None:
        bsub_threads = os.environ.get("LSB_DJOB_NUMPROC")
        if bsub_threads is not None:
            num_threads = int(bsub_threads) * 2
        else:
            num_threads = 16

    volume_path = Path(volume_path)

    # Find metadata file
    if volume_path.is_file():
        if volume_path.suffix == ".json":
            metadata_path = volume_path
            zarr_path = volume_path.parent / (
                volume_path.stem.replace("_metadata", "") + ".zarr"
            )
        else:
            # Assume it's the zarr file
            zarr_path = volume_path
            metadata_path = volume_path.parent / (volume_path.stem + "_metadata.json")
    else:
        # Assume it's a directory
        metadata_files = list(volume_path.glob("*_metadata.json"))
        if not metadata_files:
            raise ValueError(f"No metadata files found in {volume_path}")
        metadata_path = metadata_files[0]
        zarr_path = volume_path / metadata_path.stem.replace("_metadata", "") + ".zarr"

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Set up TensorStore context
    context = ts.Context(
        {
            "data_copy_concurrency": {"limit": num_threads},
        }
    )

    # Determine which datasets to load
    if load_datasets is None:
        load_datasets = ["features", "gt", "target"]
        # Add optional datasets if they exist
        if metadata["shapes"].get("boundary_weights") is not None:
            load_datasets.append("boundary_weights")
        if metadata["shapes"].get("mask") is not None:
            load_datasets.append("mask")

    result = {"metadata": metadata}

    # Load requested datasets
    for dataset_name in load_datasets:
        dataset_path = f"{dataset_name}"

        # Check if dataset exists in metadata
        if (
            dataset_name in ["boundary_weights", "mask"]
            and metadata["shapes"].get(dataset_name) is None
        ):
            print(f"  Skipping {dataset_name} (not present in volume)")
            continue

        print(f"  Loading {dataset_name}...")
        spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(zarr_path)},
            "path": dataset_path,
        }

        try:
            dataset = ts.open(spec, context=context, read=True).result()
            result[dataset_name] = dataset[:].read().result()
        except Exception as e:
            print(f"  Warning: Could not load {dataset_name}: {e}")

    return result


# %%
for i in range(1):

    boundary_weights = compute_boundary_weights(
        instance_segmentation=gt,
        mask=mask,
        boundary_weight=10,
    )

    print("  Writing boundary weights...")
    # Match chunking to actual array dimensions
    if len(boundary_weights.shape) == 4:
        weights_chunks = (min(boundary_weights.shape[0], 16), 64, 64, 64)
    else:  # 3D array
        weights_chunks = tuple(min(s, 64) for s in boundary_weights.shape)

    weights_spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": str(output_path),
        },
        "path": "boundary_weights",
        "metadata": {
            "shape": list(boundary_weights.shape),
            "chunks": list(weights_chunks),
            "dtype": "<f4",  # float32
            "compressor": compressor,
        },
    }
    weights_dataset = ts.open(weights_spec, create=True, context=context).result()
    weights_dataset[:] = boundary_weights
