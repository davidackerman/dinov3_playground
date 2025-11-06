"""
Compute and save boundary weights for preprocessed volumes.
"""

import numpy as np
import tensorstore as ts
from pathlib import Path
import json
from typing import Optional
import time

from affinity_utils import compute_boundary_weights

# Import your existing compute_boundary_weights function here
# from your_module import compute_boundary_weights


def process_volume(
    volume_path: Path, boundary_weight: float = 10.0, num_threads: int = 16
):
    """
    Process a single volume: load gt and mask, compute boundary weights, save.

    Parameters:
    -----------
    volume_path : Path
        Path to the .zarr volume directory
    boundary_weight : float
        Weight to apply to boundary pixels
    num_threads : int
        Number of threads for TensorStore operations
    """
    print(f"\nProcessing {volume_path.name}...")

    # Set up TensorStore context
    context = ts.Context(
        {
            "cache_pool": {"total_bytes_limit": 2_000_000_000},
            "data_copy_concurrency": {"limit": num_threads},
        }
    )

    # Load GT
    print("  Loading GT...")
    t0 = time.time()
    gt_spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": str(volume_path),
        },
        "path": "gt",
    }
    gt_dataset = ts.open(gt_spec, context=context, read=True).result()
    gt = gt_dataset[:].read().result()
    print(f"    GT shape: {gt.shape}, dtype: {gt.dtype}, time: {time.time()-t0:.2f}s")

    # Load mask (if it exists)
    mask_path = volume_path / "mask"
    if mask_path.exists():
        print("  Loading mask...")
        t0 = time.time()
        mask_spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(volume_path),
            },
            "path": "mask",
        }
        mask_dataset = ts.open(mask_spec, context=context, read=True).result()
        mask = mask_dataset[:].read().result()
        print(
            f"    Mask shape: {mask.shape}, dtype: {mask.dtype}, time: {time.time()-t0:.2f}s"
        )
    else:
        print("  No mask found, using None")
        mask = None

    # Compute boundary weights
    print(f"  Computing boundary weights (weight={boundary_weight})...")
    t0 = time.time()
    boundary_weights = compute_boundary_weights(
        instance_segmentation=gt,
        mask=mask,
        boundary_weight=boundary_weight,
    )
    print(
        f"    Boundary weights shape: {boundary_weights.shape}, time: {time.time()-t0:.2f}s"
    )

    # Save boundary weights
    print("  Writing boundary weights...")
    t0 = time.time()

    # Match chunking to actual array dimensions
    if len(boundary_weights.shape) == 4:
        weights_chunks = (min(boundary_weights.shape[0], 16), 64, 64, 64)
    else:  # 3D array
        weights_chunks = tuple(min(s, 64) for s in boundary_weights.shape)

    weights_spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": str(volume_path),
        },
        "path": "boundary_weights",
        "metadata": {
            "shape": list(boundary_weights.shape),
            "chunks": list(weights_chunks),
            "dtype": "<f4",  # float32
        },
    }
    weights_dataset = ts.open(weights_spec, create=True, context=context).result()
    weights_dataset[:] = boundary_weights
    print(f"    Write time: {time.time()-t0:.2f}s")

    print(f"  ✓ Done processing {volume_path.name}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute boundary weights for preprocessed volumes"
    )
    parser.add_argument(
        "--volumes-dir",
        type=str,
        required=True,
        help="Directory containing volume_*.zarr directories",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=10.0,
        help="Weight to apply to boundary pixels (default: 10.0)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=16,
        help="Number of threads for TensorStore (default: 16)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start from this volume index (optional)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End at this volume index (optional)",
    )

    args = parser.parse_args()

    volumes_dir = Path(args.volumes_dir)

    # Find all zarr volumes
    zarr_volumes = sorted(volumes_dir.glob("volume_*.zarr"))

    if not zarr_volumes:
        print(f"ERROR: No volume_*.zarr directories found in {volumes_dir}")
        return

    print(f"Found {len(zarr_volumes)} volumes in {volumes_dir}")

    # Filter by index if specified
    if args.start_index is not None or args.end_index is not None:

        def get_volume_index(path):
            # Extract index from volume_XXXXXX.zarr
            return int(path.stem.split("_")[1])

        filtered_volumes = []
        for vol in zarr_volumes:
            idx = get_volume_index(vol)
            if args.start_index is not None and idx < args.start_index:
                continue
            if args.end_index is not None and idx > args.end_index:
                continue
            filtered_volumes.append(vol)

        zarr_volumes = filtered_volumes
        print(f"Processing {len(zarr_volumes)} volumes (filtered by index range)")

    print(f"Boundary weight: {args.boundary_weight}")
    print(f"Threads: {args.num_threads}")
    print("=" * 60)

    # Process each volume
    total_start = time.time()
    for i, volume_path in enumerate(zarr_volumes):
        print(f"\n[{i+1}/{len(zarr_volumes)}]")
        try:
            process_volume(
                volume_path=volume_path,
                boundary_weight=args.boundary_weight,
                num_threads=args.num_threads,
            )
        except Exception as e:
            print(f"  ERROR processing {volume_path.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"DONE! Processed {len(zarr_volumes)} volumes in {total_time:.1f}s")
    print(f"Average time per volume: {total_time/len(zarr_volumes):.1f}s")


if __name__ == "__main__":
    main()
