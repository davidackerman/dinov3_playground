"""
PyTorch DataLoader for preprocessed DINOv3 features stored in TensorStore.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tensorstore as ts
from pathlib import Path
import json
from typing import List, Optional, Tuple, Dict


class PreprocessedDINOv3Dataset(Dataset):
    """
    Dataset for loading preprocessed DINOv3 features from TensorStore.

    This dataset loads features that were preprocessed and cached using
    the preprocess_volume.py script. Features are stored in TensorStore
    format for fast parallel loading.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        volume_indices: Optional[List[int]] = None,
        num_threads: Optional[int] = None,
        transform=None,
        return_metadata: bool = False,
        load_boundary_weights: bool = True,
        load_raw: bool = False,
        features_dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters:
        -----------
        preprocessed_dir : str
            Directory containing preprocessed volumes
        volume_indices : list of int, optional
            Specific volume indices to use. If None, uses all volumes found.
        num_threads : int, optional
            Number of threads for TensorStore parallel I/O. If None, uses 2x LSB_DJOB_NUMPROC or 16 as default
        transform : callable, optional
            Optional transform to apply to samples
        return_metadata : bool
            If True, returns metadata dict along with samples
        load_boundary_weights : bool
            If True, loads boundary weights (if available)
        features_dtype : torch.dtype
            Desired dtype for features (torch.float16 or torch.float32). Using float16 saves memory and speeds up loading.
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.load_boundary_weights = load_boundary_weights
        self.load_raw = load_raw
        self.features_dtype = features_dtype

        # Determine number of threads: 2x bsub allocation or default to 16
        if num_threads is None:
            bsub_threads = os.environ.get("LSB_DJOB_NUMPROC")
            if bsub_threads is not None:
                self.num_threads = int(bsub_threads) * 2
                print(
                    f"Using {self.num_threads} threads (2x LSB_DJOB_NUMPROC={bsub_threads})"
                )
            else:
                self.num_threads = 16
                print(
                    f"Using default {self.num_threads} threads (LSB_DJOB_NUMPROC not set)"
                )
        else:
            self.num_threads = num_threads

        self.transform = transform
        self.return_metadata = return_metadata

        # Find all metadata files
        all_metadata_files = sorted(
            self.preprocessed_dir.glob("volume_*_metadata.json")
        )

        if not all_metadata_files:
            raise ValueError(f"No preprocessed volumes found in {preprocessed_dir}")

        # Filter by volume indices if specified
        if volume_indices is not None:
            volume_index_set = set(volume_indices)
            self.metadata_files = [
                mf
                for mf in all_metadata_files
                if self._get_volume_index(mf) in volume_index_set
            ]
        else:
            self.metadata_files = all_metadata_files

        if not self.metadata_files:
            raise ValueError(f"No volumes found matching indices: {volume_indices}")

        # Load all metadata
        self.metadatas = []
        for mf in self.metadata_files:
            with open(mf, "r") as f:
                self.metadatas.append(json.load(f))

        # Set up TensorStore context for efficient loading
        self.ts_context = ts.Context(
            {
                "cache_pool": {"total_bytes_limit": 2_000_000_000},  # 2GB cache
                "data_copy_concurrency": {"limit": self.num_threads},
            }
        )

        # Pre-open all TensorStore datasets for faster access
        self._open_tensorstores()

        print(f"PreprocessedDINOv3Dataset initialized:")
        print(f"  Directory: {preprocessed_dir}")
        print(f"  Number of volumes: {len(self)}")
        print(
            f"  Volume indices: {min(self.get_volume_indices())} to {max(self.get_volume_indices())}"
        )
        print(f"  Threads: {self.num_threads}")
        print(f"  Loading boundary weights: {self.load_boundary_weights}")
        print(f"  Features dtype: {self.features_dtype}")

    def _get_volume_index(self, metadata_file: Path) -> int:
        """Extract volume index from metadata filename."""
        # Format: volume_XXXXXX_metadata.json
        return int(metadata_file.stem.split("_")[1])

    def _open_tensorstores(self):
        """Pre-open all TensorStore datasets for faster loading."""
        self.feature_stores = []
        self.gt_stores = []
        self.target_stores = []
        self.boundary_weight_stores = []
        self.mask_stores = []
        self.raw_stores = []

        for metadata in self.metadatas:
            # Get the main zarr path
            zarr_path = metadata["paths"]["volume"]

            # Open features - explicitly handle float16
            features_spec = {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": zarr_path,
                },
                "path": "features",
                "dtype": "float16",  # Explicitly specify to avoid auto-conversion
            }
            self.feature_stores.append(
                ts.open(features_spec, context=self.ts_context, read=True).result()
            )

            # Open target
            target_spec = {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": zarr_path,
                },
                "path": "target",
            }
            self.target_stores.append(
                ts.open(target_spec, context=self.ts_context, read=True).result()
            )
            # Open GT
            gt_spec = {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": zarr_path,
                },
                "path": "gt",
            }
            self.gt_stores.append(
                ts.open(gt_spec, context=self.ts_context, read=True).result()
            )
            # Open boundary weights if present and requested
            if self.load_boundary_weights and metadata["shapes"].get(
                "boundary_weights"
            ):
                weights_spec = {
                    "driver": "zarr",
                    "kvstore": {
                        "driver": "file",
                        "path": zarr_path,
                    },
                    "path": "boundary_weights",
                }
                self.boundary_weight_stores.append(
                    ts.open(weights_spec, context=self.ts_context, read=True).result()
                )
            else:
                self.boundary_weight_stores.append(None)

            if self.load_raw and metadata["shapes"].get("raw"):
                # open raw
                raw_spec = {
                    "driver": "zarr",
                    "kvstore": {
                        "driver": "file",
                        "path": zarr_path,
                    },
                    "path": "raw",
                }
                self.raw_stores.append(
                    ts.open(raw_spec, context=self.ts_context, read=True).result()
                )
            else:
                self.raw_stores.append(None)

            # Open mask if present
            if metadata["shapes"].get("mask"):
                mask_spec = {
                    "driver": "zarr",
                    "kvstore": {
                        "driver": "file",
                        "path": zarr_path,
                    },
                    "path": "mask",
                }
                self.mask_stores.append(
                    ts.open(mask_spec, context=self.ts_context, read=True).result()
                )
            else:
                self.mask_stores.append(None)

    def __len__(self) -> int:
        return len(self.metadata_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Load a preprocessed volume.

        Returns:
        --------
        If return_metadata=False:
            (features, gt, target, boundary_weights, mask)
            where boundary_weights and mask may be None
        If return_metadata=True:
            (features, gt, target, boundary_weights, mask, metadata_dict)
        """
        import time

        # Track loading times for benchmarking
        load_times = {}

        # Load features
        t0 = time.time()
        features = self.feature_stores[idx][:].read().result()
        load_times["features"] = (time.time() - t0) * 1000

        # Load target
        t0 = time.time()
        target = self.target_stores[idx][:].read().result()
        load_times["target"] = (time.time() - t0) * 1000

        # Load GT
        t0 = time.time()
        gt = self.gt_stores[idx][:].read().result()
        load_times["gt"] = (time.time() - t0) * 1000
        # GT: convert directly, TensorStore returns uint8
        gt_torch = torch.from_numpy(gt)

        # Load boundary weights
        if self.boundary_weight_stores[idx] is not None:
            t0 = time.time()
            boundary_weights = self.boundary_weight_stores[idx][:].read().result()
            load_times["boundary_weights"] = (time.time() - t0) * 1000
        else:
            boundary_weights = []
            load_times["boundary_weights"] = 0

        # Load mask
        if self.mask_stores[idx] is not None:
            t0 = time.time()
            mask = self.mask_stores[idx][:].read().result()
            load_times["mask"] = (time.time() - t0) * 1000
        else:
            mask = None
            load_times["mask"] = 0

        # Convert to torch tensors (optimized for speed)
        t0 = time.time()

        # Direct conversion without intermediate copy
        # from_numpy creates a tensor that shares memory with the numpy array
        features_torch = torch.from_numpy(features)
        # Convert to desired dtype if needed (stored as float16, user might want float32 or float16)
        if features_torch.dtype != self.features_dtype:
            features = features_torch.to(self.features_dtype)
        else:
            features = features_torch

        if self.load_raw:
            # Load Raw
            t0 = time.time()
            raw = self.raw_stores[idx][:].read().result()
            load_times["raw"] = (time.time() - t0) * 1000
            raw_torch = torch.from_numpy(raw)
        else:
            raw = []
            load_times["raw"] = 0

        # Target: should already be float32
        target_torch = torch.from_numpy(target)
        if target_torch.dtype != torch.float32:
            target = target_torch.to(torch.float32)
        else:
            target = target_torch

        if boundary_weights is not None:
            bw_torch = torch.from_numpy(boundary_weights)
            if bw_torch.dtype != torch.float32:
                boundary_weights = bw_torch.to(torch.float32)
            else:
                boundary_weights = bw_torch

        if mask is not None:
            mask_torch = torch.from_numpy(mask)
            if mask_torch.dtype != torch.float32:
                mask = mask_torch.to(torch.float32)
            else:
                mask = mask_torch

        load_times["conversion"] = (time.time() - t0) * 1000

        # Store timing info as an attribute for benchmarking
        self._last_load_times = load_times

        # Apply transform if specified
        if self.transform:
            features, gt, target, boundary_weights, mask = self.transform(
                features, gt, target, boundary_weights, mask
            )

        if self.return_metadata:
            return (
                raw,
                features,
                gt,
                target,
                boundary_weights,
                mask,
                self.metadatas[idx],
            )
        else:
            return raw, features, gt, target, boundary_weights, mask

    def get_volume_indices(self) -> List[int]:
        """Return list of volume indices in this dataset."""
        return [m["volume_index"] for m in self.metadatas]

    def get_metadata(self, idx: int) -> dict:
        """Get metadata for a specific sample."""
        return self.metadatas[idx]

    def get_dataset_sources(self) -> List[str]:
        """Get list of source dataset paths."""
        return [str(m["source_dataset"]["paths"]) for m in self.metadatas]


def create_preprocessed_dataloader(
    preprocessed_dir: str,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    volume_indices: Optional[List[int]] = None,
    num_threads: Optional[int] = None,
    load_boundary_weights: bool = True,
    features_dtype: torch.dtype = torch.float32,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for preprocessed volumes.

    Parameters:
    -----------
    preprocessed_dir : str
        Directory containing preprocessed volumes
    batch_size : int
        Batch size
    shuffle : bool
        Shuffle data
    num_workers : int
        Number of DataLoader workers (use 0 for TensorStore's internal parallelism)
    volume_indices : list of int, optional
        Specific volume indices to use
    num_threads : int, optional
        Number of threads for TensorStore operations. If None, uses 2x LSB_DJOB_NUMPROC or 16 as default
    load_boundary_weights : bool
        If True, loads boundary weights (if available)
    features_dtype : torch.dtype
        Desired dtype for features (torch.float16 or torch.float32). Using float16 saves memory and speeds up loading.
    **kwargs : additional arguments passed to DataLoader

    Returns:
    --------
    DataLoader
    """
    dataset = PreprocessedDINOv3Dataset(
        preprocessed_dir=preprocessed_dir,
        volume_indices=volume_indices,
        num_threads=num_threads,
        load_boundary_weights=load_boundary_weights,
        features_dtype=features_dtype,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )


# Example usage and benchmarking
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Benchmark preprocessed volume loading"
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        required=True,
        help="Directory containing preprocessed volumes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches to load for benchmark",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of threads for TensorStore (defaults to 2x LSB_DJOB_NUMPROC)",
    )
    parser.add_argument(
        "--features-dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Dtype for features (float16 saves memory and speeds up loading)",
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    features_dtype = (
        torch.float16 if args.features_dtype == "float16" else torch.float32
    )

    print(f"\n{'='*60}")
    print("Preprocessed Volume Loading Benchmark")
    print(f"{'='*60}")

    # Create DataLoader
    dataloader = create_preprocessed_dataloader(
        preprocessed_dir=args.preprocessed_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        num_threads=args.num_threads,
        features_dtype=features_dtype,
    )

    print(f"\nDataset size: {len(dataloader.dataset)} volumes")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(dataloader)}")

    # Benchmark loading
    print(f"\n{'='*60}")
    print(f"Loading {args.num_batches} batches...")
    print(f"{'='*60}")

    times = []
    batch_iter = iter(dataloader)
    for i in range(min(args.num_batches, len(dataloader))):
        t_total_start = time.time()

        # This is when __getitem__ gets called and actual loading happens
        batch = next(batch_iter)
        raw, features, gt, target, boundary_weights, mask = batch

        total_time = (time.time() - t_total_start) * 1000

        # Get the detailed timing info from the dataset
        load_times = dataloader.dataset._last_load_times

        # Calculate sizes
        features_size_mb = features.element_size() * features.nelement() / (1024**2)
        gt_size_mb = gt.element_size() * gt.nelement() / (1024**2)
        target_size_mb = target.element_size() * target.nelement() / (1024**2)
        bw_size_mb = (
            boundary_weights.element_size() * boundary_weights.nelement() / (1024**2)
            if boundary_weights is not None
            else 0
        )
        mask_size_mb = (
            mask.element_size() * mask.nelement() / (1024**2) if mask is not None else 0
        )
        total_size_mb = (
            features_size_mb + gt_size_mb + target_size_mb + bw_size_mb + mask_size_mb
        )

        print(f"\n{'='*60}")
        print(f"Batch {i+1}:")
        print(f"{'='*60}")

        print(f"Features:")
        print(f"  Shape: {features.shape}")
        print(f"  Dtype: {features.dtype}")
        print(f"  Size: {features_size_mb:.2f} MB")
        print(f"  TensorStore read time: {load_times['features']:.1f} ms")
        print(
            f"  Throughput: {features_size_mb / (load_times['features'] / 1000) if load_times['features'] > 0 else 0:.1f} MB/s"
        )

        print(f"GT:")
        print(f"  Shape: {gt.shape}")
        print(f"  Dtype: {gt.dtype}")
        print(f"  Size: {gt_size_mb:.2f} MB")
        print(f"  TensorStore read time: {load_times['gt']:.1f} ms")

        print(f"Target:")
        print(f"  Shape: {target.shape}")
        print(f"  Dtype: {target.dtype}")
        print(f"  Size: {target_size_mb:.2f} MB")
        print(f"  TensorStore read time: {load_times['target']:.1f} ms")

        if boundary_weights is not None:
            print(f"Boundary Weights:")
            print(f"  Shape: {boundary_weights.shape}")
            print(f"  Dtype: {boundary_weights.dtype}")
            print(f"  Size: {bw_size_mb:.2f} MB")
            print(f"  TensorStore read time: {load_times['boundary_weights']:.1f} ms")

        if mask is not None:
            print(f"Mask:")
            print(f"  Shape: {mask.shape}")
            print(f"  Dtype: {mask.dtype}")
            print(f"  Size: {mask_size_mb:.2f} MB")
            print(f"  TensorStore read time: {load_times['mask']:.1f} ms")

        tensorstore_total = sum(
            load_times[k]
            for k in ["features", "gt", "target", "boundary_weights", "mask"]
        )

        print(f"\nTiming breakdown:")
        print(f"  TensorStore reads: {tensorstore_total:.1f} ms")
        print(f"  Numpy->Torch conversion: {load_times['conversion']:.1f} ms")
        print(
            f"  Other overhead: {total_time - tensorstore_total - load_times['conversion']:.1f} ms"
        )
        print(f"  Total time: {total_time:.1f} ms")
        print(f"\nData size: {total_size_mb:.2f} MB")
        print(f"Overall throughput: {total_size_mb / (total_time / 1000):.1f} MB/s")

        times.append(total_time)

    print(f"\n{'='*60}")
    print("Benchmark Results")
    print(f"{'='*60}")
    print(f"Cold load (first batch): {times[0]:.1f}ms")
    if len(times) > 1:
        warm_times = times[1:]
        print(f"Warm loads (cached):")
        print(f"  Mean: {np.mean(warm_times):.1f}ms")
        print(f"  Std: {np.std(warm_times):.1f}ms")
        print(f"  Min: {np.min(warm_times):.1f}ms")
        print(f"  Max: {np.max(warm_times):.1f}ms")

        print(f"\nSpeedup vs cold: {times[0] / np.mean(warm_times):.1f}x")
