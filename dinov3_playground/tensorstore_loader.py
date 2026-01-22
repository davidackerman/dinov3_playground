# %%
"""
Utility functions for loading preprocessed random crops from TensorStore zarr format.

This module provides convenience functions to load raw data and DINOv3 features
from the preprocessed random crop zarr files.
"""

import tensorstore as ts
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List
import json


class RandomCropLoader:
    """
    Loader for preprocessed random crop data stored in TensorStore zarr format.

    Example usage:
    ```
    loader = RandomCropLoader("/path/to/crop_000000.zarr")

    # Load all data
    raw = loader.load_raw()
    features_high_res = loader.load_features(scale="0.0.0.0")
    features_low_res = loader.load_features(scale="1.0.0.0")

    # Load specific regions
    raw_slice = loader.load_raw(slices=(slice(10, 20), slice(0, 100), slice(0, 100)))

    # Get available scales
    scales = loader.get_available_scales()
    ```
    """

    def __init__(self, zarr_path: Union[str, Path], num_threads: Optional[int] = None):
        """
        Initialize the loader.

        Parameters:
        -----------
        zarr_path : str or Path
            Path to the zarr file (e.g., "crop_000000.zarr")
        num_threads : int, optional
            Number of threads for parallel I/O. If None, uses 16.
        """
        self.zarr_path = Path(zarr_path)

        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr file not found: {self.zarr_path}")

        # Set up TensorStore context
        if num_threads is None:
            num_threads = 16

        self.context = ts.Context(
            {
                "data_copy_concurrency": {"limit": num_threads},
            }
        )

        self._store = None
        self._metadata = None

    def _open_dataset(self, path: str):
        """Open a specific dataset within the zarr store."""
        spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(self.zarr_path / path),
            },
        }
        return ts.open(spec, context=self.context).result()

    def load_raw(self, slices: Optional[Tuple[slice, ...]] = None) -> np.ndarray:
        """
        Load raw data from the zarr file.

        Parameters:
        -----------
        slices : tuple of slices, optional
            Specific region to load (e.g., (slice(0, 10), slice(0, 100), slice(0, 100)))
            If None, loads entire volume

        Returns:
        --------
        np.ndarray : Raw data as numpy array
        """
        raw_dataset = self._open_dataset("raw")

        if slices is None:
            data = raw_dataset[:].read().result()
        else:
            data = raw_dataset[slices].read().result()

        # Ensure it's a proper numpy array (not a TensorStore array-like)
        return np.array(data, copy=False)

    def load_features(
        self,
        slices: Optional[Tuple[slice, ...]] = None,
        channels: Optional[Union[int, List[int], slice]] = None,
    ) -> np.ndarray:
        """
        Load DINOv3 features from the zarr file.

        Parameters:
        -----------
        slices : tuple of slices, optional
            Specific spatial region to load (e.g., (slice(0, 10), slice(0, 100), slice(0, 100)))
            If None, loads entire volume
        channels : int, list of int, or slice, optional
            Specific channels to load. If None, loads all channels.
            Examples: 0, [0, 1, 2], slice(0, 64)

        Returns:
        --------
        np.ndarray : Features as numpy array with shape (C, D, H, W) or subset
        """
        features_dataset = self._open_dataset("features")

        # Build indexing tuple
        if channels is None:
            channel_idx = slice(None)
        elif isinstance(channels, int):
            channel_idx = channels
        elif isinstance(channels, list):
            channel_idx = channels
        else:  # slice
            channel_idx = channels

        if slices is None:
            spatial_idx = (slice(None), slice(None), slice(None))
        else:
            spatial_idx = slices

        full_idx = (channel_idx,) + spatial_idx

        data = features_dataset[full_idx].read().result()

        # Ensure it's a proper numpy array (not a TensorStore array-like)
        return np.array(data, copy=False)

    def get_shape(self, data_type: str = "features") -> Tuple[int, ...]:
        """
        Get the shape of a dataset without loading it.

        Parameters:
        -----------
        data_type : str
            Either "raw" or "features"

        Returns:
        --------
        tuple : Shape of the dataset
        """
        if data_type == "raw":
            dataset = self._open_dataset("raw")
        elif data_type == "features":
            dataset = self._open_dataset("features")
        else:
            raise ValueError(f"data_type must be 'raw' or 'features', got {data_type}")

        return tuple(dataset.shape)

    def get_dtype(self, data_type: str = "features") -> np.dtype:
        """
        Get the dtype of a dataset without loading it.

        Parameters:
        -----------
        data_type : str
            Either "raw" or "features"

        Returns:
        --------
        np.dtype : Data type of the dataset
        """
        if data_type == "raw":
            dataset = self._open_dataset("raw")
        elif data_type == "features":
            dataset = self._open_dataset("features")
        else:
            raise ValueError(f"data_type must be 'raw' or 'features', got {data_type}")

        return dataset.dtype.numpy_dtype

    def load_metadata(self) -> Optional[Dict]:
        """
        Load metadata JSON file if it exists.

        Returns:
        --------
        dict or None : Metadata dictionary, or None if not found
        """
        if self._metadata is not None:
            return self._metadata

        # Look for metadata file
        crop_name = self.zarr_path.stem  # e.g., "crop_000000"
        metadata_path = self.zarr_path.parent / f"{crop_name}_metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self._metadata = json.load(f)
            return self._metadata
        else:
            return None

    def __repr__(self) -> str:
        """String representation of the loader."""
        raw_shape = self.get_shape("raw")
        features_shape = self.get_shape("features")

        return (
            f"RandomCropLoader(\n"
            f"  path={self.zarr_path}\n"
            f"  raw_shape={raw_shape}\n"
            f"  features_shape={features_shape}\n"
            f")"
        )


def load_crop_raw(
    zarr_path: Union[str, Path],
    slices: Optional[Tuple[slice, ...]] = None,
    num_threads: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function to load raw data from a preprocessed crop.

    Parameters:
    -----------
    zarr_path : str or Path
        Path to the zarr file
    slices : tuple of slices, optional
        Specific region to load
    num_threads : int, optional
        Number of threads for parallel I/O

    Returns:
    --------
    np.ndarray : Raw data
    """
    loader = RandomCropLoader(zarr_path, num_threads=num_threads)
    return loader.load_raw(slices=slices)


def load_crop_features(
    zarr_path: Union[str, Path],
    slices: Optional[Tuple[slice, ...]] = None,
    channels: Optional[Union[int, List[int], slice]] = None,
    num_threads: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function to load features from a preprocessed crop.

    Parameters:
    -----------
    zarr_path : str or Path
        Path to the zarr file
    slices : tuple of slices, optional
        Specific spatial region to load
    channels : int, list of int, or slice, optional
        Specific channels to load
    num_threads : int, optional
        Number of threads for parallel I/O

    Returns:
    --------
    np.ndarray : Features
    """
    loader = RandomCropLoader(zarr_path, num_threads=num_threads)
    return loader.load_features(slices=slices, channels=channels)


def load_crop_metadata(zarr_path: Union[str, Path]) -> Optional[Dict]:
    """
    Convenience function to load metadata from a preprocessed crop.

    Parameters:
    -----------
    zarr_path : str or Path
        Path to the zarr file

    Returns:
    --------
    dict or None : Metadata dictionary
    """
    loader = RandomCropLoader(zarr_path)
    return loader.load_metadata()


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tensorstore_loader.py <path_to_crop.zarr>")
        sys.exit(1)

    zarr_path = f"/nrs/cellmap/to_delete/jrc_c-elegans-bw-1/crop_000000.zarr"

    print("Loading crop data...\n")

    # Create loader
    loader = RandomCropLoader(zarr_path)
    print(loader)
    print()

    # Load metadata
    metadata = loader.load_metadata()
    if metadata:
        print("Metadata:")
        print(f"  Crop index: {metadata.get('crop_index')}")
        print(f"  Timestamp: {metadata.get('timestamp')}")
        print(
            f"  Input resolution: {metadata.get('configuration', {}).get('input_resolution')} nm"
        )
        print(
            f"  Output resolution: {metadata.get('configuration', {}).get('output_resolution')} nm"
        )
        print(f"  Use AnyUp: {metadata.get('configuration', {}).get('use_anyup')}")
        print()

    # Load raw data
    print("Loading raw data...")
    raw = loader.load_raw()
    print(f"  Shape: {raw.shape}")
    print(f"  Dtype: {raw.dtype}")
    print(f"  Min: {raw.min()}, Max: {raw.max()}")
    print()

    # Load features
    print("Loading features...")
    features = loader.load_features()
    print(f"  Shape: {features.shape}")
    print(f"  Dtype: {features.dtype}")
    print(f"  Min: {features.min():.4f}, Max: {features.max():.4f}")
    print()

    # Example: Load a subset
    print("Loading subset (first 10 slices, first 64 channels)...")
    features_subset = loader.load_features(
        slices=(slice(0, 10), slice(None), slice(None)),
        channels=slice(0, 64),
    )
    print(f"  Shape: {features_subset.shape}")
    print()

    print("Done!")

# %%
# from dinov3_playground.tensorstore_loader import RandomCropLoader

# loader = RandomCropLoader(
#     "/nrs/cellmap/to_delete/jrc_c-elegans-bw-1/anyup/crop_000000.zarr"
# )
# print(loader)

# raw = loader.load_raw()
# features = loader.load_features()
# # %%
# # plot first 10 features and raw slices
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6 * 10))
# for i in range(10):
#     plt.subplot(10, 2, i * 2 + 1)
#     plt.imshow(features[i, 64, :, :])
#     plt.title(f"Feature {i}")
#     plt.subplot(10, 2, i * 2 + 2)
#     plt.imshow(
#         (raw[256] - raw[256].min()) / (raw[256].max() - raw[256].min()), cmap="gray"
#     )
#     plt.title(f"Raw {i}")
# # %%
