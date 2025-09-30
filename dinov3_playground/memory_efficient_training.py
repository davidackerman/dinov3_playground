"""
Memory-Efficient Training Module for DINOv3 Feature Classification

This module contains all the classes and functions for memory-efficient training
of neural network classifiers on DINOv3 features, including:
- MemoryEfficientDataLoader for on-demand batch sampling
- Training functions with checkpoint saving
- Model loading and inference functions
- Checkpoint management utilities

Author: GitHub Copilot
Date: 2025-09-11
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import torch.optim as optim
import time
from tqdm import tqdm
from datetime import datetime
from .dinov3_core import (
    process,
    normalize_features,
    apply_normalization_stats,
    output_channels,
)

# Handle both relative and absolute imports
try:
    from .data_processing import sample_training_data
    from .models import ImprovedClassifier, SimpleClassifier
    from .model_training import balance_classes
except ImportError:
    from dinov3_playground.data_processing import sample_training_data
    from dinov3_playground.models import ImprovedClassifier, SimpleClassifier
    from dinov3_playground.model_training import balance_classes
from torch.amp import autocast  # Updated import
from torch.cuda.amp import GradScaler  # Keep GradScaler from cuda.amp

# Add these imports at the top if not already present:
import pickle
import glob
import os
from datetime import datetime


class MemoryEfficientDataLoader3D:
    """
    Memory-efficient data loader for 3D DINOv3 UNet training.
    Handles 3D volumes instead of 2D images.
    """

    def __init__(
        self,
        raw_data,
        gt_data,
        train_volume_pool_size=20,
        val_volume_pool_size=5,
        target_volume_size=(64, 64, 64),
        dinov3_slice_size=896,
        seed=42,
        model_id=None,
        learn_upsampling=False,  # NEW PARAMETER
        dinov3_stride=None,  # NEW PARAMETER for sliding window inference
    ):
        """
        Initialize memory-efficient 3D data loader.

        Parameters:
        -----------
        raw_data : numpy.ndarray
            4D raw volume data (num_volumes, D, H, W)
        gt_data : numpy.ndarray
            4D ground truth data (num_volumes, D, H, W)
        train_volume_pool_size : int, default=20
            Number of possible training volumes to sample from
        val_volume_pool_size : int, default=5
            Number of validation volumes (fixed)
        target_volume_size : tuple, default=(64, 64, 64)
            Target 3D volume size (D, H, W)
        dinov3_slice_size : int, default=896
            Size for processing each 2D slice through DINOv3
        seed : int, default=42
            Random seed for reproducibility
        model_id : str, optional
            DINOv3 model identifier to use
        learn_upsampling : bool, default=False
            Whether to use learned upsampling in the UNet
        dinov3_stride : int, optional
            Stride for DINOv3 sliding window inference. If None, uses patch_size (16).
            Use smaller values (e.g., 8, 4) for higher resolution features.
        """

        # Validate inputs with ROI-level padding support
        if raw_data.shape[0] != gt_data.shape[0]:
            raise ValueError(
                f"Number of volumes must match: {raw_data.shape[0]} vs {gt_data.shape[0]}"
            )

        if len(raw_data.shape) >= 2 and raw_data.shape[1] != gt_data.shape[1]:
            raise ValueError(
                f"Depth dimension must match: {raw_data.shape[1]} vs {gt_data.shape[1]}"
            )

        # Allow spatial dimensions to differ due to ROI-level padding
        if len(raw_data.shape) >= 4:
            raw_spatial = raw_data.shape[2:]
            gt_spatial = gt_data.shape[2:]
            if raw_spatial != gt_spatial:
                roi_padding = 0
                if dinov3_stride is not None and dinov3_stride < 16:
                    roi_padding = 16 - dinov3_stride
                expected_raw_spatial = tuple(
                    dim + 2 * roi_padding for dim in gt_spatial
                )
                if raw_spatial != expected_raw_spatial:
                    raise ValueError(
                        f"Raw spatial dimensions {raw_spatial} don't match GT {gt_spatial} "
                        f"or expected padded dimensions {expected_raw_spatial} for stride {dinov3_stride}"
                    )

        if len(raw_data.shape) != 4:
            raise ValueError(
                f"Expected 4D data (num_volumes, D, H, W), got {raw_data.shape}"
            )

        total_volumes = len(raw_data)
        if total_volumes < train_volume_pool_size + val_volume_pool_size:
            raise ValueError(
                f"Not enough volumes: need {train_volume_pool_size + val_volume_pool_size}, "
                f"got {total_volumes}"
            )

        self.raw_data = raw_data
        self.gt_data = gt_data
        self.train_volume_pool_size = train_volume_pool_size
        self.val_volume_pool_size = val_volume_pool_size
        self.target_volume_size = target_volume_size
        self.dinov3_slice_size = dinov3_slice_size
        self.model_id = model_id
        self.seed = seed  # Store seed for config saving
        self.learn_upsampling = learn_upsampling  # Store upsampling mode
        self.dinov3_stride = dinov3_stride  # Store sliding window stride

        # Calculate padding requirements for sliding window
        self._calculate_padding_requirements()

        # Set up random state
        self.rng = np.random.RandomState(seed)

        # Split data into train/val pools
        total_volumes = len(self.raw_data)
        all_indices = np.arange(total_volumes)
        self.rng.shuffle(all_indices)

        # Fixed validation volumes
        self.val_indices = all_indices[:val_volume_pool_size]

        # Pool of possible training volumes
        self.train_pool_indices = all_indices[
            val_volume_pool_size : val_volume_pool_size + train_volume_pool_size
        ]

        print(f"3D Data loader initialized:")
        print(f"  - Total volumes: {total_volumes}")
        print(f"  - Training pool: {len(self.train_pool_indices)} volumes")
        print(f"  - Validation set: {len(self.val_indices)} volumes")
        print(f"  - Target volume size: {target_volume_size}")
        print(f"  - DINOv3 slice size: {dinov3_slice_size}")
        if dinov3_stride is not None:
            print(
                f"  - DINOv3 stride: {dinov3_stride} (sliding window inference enabled)"
            )
            # Calculate padding requirements
            self._calculate_padding_requirements()
        else:
            print(f"  - DINOv3 stride: 16 (standard inference)")
            self.sliding_window_padding = 0

    def _calculate_padding_requirements(self):
        """Calculate padding requirements for sliding window inference."""
        if self.dinov3_stride is not None and self.dinov3_stride < 16:
            # Calculate maximum shift needed for info
            patch_size = 16  # DINOv3 patch size
            max_shift = patch_size - self.dinov3_stride
            print(
                f"  - Sliding window context: {max_shift} pixels (handled at ROI level)"
            )
            self.sliding_window_padding = (
                0  # No per-slice padding needed with ROI-level padding
            )
        else:
            self.sliding_window_padding = 0

    def sample_training_batch(self, batch_size):
        """
        Sample a new batch of training volumes.

        Parameters:
        -----------
        batch_size : int
            Number of volumes to sample

        Returns:
        --------
        tuple: (batch_volumes, batch_gt) of shape (batch_size, D, H, W)
        """
        # Sample random volumes from training pool
        sampled_indices = self.rng.choice(
            self.train_pool_indices,
            size=batch_size,
            replace=len(self.train_pool_indices) < batch_size,
        )

        batch_volumes = self.raw_data[sampled_indices]
        batch_gt = self.gt_data[sampled_indices]

        return batch_volumes, batch_gt

    def get_validation_data(self):
        """
        Get the fixed validation dataset.

        Returns:
        --------
        tuple: (val_volumes, val_gt) of shape (val_pool_size, D, H, W)
        """
        val_volumes = self.raw_data[self.val_indices]
        val_gt = self.gt_data[self.val_indices]

        return val_volumes, val_gt

    def extract_dinov3_features_3d(self, volumes, epoch=None, batch=None):
        """
        Extract DINOv3 features from 3D volumes by processing each slice.
        """
        from .dinov3_core import process, get_current_model_info
        from skimage.transform import resize

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if volumes.ndim == 3:
            volumes = volumes[np.newaxis, ...]

        batch_size, depth, height, width = volumes.shape
        target_d, target_h, target_w = self.target_volume_size

        # Get current model info to determine output channels
        model_info = get_current_model_info()
        current_output_channels = model_info["output_channels"]

        # Check if input has ROI-level padding
        has_roi_padding = (height != target_h) or (width != target_w)
        if has_roi_padding:
            roi_padding = max((height - target_h) // 2, (width - target_w) // 2)
            # print(f"Detected ROI-level padding: {roi_padding} pixels, input {(height, width)} -> target {(target_h, target_w)}")

        # if epoch is not None and batch is not None:
        #     print(f"Epoch {epoch}, Batch {batch}:")
        # print(f"Processing {batch_size} volumes for DINOv3 feature extraction...")
        # print(f"  - Volume shape: {volumes.shape}")
        # print(f"  - Target volume size: {self.target_volume_size}")
        # print(f"  - Expected output channels: {current_output_channels}")
        # print(f"  - DINOv3 slice size: {self.dinov3_slice_size}")

        all_features = []

        for b in range(batch_size):
            volume_features = []

            for z in range(target_d):
                # Extract slice from potentially padded volume
                if volumes.shape[1] == target_d:
                    slice_2d = volumes[b, z]
                else:
                    # Need to resize the volume first (depth dimension)
                    volume_resized = resize(
                        volumes[b],
                        (target_d, height, width),  # Keep spatial dimensions for now
                        preserve_range=True,
                        anti_aliasing=True,
                    )
                    slice_2d = volume_resized[z]

                # Handle ROI-level padding: crop to target spatial size
                if has_roi_padding:
                    # Crop from center of padded slice
                    start_h = (height - target_h) // 2
                    start_w = (width - target_w) // 2
                    slice_2d = slice_2d[
                        start_h : start_h + target_h, start_w : start_w + target_w
                    ]

                # Ensure slice is exactly the target spatial size
                if slice_2d.shape != (target_h, target_w):
                    slice_2d = resize(
                        slice_2d,
                        (target_h, target_w),
                        preserve_range=True,
                        anti_aliasing=True,
                    ).astype(volumes.dtype)

                # FIXED: Upsample to the correct DINOv3 input size
                # The issue was here - we need to match the size that DINOv3 was initialized with
                slice_upsampled = resize(
                    slice_2d,
                    (self.dinov3_slice_size, self.dinov3_slice_size),
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype(volumes.dtype)

                # Note: Sliding window padding is now handled at ROI level, not per-slice
                # No additional padding needed here since ROI-level padding already provides context

                # Extract features
                slice_batch = slice_upsampled[np.newaxis, ...]

                # Use updated autocast
                with autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu",
                    enabled=False,
                ):  # Disable autocast for metric calculations
                    # DEBUG: Print stride processing info
                    if not hasattr(self, "_stride_debug_printed") and b == 0 and z == 0:
                        from tqdm import tqdm

                        tqdm.write(
                            f"  - DEBUG: Calling process() with stride={self.dinov3_stride}"
                        )
                        tqdm.write(f"  - DEBUG: Input slice shape: {slice_batch.shape}")
                        tqdm.write(
                            f"  - DEBUG: Expected feature size with stride={self.dinov3_stride}: {self.dinov3_slice_size // self.dinov3_stride if self.dinov3_stride else self.dinov3_slice_size // 16}x{self.dinov3_slice_size // self.dinov3_stride if self.dinov3_stride else self.dinov3_slice_size // 16}"
                        )
                        self._stride_debug_printed = True

                    dinov3_features = process(
                        slice_batch,
                        model_id=self.model_id,
                        image_size=self.dinov3_slice_size,  # Use original size - ROI padding handles context
                        stride=self.dinov3_stride,  # Sliding window inference enabled
                    )

                # Convert to torch tensor if it's numpy (ensure float32 for model compatibility)
                if isinstance(dinov3_features, np.ndarray):
                    dinov3_features = torch.from_numpy(dinov3_features).float()

                # DEBUG: Print feature information (only once per training session)
                if not hasattr(self, "_debug_printed") and b == 0 and z == 0:
                    from tqdm import tqdm

                    tqdm.write(f"  - DINOv3 features shape: {dinov3_features.shape}")
                    tqdm.write(f"  - DINOv3 features size: {dinov3_features.numel()}")
                    tqdm.write(f"  - DINOv3 features type: {type(dinov3_features)}")
                    self._debug_printed = True

                # FIXED: Handle the actual DINOv3 output format
                # DINOv3 returns (channels, batch_size, patch_h, patch_w)
                if len(dinov3_features.shape) == 4:
                    # Expected format: (channels, batch_size, patch_h, patch_w)
                    channels, batch_dim, patch_h, patch_w = dinov3_features.shape

                    if batch_dim != 1:
                        raise ValueError(f"Expected batch size 1, got {batch_dim}")

                    # Remove batch dimension: (channels, patch_h, patch_w)
                    features_2d = dinov3_features.squeeze(1)

                    # With sliding window inference, we get higher resolution features
                    if self.dinov3_stride is not None and self.dinov3_stride < 16:
                        # Higher resolution due to sliding window (ROI-level padding provides context)
                        expected_features_size = (
                            self.dinov3_slice_size // self.dinov3_stride
                        )

                        # DEBUG: Print sizing information
                        if (
                            not hasattr(self, "_size_debug_printed")
                            and b == 0
                            and z == 0
                        ):
                            from tqdm import tqdm

                            tqdm.write(f"  - Slice size: {self.dinov3_slice_size}")
                            tqdm.write(f"  - Stride: {self.dinov3_stride}")
                            tqdm.write(
                                f"  - Sliding window context handled at ROI level"
                            )
                            expected_size = (
                                self.dinov3_slice_size // self.dinov3_stride
                                if self.dinov3_stride
                                else self.dinov3_slice_size // 16
                            )
                            tqdm.write(
                                f"  - Feature size: {patch_h}x{patch_w} (expected: {expected_size}x{expected_size})"
                            )
                            if patch_h != expected_size:
                                tqdm.write(
                                    f"  - ⚠️  WARNING: Feature size mismatch! Stride processing may not be working."
                                )
                            self._size_debug_printed = True

                        # Don't crop the features - keep the higher resolution
                        # The UNet will handle the size differences through learned upsampling
                        pass  # Keep features_2d as is

                elif len(dinov3_features.shape) == 3:
                    # Alternative format: (batch_size, tokens, channels) or similar
                    # This shouldn't happen with the current process() function, but handle just in case
                    if dinov3_features.shape[0] == 1:
                        # Batch first format
                        features_flat = dinov3_features.squeeze(0)  # Remove batch dim
                        # Reshape based on expected channels
                        if features_flat.shape[1] == current_output_channels:
                            # (tokens, channels) -> need to reshape to spatial
                            tokens = features_flat.shape[0]
                            spatial_size = int(np.sqrt(tokens))
                            features_2d = features_flat.transpose(0, 1).reshape(
                                current_output_channels, spatial_size, spatial_size
                            )
                        else:
                            raise ValueError(
                                f"Unexpected feature shape: {dinov3_features.shape}"
                            )
                    else:
                        raise ValueError(
                            f"Unexpected feature shape: {dinov3_features.shape}"
                        )

                else:
                    raise ValueError(
                        f"Unexpected DINOv3 features shape: {dinov3_features.shape}"
                    )

                # DEBUG: Print reshaped features (only once per training session)
                # Debug output suppressed during training to show progress bars clearly
                if not hasattr(self, "_reshape_debug_printed") and b == 0 and z == 0:
                    self._reshape_debug_printed = True

                # Resize to target spatial dimensions (skip if learning upsampling)
                if self.learn_upsampling:
                    # Keep features at DINOv3 native resolution
                    features_resized = features_2d

                    # DEBUG: Print native resolution features (only once per training session)
                    # Debug output suppressed during training to show progress bars clearly
                    if not hasattr(self, "_native_debug_printed") and b == 0 and z == 0:
                        self._native_debug_printed = True
                else:
                    # Traditional interpolation upsampling
                    features_resized = torch.nn.functional.interpolate(
                        features_2d.unsqueeze(0),  # Add batch dimension
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(
                        0
                    )  # Remove batch dimension

                    # DEBUG: Print interpolated features (only once per training session)
                    if not hasattr(self, "_resize_debug_printed") and b == 0 and z == 0:
                        from tqdm import tqdm

                        tqdm.write(
                            f"  - Interpolated features shape: {features_resized.shape}"
                        )
                        self._resize_debug_printed = True

                volume_features.append(features_resized)

            # Stack along depth dimension
            volume_features_3d = torch.stack(
                volume_features, dim=1
            )  # (channels, depth, height, width)
            all_features.append(volume_features_3d)

        batch_features = torch.stack(all_features, dim=0)

        # DEBUG: Print final batch shape (only once per training session)
        if not hasattr(self, "_batch_debug_printed"):
            from tqdm import tqdm

            tqdm.write(f"  - Final batch features shape: {batch_features.shape}")
            self._batch_debug_printed = True

        return batch_features.to(device)

    def get_data_info(self):
        """
        Get information about the loaded data.

        Returns:
        --------
        dict: Data information
        """
        return {
            "total_volumes": len(self.raw_data),
            "raw_shape": self.raw_data.shape,
            "gt_shape": self.gt_data.shape,
            "train_pool_size": len(self.train_pool_indices),
            "val_set_size": len(self.val_indices),
            "target_volume_size": self.target_volume_size,
            "dinov3_slice_size": self.dinov3_slice_size,
            "unique_gt_values": np.unique(self.gt_data),
        }


class MemoryEfficientDataLoader:
    """
    Memory-efficient data loader for DINOv3 training.
    Updated to use configurable model_id.
    """

    def __init__(
        self,
        raw_data,
        gt_data,
        train_image_pool_size=50,
        val_image_pool_size=10,
        target_size=224,
        seed=42,
        model_id=None,
    ):
        """
        Initialize memory-efficient data loader.

        Parameters:
        -----------
        raw_data : numpy.ndarray
            3D raw image data (z, y, x)
        gt_data : numpy.ndarray
            3D ground truth data (z, y, x)
        train_image_pool_size : int, default=50
            Number of possible training images to sample from
        val_image_pool_size : int, default=10
            Number of validation images (fixed)
        target_size : int, default=224
            Final image patch size
        seed : int, default=42
            Random seed for reproducibility
        model_id : str, optional
            DINOv3 model identifier to use
        """
        self.raw_data = raw_data
        self.gt_data = gt_data
        self.target_size = target_size
        self.train_image_pool_size = train_image_pool_size
        self.val_image_pool_size = val_image_pool_size
        self.seed = seed
        self.model_id = model_id

        # Initialize DINOv3 with specified model
        if model_id:
            from .dinov3_core import initialize_dinov3

            initialize_dinov3(model_id=model_id, image_size=896)

        # Calculate upsampling parameters for DINOv3
        self.dinov3_input_size = 896  # DINOv3 input size

        # Set random seed
        np.random.seed(seed)

        # Pre-sample validation data (fixed throughout training)
        print(f"Sampling {val_image_pool_size} validation images...")
        self.val_images, self.val_gt = sample_training_data(
            self.raw_data,
            self.gt_data,
            target_size=target_size,
            num_samples=val_image_pool_size,
            method="flexible",
            seed=seed,
            use_augmentation=False,  # No augmentation for validation
        )

        print(f"Validation data shape: {self.val_images.shape}")
        unique_val_classes, val_counts = np.unique(self.val_gt, return_counts=True)
        print(
            f"Validation class distribution: {dict(zip(unique_val_classes, val_counts))}"
        )

    def sample_training_batch(self, batch_size):
        """
        Sample a new batch of training images.

        Parameters:
        -----------
        batch_size : int
            Number of images to sample

        Returns:
        --------
        tuple: (batch_images, batch_gt) of shape (batch_size, target_size, target_size)
        """
        # Use the imported function, not self.sample_training_data
        batch_images, batch_gt = sample_training_data(
            self.raw_data,
            self.gt_data,
            target_size=self.target_size,
            num_samples=batch_size,
            method="flexible",
            use_augmentation=True,  # Use augmentation for training
        )

        return batch_images, batch_gt

    def get_validation_data(self):
        """
        Get the fixed validation dataset.

        Returns:
        --------
        tuple: (val_images, val_gt) of shape (val_pool_size, target_size, target_size)
        """
        return self.val_images, self.val_gt

    def get_batch_features_and_labels(
        self, images, labels, norm_stats=None, device=None
    ):
        """
        Extract DINOv3 features and prepare labels for a batch of images.
        Uses the configured model_id.

        Parameters:
        -----------
        images : numpy.ndarray
            Images of shape (batch_size, target_size, target_size)
        labels : numpy.ndarray
            Labels of shape (batch_size, target_size, target_size)
        norm_stats : dict, optional
            Pre-computed normalization statistics
        device : torch.device, optional
            Device to put tensors on

        Returns:
        --------
        tuple: (features, labels, coordinates, norm_stats)
        """
        from .dinov3_core import process

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_size = images.shape[0]

        # Step 1: Upsample images to DINOv3 input size
        from skimage.transform import resize

        upsampled_images = np.zeros(
            (batch_size, self.dinov3_input_size, self.dinov3_input_size),
            dtype=images.dtype,
        )

        for i in range(batch_size):
            upsampled_images[i] = resize(
                images[i],
                (self.dinov3_input_size, self.dinov3_input_size),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(images.dtype)

        # Step 2: Extract DINOv3 features using the configured model
        dinov3_features = process(
            upsampled_images, model_id=self.model_id
        )  # Pass model_id

        # Step 3: Flatten features for each image
        features_list = []
        labels_list = []
        coords_list = []

        for img_idx in range(batch_size):
            # Get features for this image: (output_channels, patch_h, patch_w)
            img_features = dinov3_features[:, img_idx, :, :]
            img_labels = labels[img_idx]

            # Flatten spatial dimensions
            patch_h, patch_w = img_features.shape[1], img_features.shape[2]
            features_flat = img_features.reshape(
                img_features.shape[0], -1
            ).T  # (num_patches, output_channels)

            # Downsample labels to match patch resolution
            labels_downsampled = resize(
                img_labels.astype(float),
                (patch_h, patch_w),
                order=0,  # Nearest neighbor for labels
                preserve_range=True,
                anti_aliasing=False,
            ).astype(int)

            labels_flat = labels_downsampled.flatten()  # (num_patches,)

            # Create coordinate grid
            coords = np.array([(i, j) for i in range(patch_h) for j in range(patch_w)])

            features_list.append(features_flat)
            labels_list.append(labels_flat)
            coords_list.append(coords)

        # Concatenate all features and labels
        all_features = np.concatenate(features_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        all_coords = np.concatenate(coords_list, axis=0)

        # Apply normalization
        if norm_stats is None:
            normalized_features, norm_stats = normalize_features(
                all_features, method="standardize"
            )
        else:
            normalized_features = apply_normalization_stats(all_features, norm_stats)

        # Convert to tensors
        features_tensor = torch.tensor(normalized_features, dtype=torch.float32).to(
            device
        )
        labels_tensor = torch.tensor(all_labels, dtype=torch.long).to(device)

        return features_tensor, labels_tensor, all_coords, norm_stats

    def extract_dinov3_features_for_unet(self, images):
        """
        Extract DINOv3 features for UNet training (returns features in format needed for UNet).
        Uses the configured model_id.

        Parameters:
        -----------
        images : numpy.ndarray
            Images of shape (batch_size, target_size, target_size)

        Returns:
        --------
        torch.Tensor: DINOv3 features (batch_size, output_channels, target_size, target_size)
        """
        from .dinov3_core import process, get_current_model_info

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = images.shape[0]

        # Step 1: Upsample to DINOv3 input size
        from skimage.transform import resize

        upsampled_images = np.zeros(
            (batch_size, self.dinov3_input_size, self.dinov3_input_size),
            dtype=images.dtype,
        )

        for i in range(batch_size):
            upsampled_images[i] = resize(
                images[i],
                (self.dinov3_input_size, self.dinov3_input_size),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(images.dtype)

        # Step 2: Extract DINOv3 features using the configured model
        dinov3_features = process(
            upsampled_images, model_id=self.model_id
        )  # Pass model_id

        # Get current output channels
        model_info = get_current_model_info()
        current_output_channels = model_info["output_channels"]

        # Step 3: Rearrange to (batch_size, output_channels, H_feat, W_feat)
        if dinov3_features.shape[0] == current_output_channels:
            features_rearranged = np.transpose(dinov3_features, (1, 0, 2, 3))
        else:
            raise ValueError(
                f"Unexpected DINOv3 output format: {dinov3_features.shape}"
            )

        # Step 4: Upsample features to target resolution
        features_tensor = torch.tensor(features_rearranged, dtype=torch.float32)
        features_upsampled = F.interpolate(
            features_tensor,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        return features_upsampled.to(device)


def get_checkpoint_base_path(model_type, model_id=None, export_base_dir=None):
    """
    Get the base checkpoint path with model type and model ID, then current date/time structure.

    Parameters:
    -----------
    model_type : str
        Type of model (e.g., 'improved_classifier', 'simple_classifier', 'dinov3_unet')
    model_id : str, optional
        Model identifier (e.g., 'facebook/dinov3-vits16-pretrain-lvd1689m')
    export_base_dir : str, optional
        Base directory for exports

    Returns:
    --------
    str: Base path in format /groups/cellmap/.../tmp/results/MODEL_TYPE_MODEL_ID/YYYYMMDD_HHMMSS/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = "/groups/cellmap/cellmap/ackermand/Programming/cellmap-flow/dinov3_playground/tmp/results"

    if export_base_dir is not None:
        base_path = export_base_dir

    # Clean model_id for filesystem compatibility
    if model_id:
        # Convert model ID to filesystem-safe string
        model_id_clean = model_id.replace("/", "_").replace("-", "_")
        model_dir = f"{model_type}_{model_id_clean}"
    else:
        model_dir = model_type

    return os.path.join(base_path, model_dir, timestamp)


def create_model_checkpoint_dir(base_path):
    """
    Create model checkpoint directory.

    Parameters:
    -----------
    base_path : str
        Base checkpoint path (already includes model type and timestamp)

    Returns:
    --------
    str: Full model checkpoint directory path
    """
    model_dir = os.path.join(base_path, "model")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def train_classifier_memory_efficient(
    data_loader,
    num_classes,
    device,
    epochs=500,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=50,
    min_delta=0.001,
    use_improved_classifier=True,
    images_per_batch=4,
    batches_per_epoch=10,
    save_checkpoints=True,
    model_id=None,
):
    """
    Train classifier using memory-efficient data loading.
    Updated to include model_id in checkpoint path.
    """

    print(f"Memory-efficient training setup:")
    print(f"  - Images per batch: {images_per_batch}")
    print(f"  - Batches per epoch: {batches_per_epoch}")
    print(f"  - Total images per epoch: {images_per_batch * batches_per_epoch}")

    # Setup checkpoint saving with new directory structure
    checkpoint_dir = None
    if save_checkpoints:
        model_name = (
            "improved_classifier" if use_improved_classifier else "simple_classifier"
        )
        base_path = get_checkpoint_base_path(model_name, model_id)
        checkpoint_dir = create_model_checkpoint_dir(base_path)
        print(f"  - Model ID: {model_id}")
        print(f"  - Checkpoints will be saved to: {checkpoint_dir}")

    # Get validation data (fixed throughout training)
    val_images, val_gt = data_loader.get_validation_data()

    # Process validation data
    print("Processing validation data...")
    val_X, val_y, _, norm_stats = data_loader.get_batch_features_and_labels(
        val_images, val_gt, device=device
    )

    print(f"Validation set: {val_X.shape} features from {len(val_images)} images")

    # Initialize classifier
    from .models import ImprovedClassifier, SimpleClassifier

    if use_improved_classifier:
        classifier = ImprovedClassifier(
            input_dim=val_X.shape[1], num_classes=num_classes
        ).to(device)
        model_name = "improved_classifier"
    else:
        classifier = SimpleClassifier(
            input_dim=val_X.shape[1], num_classes=num_classes
        ).to(device)
        model_name = "simple_classifier"

    print(f"Using {model_name}")

    # Get current model info for config saving
    from .dinov3_core import get_current_model_info

    model_info = get_current_model_info()
    current_output_channels = model_info["output_channels"]

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20
    )

    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    epochs_without_improvement = 0

    print(f"Starting memory-efficient training for up to {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase - sample new images each epoch
        classifier.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch_idx in range(batches_per_epoch):
            # Sample new training images for this batch
            train_images, train_gt = data_loader.sample_training_batch(images_per_batch)

            # Process training data
            train_X, train_y, _, _ = data_loader.get_batch_features_and_labels(
                train_images, train_gt, norm_stats=norm_stats, device=device
            )

            # Training step
            optimizer.zero_grad()
            outputs = classifier(train_X)
            loss = criterion(outputs, train_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            _, predicted = torch.max(outputs.data, 1)
            epoch_train_loss += loss.item() * train_y.size(0)
            epoch_train_correct += (predicted == train_y).sum().item()
            epoch_train_total += train_y.size(0)

        # Average training metrics for this epoch
        train_loss = epoch_train_loss / epoch_train_total
        train_acc = epoch_train_correct / epoch_train_total

        # Validation phase (using fixed validation set)
        classifier.eval()
        with torch.no_grad():
            val_outputs = classifier(val_X)
            val_loss = criterion(val_outputs, val_y).item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_correct = (val_predicted == val_y).sum().item()
            val_acc = val_correct / val_y.size(0)

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]["lr"]

        # Check if this is the best validation accuracy so far
        is_best_model = False
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            is_best_model = True
        else:
            epochs_without_improvement += 1

        # Print progress every epoch
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs}")
        print(
            f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} (from {epoch_train_total} samples)"
        )
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"  Best Val Acc: {best_val_acc:.4f}, LR: {current_lr:.6f}")
        print(f"  Epochs without improvement: {epochs_without_improvement}")
        if new_lr < old_lr:
            print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Save checkpoints
        if save_checkpoints:
            # Always save training stats (lightweight)
            stats_data = {
                "epoch": epoch + 1,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
                "normalization_stats": norm_stats,
                "training_config": {
                    "num_classes": num_classes,
                    "use_improved_classifier": use_improved_classifier,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "patience": patience,
                    "min_delta": min_delta,
                    "images_per_batch": images_per_batch,
                    "batches_per_epoch": batches_per_epoch,
                    "model_type": model_name,
                    "model_id": model_id,
                    "target_size": target_size,  # 2D image size
                    "image_size": target_size,  # For DINOv3 initialization
                    "seed": data_loader.seed,  # Add seed
                    "train_image_pool_size": data_loader.train_image_pool_size,  # Add pool sizes
                    "val_image_pool_size": data_loader.val_image_pool_size,
                },
                "model_config": {
                    "num_classes": num_classes,
                    "input_size": (target_size, target_size),  # 2D input size
                    "model_id": model_id,
                    "model_type": model_name,
                    "use_improved_classifier": use_improved_classifier,
                    "target_size": target_size,  # Add target size
                    "image_size": target_size,  # Add image size for DINOv3
                    "input_channels": current_output_channels,  # Add input channels
                },
            }

            # Save stats every epoch
            stats_path = os.path.join(checkpoint_dir, f"stats_epoch_{epoch+1:04d}.pkl")
            with open(stats_path, "wb") as f:
                pickle.dump(stats_data, f)

            # Always save latest stats
            latest_stats_path = os.path.join(checkpoint_dir, "latest_stats.pkl")
            with open(latest_stats_path, "wb") as f:
                pickle.dump(stats_data, f)

            # Only save full model checkpoint if this is the best performance
            if is_best_model:
                checkpoint_data = {
                    **stats_data,  # Include all stats
                    "classifier_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }

                # Save best model checkpoint (with state dicts)
                best_path = os.path.join(checkpoint_dir, "best.pkl")
                with open(best_path, "wb") as f:
                    pickle.dump(checkpoint_data, f)

                print(f"  *** NEW BEST MODEL SAVED: {val_acc:.4f} ***")

            print(f"  Stats saved: {os.path.basename(stats_path)}")

        print()  # Add blank line for readability

        # Early stopping
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)"
            )
            break

    return {
        "classifier": classifier,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(train_losses),
        "normalization_stats": norm_stats,
        "checkpoint_dir": checkpoint_dir,
    }


def train_unet_memory_efficient(
    data_loader,
    num_classes,
    device,
    epochs=500,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=50,
    min_delta=0.001,
    base_channels=64,
    images_per_batch=2,
    batches_per_epoch=10,
    save_checkpoints=True,
    model_id=None,
    export_base_dir=None,
    use_class_weighting=True,
):
    """
    Train UNet using memory-efficient data loading with DINOv3 features.
    Updated to use configurable export directory and selective checkpoint saving.
    Now includes class balancing for loss computation.
    """

    print(f"Memory-efficient UNet training setup:")
    print(f"  - Images per batch: {images_per_batch}")
    print(f"  - Batches per epoch: {batches_per_epoch}")
    print(f"  - Total images per epoch: {images_per_batch * batches_per_epoch}")
    print(f"  - Class weighting: {use_class_weighting}")

    # Setup checkpoint saving with new directory structure
    checkpoint_dir = None
    if save_checkpoints:
        base_path = get_checkpoint_base_path("dinov3_unet", model_id, export_base_dir)
        checkpoint_dir = create_model_checkpoint_dir(base_path)
        print(f"  - Model ID: {model_id}")
        print(f"  - Export base: {export_base_dir}")
        print(f"  - Checkpoints will be saved to: {checkpoint_dir}")

    # Get validation data (fixed throughout training)
    val_images, val_gt = data_loader.get_validation_data()

    # Process validation data to get features
    print("Processing validation data...")
    val_features = data_loader.extract_dinov3_features_for_unet(val_images)
    val_labels = torch.tensor(val_gt, dtype=torch.long).to(device)

    print(
        f"Validation set: {val_features.shape} features from {len(val_images)} images"
    )
    print(f"Validation labels shape: {val_labels.shape}")

    # Calculate class weights from validation data for consistent weighting
    if use_class_weighting:
        print("Calculating class weights from validation data...")
        val_labels_flat = val_labels.view(-1).cpu().numpy()

        # Count pixels per class
        unique_classes, class_counts = np.unique(val_labels_flat, return_counts=True)
        total_pixels = len(val_labels_flat)

        # Calculate inverse frequency weights
        class_weights = np.zeros(num_classes)
        for i, (class_id, count) in enumerate(zip(unique_classes, class_counts)):
            if class_id < num_classes:  # Safety check
                class_weights[class_id] = total_pixels / (num_classes * count)

        # Normalize weights so they sum to num_classes
        class_weights = class_weights * num_classes / np.sum(class_weights)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
            device
        )

        print("Class distribution in validation data:")
        for i, (class_id, count) in enumerate(zip(unique_classes, class_counts)):
            if class_id < num_classes:
                percentage = (count / total_pixels) * 100
                weight = class_weights[class_id]
                print(
                    f"  Class {class_id}: {count:,} pixels ({percentage:.2f}%), weight: {weight:.3f}"
                )
    else:
        class_weights_tensor = None
        print("Using unweighted loss (no class balancing)")

    # Get the current output channels from the DINOv3 model
    from .dinov3_core import get_current_model_info

    model_info = get_current_model_info()
    current_output_channels = model_info["output_channels"]

    print(
        f"Using {current_output_channels} input channels for UNet (from .dinov3 {model_info['model_id']})"
    )

    # Initialize UNet with correct input channels
    from .models import DINOv3UNet

    unet = DINOv3UNet(
        input_channels=current_output_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    ).to(device)

    print(
        f"Using DINOv3UNet with {base_channels} base channels and {current_output_channels} input channels"
    )

    # Loss and optimizer with class weighting
    if use_class_weighting and class_weights_tensor is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print(
            f"Using weighted CrossEntropyLoss with class weights: {class_weights_tensor.cpu().numpy()}"
        )
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")

    optimizer = optim.Adam(
        unet.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20
    )

    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_class_accs, val_class_accs = [], []  # Per-class accuracies
    best_val_acc = 0.0
    epochs_without_improvement = 0

    print(f"Starting memory-efficient UNet training for up to {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase - sample new images each epoch
        unet.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        epoch_train_class_correct = np.zeros(num_classes)
        epoch_train_class_total = np.zeros(num_classes)

        for batch_idx in range(batches_per_epoch):
            # Sample new training images for this batch
            train_images, train_gt = data_loader.sample_training_batch(images_per_batch)

            # Extract DINOv3 features for UNet
            train_features = data_loader.extract_dinov3_features_for_unet(train_images)
            train_labels = torch.tensor(train_gt, dtype=torch.long).to(device)

            # Training step
            optimizer.zero_grad()
            logits = unet(train_features)  # (batch, num_classes, H, W)

            loss = criterion(logits, train_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == train_labels).sum().item()
            total = train_labels.numel()

            epoch_train_loss += loss.item() * total
            epoch_train_correct += correct
            epoch_train_total += total

            # Track per-class accuracy
            for class_id in range(num_classes):
                class_mask = train_labels == class_id
                if class_mask.sum() > 0:
                    class_correct = (predictions[class_mask] == class_id).sum().item()
                    class_total = class_mask.sum().item()
                    epoch_train_class_correct[class_id] += class_correct
                    epoch_train_class_total[class_id] += class_total

        # Average training metrics for this epoch
        train_loss = epoch_train_loss / epoch_train_total
        train_acc = epoch_train_correct / epoch_train_total

        # Calculate per-class training accuracies
        train_class_acc = []
        for class_id in range(num_classes):
            if epoch_train_class_total[class_id] > 0:
                class_acc = (
                    epoch_train_class_correct[class_id]
                    / epoch_train_class_total[class_id]
                )
                train_class_acc.append(class_acc)
            else:
                train_class_acc.append(0.0)

        # Validation phase (using fixed validation set)
        unet.eval()
        with torch.no_grad():
            val_logits = unet(val_features)
            val_loss = criterion(val_logits, val_labels).item()
            val_predictions = torch.argmax(val_logits, dim=1)
            val_correct = (val_predictions == val_labels).sum().item()
            val_total = val_labels.numel()
            val_acc = val_correct / val_total

            # Calculate per-class validation accuracies
            val_class_acc = []
            for class_id in range(num_classes):
                class_mask = val_labels == class_id
                if class_mask.sum() > 0:
                    class_correct = (
                        (val_predictions[class_mask] == class_id).sum().item()
                    )
                    class_total = class_mask.sum().item()
                    class_acc = class_correct / class_total
                    val_class_acc.append(class_acc)
                else:
                    val_class_acc.append(0.0)

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_class_accs.append(train_class_acc)
        val_class_accs.append(val_class_acc)

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]["lr"]

        # Check if this is the best validation accuracy so far
        is_best_model = False
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            is_best_model = True
        else:
            epochs_without_improvement += 1

        # Print progress every epoch
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs}")
        print(
            f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} (from {epoch_train_total} pixels)"
        )
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        # Print per-class accuracies
        print(
            f"  Train Class Accs: ["
            + ", ".join([f"{acc:.3f}" for acc in train_class_acc])
            + "]"
        )
        print(
            f"  Val Class Accs:   ["
            + ", ".join([f"{acc:.3f}" for acc in val_class_acc])
            + "]"
        )

        print(f"  Best Val Acc: {best_val_acc:.4f}, LR: {current_lr:.6f}")
        print(f"  Epochs without improvement: {epochs_without_improvement}")
        if new_lr < old_lr:
            print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Save checkpoints
        if save_checkpoints:
            # Always save training stats (lightweight)
            stats_data = {
                "epoch": epoch + 1,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
                "train_class_accs": train_class_accs,
                "val_class_accs": val_class_accs,
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
                "class_weights": (
                    class_weights_tensor.cpu().numpy()
                    if class_weights_tensor is not None
                    else None
                ),
                "training_config": {
                    "num_classes": num_classes,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "patience": patience,
                    "min_delta": min_delta,
                    "base_channels": base_channels,
                    "images_per_batch": images_per_batch,
                    "batches_per_epoch": batches_per_epoch,
                    "model_type": "dinov3_unet",
                    "model_id": model_id,
                    "input_channels": current_output_channels,
                    "use_class_weighting": use_class_weighting,
                    "target_size": data_loader.target_size,  # Add target size
                    "image_size": data_loader.target_size,  # For DINOv3 initialization
                    "seed": data_loader.seed,  # Add seed
                    "train_image_pool_size": data_loader.train_image_pool_size,  # Add pool sizes
                    "val_image_pool_size": data_loader.val_image_pool_size,
                },
                "model_config": {
                    "num_classes": num_classes,
                    "base_channels": base_channels,
                    "input_size": (
                        data_loader.target_size,
                        data_loader.target_size,
                    ),  # 2D input size
                    "input_channels": current_output_channels,
                    "model_id": model_id,
                    "model_type": "dinov3_unet",
                    "target_size": data_loader.target_size,  # Add target size
                    "image_size": data_loader.target_size,  # Add image size for DINOv3
                },
            }

            # Save stats every epoch
            stats_path = os.path.join(checkpoint_dir, f"stats_epoch_{epoch+1:04d}.pkl")
            with open(stats_path, "wb") as f:
                pickle.dump(stats_data, f)

            # Always save latest stats
            latest_stats_path = os.path.join(checkpoint_dir, "latest_stats.pkl")
            with open(latest_stats_path, "wb") as f:
                pickle.dump(stats_data, f)

            # Only save full model checkpoint if this is the best performance
            if is_best_model:
                checkpoint_data = {
                    **stats_data,  # Include all stats
                    "unet_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }

                # Save best model checkpoint (with state dicts)
                best_path = os.path.join(checkpoint_dir, "best.pkl")
                with open(best_path, "wb") as f:
                    pickle.dump(checkpoint_data, f)

                print(f"  *** NEW BEST MODEL SAVED: {val_acc:.4f} ***")

            print(f"  Stats saved: {os.path.basename(stats_path)}")

        print()  # Add blank line for readability

        # Early stopping
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)"
            )
            break

    return {
        "unet": unet,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "train_class_accs": train_class_accs,
        "val_class_accs": val_class_accs,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(train_losses),
        "checkpoint_dir": checkpoint_dir,
        "export_base_dir": export_base_dir,
        "class_weights": (
            class_weights_tensor.cpu().numpy()
            if class_weights_tensor is not None
            else None
        ),
    }


def train_3d_unet_memory_efficient_v2(
    data_loader_3d,
    num_classes,
    device,
    epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=20,
    min_delta=0.001,
    base_channels=32,
    volumes_per_batch=1,
    batches_per_epoch=10,
    save_checkpoints=True,
    model_id=None,
    export_base_dir=None,
    use_class_weighting=True,
    use_mixed_precision=True,  # Add this parameter
    # Additional parameters for complete config saving
    use_half_precision=False,
    use_gradient_checkpointing=False,
    memory_efficient_mode="auto",
    learn_upsampling=False,  # NEW PARAMETER
):
    """
    Train 3D UNet using memory-efficient data loading with DINOv3 features.
    Now supports mixed precision training for reduced memory usage.
    """

    print(f"Memory-efficient 3D UNet training setup:")
    print(f"  - Volumes per batch: {volumes_per_batch}")
    print(f"  - Batches per epoch: {batches_per_epoch}")
    print(f"  - Total volumes per epoch: {volumes_per_batch * batches_per_epoch}")
    print(f"  - Class weighting: {use_class_weighting}")
    print(f"  - Mixed precision: {use_mixed_precision}")
    print(f"  - Learn upsampling: {learn_upsampling}")  # NEW

    # Initialize mixed precision scaler
    scaler = GradScaler() if use_mixed_precision else None

    # Setup checkpoint saving
    checkpoint_dir = None
    if save_checkpoints:
        base_path = get_checkpoint_base_path("dinov3_unet3d", model_id, export_base_dir)
        checkpoint_dir = create_model_checkpoint_dir(base_path)
        print(f"  - Model ID: {model_id}")
        print(f"  - Export base: {export_base_dir}")
        print(f"  - Checkpoints will be saved to: {checkpoint_dir}")

    # Get validation data (keep on CPU for memory efficiency)
    val_volumes, val_gt_volumes = data_loader_3d.get_validation_data()

    print(f"Validation set: {len(val_volumes)} volumes prepared")
    print(f"Validation labels will be processed volume-by-volume for memory efficiency")
    print(f"Expected validation shape per volume: {val_gt_volumes[0].shape}")
    print(
        "Note: Validation features and labels will be computed volume-by-volume on-demand"
    )

    # Calculate class weights from validation data (process on CPU to avoid GPU memory issues)
    if use_class_weighting:
        print("Calculating class weights from validation data...")
        # Process validation volumes on CPU to calculate class distribution
        all_val_labels = []
        for vol_gt in val_gt_volumes:
            all_val_labels.append(vol_gt.flatten())
        val_labels_flat = np.concatenate(all_val_labels)

        unique_classes, class_counts = np.unique(val_labels_flat, return_counts=True)
        total_voxels = len(val_labels_flat)

        class_weights = np.zeros(num_classes)
        for i, (class_id, count) in enumerate(zip(unique_classes, class_counts)):
            if class_id < num_classes:
                class_weights[class_id] = total_voxels / (num_classes * count)

        class_weights = class_weights * num_classes / np.sum(class_weights)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
            device
        )

        print("Class distribution in validation data:")
        for i, (class_id, count) in enumerate(zip(unique_classes, class_counts)):
            if class_id < num_classes:
                percentage = (count / total_voxels) * 100
                weight = class_weights[class_id]
                print(
                    f"  Class {class_id}: {count:,} voxels ({percentage:.2f}%), weight: {weight:.3f}"
                )

        # Clean up temporary data
        del all_val_labels, val_labels_flat
    else:
        class_weights_tensor = None
        print("Using unweighted loss (no class balancing)")

    # Get current model info
    from .dinov3_core import get_current_model_info

    model_info = get_current_model_info()
    current_output_channels = model_info["output_channels"]

    print(
        f"Using {current_output_channels} input channels for 3D UNet (from .dinov3 {model_info['model_id']})"
    )

    # Initialize 3D UNet
    from .models import DINOv3UNet3D

    # Calculate DINOv3 feature size if learning upsampling
    dinov3_feature_size = None
    if learn_upsampling:
        # DINOv3 features are typically 1/16 of the slice size
        slice_size = data_loader_3d.dinov3_slice_size
        feature_spatial_size = slice_size // 16  # e.g., 896//16 = 56
        dinov3_feature_size = (
            data_loader_3d.target_volume_size[0],  # Keep depth dimension
            feature_spatial_size,
            feature_spatial_size,
        )

    unet3d = DINOv3UNet3D(
        input_channels=current_output_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        input_size=data_loader_3d.target_volume_size,
        use_half_precision=use_half_precision,
        learn_upsampling=learn_upsampling,
        dinov3_feature_size=dinov3_feature_size,
        use_gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    ).to(device)

    print(
        f"Using DINOv3UNet3D with {base_channels} base channels and {current_output_channels} input channels"
    )

    # Loss and optimizer with class weighting
    if use_class_weighting and class_weights_tensor is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print(
            f"Using weighted CrossEntropyLoss with class weights: {class_weights_tensor.cpu().numpy()}"
        )
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")

    optimizer = optim.Adam(
        unet3d.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_class_accs, val_class_accs = [], []
    val_class_precisions, val_class_f1s, val_class_ious = [], [], []
    best_mean_iou = 0.0  # Changed from best_val_acc to best_mean_iou
    epochs_without_improvement = 0

    print(f"Starting memory-efficient 3D UNet training for up to {epochs} epochs...")

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        unet3d.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        epoch_train_class_correct = np.zeros(num_classes)
        epoch_train_class_total = np.zeros(num_classes)

        # Progress bar for batches within epoch
        batch_pbar = tqdm(
            range(batches_per_epoch),
            desc=f"Epoch {epoch+1}/{epochs} - Training",
            leave=False,
            ncols=100,
            position=0,
            dynamic_ncols=True,
        )

        for batch_idx in batch_pbar:
            # Sample new training volumes for this batch
            train_volumes, train_gt_volumes = data_loader_3d.sample_training_batch(
                volumes_per_batch
            )

            # Extract DINOv3 features for 3D UNet
            train_features = data_loader_3d.extract_dinov3_features_3d(
                train_volumes, epoch, batch_idx
            )
            train_labels = torch.tensor(train_gt_volumes, dtype=torch.long).to(device)

            # Training step
            optimizer.zero_grad()

            if use_mixed_precision:
                # Fix 1: Update autocast usage
                with autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu",
                    enabled=False,
                ):  # Disable autocast for metric calculations
                    logits = unet3d(train_features)
                    loss = criterion(logits, train_labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet3d.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = unet3d(train_features)
                loss = criterion(logits, train_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet3d.parameters(), max_norm=1.0)
                optimizer.step()

            # Clear GPU cache to prevent memory accumulation
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Track metrics
            with autocast(
                device_type="cuda" if device.type == "cuda" else "cpu", enabled=False
            ):  # Disable autocast for metric calculations
                predictions = torch.argmax(logits.float(), dim=1)
                correct = (predictions == train_labels).sum().item()
                total = train_labels.numel()

            epoch_train_loss += loss.item() * total
            epoch_train_correct += correct
            epoch_train_total += total

            # Track per-class accuracy
            for class_id in range(num_classes):
                class_mask = train_labels == class_id
                if class_mask.sum() > 0:
                    class_correct = (predictions[class_mask] == class_id).sum().item()
                    class_total = class_mask.sum().item()
                    epoch_train_class_correct[class_id] += class_correct
                    epoch_train_class_total[class_id] += class_total

            # Update progress bar with current metrics
            current_loss = epoch_train_loss / max(epoch_train_total, 1)
            current_acc = epoch_train_correct / max(epoch_train_total, 1)
            batch_pbar.set_postfix(
                {"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.4f}"}
            )

        batch_pbar.close()
        training_time = time.time() - epoch_start_time

        # Average training metrics for this epoch
        train_loss = epoch_train_loss / epoch_train_total
        train_acc = epoch_train_correct / epoch_train_total

        # Calculate per-class training accuracies
        train_class_acc = []
        for class_id in range(num_classes):
            if epoch_train_class_total[class_id] > 0:
                class_acc = (
                    epoch_train_class_correct[class_id]
                    / epoch_train_class_total[class_id]
                )
                train_class_acc.append(class_acc)
            else:
                train_class_acc.append(0.0)

        # Validation phase with timing - VOLUME-BY-VOLUME for memory efficiency
        # This approach processes each validation volume independently to avoid GPU memory spikes
        val_start_time = time.time()
        unet3d.eval()
        with torch.no_grad():
            # Initialize validation metrics accumulators
            val_total_loss = 0.0
            val_total_correct = 0
            val_total_voxels = 0

            # Per-class metrics accumulators
            val_class_true_positives = np.zeros(num_classes)
            val_class_false_positives = np.zeros(num_classes)
            val_class_false_negatives = np.zeros(num_classes)
            val_class_total_gt = np.zeros(num_classes)  # For recall calculation
            val_class_total_pred = np.zeros(num_classes)  # For precision calculation

            # Process each validation volume independently
            for vol_idx in range(len(val_volumes)):
                # Extract features for single volume (keeps GPU memory minimal)
                single_vol = np.array(
                    [val_volumes[vol_idx]]
                )  # Convert to numpy array with batch dimension
                vol_features = data_loader_3d.extract_dinov3_features_3d(
                    single_vol, epoch=epoch
                )

                # Get ground truth for this volume and move to GPU
                vol_gt = (
                    torch.tensor(val_gt_volumes[vol_idx], dtype=torch.long)
                    .unsqueeze(0)
                    .to(device)
                )

                # Run inference on single volume
                if use_mixed_precision:
                    with autocast(
                        device_type="cuda" if device.type == "cuda" else "cpu"
                    ):
                        vol_logits = unet3d(vol_features)
                        vol_loss = criterion(vol_logits, vol_gt).item()
                else:
                    vol_logits = unet3d(vol_features)
                    vol_loss = criterion(vol_logits, vol_gt).item()

                # Convert to predictions
                vol_predictions = torch.argmax(vol_logits.float(), dim=1)

                # Accumulate basic metrics
                vol_correct = (vol_predictions == vol_gt).sum().item()
                vol_voxels = vol_gt.numel()

                val_total_loss += vol_loss * vol_voxels
                val_total_correct += vol_correct
                val_total_voxels += vol_voxels

                # Accumulate per-class metrics (convert to CPU for efficiency)
                vol_gt_cpu = vol_gt.cpu().numpy()
                vol_pred_cpu = vol_predictions.cpu().numpy()

                for class_id in range(num_classes):
                    # Ground truth and prediction masks
                    gt_mask = vol_gt_cpu == class_id
                    pred_mask = vol_pred_cpu == class_id

                    # Confusion matrix components
                    true_positives = np.sum(gt_mask & pred_mask)
                    false_positives = np.sum(~gt_mask & pred_mask)
                    false_negatives = np.sum(gt_mask & ~pred_mask)

                    # Accumulate
                    val_class_true_positives[class_id] += true_positives
                    val_class_false_positives[class_id] += false_positives
                    val_class_false_negatives[class_id] += false_negatives
                    val_class_total_gt[class_id] += np.sum(gt_mask)
                    val_class_total_pred[class_id] += np.sum(pred_mask)

                # Immediately free GPU memory for this volume
                del vol_features, vol_logits, vol_predictions, vol_gt

                # Clear GPU cache after each volume
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # Calculate final validation metrics from accumulated values
            val_loss = val_total_loss / val_total_voxels
            val_acc = val_total_correct / val_total_voxels

            # Calculate comprehensive per-class validation metrics
            val_class_acc = []  # Recall/Sensitivity
            val_class_precision = []
            val_class_f1 = []
            val_class_iou = []

            for class_id in range(num_classes):
                tp = val_class_true_positives[class_id]
                fp = val_class_false_positives[class_id]
                fn = val_class_false_negatives[class_id]

                # Recall (Sensitivity) = TP / (TP + FN)
                if val_class_total_gt[class_id] > 0:
                    recall = tp / val_class_total_gt[class_id]
                else:
                    recall = 0.0

                # Precision = TP / (TP + FP)
                if val_class_total_pred[class_id] > 0:
                    precision = tp / val_class_total_pred[class_id]
                else:
                    precision = 0.0

                # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                # IoU (Intersection over Union) = TP / (TP + FP + FN)
                union = tp + fp + fn
                if union > 0:
                    iou = tp / union
                else:
                    iou = 0.0

                val_class_acc.append(recall)
                val_class_precision.append(precision)
                val_class_f1.append(f1)
                val_class_iou.append(iou)

        val_time = time.time() - val_start_time
        total_epoch_time = time.time() - epoch_start_time

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_class_accs.append(train_class_acc)
        val_class_accs.append(val_class_acc)
        val_class_precisions.append(val_class_precision)
        val_class_f1s.append(val_class_f1)
        val_class_ious.append(val_class_iou)

        # Calculate mean IoU for both scheduler and best model selection
        current_mean_iou = np.mean(val_class_iou)

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(current_mean_iou)
        new_lr = optimizer.param_groups[0]["lr"]

        # Check if this is the best mean IoU so far
        is_best_model = False
        if current_mean_iou > best_mean_iou + min_delta:
            best_mean_iou = current_mean_iou
            epochs_without_improvement = 0
            is_best_model = True
        else:
            epochs_without_improvement += 1

        # Print progress every epoch with timing
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs} - Time: {total_epoch_time:.1f}s (Train: {training_time:.1f}s, Val: {val_time:.1f}s)"
        )
        print(
            f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} (from {epoch_train_total} voxels)"
        )
        print(
            f"  Val:   Loss={val_loss:.4f}, IoU={current_mean_iou:.4f}, Acc={val_acc:.4f}"
        )

        # Print comprehensive per-class metrics
        print(
            f"  Train Class Recall: ["
            + ", ".join([f"{acc:.3f}" for acc in train_class_acc])
            + "]"
        )
        print(
            f"  Val Class Recall:   ["
            + ", ".join([f"{acc:.3f}" for acc in val_class_acc])
            + "]"
        )
        print(
            f"  Val Class Precision:["
            + ", ".join([f"{prec:.3f}" for prec in val_class_precision])
            + "]"
        )
        print(
            f"  Val Class F1:       ["
            + ", ".join([f"{f1:.3f}" for f1 in val_class_f1])
            + "]"
        )
        print(
            f"  Val Class IoU:      ["
            + ", ".join([f"{iou:.3f}" for iou in val_class_iou])
            + "]"
        )
        print(f"  Mean IoU: {current_mean_iou:.4f}")

        # Print class distribution to understand accuracy weighting
        val_class_counts = val_class_total_gt  # Use the counts we already calculated
        val_class_percentages = [
            count / val_total_voxels * 100 for count in val_class_counts
        ]
        print(
            f"  Val Class Dist:   ["
            + ", ".join([f"{pct:.1f}%" for pct in val_class_percentages])
            + "]"
        )

        print(f"  Best Mean IoU: {best_mean_iou:.4f}, LR: {current_lr:.6f}")
        print(f"  Epochs without improvement: {epochs_without_improvement}")
        if new_lr < old_lr:
            print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Save checkpoints
        if save_checkpoints:
            # For backward compatibility with checkpoint system, alias the best metric
            best_val_acc = best_mean_iou
            stats_data = {
                "epoch": epoch + 1,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
                "train_class_accs": train_class_accs,
                "val_class_accs": val_class_accs,
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
                "class_weights": (
                    class_weights_tensor.cpu().numpy()
                    if class_weights_tensor is not None
                    else None
                ),
                "training_config": {
                    "num_classes": num_classes,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "patience": patience,
                    "min_delta": min_delta,
                    "base_channels": base_channels,
                    "volumes_per_batch": volumes_per_batch,
                    "batches_per_epoch": batches_per_epoch,
                    "model_type": "dinov3_unet3d",
                    "model_id": model_id,
                    "input_channels": current_output_channels,
                    "use_class_weighting": use_class_weighting,
                    "target_volume_size": data_loader_3d.target_volume_size,
                    "dinov3_slice_size": data_loader_3d.dinov3_slice_size,
                    "image_size": data_loader_3d.dinov3_slice_size,  # For DINOv3 initialization
                    "seed": data_loader_3d.seed,  # Add seed
                    "use_mixed_precision": use_mixed_precision,  # Add memory efficiency params
                    "use_half_precision": use_half_precision,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                    "memory_efficient_mode": memory_efficient_mode,
                    "learn_upsampling": learn_upsampling,  # Add upsampling mode
                    "train_volume_pool_size": data_loader_3d.train_volume_pool_size,
                    "val_volume_pool_size": data_loader_3d.val_volume_pool_size,
                },
                "model_config": {
                    "num_classes": num_classes,
                    "base_channels": base_channels,
                    "input_size": data_loader_3d.target_volume_size,  # 3D volume size
                    "input_channels": current_output_channels,
                    "model_id": model_id,
                    "model_type": "dinov3_unet3d",
                    "dinov3_slice_size": data_loader_3d.dinov3_slice_size,
                },
            }

            # Save stats every epoch
            stats_path = os.path.join(checkpoint_dir, f"stats_epoch_{epoch+1:04d}.pkl")
            with open(stats_path, "wb") as f:
                pickle.dump(stats_data, f)

            # Always save latest stats
            latest_stats_path = os.path.join(checkpoint_dir, "latest_stats.pkl")
            with open(latest_stats_path, "wb") as f:
                pickle.dump(stats_data, f)

            # Only save full model checkpoint if this is the best performance
            if is_best_model:
                checkpoint_data = {
                    **stats_data,
                    "unet3d_state_dict": unet3d.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }

                best_path = os.path.join(checkpoint_dir, "best.pkl")
                with open(best_path, "wb") as f:
                    pickle.dump(checkpoint_data, f)

                print(
                    f"  *** NEW BEST 3D MODEL SAVED: IoU={current_mean_iou:.4f} (Acc={val_acc:.4f}) ***"
                )

            print(f"  Stats saved: {os.path.basename(stats_path)}")

        print()

        # Early stopping
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)"
            )
            break

    return {
        "unet3d": unet3d,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "train_class_accs": train_class_accs,
        "val_class_accs": val_class_accs,
        "val_class_precisions": val_class_precisions,
        "val_class_f1s": val_class_f1s,
        "val_class_ious": val_class_ious,
        "best_mean_iou": best_mean_iou,
        "epochs_trained": len(train_losses),
        "checkpoint_dir": checkpoint_dir,
        "export_base_dir": export_base_dir,
        "class_weights": (
            class_weights_tensor.cpu().numpy()
            if class_weights_tensor is not None
            else None
        ),
    }


def train_with_memory_efficient_loader(
    raw_data,
    gt_data,
    train_pool_size=100,
    val_pool_size=20,
    num_classes=None,
    target_size=(224, 224),
    batch_size=8,
    epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=20,
    min_delta=0.001,
    base_channels=64,
    device=None,
    seed=42,
    model_id=None,
    export_base_dir=None,
    save_checkpoints=True,
    use_class_weighting=True,
):
    """
    High-level function to train a 2D UNet using memory-efficient data loading.
    This is the 2D version (for backward compatibility).
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_classes is None:
        num_classes = len(np.unique(gt_data))

    if export_base_dir is None:
        export_base_dir = "/tmp/dinov3_2d_results"

    print(f"Setting up memory-efficient 2D UNet training:")
    print(f"  Raw data shape: {raw_data.shape}")
    print(f"  GT data shape: {gt_data.shape}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Target size: {target_size}")
    print(f"  Training pool: {train_pool_size}")
    print(f"  Validation pool: {val_pool_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {epochs}")
    print(f"  Base channels: {base_channels}")
    print(f"  Device: {device}")
    print(f"  Model ID: {model_id}")
    print(f"  Export directory: {export_base_dir}")
    print()

    # Create 2D data loader
    data_loader = MemoryEfficientDataLoader(
        raw_data=raw_data,
        gt_data=gt_data,
        train_pool_size=train_pool_size,
        val_pool_size=val_pool_size,
        target_size=target_size,
        seed=seed,
        model_id=model_id,
    )

    # Train 2D UNet
    results = train_unet_memory_efficient(
        data_loader=data_loader,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        base_channels=base_channels,
        batch_size=batch_size,
        model_id=model_id,
        export_base_dir=export_base_dir,
        save_checkpoints=save_checkpoints,
        use_class_weighting=use_class_weighting,
    )

    print(f"\n2D UNet training completed!")
    print(f"  Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"  Epochs trained: {results['epochs_trained']}")

    if save_checkpoints:
        print(f"  Checkpoints saved to: {results['checkpoint_dir']}")

    return results


def train_3d_unet_memory_efficient(
    raw_data,
    gt_data,
    pipeline,
    epochs=100,
    learning_rate=1e-3,
    batch_size=1,
    device=None,
):
    """
    Train 3D UNet with memory-efficient approach.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split data into train/val
    n_volumes = raw_data.shape[0]
    n_train = int(0.8 * n_volumes)

    train_raw = raw_data[:n_train]
    train_gt = gt_data[:n_train]
    val_raw = raw_data[n_train:]
    val_gt = gt_data[n_train:]

    print(f"Training volumes: {train_raw.shape[0]}")
    print(f"Validation volumes: {val_raw.shape[0]}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pipeline.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    print(f"Starting 3D UNet training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        pipeline.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        # Process training volumes
        for i in range(0, len(train_raw), batch_size):
            batch_raw = train_raw[i : i + batch_size]
            batch_gt = train_gt[i : i + batch_size]

            # Convert to tensors
            gt_tensor = torch.tensor(batch_gt, dtype=torch.long).to(device)

            optimizer.zero_grad()

            # Forward pass through pipeline
            logits = pipeline(batch_raw)
            loss = criterion(logits, gt_tensor)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == gt_tensor).sum().item()
            total = gt_tensor.numel()

            epoch_train_loss += loss.item() * total
            epoch_train_correct += correct
            epoch_train_total += total

        # Validation phase
        pipeline.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0

        with torch.no_grad():
            for i in range(0, len(val_raw), batch_size):
                batch_raw = val_raw[i : i + batch_size]
                batch_gt = val_gt[i : i + batch_size]

                gt_tensor = torch.tensor(batch_gt, dtype=torch.long).to(device)
                logits = pipeline(batch_raw)
                loss = criterion(logits, gt_tensor)

                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == gt_tensor).sum().item()
                total = gt_tensor.numel()

                epoch_val_loss += loss.item() * total
                epoch_val_correct += correct
                epoch_val_total += total

        # Calculate metrics
        train_loss = epoch_train_loss / epoch_train_total
        train_acc = epoch_train_correct / epoch_train_total
        val_loss = epoch_val_loss / epoch_val_total
        val_acc = epoch_val_correct / epoch_val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            print(f"  Best Val Acc: {best_val_acc:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            print()

    return {
        "pipeline": pipeline,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc,
    }


def train_3d_unet_with_memory_efficient_loader(
    raw_data,
    gt_data,
    train_volume_pool_size=20,
    val_volume_pool_size=5,
    num_classes=None,
    target_volume_size=(64, 64, 64),
    volumes_per_batch=1,
    batches_per_epoch=10,
    epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=20,
    min_delta=0.001,
    base_channels=32,
    dinov3_slice_size=256,
    device=None,
    seed=42,
    model_id=None,
    export_base_dir=None,
    save_checkpoints=True,
    use_class_weighting=True,
    # NEW MEMORY EFFICIENCY PARAMETERS
    use_mixed_precision=True,
    use_half_precision=False,
    use_gradient_checkpointing=False,
    memory_efficient_mode="auto",  # "auto", "aggressive", "conservative"
    learn_upsampling=False,  # NEW PARAMETER
    dinov3_stride=None,  # NEW PARAMETER for sliding window inference
):
    """
    Memory-efficient 3D UNet training with multiple precision and memory optimization options.

    Parameters:
    -----------
    dinov3_stride : int, optional
        Stride for DINOv3 sliding window inference. If None, uses patch_size (16) for standard inference.
        Use smaller values (e.g., 8, 4) for higher resolution features at the cost of increased computation:
        - stride=8: 4x more features, 4x slower
        - stride=4: 16x more features, 16x slower
        Best used with learn_upsampling=True to avoid downsampling high-res features.
    """

    # Auto-detect best memory settings
    if memory_efficient_mode == "auto":
        gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (
            1024**3
        )
        if gpu_memory_gb < 8:
            use_mixed_precision = True
            use_half_precision = True
            volumes_per_batch = 1
            base_channels = min(base_channels, 16)
            print(
                f"Auto-detected GPU memory: {gpu_memory_gb:.1f}GB - Using aggressive memory optimization"
            )
        elif gpu_memory_gb < 16:
            use_mixed_precision = True
            use_half_precision = False
            print(
                f"Auto-detected GPU memory: {gpu_memory_gb:.1f}GB - Using moderate memory optimization"
            )
        else:
            print(
                f"Auto-detected GPU memory: {gpu_memory_gb:.1f}GB - Using standard settings"
            )

    elif memory_efficient_mode == "aggressive":
        use_mixed_precision = True
        use_half_precision = True
        use_gradient_checkpointing = True
        volumes_per_batch = 1
        base_channels = min(base_channels, 16)
        print("Using aggressive memory optimization settings")

    print(f"Memory optimization settings:")
    print(f"  - Mixed precision (AMP): {use_mixed_precision}")
    print(f"  - Half precision model: {use_half_precision}")
    print(f"  - Gradient checkpointing: {use_gradient_checkpointing}")
    print(f"  - Adjusted base channels: {base_channels}")
    print(f"  - Volumes per batch: {volumes_per_batch}")
    # Auto-detect number of classes if not provided
    if num_classes is None:
        unique_classes = np.unique(gt_data)
        num_classes = len(unique_classes)
        print(
            f"Auto-detected {num_classes} classes from ground truth data: {unique_classes}"
        )

    print(f"Setting up memory-efficient 3D UNet training:")
    print(f"  Raw data shape: {raw_data.shape}")
    print(f"  GT data shape: {gt_data.shape}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes in data: {np.unique(gt_data)}")
    print(f"  Target volume size: {target_volume_size}")
    print(f"  Training volume pool: {train_volume_pool_size}")
    print(f"  Validation volumes: {val_volume_pool_size}")
    print(f"  Volumes per batch: {volumes_per_batch}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Total volumes per epoch: {volumes_per_batch * batches_per_epoch}")
    print(f"  Max epochs: {epochs}")
    print(f"  Base channels: {base_channels}")
    print(f"  Device: {device}")
    print(f"  Model ID: {model_id}")
    print(f"  Export directory: {export_base_dir}")
    print(f"  Save checkpoints: {save_checkpoints}")
    print(f"  Use class weighting: {use_class_weighting}")
    print()

    # Validate input data
    if len(raw_data) < val_volume_pool_size + 2:
        raise ValueError(
            f"Need at least {val_volume_pool_size + 2} volumes for training"
        )

    # Validate shapes with ROI-level padding support
    if raw_data.shape[0] != gt_data.shape[0]:
        raise ValueError(
            f"Number of volumes must match: {raw_data.shape[0]} vs {gt_data.shape[0]}"
        )

    if raw_data.shape[1] != gt_data.shape[1]:
        raise ValueError(
            f"Depth dimension must match: {raw_data.shape[1]} vs {gt_data.shape[1]}"
        )

    # Check if raw data has ROI-level padding (different spatial dimensions)
    raw_spatial = raw_data.shape[2:]  # (H, W)
    gt_spatial = gt_data.shape[2:]  # (H, W)

    if raw_spatial != gt_spatial:
        # Calculate expected padding based on dinov3_stride
        roi_padding = 0
        if dinov3_stride is not None and dinov3_stride < 16:
            roi_padding = 16 - dinov3_stride

        expected_raw_spatial = tuple(dim + 2 * roi_padding for dim in gt_spatial)

        if raw_spatial != expected_raw_spatial:
            raise ValueError(
                f"Raw spatial dimensions {raw_spatial} don't match GT {gt_spatial} "
                f"or expected padded dimensions {expected_raw_spatial} for stride {dinov3_stride}"
            )

        print(
            f"✓ Detected ROI-level padding: Raw {raw_spatial} vs GT {gt_spatial} (padding={roi_padding})"
        )
    else:
        print(f"✓ No ROI-level padding: Raw and GT shapes match {raw_data.shape}")

    if len(raw_data.shape) != 4:
        raise ValueError(
            f"Expected 4D data (num_volumes, D, H, W), got {raw_data.shape}"
        )

    # Create 3D data loader
    print("Creating memory-efficient 3D data loader...")
    data_loader_3d = MemoryEfficientDataLoader3D(
        raw_data=raw_data,
        gt_data=gt_data,
        train_volume_pool_size=train_volume_pool_size,
        val_volume_pool_size=val_volume_pool_size,
        target_volume_size=target_volume_size,
        dinov3_slice_size=dinov3_slice_size,
        seed=seed,
        model_id=model_id,
        learn_upsampling=learn_upsampling,
        dinov3_stride=dinov3_stride,  # NEW: Sliding window parameter
    )

    print("Data loader created successfully!")
    print()

    # Train 3D UNet
    print("Starting 3D UNet training...")
    results = train_3d_unet_memory_efficient_v2(
        data_loader_3d=data_loader_3d,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        base_channels=base_channels,
        volumes_per_batch=volumes_per_batch,
        batches_per_epoch=batches_per_epoch,
        save_checkpoints=save_checkpoints,
        model_id=model_id,
        export_base_dir=export_base_dir,
        use_class_weighting=use_class_weighting,
        use_mixed_precision=use_mixed_precision,  # Pass through
        use_half_precision=use_half_precision,  # Pass through additional params
        use_gradient_checkpointing=use_gradient_checkpointing,
        memory_efficient_mode=memory_efficient_mode,
        learn_upsampling=learn_upsampling,  # Pass through new parameter
    )

    print(f"\n3D UNet training completed!")
    print(f"  Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"  Epochs trained: {results['epochs_trained']}")

    if save_checkpoints:
        print(f"  Checkpoints saved to: {results['checkpoint_dir']}")

    if "class_weights" in results and results["class_weights"] is not None:
        print(f"  Class weights used: {results['class_weights']}")

    return results


# Alias for consistency with 2D naming convention
train_with_3d_unet_memory_efficient_loader = train_3d_unet_with_memory_efficient_loader


# Add this function at the end of the file:


def train_with_unet_memory_efficient_loader(
    raw_data,
    gt_data,
    train_pool_size=100,
    val_pool_size=20,
    num_classes=None,
    target_size=(224, 224),
    batch_size=8,
    epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=20,
    min_delta=0.001,
    base_channels=64,
    device=None,
    seed=42,
    model_id=None,
    export_base_dir=None,
    save_checkpoints=True,
    use_class_weighting=True,
):
    """
    High-level function to train a 2D UNet using memory-efficient data loading.
    This is the 2D version with the exact naming expected by __init__.py.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_classes is None:
        num_classes = len(np.unique(gt_data))

    if export_base_dir is None:
        export_base_dir = "/tmp/dinov3_2d_results"

    print(f"Setting up memory-efficient 2D UNet training:")
    print(f"  Raw data shape: {raw_data.shape}")
    print(f"  GT data shape: {gt_data.shape}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Target size: {target_size}")
    print(f"  Training pool: {train_pool_size}")
    print(f"  Validation pool: {val_pool_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {epochs}")
    print(f"  Base channels: {base_channels}")
    print(f"  Device: {device}")
    print(f"  Model ID: {model_id}")
    print(f"  Export directory: {export_base_dir}")
    print()

    # Validate input data
    if len(raw_data) < val_pool_size + 2:
        raise ValueError(f"Need at least {val_pool_size + 2} images for training")

    if raw_data.shape != gt_data.shape:
        raise ValueError(
            f"Raw and GT data shapes must match: {raw_data.shape} vs {gt_data.shape}"
        )

    # Create 2D data loader
    print("Creating memory-efficient 2D data loader...")
    data_loader = MemoryEfficientDataLoader(
        raw_data=raw_data,
        gt_data=gt_data,
        train_pool_size=train_pool_size,
        val_pool_size=val_pool_size,
        target_size=target_size,
        seed=seed,
        model_id=model_id,
    )

    print("Data loader created successfully!")
    print()

    # Train 2D UNet
    print("Starting 2D UNet training...")
    results = train_unet_memory_efficient(
        data_loader=data_loader,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        base_channels=base_channels,
        batch_size=batch_size,
        model_id=model_id,
        export_base_dir=export_base_dir,
        save_checkpoints=save_checkpoints,
        use_class_weighting=use_class_weighting,
    )

    print(f"\n2D UNet training completed!")
    print(f"  Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"  Epochs trained: {results['epochs_trained']}")

    if save_checkpoints:
        print(f"  Checkpoints saved to: {results['checkpoint_dir']}")

    return results


# Alias for backward compatibility
train_with_memory_efficient_loader = train_with_unet_memory_efficient_loader


# Add these checkpoint utility functions at the end of the file:

import pickle
import glob


def load_checkpoint(checkpoint_path, device=None):
    """
    Load a training checkpoint.

    Parameters:
    -----------
    checkpoint_path : str
        Path to the checkpoint file (.pkl)
    device : torch.device, optional
        Device to load the checkpoint to

    Returns:
    --------
    dict: Checkpoint data containing model state, training stats, etc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from: {checkpoint_path}")

    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        print(f"✅ Checkpoint loaded successfully!")

        # Print basic info about the checkpoint
        if "epoch" in checkpoint:
            print(f"  - Epoch: {checkpoint['epoch']}")
        if "best_val_acc" in checkpoint:
            print(f"  - Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
        if "training_config" in checkpoint:
            config = checkpoint["training_config"]
            print(f"  - Model type: {config.get('model_type', 'unknown')}")
            print(f"  - Number of classes: {config.get('num_classes', 'unknown')}")
            print(f"  - Base channels: {config.get('base_channels', 'unknown')}")

        return checkpoint

    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def list_checkpoints(checkpoint_dir):
    """
    List available checkpoints in a directory.

    Parameters:
    -----------
    checkpoint_dir : str
        Directory containing checkpoint files

    Returns:
    --------
    dict: Dictionary with checkpoint types and their paths
    """

    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
        return {}

    checkpoints = {}

    # Look for different types of checkpoints
    best_path = os.path.join(checkpoint_dir, "best.pkl")
    if os.path.exists(best_path):
        checkpoints["best"] = best_path

    latest_stats_path = os.path.join(checkpoint_dir, "latest_stats.pkl")
    if os.path.exists(latest_stats_path):
        checkpoints["latest_stats"] = latest_stats_path

    # Look for epoch-specific checkpoints
    epoch_pattern = os.path.join(checkpoint_dir, "stats_epoch_*.pkl")
    epoch_files = glob.glob(epoch_pattern)
    if epoch_files:
        # Sort by epoch number
        epoch_files.sort(key=lambda x: int(x.split("_epoch_")[1].split(".")[0]))
        checkpoints["epoch_stats"] = epoch_files

    print(f"Found checkpoints in {checkpoint_dir}:")

    if "best" in checkpoints:
        print(f"  ✅ Best model: {checkpoints['best']}")

    if "latest_stats" in checkpoints:
        print(f"  📊 Latest stats: {checkpoints['latest_stats']}")

    if "epoch_stats" in checkpoints:
        print(f"  📈 Epoch stats: {len(checkpoints['epoch_stats'])} files")
        print(f"     Latest: {os.path.basename(checkpoints['epoch_stats'][-1])}")

    if not checkpoints:
        print(f"  ❌ No checkpoints found")

    return checkpoints


def get_checkpoint_base_path(model_type, model_id=None, export_base_dir=None):
    """
    Generate a base path for checkpoints based on model type and ID.

    Parameters:
    -----------
    model_type : str
        Type of model (e.g., "dinov3_unet", "dinov3_unet3d")
    model_id : str, optional
        DINOv3 model identifier
    export_base_dir : str, optional
        Base directory for exports

    Returns:
    --------
    str: Base path for checkpoints
    """
    if export_base_dir is None:
        export_base_dir = "/tmp/dinov3_results"

    # Create a clean model identifier
    if model_id:
        # Extract model name from full path
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        model_name = model_name.replace("-", "_")
    else:
        model_name = "default"

    base_path = os.path.join(export_base_dir, f"{model_type}_{model_name}")
    return base_path


def create_model_checkpoint_dir(base_path):
    """
    Create a timestamped checkpoint directory.

    Parameters:
    -----------
    base_path : str
        Base path for the model checkpoints

    Returns:
    --------
    str: Full path to the created checkpoint directory
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(base_path, f"run_{timestamp}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    return checkpoint_dir


def load_latest_checkpoint(checkpoint_dir, checkpoint_type="best"):
    """
    Load the latest checkpoint of a specific type.

    Parameters:
    -----------
    checkpoint_dir : str
        Directory containing checkpoints
    checkpoint_type : str, default="best"
        Type of checkpoint to load ("best", "latest_stats", "latest_epoch")

    Returns:
    --------
    dict: Checkpoint data, or None if not found
    """
    checkpoints = list_checkpoints(checkpoint_dir)

    if checkpoint_type == "best" and "best" in checkpoints:
        return load_checkpoint(checkpoints["best"])

    elif checkpoint_type == "latest_stats" and "latest_stats" in checkpoints:
        return load_checkpoint(checkpoints["latest_stats"])

    elif checkpoint_type == "latest_epoch" and "epoch_stats" in checkpoints:
        latest_epoch_file = checkpoints["epoch_stats"][-1]
        return load_checkpoint(latest_epoch_file)

    else:
        print(f"❌ No checkpoint of type '{checkpoint_type}' found in {checkpoint_dir}")
        return None


def restore_model_from_checkpoint(
    checkpoint_path, model_class, device=None, **model_kwargs
):
    """
    Restore a model from a checkpoint.

    Parameters:
    -----------
    checkpoint_path : str
        Path to checkpoint file
    model_class : class
        Model class to instantiate
    device : torch.device, optional
        Device to load model to
    **model_kwargs : dict
        Arguments for model constructor

    Returns:
    --------
    tuple: (model, checkpoint_data)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)

    # Create model with saved configuration
    if "training_config" in checkpoint:
        config = checkpoint["training_config"]

        # Update model_kwargs with saved config
        if "input_channels" in config:
            model_kwargs["input_channels"] = config["input_channels"]
        if "num_classes" in config:
            model_kwargs["num_classes"] = config["num_classes"]
        if "base_channels" in config:
            model_kwargs["base_channels"] = config["base_channels"]

    # Instantiate model
    model = model_class(**model_kwargs).to(device)

    # Load model weights
    if "unet_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["unet_state_dict"])
    elif "unet3d_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["unet3d_state_dict"])
    else:
        raise KeyError("No model state dict found in checkpoint")

    print(f"✅ Model restored from checkpoint")
    print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  - Best val acc: {checkpoint.get('best_val_acc', 'unknown')}")

    return model, checkpoint
