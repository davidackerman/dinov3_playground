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


def print_class_metrics_table(
    class_names,
    train_recall=None,
    train_dist=None,
    val_recall=None,
    val_precision=None,
    val_f1=None,
    val_iou=None,
    val_dist=None,
):
    """
    Print per-class metrics in a nicely formatted table.

    Parameters:
    -----------
    class_names : list of str
        Names of classes (e.g., ['background', 'nuc', 'mito', 'er'])
    train_recall, train_dist, val_recall, val_precision, val_f1, val_iou, val_dist : list of float, optional
        Metric values for each class. All should have same length as class_names.
    """
    if not class_names:
        return

    # Determine which metrics to display
    metrics_to_show = []
    if train_recall is not None:
        metrics_to_show.append(("Train Recall", train_recall))
    if train_dist is not None:
        metrics_to_show.append(("Train Dist", train_dist))
    if val_recall is not None:
        metrics_to_show.append(("Val Recall", val_recall))
    if val_precision is not None:
        metrics_to_show.append(("Val Precision", val_precision))
    if val_f1 is not None:
        metrics_to_show.append(("Val F1", val_f1))
    if val_iou is not None:
        metrics_to_show.append(("Val IoU", val_iou))
    if val_dist is not None:
        metrics_to_show.append(("Val Dist", val_dist))

    if not metrics_to_show:
        return

    # Calculate column widths
    class_col_width = max(max(len(name) for name in class_names), len("Class"))
    metric_col_width = 12

    # Print header
    header = f"  {'Class':<{class_col_width}}"
    for metric_name, _ in metrics_to_show:
        header += f" {metric_name:>{metric_col_width}}"
    print(header)
    print("  " + "-" * (class_col_width + metric_col_width * len(metrics_to_show)))

    # Print rows for each class
    for i, class_name in enumerate(class_names):
        row = f"  {class_name:<{class_col_width}}"
        for metric_name, metric_values in metrics_to_show:
            if i < len(metric_values):
                value = metric_values[i]
                # Format differently for distribution percentages vs metrics
                if "Dist" in metric_name:
                    row += f" {value:>{metric_col_width - 1}.1f}%"
                else:
                    row += f" {value:>{metric_col_width}.3f}"
            else:
                row += f" {'N/A':>{metric_col_width}}"
        print(row)
    print()  # Blank line after table


class MemoryEfficientDataLoader3D:
    """
    Memory-efficient data loader for 3D DINOv3 UNet training.
    Handles 3D volumes instead of 2D images.
    """

    def __init__(
        self,
        raw_data,
        gt_data,
        gt_masks=None,  # NEW PARAMETER for GT extension masks
        context_data=None,  # NEW PARAMETER for context volumes at lower resolution
        context_scale=None,  # NEW PARAMETER for context resolution (e.g., 8 for 8nm)
        train_volume_pool_size=20,
        val_volume_pool_size=5,
        target_volume_size=(64, 64, 64),
        dinov3_slice_size=896,
        seed=42,
        model_id=None,
        learn_upsampling=False,  # NEW PARAMETER
        dinov3_stride=None,  # NEW PARAMETER for sliding window inference
        use_orthogonal_planes=True,  # NEW PARAMETER for 3-plane processing
        enable_detailed_timing=True,  # NEW PARAMETER for detailed timing
        verbose=True,  # NEW PARAMETER to control verbose output
        output_type="labels",  # NEW PARAMETER: "labels", "affinities", or "affinities_lsds"
        affinity_offsets=None,  # NEW PARAMETER: list of (z,y,x) offsets for affinities
        lsds_sigma=20.0,  # NEW PARAMETER: sigma for LSDS computation
    ):
        """
        Initialize memory-efficient 3D data loader.

        Parameters:
        -----------
        raw_data : numpy.ndarray
            4D raw volume data (num_volumes, D, H, W)
        gt_data : numpy.ndarray
            4D ground truth data (num_volumes, D, H, W)
        gt_masks : numpy.ndarray, optional
            4D mask data (num_volumes, D, H, W) with 1 where GT is valid, 0 elsewhere.
            If None, assumes all GT regions are valid (full mask of 1s).
        context_data : numpy.ndarray, optional
            4D context volume data (num_volumes, D, H, W) at lower resolution for spatial context.
            If None, no contextual information will be used.
        context_scale : int or float, optional
            Resolution of context data in nm (e.g., 8 for 8nm). Used for feature processing.
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
        use_orthogonal_planes : bool, default=True
            If True, processes slices in all 3 orthogonal planes (XY, XZ, YZ) and averages them.
            If False, processes only XY planes (original behavior).
        output_type : str, default="labels"
            Type of output target: "labels" for class labels, "affinities" for affinity graphs,
            or "affinities_lsds" for combined affinities and LSDs.
            When "affinities", GT will be converted from instance segmentation to affinities.
            When "affinities_lsds", GT will be converted to both affinities and LSDs.
        affinity_offsets : list of tuples, optional
            List of (z, y, x) offsets for computing affinities. If None and output_type="affinities"
            or "affinities_lsds", defaults to [(1,0,0), (0,1,0), (0,0,1)] for +z, +y, +x directions.
        lsds_sigma : float, default=20.0
            Sigma parameter for LSDS computation (only used when output_type="affinities_lsds").
            Controls the smoothing scale for Local Shape Descriptors.
        """

        # Validate inputs with ROI-level padding support
        if raw_data.shape[0] != gt_data.shape[0]:
            raise ValueError(
                f"Number of volumes must match: {raw_data.shape[0]} vs {gt_data.shape[0]}"
            )

        # Allow different dimensions for multi-resolution training
        # Raw data can be at higher resolution than GT data
        if len(raw_data.shape) >= 2:
            raw_shape = raw_data.shape[1:]  # (D, H, W)
            gt_shape = gt_data.shape[1:]  # (D, H, W)

            if raw_shape == gt_shape:
                print(f"✓ DataLoader: Same resolution mode - {raw_shape}")
            else:
                print(
                    f"✓ DataLoader: Multi-resolution mode - Raw {raw_shape} → GT {gt_shape}"
                )

                # Validate that dimensions are reasonable ratios
                for dim_name, (raw_dim, gt_dim) in zip(
                    ["D", "H", "W"], zip(raw_shape, gt_shape)
                ):
                    ratio = raw_dim / gt_dim
                    if (
                        ratio < 0.1 or ratio > 16.0
                    ):  # Allow wide range but catch obvious errors
                        print(
                            f"Warning: {dim_name} dimension ratio ({ratio:.2f}) is very extreme"
                        )

        # Multi-resolution training allows different spatial dimensions
        # This replaces the old ROI padding validation with more flexible logic
        if len(raw_data.shape) >= 4:
            raw_spatial = raw_data.shape[2:]
            gt_spatial = gt_data.shape[2:]
            if raw_spatial != gt_spatial:
                print(
                    f"✓ DataLoader: Different spatial dimensions - Raw {raw_spatial} vs GT {gt_spatial}"
                )
                print("  This is expected for multi-resolution training")
            else:
                print(f"✓ DataLoader: Same spatial dimensions - {raw_spatial}")

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

        # Handle GT masks - create full masks if not provided
        if gt_masks is None:
            self.gt_masks = np.ones_like(gt_data, dtype=np.uint8)
            print("  - GT masks: Created full masks (all regions valid)")
        else:
            if gt_masks.shape != gt_data.shape:
                raise ValueError(
                    f"GT masks shape {gt_masks.shape} doesn't match GT data shape {gt_data.shape}"
                )
            self.gt_masks = gt_masks
            valid_fraction = np.mean(gt_masks)
            print(
                f"  - GT masks: Using provided masks ({valid_fraction:.3f} average valid fraction)"
            )

        # Handle context data
        self.context_data = context_data
        self.context_scale = context_scale
        self.has_context = context_data is not None and context_scale is not None

        if self.has_context:
            if context_data.shape[0] != raw_data.shape[0]:
                raise ValueError(
                    f"Context data volume count {context_data.shape[0]} doesn't match raw data {raw_data.shape[0]}"
                )
            print(
                f"  - Context data: Using {context_scale}nm context data, shape {context_data.shape}"
            )
        else:
            print("  - Context data: No context data provided")

        self.train_volume_pool_size = train_volume_pool_size
        self.val_volume_pool_size = val_volume_pool_size
        self.target_volume_size = target_volume_size
        self.dinov3_slice_size = dinov3_slice_size
        self.model_id = model_id
        self.seed = seed  # Store seed for config saving
        self.learn_upsampling = learn_upsampling  # Store upsampling mode
        self.dinov3_stride = dinov3_stride  # Store sliding window stride
        self.use_orthogonal_planes = (
            use_orthogonal_planes  # Store orthogonal planes mode
        )
        self.verbose = verbose  # Store verbose output mode

        # Handle affinity output
        self.output_type = output_type
        self.lsds_sigma = lsds_sigma  # Store LSDS sigma parameter

        if affinity_offsets is None and output_type in [
            "affinities",
            "affinities_lsds",
        ]:
            # Default: affinities in +z, +y, +x directions
            self.affinity_offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        else:
            self.affinity_offsets = affinity_offsets

        if output_type == "affinities":
            print(f"  - Output type: Affinities with offsets {self.affinity_offsets}")
        elif output_type == "affinities_lsds":
            print(f"  - Output type: Affinities + LSDs")
            print(f"    - Affinity offsets: {self.affinity_offsets}")
            print(f"    - LSDS sigma: {lsds_sigma}")
            print(
                f"    - Total output channels: 10 (LSDs) + {len(self.affinity_offsets)} (affinities) = {10 + len(self.affinity_offsets)}"
            )
        else:
            print(f"  - Output type: Class labels")

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
        tuple: (batch_volumes, batch_gt, batch_masks, batch_context)
               - batch_volumes: shape (batch_size, D, H, W)
               - batch_gt: shape (batch_size, D, H, W) for labels or (batch_size, num_offsets, D, H, W) for affinities
               - batch_masks: shape (batch_size, D, H, W)
               - batch_context: None or shape (batch_size, D, H, W)
        """
        # Sample random volumes from training pool
        sampled_indices = self.rng.choice(
            self.train_pool_indices,
            size=batch_size,
            replace=len(self.train_pool_indices) < batch_size,
        )

        batch_volumes = self.raw_data[sampled_indices]
        batch_gt = self.gt_data[sampled_indices]
        batch_masks = self.gt_masks[sampled_indices]

        # Convert to affinities or affinities+lsds if needed
        if self.output_type == "affinities":
            from .affinity_utils import compute_affinities_3d

            # Convert each volume's instance segmentation to affinities
            batch_targets = []
            for gt_volume in batch_gt:
                affinities = compute_affinities_3d(
                    gt_volume, offsets=self.affinity_offsets
                )
                batch_targets.append(affinities)

            # Stack into batch: (batch_size, num_offsets, D, H, W)
            batch_targets = np.stack(batch_targets, axis=0)

        elif self.output_type == "affinities_lsds":
            from .affinity_utils import compute_affinities_and_lsds_3d

            # Convert each volume's instance segmentation to affinities and LSDs
            batch_targets = []
            for gt_volume in batch_gt:
                combined = compute_affinities_and_lsds_3d(
                    gt_volume, offsets=self.affinity_offsets, lsds_sigma=self.lsds_sigma
                )
                batch_targets.append(combined)

            # Stack into batch: (batch_size, 10 + num_offsets, D, H, W)
            batch_targets = np.stack(batch_targets, axis=0)

        # Add context data if available
        if self.has_context:
            batch_context = self.context_data[sampled_indices]
        else:
            batch_context = None

        return batch_volumes, batch_gt, batch_targets, batch_masks, batch_context

    def get_validation_data(self):
        """
        Get the fixed validation dataset.

        Returns:
        --------
        tuple: (val_volumes, val_gt, val_masks, val_context)
               - val_volumes: shape (val_pool_size, D, H, W)
               - val_gt: shape (val_pool_size, D, H, W) for labels,
                         (val_pool_size, num_offsets, D, H, W) for affinities,
                         or (val_pool_size, 10 + num_offsets, D, H, W) for affinities_lsds
               - val_masks: shape (val_pool_size, D, H, W)
               - val_context: None or shape (val_pool_size, D, H, W)
        """
        val_volumes = self.raw_data[self.val_indices]
        val_gt = self.gt_data[self.val_indices]
        val_masks = self.gt_masks[self.val_indices]

        # Convert to affinities or affinities+lsds if needed
        if self.output_type == "affinities":
            from .affinity_utils import compute_affinities_3d

            # Convert each volume's instance segmentation to affinities
            val_targets = []
            for gt_volume in val_gt:
                affinities = compute_affinities_3d(
                    gt_volume, offsets=self.affinity_offsets
                )
                val_targets.append(affinities)

            # Stack into batch: (val_pool_size, num_offsets, D, H, W)
            val_targets = np.stack(val_targets, axis=0)

        elif self.output_type == "affinities_lsds":
            from .affinity_utils import compute_affinities_and_lsds_3d

            # Convert each volume's instance segmentation to affinities and LSDs
            val_targets = []
            for gt_volume in val_gt:
                combined = compute_affinities_and_lsds_3d(
                    gt_volume, offsets=self.affinity_offsets, lsds_sigma=self.lsds_sigma
                )
                val_targets.append(combined)

            # Stack into batch: (val_pool_size, 10 + num_offsets, D, H, W)
            val_targets = np.stack(val_targets, axis=0)

        # Add context data if available
        if self.has_context:
            val_context = self.context_data[self.val_indices]
        else:
            val_context = None

        return val_volumes, val_gt, val_targets, val_masks, val_context

    def extract_dinov3_features_3d(
        self, volumes, epoch=None, batch=None, enable_detailed_timing=False
    ):
        """
        Extract DINOv3 features from 3D volumes by processing slices.
        Uses orthogonal planes if enabled.
        """
        from .models import DINOv3UNet3DPipeline
        from .dinov3_core import get_current_model_info

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if volumes.ndim == 3:
            volumes = volumes[np.newaxis, ...]

        batch_size, depth, height, width = volumes.shape

        # Get current model info to determine output channels
        model_info = get_current_model_info()
        current_output_channels = model_info["output_channels"]

        # Create a temporary pipeline instance for feature extraction
        temp_pipeline = DINOv3UNet3DPipeline(
            num_classes=2,  # Dummy value, only using for feature extraction
            input_size=self.target_volume_size,
            dinov3_slice_size=self.dinov3_slice_size,
            base_channels=32,  # Dummy value
            input_channels=current_output_channels,
            use_orthogonal_planes=self.use_orthogonal_planes,
            device=device,
            verbose=self.verbose,  # Pass verbose flag to suppress messages
        )

        # GPU-accelerated batch processing optimization
        import time

        batch_processing_start = time.time()

        # Simple GPU-optimized stacking - no complex batch processing
        # Pre-allocate GPU tensor for batch features to avoid repeated memory allocation
        expected_shape = (
            batch_size,
            current_output_channels,
            *self.target_volume_size,
        )
        batch_features = torch.zeros(expected_shape, device=device, dtype=torch.float32)

        # Process each volume through the pipeline's feature extractor
        # with GPU-optimized direct assignment instead of list append + stack
        for b in range(batch_size):
            # Extract a single volume
            single_volume = volumes[b : b + 1]  # Keep batch dimension

            # Use the pipeline's orthogonal feature extraction
            # Pass target_volume_size as target output to ensure features match GT size
            if enable_detailed_timing:
                volume_features, volume_timing = (
                    temp_pipeline.extract_dinov3_features_3d(
                        single_volume,
                        use_orthogonal_planes=self.use_orthogonal_planes,
                        enable_timing=True,
                        target_output_size=self.target_volume_size,
                    )
                )
                # Store timing for this volume
                if b == 0:  # First volume, initialize timing aggregation
                    detailed_timing = volume_timing
                else:  # Subsequent volumes, aggregate timing
                    for key in [
                        "total_upsampling_time",
                        "total_stacking_time",
                        "total_dinov3_inference_time",
                        "total_feature_extraction_time",
                        "total_downsampling_time",
                        "total_batches",
                        "total_slices",
                    ]:
                        if key in detailed_timing:
                            detailed_timing[key] += volume_timing.get(key, 0)
            else:
                volume_features = temp_pipeline.extract_dinov3_features_3d(
                    single_volume,
                    use_orthogonal_planes=self.use_orthogonal_planes,
                    target_output_size=self.target_volume_size,
                )

            # GPU-optimized direct assignment instead of list append + stack
            # This is much faster than: all_features.append() + torch.stack(all_features)
            # volume_features shape: (1, output_channels, D, H, W)
            if volume_features.dim() == 5 and volume_features.shape[0] == 1:
                # Direct GPU tensor assignment - eliminates CPU-GPU transfer overhead
                batch_features[b] = volume_features.squeeze(0).to(device)
            else:
                print(
                    f"WARNING: Unexpected volume_features shape: {volume_features.shape}"
                )
                batch_features[b] = volume_features.to(device)

        batch_processing_time = time.time() - batch_processing_start
        processing_method = "gpu_optimized_stacking"

        if enable_detailed_timing and "detailed_timing" in locals():
            detailed_timing["batch_processing_time"] = batch_processing_time
            detailed_timing["batch_processing_method"] = processing_method
            detailed_timing["batch_size_processed"] = batch_size

        # Clear temporary pipeline and input volumes from GPU memory
        del temp_pipeline
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if enable_detailed_timing:
            return batch_features, detailed_timing
        else:
            return batch_features

    def extract_multi_scale_dinov3_features_3d(
        self,
        local_volumes,
        context_volumes=None,
        epoch=None,
        batch=None,
        enable_detailed_timing=False,
    ):
        """
        Extract both local high-resolution and contextual low-resolution DINOv3 features.

        IMPORTANT: Returns features SEPARATELY for proper context fusion architecture.
        Context features should NOT be mixed with raw features via simple concatenation.
        Instead, they should be passed separately to the UNet for attention-based fusion.

        Parameters:
        -----------
        local_volumes : numpy.ndarray, shape (batch_size, D, H, W)
            High-resolution local volumes
        context_volumes : numpy.ndarray, optional
            Low-resolution context volumes, same batch size
        epoch : int, optional
            Current training epoch (for logging)
        batch : int, optional
            Current batch index (for logging)
        enable_detailed_timing : bool
            Whether to return detailed timing information

        Returns:
        --------
        If context_volumes is None:
            - local_features: torch.Tensor (batch_size, channels, D, H, W)
            - None: No context features
        If context_volumes is provided:
            - local_features: torch.Tensor (batch_size, channels, D, H, W)
            - context_features: torch.Tensor (batch_size, channels, D, H, W)

        If enable_detailed_timing=True, also returns timing dict
        """
        import torch.nn.functional as F

        # Extract local high-resolution features
        if enable_detailed_timing:
            local_features, local_timing = self.extract_dinov3_features_3d(
                local_volumes, epoch, batch, enable_detailed_timing=True
            )
        else:
            local_features = self.extract_dinov3_features_3d(
                local_volumes, epoch, batch, enable_detailed_timing=False
            )

        # If no context data, return just local features and None for context
        if context_volumes is None or not self.has_context:
            if enable_detailed_timing:
                return local_features, None, local_timing
            else:
                return local_features, None

        # Extract context features
        if enable_detailed_timing:
            context_features, context_timing = self.extract_dinov3_features_3d(
                context_volumes, epoch, batch, enable_detailed_timing=True
            )
        else:
            context_features = self.extract_dinov3_features_3d(
                context_volumes, epoch, batch, enable_detailed_timing=False
            )

        # Clear input context volumes from memory after feature extraction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Resize context features to match local feature spatial dimensions if needed
        if context_features.shape[2:] != local_features.shape[2:]:
            context_features = F.interpolate(
                context_features,
                size=local_features.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        # Return SEPARATE features for proper context fusion architecture
        if enable_detailed_timing:
            # Combine timing information
            combined_timing = local_timing.copy()
            combined_timing.update(
                {f"context_{k}": v for k, v in context_timing.items()}
            )
            combined_timing["local_channels"] = local_features.shape[1]
            combined_timing["context_channels"] = context_features.shape[1]
            return local_features, context_features, combined_timing
        else:
            return local_features, context_features

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
    checkpoint_every_n_epochs=None,
    model_id=None,
    export_base_dir=None,
    use_class_weighting=True,
    use_mixed_precision=True,  # Add this parameter
    # Loss function parameters
    loss_type="weighted_ce",  # NEW: 'ce', 'weighted_ce', 'focal', 'dice', 'focal_dice', 'tversky'
    focal_gamma=2.0,  # NEW: Focusing parameter for Focal Loss
    focal_weight=0.5,  # NEW: Weight for focal component in combined loss
    dice_weight=0.5,  # NEW: Weight for dice component in combined loss
    dice_smooth=1.0,  # NEW: Smoothing for Dice loss
    tversky_alpha=0.5,  # NEW: Tversky alpha (FP weight)
    tversky_beta=0.5,  # NEW: Tversky beta (FN weight)
    # Additional parameters for complete config saving
    use_half_precision=False,
    use_gradient_checkpointing=False,
    memory_efficient_mode="auto",
    learn_upsampling=False,  # NEW PARAMETER
    enable_detailed_timing=True,
    # Data resolution parameters
    min_resolution_for_raw=None,
    base_resolution=None,
    class_names=None,  # NEW PARAMETER for class names
    # Boundary weighting parameters (for boundary_affinity loss)
    boundary_weight=5.0,  # Maximum weight at instance boundaries
    boundary_sigma=5.0,  # Distance decay for boundary weights
    boundary_anisotropy=None,  # Voxel anisotropy (z,y,x) for EDT
    mask_clip_distance=None,
    use_batchrenorm=False,  # Whether to use BatchRenorm instead of BatchNorm
):
    """
    Train 3D UNet using memory-efficient data loading with DINOv3 features.
    Now supports mixed precision training for reduced memory usage.
    """
    print("******************************Using batchrenorm in V2:", use_batchrenorm)
    # Generate default class names if not provided
    if class_names is None:
        if num_classes == 2:
            class_names = ["background", "foreground"]
        else:
            class_names = ["background"] + [f"class_{i}" for i in range(1, num_classes)]
        print(f"Generated default class names: {class_names}")
    else:
        print(f"Using provided class names: {class_names}")

    print(f"Memory-efficient 3D UNet training setup:")
    print(f"  - Volumes per batch: {volumes_per_batch}")
    print(f"  - Batches per epoch: {batches_per_epoch}")
    print(f"  - Total volumes per epoch: {volumes_per_batch * batches_per_epoch}")
    print(f"  - Loss type: {loss_type}")
    print(f"  - Class weighting: {use_class_weighting}")
    print(f"  - Mixed precision: {use_mixed_precision}")
    print(f"  - Learn upsampling: {learn_upsampling}")  # NEW
    print(f"  - GT extension masks: Conditional masking (auto-detected per batch)")

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
    val_volumes, val_segmentations, val_targets, val_masks, val_context = (
        data_loader_3d.get_validation_data()
    )

    print(f"Validation set: {len(val_volumes)} volumes prepared")
    print(f"Validation labels will be processed volume-by-volume for memory efficiency")
    print(f"Expected validation shape per volume: {val_segmentations[0].shape}")
    print(
        "Note: Validation features and labels will be computed volume-by-volume on-demand"
    )

    # Calculate class weights from training data (process volumes on CPU to avoid GPU memory issues)
    if use_class_weighting:
        print("Calculating class weights from training data...")
        # Sample training data to calculate class distribution from masked regions
        print("Sampling training volumes to estimate class distribution...")

        # Use a reasonable sample of training volumes for class weight calculation
        sample_size = min(
            10, len(data_loader_3d.train_pool_indices)
        )  # Sample up to 10 volumes
        sample_indices = np.random.choice(
            data_loader_3d.train_pool_indices, size=sample_size, replace=False
        )

        all_train_labels = []
        total_masked_voxels = 0

        for idx in sample_indices:
            train_segmentations = data_loader_3d.gt_data[idx]
            train_mask = data_loader_3d.gt_masks[idx]

            # Check if mask is actually restricting regions (not all 1s)
            if not np.all(train_mask == 1):
                # Only consider masked regions for class distribution
                mask_bool = train_mask.astype(bool)
                masked_labels = train_segmentations[mask_bool]
                all_train_labels.append(masked_labels.flatten())
                total_masked_voxels += np.sum(mask_bool)
            else:
                # If mask is all 1s, use all voxels
                all_train_labels.append(train_segmentations.flatten())
                total_masked_voxels += train_segmentations.size

        train_labels_flat = np.concatenate(all_train_labels)

        unique_classes, class_counts = np.unique(train_labels_flat, return_counts=True)
        total_voxels = len(train_labels_flat)

        class_weights = np.zeros(num_classes)
        for i, (class_id, count) in enumerate(zip(unique_classes, class_counts)):
            if class_id < num_classes:
                class_weights[class_id] = total_voxels / (num_classes * count)

        class_weights = class_weights * num_classes / np.sum(class_weights)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
            device
        )

        print(
            f"Class distribution in training data (from {sample_size} sampled volumes, {total_masked_voxels:,} masked voxels):"
        )
        for i, (class_id, count) in enumerate(zip(unique_classes, class_counts)):
            if class_id < num_classes:
                percentage = (count / total_voxels) * 100
                weight = class_weights[class_id]
                print(
                    f"  Class {class_id}: {count:,} voxels ({percentage:.2f}%), weight: {weight:.3f}"
                )

        # Clean up temporary data
        del all_train_labels, train_labels_flat
    else:
        class_weights_tensor = None
        print("Using unweighted loss (no class balancing)")

    # Get current model info
    from .dinov3_core import get_current_model_info

    model_info = get_current_model_info()
    current_output_channels = model_info["output_channels"]

    # Context fusion setup
    use_context_fusion = data_loader_3d.has_context
    if use_context_fusion:
        print(f"✓ Context fusion ENABLED: Using attention-based multi-scale fusion")
        print(
            f"  - Raw features: {current_output_channels} channels from {model_info['model_id']}"
        )
        print(
            f"  - Context features: {current_output_channels} channels at {data_loader_3d.context_scale}nm"
        )
        print(
            f"  - Fusion: Context guides raw features via attention at skip connections"
        )
        model_input_channels = (
            current_output_channels  # Only raw goes through main encoder
        )
        context_channels = current_output_channels  # Context processed separately
    else:
        print(
            f"Using {current_output_channels} input channels for 3D UNet (from DINOv3 {model_info['model_id']})"
        )
        model_input_channels = current_output_channels
        context_channels = None

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
        input_channels=model_input_channels,  # Raw feature channels
        num_classes=num_classes,
        base_channels=base_channels,
        input_size=data_loader_3d.target_volume_size,
        use_half_precision=use_half_precision,
        learn_upsampling=learn_upsampling,
        dinov3_feature_size=dinov3_feature_size,
        use_gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        use_context_fusion=use_context_fusion,  # Enable context fusion architecture
        context_channels=context_channels
        or current_output_channels,  # Context feature dimension
        use_batchrenorm=use_batchrenorm,
    ).to(device)

    print(
        f"Using DINOv3UNet3D with {base_channels} base channels and {model_input_channels} input channels"
    )
    if use_context_fusion:
        print(
            f"  - Context fusion layers: 4 attention modules at encoder skip connections"
        )

    # Loss function setup with configurable loss type
    from .losses import get_loss_function

    # Prepare class weights if using weighted loss
    loss_class_weights = None
    if use_class_weighting and class_weights_tensor is not None:
        loss_class_weights = class_weights_tensor

    # Prepare boundary weighting parameters
    # Set anisotropy based on resolutions if not provided
    if (
        boundary_anisotropy is None
        and base_resolution is not None
        and min_resolution_for_raw is not None
    ):
        # Use resolution ratio as anisotropy hint
        # For isotropic data, this will be (1.0, 1.0, 1.0)
        boundary_anisotropy = (1.0, 1.0, 1.0)

    # Get loss function based on type
    criterion = get_loss_function(
        loss_type=loss_type,
        class_weights=loss_class_weights,
        gamma=focal_gamma,
        focal_weight=focal_weight,
        dice_weight=dice_weight,
        dice_smooth=dice_smooth,
        alpha=tversky_alpha,
        beta=tversky_beta,
        # Boundary-weighted affinity parameters
        boundary_weight=boundary_weight,
        sigma=boundary_sigma,
        anisotropy=boundary_anisotropy,
        mask_clip_distance=mask_clip_distance,
    )

    # Print loss configuration
    print(f"\nLoss Configuration:")
    print(f"  - Loss type: {loss_type}")
    if loss_type == "focal":
        print(f"  - Focal gamma: {focal_gamma}")
        if loss_class_weights is not None:
            print(f"  - Class weights: {loss_class_weights.cpu().numpy()}")
    elif loss_type == "focal_dice":
        print(f"  - Focal gamma: {focal_gamma}")
        print(f"  - Focal weight: {focal_weight}")
        print(f"  - Dice weight: {dice_weight}")
        print(f"  - Dice smooth: {dice_smooth}")
        if loss_class_weights is not None:
            print(f"  - Class weights: {loss_class_weights.cpu().numpy()}")
    elif loss_type == "dice":
        print(f"  - Dice smooth: {dice_smooth}")
    elif loss_type == "tversky":
        print(f"  - Tversky alpha (FP weight): {tversky_alpha}")
        print(f"  - Tversky beta (FN weight): {tversky_beta}")
        print(f"  - Dice smooth: {dice_smooth}")
    elif loss_type == "weighted_ce":
        if loss_class_weights is not None:
            print(f"  - Class weights: {loss_class_weights.cpu().numpy()}")
    elif loss_type == "affinity":
        print(f"  - Using class weighting: {use_class_weighting}")
    elif loss_type == "boundary_affinity":
        print(f"  - Boundary weight: {boundary_weight}")
        print(f"  - Boundary sigma: {boundary_sigma}")
        print(f"  - Boundary anisotropy: {boundary_anisotropy}")
        print(f"  - Using class weighting: {use_class_weighting}")
    elif loss_type == "affinity_lsds":
        print(f"  - LSDS channels: 10")
        print(f"  - Affinity channels: {num_classes - 10}")
        print(f"  - Using class weighting for affinities: {use_class_weighting}")

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
    # Best validation metric: for segmentation this is IoU/accuracy (start at 0.0).
    # For affinity/affinities_lsds we use negative val_loss as the working metric (higher is better),
    # so initialize to -inf so the first measured val_loss will be able to improve it.
    if data_loader_3d.output_type in ["affinities", "affinities_lsds"]:
        best_metric = -float("inf")
    else:
        best_metric = 0.0
    best_val_loss = float("inf")  # Track best validation loss for affinities/LSDs
    epochs_without_improvement = 0

    # Helper function to compute loss (handles both CE-based and Dice-based losses)
    def compute_loss(criterion, logits, targets, masks=None, instance_seg=None):
        """
        Compute loss, handling different loss function signatures.

        Some losses (Dice, Focal+Dice, Tversky) need num_classes parameter.
        Affinity losses (affinity, boundary_affinity, affinity_lsds) need mask and instance_seg.
        """
        if loss_type in ["dice", "focal_dice", "tversky"]:
            return criterion(logits, targets, num_classes)
        elif loss_type in [
            "affinity",
            "boundary_affinity",
            "affinity_lsds",
            "boundary_affinity_focal_lsds",
        ]:
            # Affinity losses expect different signature
            if loss_type in ["boundary_affinity", "boundary_affinity_focal_lsds"]:
                # Boundary-weighted loss needs instance segmentation
                return criterion(logits, targets, instance_seg=instance_seg, mask=masks)

            else:
                # Standard affinity loss or affinity_lsds loss
                return criterion(logits, targets, mask=masks)
        else:
            return criterion(logits, targets)

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

            (
                train_volumes,
                train_segmentations,
                train_targets,
                train_masks,
                train_context,
            ) = data_loader_3d.sample_training_batch(volumes_per_batch)
            # TIMING: Start DINOv3 feature extraction
            dinov3_start = time.time()

            # Extract DINOv3 features (separate for proper context fusion)
            if enable_detailed_timing:
                train_features, train_context_features, detailed_timing = (
                    data_loader_3d.extract_multi_scale_dinov3_features_3d(
                        train_volumes,
                        train_context,
                        epoch,
                        batch_idx,
                        enable_detailed_timing=enable_detailed_timing,
                    )
                )
            else:
                train_features, train_context_features = (
                    data_loader_3d.extract_multi_scale_dinov3_features_3d(
                        train_volumes,
                        train_context,
                        epoch,
                        batch_idx,
                        enable_detailed_timing=enable_detailed_timing,
                    )
                )
                detailed_timing = {}

            # TIMING: End DINOv3 feature extraction
            dinov3_end = time.time()
            dinov3_time = dinov3_end - dinov3_start

            # Handle both label and affinity formats
            if data_loader_3d.output_type in ["affinities", "affinities_lsds"]:
                # Affinities/LSDs are float32: (batch, num_offsets, D, H, W) or (batch, 10+num_offsets, D, H, W)
                train_labels = torch.tensor(train_targets, dtype=torch.float32).to(
                    device
                )
            else:
                # Class labels are long: (batch, D, H, W)
                train_labels = torch.tensor(train_targets, dtype=torch.long).to(device)

            train_masks_tensor = torch.tensor(train_masks, dtype=torch.float32).to(
                device
            )

            # Debug: Print shapes for first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"  DEBUG - Tensor shapes:")
                print(f"    train_features: {train_features.shape}")
                if train_context_features is not None:
                    print(f"    train_context_features: {train_context_features.shape}")
                print(f"    train_labels: {train_labels.shape}")
                print(f"    train_masks_tensor: {train_masks_tensor.shape}")
                print(
                    f"    Mask min/max: {train_masks_tensor.min().item():.3f}/{train_masks_tensor.max().item():.3f}"
                )
                print(
                    f"    Mask fraction valid: {train_masks_tensor.mean().item():.3f}"
                )

            # Check if masks are actually being used (not all 1s)
            use_masks = not torch.all(train_masks_tensor == 1.0)

            # TIMING: Start training step (UNet forward/backward)
            training_start = time.time()

            # Training step
            optimizer.zero_grad()

            if use_mixed_precision:
                # Fix 1: Update autocast usage
                with autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu",
                    enabled=False,
                ):  # Disable autocast for metric calculations
                    # Pass raw and context features separately for proper fusion
                    logits = unet3d(
                        train_features, context_features=train_context_features
                    )

                    # Debug: Print logits shape for first batch of first epoch
                    if epoch == 0 and batch_idx == 0:
                        print(f"    logits: {logits.shape}")
                        print(
                            f"    logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]"
                        )
                        # Check if logits sum to 1 (they shouldn't for raw logits)
                        logits_sum = logits.sum(dim=1)  # Sum across class dimension
                        print(
                            f"    logits sum across classes: [{logits_sum.min().item():.3f}, {logits_sum.max().item():.3f}]"
                        )
                        # Show softmax probabilities for comparison
                        probs = F.softmax(logits, dim=1)
                        probs_sum = probs.sum(dim=1)
                        print(
                            f"    probabilities (after softmax) sum: [{probs_sum.min().item():.3f}, {probs_sum.max().item():.3f}]"
                        )
                        print(
                            f"    probabilities range: [{probs.min().item():.3f}, {probs.max().item():.3f}]"
                        )

                    if use_masks:
                        # Apply masks to loss calculation
                        # For CE-based losses, we can use per-pixel masking
                        if loss_type in ["ce", "weighted_ce", "focal"]:
                            per_pixel_loss = F.cross_entropy(
                                logits,
                                train_labels,
                                reduction="none",
                                weight=loss_class_weights,
                            )
                            if loss_type == "focal":
                                # Apply focal term
                                p_t = torch.exp(-per_pixel_loss)
                                focal_weight = (1 - p_t) ** focal_gamma
                                per_pixel_loss = focal_weight * per_pixel_loss

                            masked_loss = per_pixel_loss * train_masks_tensor
                            loss = masked_loss.sum() / train_masks_tensor.sum()
                        elif loss_type in [
                            "affinity",
                            "boundary_affinity",
                            "boundary_affinity_focal_lsds",
                        ]:
                            # Affinity losses handle masking internally
                            # For boundary_affinity, pass the instance segmentation (train_gt_volumes)
                            loss = compute_loss(
                                criterion,
                                logits,
                                train_labels,
                                masks=train_masks_tensor,
                                instance_seg=train_segmentations,  # Original instance seg before affinity conversion
                            )
                        else:
                            # For Dice-based losses, compute on full volume and weight the loss
                            # This is an approximation; ideally Dice should only use masked regions
                            loss_full = compute_loss(
                                criterion,
                                logits,
                                train_labels,
                                masks=train_masks_tensor,
                                instance_seg=(
                                    train_segmentations
                                    if loss_type
                                    in [
                                        "boundary_affinity",
                                        "boundary_affinity_focal_lsds",
                                    ]
                                    else None
                                ),
                            )
                            # Weight by fraction of valid voxels
                            loss = loss_full

                        # Debug: Print loss calculation details for first batch of first epoch
                        if epoch == 0 and batch_idx == 0:
                            print(f"    final loss: {loss.item():.4f}")
                    else:
                        # Use standard loss when no masking needed
                        loss = compute_loss(
                            criterion,
                            logits,
                            train_labels,
                            masks=None,
                            instance_seg=(
                                train_segmentations
                                if loss_type
                                in ["boundary_affinity", "boundary_affinity_focal_lsds"]
                                else None
                            ),
                        )

                        # Debug: Print unmasked loss for first batch of first epoch
                        if epoch == 0 and batch_idx == 0:
                            print(f"    unmasked loss: {loss.item():.4f}")

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet3d.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            # else:
            #     # Pass raw and context features separately for proper fusion
            #     logits = unet3d(train_features, context_features=train_context_features)
            #     if use_masks:
            #         # Apply masks to loss calculation
            #         if loss_type in ["ce", "weighted_ce", "focal"]:
            #             per_pixel_loss = F.cross_entropy(
            #                 logits,
            #                 train_labels,
            #                 reduction="none",
            #                 weight=loss_class_weights,
            #             )
            #             if loss_type == "focal":
            #                 p_t = torch.exp(-per_pixel_loss)
            #                 focal_weight = (1 - p_t) ** focal_gamma
            #                 per_pixel_loss = focal_weight * per_pixel_loss

            #             masked_loss = per_pixel_loss * train_masks_tensor
            #             loss = masked_loss.sum() / train_masks_tensor.sum()
            #         else:
            #             loss_full = compute_loss(criterion, logits, train_labels)
            #             loss = loss_full
            #     else:
            #         # Use standard loss when no masking needed
            #         loss = compute_loss(criterion, logits, train_labels)
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(unet3d.parameters(), max_norm=1.0)
            #     optimizer.step()

            # Clear GPU cache to prevent memory accumulation
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # TIMING: End training step
            training_end = time.time()
            training_time = training_end - training_start

            # Track metrics (conditional on mask usage)
            with autocast(
                device_type="cuda" if device.type == "cuda" else "cpu", enabled=False
            ):  # Disable autocast for metric calculations

                # Handle different output types
                if data_loader_3d.output_type == "affinities":
                    # For affinities: apply sigmoid and threshold at 0.5
                    predictions = (torch.sigmoid(logits.float()) > 0.5).float()
                    # Compare binary predictions with binary targets
                    if use_masks:
                        # Expand mask to match affinity dimensions: (batch, D, H, W) -> (batch, num_offsets, D, H, W)
                        train_mask_expanded = (
                            train_masks_tensor.unsqueeze(1)
                            .expand_as(predictions)
                            .bool()
                        )
                        correct = (
                            ((predictions == train_labels) & train_mask_expanded)
                            .sum()
                            .item()
                        )
                        total = train_mask_expanded.sum().item()
                    else:
                        correct = (predictions == train_labels).sum().item()
                        total = train_labels.numel()
                elif data_loader_3d.output_type == "affinities_lsds":
                    # For affinities+LSDs: don't compute accuracy (use loss as metric instead)
                    # This is a regression+binary task, so accuracy doesn't make sense
                    correct = 0
                    total = 1  # Avoid division by zero, accuracy will be 0
                else:
                    # For class labels: use argmax
                    predictions = torch.argmax(logits.float(), dim=1)
                    if use_masks:
                        # Only for masked regions
                        train_mask_bool = train_masks_tensor.bool()
                        correct = (
                            ((predictions == train_labels) & train_mask_bool)
                            .sum()
                            .item()
                        )
                        total = train_masks_tensor.sum().item()
                    else:
                        # All regions when no masking
                        correct = (predictions == train_labels).sum().item()
                        total = train_labels.numel()

            epoch_train_loss += loss.item() * total
            epoch_train_correct += correct
            epoch_train_total += total

            # Track per-class accuracy (conditional on mask usage)
            # Skip per-class metrics for affinity and affinity_lsds outputs since they're not classification
            if data_loader_3d.output_type not in ["affinities", "affinities_lsds"]:
                for class_id in range(num_classes):
                    if use_masks:
                        # Only in masked regions
                        train_mask_bool = train_masks_tensor.bool()
                        class_mask = (train_labels == class_id) & train_mask_bool
                    else:
                        # All regions when no masking
                        class_mask = train_labels == class_id

                    if class_mask.sum() > 0:
                        class_correct = (
                            (predictions[class_mask] == class_id).sum().item()
                        )
                        class_total = class_mask.sum().item()
                        epoch_train_class_correct[class_id] += class_correct
                        epoch_train_class_total[class_id] += class_total

            # TIMING: Print timing breakdown for this batch
            total_batch_time = dinov3_time + training_time
            dinov3_pct = (dinov3_time / total_batch_time) * 100
            training_pct = (training_time / total_batch_time) * 100

            # Print timing every 5 batches or on first batch
            if (batch_idx % 5 == 0 or batch_idx == 0) and enable_detailed_timing:
                tqdm.write(f"  Batch {batch_idx+1} Timing:")
                tqdm.write(
                    f"    DINOv3 extraction: {dinov3_time:.3f}s ({dinov3_pct:.1f}%)"
                )
                tqdm.write(
                    f"    UNet training:      {training_time:.3f}s ({training_pct:.1f}%)"
                )
                tqdm.write(f"    Total batch time:   {total_batch_time:.3f}s")

                # Print detailed DINOv3 timing breakdown if available
                if "detailed_timing" in locals() and detailed_timing:
                    tqdm.write(f"    DINOv3 Detailed Breakdown:")
                    tqdm.write(
                        f"      Slice upsampling:    {detailed_timing.get('total_upsampling_time', 0):.3f}s ({detailed_timing.get('upsampling_percentage', 0):.1f}%)"
                    )
                    tqdm.write(
                        f"      Batch stacking:      {detailed_timing.get('total_stacking_time', 0):.3f}s ({detailed_timing.get('stacking_percentage', 0):.1f}%)"
                    )
                    tqdm.write(
                        f"      DINOv3 inference:    {detailed_timing.get('total_dinov3_inference_time', 0):.3f}s ({detailed_timing.get('dinov3_inference_percentage', 0):.1f}%)"
                    )
                    tqdm.write(
                        f"      Feature extraction:  {detailed_timing.get('total_feature_extraction_time', 0):.3f}s ({detailed_timing.get('feature_extraction_percentage', 0):.1f}%)"
                    )
                    tqdm.write(
                        f"      Feature downsampling:{detailed_timing.get('total_downsampling_time', 0):.3f}s ({detailed_timing.get('downsampling_percentage', 0):.1f}%)"
                    )
                    tqdm.write(
                        f"      Total slices processed: {detailed_timing.get('total_slices', 0)} in {detailed_timing.get('total_batches', 0)} batches"
                    )

            # Update progress bar with current metrics
            current_loss = epoch_train_loss / max(epoch_train_total, 1)
            current_acc = epoch_train_correct / max(epoch_train_total, 1)
            batch_pbar.set_postfix(
                {"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.4f}"}
            )

            # Clear processed volumes from GPU memory after each batch
            del train_features, train_labels, logits, train_masks_tensor
            if "predictions" in locals():
                del predictions
            if train_context_features is not None:
                del train_context_features
            if "train_volumes" in locals():
                del train_volumes, train_targets, train_masks
            if "train_context" in locals() and train_context is not None:
                del train_context
            if device.type == "cuda":
                torch.cuda.empty_cache()

        batch_pbar.close()
        training_time = time.time() - epoch_start_time

        # Clear all training-related GPU memory before validation
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
        for train_or_eval in ["eval"]:
            print(
                f"\n****************Validation phase: {train_or_eval.upper()}****************"
            )
            if train_or_eval == "train":
                unet3d.train()

                def freeze_batch_layers(m):
                    if hasattr(m, "track_running_stats") and hasattr(m, "running_mean"):
                        m.eval()  # switch BN/BRN to eval behavior
                        m.track_running_stats = False  # prevent updates

                unet3d.apply(freeze_batch_layers)
            else:
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
                val_class_total_pred = np.zeros(
                    num_classes
                )  # For precision calculation

                # Initialize validation timing accumulators
                total_val_dinov3_time = 0.0
                total_val_inference_time = 0.0

                # Process each validation volume independently
                for vol_idx in range(len(val_volumes)):
                    # TIMING: Start DINOv3 validation feature extraction
                    val_dinov3_start = time.time()

                    # Extract features for single volume (keeps GPU memory minimal)
                    single_vol = np.array(
                        [val_volumes[vol_idx]]
                    )  # Convert to numpy array with batch dimension

                    # Get context volume if available
                    single_context = None
                    if val_context is not None:
                        single_context = np.array([val_context[vol_idx]])

                    # Extract features with detailed timing for first validation volume
                    if (
                        vol_idx == 0
                    ):  # Only get detailed timing for first volume to avoid spam
                        result = data_loader_3d.extract_multi_scale_dinov3_features_3d(
                            single_vol,
                            single_context,
                            epoch=epoch,
                            enable_detailed_timing=enable_detailed_timing,
                        )
                        if enable_detailed_timing:
                            vol_features, vol_context_features, val_detailed_timing = (
                                result
                            )
                        else:
                            vol_features, vol_context_features = result
                            val_detailed_timing = None
                    else:
                        vol_features, vol_context_features = (
                            data_loader_3d.extract_multi_scale_dinov3_features_3d(
                                single_vol, single_context, epoch=epoch
                            )
                        )

                    # TIMING: End DINOv3 validation feature extraction
                    val_dinov3_end = time.time()
                    val_dinov3_time = val_dinov3_end - val_dinov3_start

                    # Get ground truth and mask for this volume and move to GPU
                    # Handle both label and affinity formats
                    if data_loader_3d.output_type in ["affinities", "affinities_lsds"]:
                        # Affinities/LSDs are float32: (1, num_channels, D, H, W)
                        current_val_targets = (
                            torch.tensor(val_targets[vol_idx], dtype=torch.float32)
                            .unsqueeze(0)
                            .to(device)
                        )
                    else:
                        # Class labels are long: (1, D, H, W)
                        current_val_targets = (
                            torch.tensor(val_targets[vol_idx], dtype=torch.long)
                            .unsqueeze(0)
                            .to(device)
                        )

                    current_val_mask = (
                        torch.tensor(val_masks[vol_idx], dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )

                    # Check if masks are actually being used (not all 1s)
                    use_vol_masks = not torch.all(current_val_mask == 1.0)

                    # TIMING: Start validation inference
                    val_inference_start = time.time()

                    # Run inference on single volume
                    if use_mixed_precision:
                        with autocast(
                            device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=False,
                        ):
                            # Pass raw and context features separately for proper fusion
                            vol_logits = unet3d(
                                vol_features, context_features=vol_context_features
                            )
                            if use_vol_masks:
                                # Apply masks to validation loss calculation
                                if loss_type in ["ce", "weighted_ce", "focal"]:
                                    vol_per_pixel_loss = F.cross_entropy(
                                        vol_logits,
                                        current_val_targets,
                                        reduction="none",
                                        weight=loss_class_weights,
                                    )
                                    if loss_type == "focal":
                                        p_t = torch.exp(-vol_per_pixel_loss)
                                        focal_weight = (1 - p_t) ** focal_gamma
                                        vol_per_pixel_loss = (
                                            focal_weight * vol_per_pixel_loss
                                        )

                                    vol_masked_loss = (
                                        vol_per_pixel_loss * current_val_mask
                                    )
                                    vol_loss = (
                                        vol_masked_loss.sum() / current_val_mask.sum()
                                    ).item()
                                elif loss_type in [
                                    "affinity",
                                    "boundary_affinity",
                                    "boundary_affinity_focal_lsds",
                                ]:
                                    # Affinity losses handle masking internally
                                    vol_loss = compute_loss(
                                        criterion,
                                        vol_logits,
                                        current_val_targets,
                                        masks=current_val_mask,
                                        instance_seg=val_segmentations[
                                            vol_idx
                                        ],  # Original instance seg
                                    ).item()
                                else:
                                    vol_loss = compute_loss(
                                        criterion,
                                        vol_logits,
                                        current_val_targets,
                                        masks=current_val_mask,
                                        instance_seg=(
                                            val_segmentations[vol_idx]
                                            if loss_type
                                            in [
                                                "boundary_affinity",
                                                "boundary_affinity_focal_lsds",
                                            ]
                                            else None
                                        ),
                                    ).item()
                            else:
                                # Use standard loss when no masking needed
                                vol_loss = compute_loss(
                                    criterion,
                                    vol_logits,
                                    current_val_targets,
                                    masks=None,
                                    instance_seg=(
                                        train_segmentations
                                        if loss_type
                                        in [
                                            "boundary_affinity",
                                            "boundary_affinity_focal_lsds",
                                        ]
                                        else None
                                    ),
                                ).item()
                    # else:
                    #     # Pass raw and context features separately for proper fusion
                    #     vol_logits = unet3d(
                    #         vol_features, context_features=vol_context_features
                    #     )
                    #     if use_vol_masks:
                    #         # Apply masks to validation loss calculation
                    #         if loss_type in ["ce", "weighted_ce", "focal"]:
                    #             vol_per_pixel_loss = F.cross_entropy(
                    #                 vol_logits,
                    #                 vol_gt,
                    #                 reduction="none",
                    #                 weight=loss_class_weights,
                    #             )
                    #             if loss_type == "focal":
                    #                 p_t = torch.exp(-vol_per_pixel_loss)
                    #                 focal_weight = (1 - p_t) ** focal_gamma
                    #                 vol_per_pixel_loss = (
                    #                     focal_weight * vol_per_pixel_loss
                    #                 )

                    #             vol_masked_loss = vol_per_pixel_loss * vol_mask
                    #             vol_loss = (
                    #                 vol_masked_loss.sum() / vol_mask.sum()
                    #             ).item()
                    #         elif loss_type in ["affinity", "boundary_affinity"]:
                    #             # Affinity losses handle masking internally
                    #             vol_loss = compute_loss(
                    #                 criterion,
                    #                 vol_logits,
                    #                 vol_gt,
                    #                 masks=vol_mask,
                    #                 instance_seg=val_gt_volume,  # Original instance seg
                    #             ).item()
                    #         else:
                    #             vol_loss = compute_loss(
                    #                 criterion,
                    #                 vol_logits,
                    #                 vol_gt,
                    #                 masks=vol_mask,
                    #                 instance_seg=(
                    #                     val_gt_volume
                    #                     if loss_type == "boundary_affinity"
                    #                     else None
                    #                 ),
                    #             ).item()
                    #     else:
                    #         # Use standard loss when no masking needed
                    #         vol_loss = compute_loss(
                    #             criterion,
                    #             vol_logits,
                    #             vol_gt,
                    #             masks=None,
                    #             instance_seg=(
                    #                 val_gt_volume
                    #                 if loss_type == "boundary_affinity"
                    #                 else None
                    #             ),
                    #         ).item()

                    # TIMING: End validation inference
                    val_inference_end = time.time()
                    val_inference_time = val_inference_end - val_inference_start

                    # Accumulate validation timing
                    total_val_dinov3_time += val_dinov3_time
                    total_val_inference_time += val_inference_time

                    # Convert to predictions
                    if data_loader_3d.output_type == "affinities":
                        # For affinities: apply sigmoid and threshold at 0.5
                        vol_predictions = (
                            torch.sigmoid(vol_logits.float()) > 0.5
                        ).float()
                    elif data_loader_3d.output_type == "affinities_lsds":
                        # For affinities+LSDs: don't compute predictions (use loss as metric)
                        vol_predictions = None
                    else:
                        # For class labels: use argmax
                        vol_predictions = torch.argmax(vol_logits.float(), dim=1)

                    # Accumulate basic metrics (conditional on mask usage)
                    if data_loader_3d.output_type == "affinities":
                        # For affinities, expand mask to match affinity dimensions
                        if use_vol_masks:
                            vol_mask_expanded = (
                                current_val_mask.unsqueeze(1)
                                .expand_as(vol_predictions)
                                .bool()
                            )
                            vol_correct = (
                                ((vol_predictions == vol_gt) & vol_mask_expanded)
                                .sum()
                                .item()
                            )
                            vol_valid_voxels = vol_mask_expanded.sum().item()
                        else:
                            vol_correct = (vol_predictions == vol_gt).sum().item()
                            vol_valid_voxels = vol_gt.numel()
                    elif data_loader_3d.output_type == "affinities_lsds":
                        # For affinities+LSDs: don't compute accuracy (use loss only)
                        vol_correct = 0
                        vol_valid_voxels = 1  # Avoid division by zero
                    else:
                        # For class labels
                        if use_vol_masks:
                            # Only for masked regions
                            vol_mask_bool = current_val_mask.bool()
                            vol_correct = (
                                ((vol_predictions == vol_gt) & vol_mask_bool)
                                .sum()
                                .item()
                            )
                            vol_valid_voxels = current_val_mask.sum().item()
                        else:
                            # All regions when no masking
                            vol_correct = (vol_predictions == vol_gt).sum().item()
                            vol_valid_voxels = vol_gt.numel()

                    val_total_loss += vol_loss * vol_valid_voxels
                    val_total_correct += vol_correct
                    val_total_voxels += vol_valid_voxels

                    # Accumulate per-class metrics (convert to CPU for efficiency, conditional on mask usage)
                    # Skip per-class metrics for affinity and affinity_lsds outputs
                    if data_loader_3d.output_type not in [
                        "affinities",
                        "affinities_lsds",
                    ]:
                        vol_gt_cpu = vol_gt.cpu().numpy()
                        vol_pred_cpu = vol_predictions.cpu().numpy()

                        if use_vol_masks:
                            vol_mask_cpu = current_val_mask.cpu().numpy().astype(bool)
                        else:
                            # Create a full mask when no masking is needed
                            vol_mask_cpu = np.ones_like(vol_gt_cpu, dtype=bool)

                        for class_id in range(num_classes):
                            # For confusion matrix, we need to consider the valid region
                            valid_gt_mask = (vol_gt_cpu == class_id) & vol_mask_cpu
                            valid_not_gt_mask = (vol_gt_cpu != class_id) & vol_mask_cpu
                            valid_pred_mask = (vol_pred_cpu == class_id) & vol_mask_cpu

                            # Confusion matrix components (only in valid regions)
                            true_positives = np.sum(valid_gt_mask & valid_pred_mask)
                            false_positives = np.sum(
                                valid_not_gt_mask & valid_pred_mask
                            )
                            false_negatives = np.sum(valid_gt_mask & ~valid_pred_mask)

                            # Accumulate
                            val_class_true_positives[class_id] += true_positives
                            val_class_false_positives[class_id] += false_positives
                            val_class_false_negatives[class_id] += false_negatives
                            val_class_total_gt[class_id] += np.sum(valid_gt_mask)
                            val_class_total_pred[class_id] += np.sum(valid_pred_mask)

                    # Immediately free GPU memory for this volume
                    del vol_features, vol_logits, current_val_mask
                    if "vol_predictions" in locals():
                        del vol_predictions
                    if vol_context_features is not None:
                        del vol_context_features
                    if "single_vol" in locals():
                        del single_vol
                    if "single_context" in locals() and single_context is not None:
                        del single_context

                    # Clear GPU cache after each volume
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                # Calculate final validation metrics from accumulated values
                val_loss = val_total_loss / val_total_voxels
                val_acc = val_total_correct / val_total_voxels

                # TIMING: Print validation timing summary
                total_val_time = total_val_dinov3_time + total_val_inference_time
                val_dinov3_pct = (
                    (total_val_dinov3_time / total_val_time) * 100
                    if total_val_time > 0
                    else 0
                )
                val_inference_pct = (
                    (total_val_inference_time / total_val_time) * 100
                    if total_val_time > 0
                    else 0
                )
                avg_val_dinov3_time = (
                    total_val_dinov3_time / len(val_volumes)
                    if len(val_volumes) > 0
                    else 0
                )
                avg_val_inference_time = (
                    total_val_inference_time / len(val_volumes)
                    if len(val_volumes) > 0
                    else 0
                )

                print(f"\n  Validation Timing Summary ({len(val_volumes)} volumes):")
                print(
                    f"    Total DINOv3 extraction: {total_val_dinov3_time:.3f}s ({val_dinov3_pct:.1f}%) - Avg: {avg_val_dinov3_time:.3f}s/vol"
                )
                print(
                    f"    Total UNet inference:     {total_val_inference_time:.3f}s ({val_inference_pct:.1f}%) - Avg: {avg_val_inference_time:.3f}s/vol"
                )
                print(f"    Total validation time:    {total_val_time:.3f}s")

                # Print detailed DINOv3 breakdown if available (from first validation volume)
                if "val_detailed_timing" in locals() and val_detailed_timing:
                    print(f"    DINOv3 Detailed Breakdown (first volume):")
                    print(
                        f"      Slice upsampling:    {val_detailed_timing.get('total_upsampling_time', 0):.3f}s ({val_detailed_timing.get('upsampling_percentage', 0):.1f}%)"
                    )
                    print(
                        f"      Batch stacking:      {val_detailed_timing.get('total_stacking_time', 0):.3f}s ({val_detailed_timing.get('stacking_percentage', 0):.1f}%)"
                    )
                    print(
                        f"      DINOv3 inference:    {val_detailed_timing.get('total_dinov3_inference_time', 0):.3f}s ({val_detailed_timing.get('dinov3_inference_percentage', 0):.1f}%)"
                    )
                    print(
                        f"      Feature extraction:  {val_detailed_timing.get('total_feature_extraction_time', 0):.3f}s ({val_detailed_timing.get('feature_extraction_percentage', 0):.1f}%)"
                    )
                    print(
                        f"      Feature downsampling:{val_detailed_timing.get('total_downsampling_time', 0):.3f}s ({val_detailed_timing.get('downsampling_percentage', 0):.1f}%)"
                    )
                    print(
                        f"      Total slices processed: {val_detailed_timing.get('total_slices', 0)} in {val_detailed_timing.get('total_batches', 0)} batches"
                    )

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

            # Clear all validation-related GPU memory after validation phase
            if device.type == "cuda":
                torch.cuda.empty_cache()

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
            # For affinities/LSDs, use validation loss instead of IoU or accuracy
            if data_loader_3d.output_type in ["affinities", "affinities_lsds"]:
                # For affinity/LSDS training, use validation LOSS as the metric (lower is better)
                current_metric = (
                    -val_loss
                )  # Negative so we can still use "higher is better" logic
                metric_name = "Val Loss"
                current_mean_iou = None  # Not applicable for affinities
            elif data_loader_3d.output_type == "affinities":
                # For affinity training, use validation accuracy as the metric
                current_metric = val_acc
                metric_name = "Val Accuracy"
                current_mean_iou = None  # Not applicable for affinities
            else:
                # For segmentation, use mean IoU
                current_mean_iou = np.mean(val_class_iou)
                current_metric = current_mean_iou
                metric_name = "Mean IoU"

            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(current_metric)
            new_lr = optimizer.param_groups[0]["lr"]

            # Check if this is the best metric so far.
            # For affinities/affinities_lsds we prefer lower validation LOSS (val_loss).
            is_best_model = False
            if train_or_eval == "eval":
                if data_loader_3d.output_type in ["affinities", "affinities_lsds"]:
                    # Lower val_loss is better
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        best_metric = -best_val_loss
                        epochs_without_improvement = 0
                        is_best_model = True
                    else:
                        epochs_without_improvement += 1
                    # Keep current_metric consistent (scheduler expects higher-is-better)
                    current_metric = -val_loss
                else:
                    if current_metric > best_metric + min_delta:
                        best_metric = (
                            current_metric  # Store best metric (IoU or accuracy)
                        )
                        epochs_without_improvement = 0
                        is_best_model = True
                    else:
                        epochs_without_improvement += 1

            # Print progress every epoch with timing
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{epochs} - Time: {total_epoch_time:.1f}s (Train: {training_time:.1f}s, Val: {val_time:.1f}s)"
            )

            # Different output formats based on output type
            if data_loader_3d.output_type in ["affinities", "affinities_lsds"]:
                # For affinities/LSDs: only show loss (accuracy not meaningful)
                print(f"  Train: Loss={train_loss:.4f}")
                print(f"  Val:   Loss={val_loss:.4f}")
                print(f"  Best Val Loss: {best_val_loss:.4f}, LR: {current_lr:.6f}")
            else:
                # For segmentation: show loss, accuracy, and IoU
                print(
                    f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} (from {epoch_train_total} voxels)"
                )
                print(
                    f"  Val:   Loss={val_loss:.4f}, IoU={current_metric:.4f}, Acc={val_acc:.4f}"
                )
                print(f"  Best Val Acc: {best_val_acc:.4f}, LR: {current_lr:.6f}")

        # Print comprehensive per-class metrics in a nice table (only for segmentation)
        if data_loader_3d.output_type not in ["affinities", "affinities_lsds"]:
            train_class_percentages = [
                count / epoch_train_total * 100 for count in epoch_train_class_total
            ]
            val_class_counts = (
                val_class_total_gt  # Use the counts we already calculated
            )
            val_class_percentages = [
                count / val_total_voxels * 100 for count in val_class_counts
            ]

            print_class_metrics_table(
                class_names=class_names,
                train_recall=train_class_acc,
                train_dist=train_class_percentages,
                val_recall=val_class_acc,
                val_precision=val_class_precision,
                val_f1=val_class_f1,
                val_iou=val_class_iou,
                val_dist=val_class_percentages,
            )

            print(f"  Mean IoU: {current_metric:.4f}")

        # For user-facing display, show the human-friendly best value:
        # - For affinity/LSDS: show positive best_val_loss
        # - Otherwise show best_metric (e.g., IoU or accuracy)
        if data_loader_3d.output_type in ["affinities", "affinities_lsds"]:
            display_best = best_val_loss
        else:
            display_best = best_metric

        print(f"  Best {metric_name}: {display_best:.4f}, LR: {current_lr:.6f}")
        print(f"  Epochs without improvement: {epochs_without_improvement}")
        if new_lr < old_lr:
            print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Save checkpoints
        if save_checkpoints:
            # For backward compatibility with checkpoint system, alias the best metric
            best_val_acc = best_metric
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
                    "class_names": class_names,  # Add class names
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "patience": patience,
                    "min_delta": min_delta,
                    "base_channels": base_channels,
                    "volumes_per_batch": volumes_per_batch,
                    "batches_per_epoch": batches_per_epoch,
                    "epochs": epochs,  # Add total epochs
                    "model_type": "dinov3_unet3d",
                    "model_id": model_id,
                    "input_channels": current_output_channels,
                    "use_class_weighting": use_class_weighting,
                    # Loss function parameters
                    "loss_type": loss_type,
                    "focal_gamma": focal_gamma,
                    "focal_weight": focal_weight,
                    "dice_weight": dice_weight,
                    "dice_smooth": dice_smooth,
                    "tversky_alpha": tversky_alpha,
                    "tversky_beta": tversky_beta,
                    # Volume and feature sizes
                    "target_volume_size": data_loader_3d.target_volume_size,
                    "dinov3_slice_size": data_loader_3d.dinov3_slice_size,
                    "image_size": data_loader_3d.dinov3_slice_size,  # For DINOv3 initialization
                    "seed": data_loader_3d.seed,  # Add seed
                    "use_mixed_precision": use_mixed_precision,  # Add memory efficiency params
                    "use_half_precision": use_half_precision,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                    "memory_efficient_mode": memory_efficient_mode,
                    "learn_upsampling": learn_upsampling,  # Add upsampling mode
                    "use_orthogonal_planes": data_loader_3d.use_orthogonal_planes,  # Add orthogonal planes setting
                    "train_volume_pool_size": data_loader_3d.train_volume_pool_size,
                    "val_volume_pool_size": data_loader_3d.val_volume_pool_size,
                    "checkpoint_every_n_epochs": checkpoint_every_n_epochs,  # Add checkpoint frequency
                    "enable_detailed_timing": enable_detailed_timing,  # Add timing control
                    # Data resolution parameters
                    "min_resolution_for_raw": min_resolution_for_raw,  # Add raw data resolution
                    "base_resolution": base_resolution,  # Add ground truth resolution
                    # Context fusion parameters
                    "use_context_fusion": use_context_fusion,  # Add context fusion flag
                    "context_scale": getattr(
                        data_loader_3d, "context_scale", None
                    ),  # Add context resolution if available
                    # Additional data loader parameters
                    "dinov3_stride": getattr(
                        data_loader_3d, "dinov3_stride", None
                    ),  # Add DINOv3 stride if available
                    "verbose": getattr(
                        data_loader_3d, "verbose", True
                    ),  # Add verbose setting
                    # Affinity parameters
                    "output_type": getattr(
                        data_loader_3d, "output_type", "labels"
                    ),  # Add output type
                    "affinity_offsets": getattr(
                        data_loader_3d, "affinity_offsets", None
                    ),  # Add affinity offsets
                    "lsds_sigma": getattr(
                        data_loader_3d, "lsds_sigma", 20.0
                    ),  # Add LSDS sigma parameter
                    "compute_lsds": getattr(data_loader_3d, "output_type", "labels")
                    == "affinities_lsds",
                },
                "model_config": {
                    "num_classes": num_classes,
                    "base_channels": base_channels,
                    "input_size": data_loader_3d.target_volume_size,  # 3D volume size
                    "input_channels": current_output_channels,
                    "model_id": model_id,
                    "model_type": "dinov3_unet3d",
                    "dinov3_slice_size": data_loader_3d.dinov3_slice_size,
                    "learn_upsampling": learn_upsampling,  # Add upsampling mode to model config
                    "use_orthogonal_planes": data_loader_3d.use_orthogonal_planes,  # Add orthogonal planes
                    "use_half_precision": use_half_precision,  # Add precision settings
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                    # Data resolution parameters for model recreation
                    "min_resolution_for_raw": min_resolution_for_raw,
                    "base_resolution": base_resolution,
                    # Context fusion parameters
                    "use_context_fusion": use_context_fusion,  # Add context fusion flag
                    "context_scale": getattr(
                        data_loader_3d, "context_scale", None
                    ),  # Add context resolution
                    # Affinity parameters
                    "output_type": getattr(
                        data_loader_3d, "output_type", "labels"
                    ),  # Add output type
                    "affinity_offsets": getattr(
                        data_loader_3d, "affinity_offsets", None
                    ),  # Add affinity offsets
                    "lsds_sigma": getattr(
                        data_loader_3d, "lsds_sigma", 20.0
                    ),  # Add LSDS sigma parameter
                    "compute_lsds": getattr(data_loader_3d, "output_type", "labels")
                    == "affinities_lsds",
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

            # Save full model checkpoint if this is the best performance
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

                if data_loader_3d.output_type in ["affinities", "affinities_lsds"]:
                    print(f"  *** NEW BEST 3D MODEL SAVED: Loss={val_loss:.4f} ***")
                else:
                    print(
                        f"  *** NEW BEST 3D MODEL SAVED: IoU={current_metric:.4f} (Acc={val_acc:.4f}) ***"
                    )

            # Also save model checkpoint every N epochs if specified
            elif (
                checkpoint_every_n_epochs is not None
                and (epoch + 1) % checkpoint_every_n_epochs == 0
            ):
                checkpoint_data = {
                    **stats_data,
                    "unet3d_state_dict": unet3d.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }

                epoch_path = os.path.join(
                    checkpoint_dir, f"checkpoint_epoch_{epoch+1:04d}.pkl"
                )
                with open(epoch_path, "wb") as f:
                    pickle.dump(checkpoint_data, f)

                if data_loader_3d.output_type == "affinities":
                    print(
                        f"  *** PERIODIC CHECKPOINT SAVED: Epoch {epoch+1} (Acc={current_metric:.4f}) ***"
                    )
                else:
                    print(
                        f"  *** PERIODIC CHECKPOINT SAVED: Epoch {epoch+1} (IoU={current_metric:.4f}) ***"
                    )

            print(f"  Stats saved: {os.path.basename(stats_path)}")

        print()

        # Final GPU cleanup at end of epoch to ensure fresh start for next epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
        "best_metric": best_metric,  # Best validation metric (IoU or accuracy or -loss)
        "best_mean_iou": best_metric,  # Backward compatibility
        "best_val_acc": best_metric,  # Backward compatibility (note: for affinities_lsds this is -loss)
        "best_val_loss": best_val_loss,  # Best validation loss for affinities/LSDs
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
    gt_masks=None,  # NEW PARAMETER for GT extension masks
    context_data=None,  # NEW PARAMETER for context volumes at lower resolution
    context_scale=None,  # NEW PARAMETER for context resolution (e.g., 8 for 8nm)
    train_volume_pool_size=20,
    val_volume_pool_size=5,
    num_classes=None,
    class_names=None,  # NEW PARAMETER for class names (e.g., ['background', 'nuc', 'mito', 'er'])
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
    checkpoint_every_n_epochs=None,  # Save model checkpoint every N epochs (in addition to best)
    use_class_weighting=True,
    # NEW MEMORY EFFICIENCY PARAMETERS
    use_mixed_precision=True,
    use_half_precision=False,
    use_gradient_checkpointing=False,
    memory_efficient_mode="auto",  # "auto", "aggressive", "conservative"
    learn_upsampling=False,  # NEW PARAMETER
    dinov3_stride=None,  # NEW PARAMETER for sliding window inference
    use_orthogonal_planes=False,  # NEW PARAMETER for orthogonal plane processing
    enable_detailed_timing=True,  # NEW PARAMETER for detailed timing
    verbose=True,  # NEW PARAMETER to control verbose output
    # DATA RESOLUTION PARAMETERS
    min_resolution_for_raw=None,  # NEW PARAMETER for raw data resolution
    base_resolution=None,  # NEW PARAMETER for ground truth resolution
    # LOSS FUNCTION PARAMETERS
    loss_type="weighted_ce",  # 'ce', 'weighted_ce', 'focal', 'dice', 'focal_dice', 'tversky', 'affinity', 'boundary_affinity'
    focal_gamma=2.0,  # Focusing parameter for Focal Loss
    focal_weight=0.5,  # Weight for focal component in combined loss
    dice_weight=0.5,  # Weight for dice component in combined loss
    dice_smooth=1.0,  # Smoothing for Dice loss
    tversky_alpha=0.5,  # Tversky alpha (FP weight)
    tversky_beta=0.5,  # Tversky beta (FN weight)
    # BOUNDARY WEIGHTING PARAMETERS (for boundary_affinity loss)
    boundary_weight=10.0,  # Maximum weight at instance boundaries (1.0 = no weighting)
    boundary_sigma=5.0,  # Distance decay for boundary weights in pixels
    boundary_anisotropy=None,  # Voxel anisotropy (z,y,x) for EDT; None = isotropic
    # AFFINITY PARAMETERS
    output_type="labels",  # 'labels' or 'affinities' or 'affinities_lsds'
    affinity_offsets=None,  # List of (z,y,x) tuples for affinity computation
    lsds_sigma=20.0,  # Sigma parameter for LSDS computation (only used when output_type='affinities_lsds')
    use_batchrenorm=False,  # Whether to use BatchRenorm instead of BatchNorm (more stable for small batches
    mask_clip_distance=None,
):
    """
    Memory-efficient 3D UNet training with multiple precision and memory optimization options.

    Parameters:
    -----------
    gt_masks : numpy.ndarray, optional
        Binary masks indicating valid ground truth regions (1) vs extended regions (0).
        Shape must match gt_data. Used for GT extension functionality where raw volumes
        extend beyond ground truth boundaries. If None, all regions are treated as valid.
    checkpoint_every_n_epochs : int, optional
        Save full model checkpoint every N epochs, regardless of performance.
        In addition to always saving the best model, this allows periodic backups.
        If None (default), only saves the best model and training statistics every epoch.
        Example: checkpoint_every_n_epochs=10 saves model every 10 epochs.
    dinov3_stride : int, optional
        Stride for DINOv3 sliding window inference. If None, uses patch_size (16) for standard inference.
        Use smaller values (e.g., 8, 4) for higher resolution features at the cost of increased computation:
        - stride=8: 4x more features, 4x slower
        - stride=4: 16x more features, 16x slower
        Best used with learn_upsampling=True to avoid downsampling high-res features.
    use_orthogonal_planes : bool, optional
        Whether to use orthogonal plane processing (XY, XZ, YZ slices) for more comprehensive 3D features.
        When True, processes slices in all three orientations and averages the features.
        Default is False (standard Z-slice only processing).
    output_type : str, optional
        Type of output target: 'labels' for class labels or 'affinities' for affinity graphs.
        When 'affinities', GT is converted from instance segmentation to binary affinities.
        Default is 'labels'.
    affinity_offsets : list of tuples, optional
        List of (z, y, x) offsets for computing affinities. Only used when output_type='affinities'.
        If None and output_type='affinities', defaults to [(1,0,0), (0,1,0), (0,0,1)] for +z, +y, +x directions.
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

    # Generate default class names if not provided
    if class_names is None:
        if num_classes == 2:
            class_names = ["background", "foreground"]
        else:
            class_names = ["background"] + [f"class_{i}" for i in range(1, num_classes)]
        print(f"Generated default class names: {class_names}")
    else:
        # Validate class names
        if len(class_names) != num_classes:
            print(
                f"WARNING: Number of class names ({len(class_names)}) doesn't match num_classes ({num_classes})"
            )
            print(f"  Provided names: {class_names}")
            # Adjust class_names to match num_classes
            if len(class_names) < num_classes:
                class_names = class_names + [
                    f"class_{i}" for i in range(len(class_names), num_classes)
                ]
            else:
                class_names = class_names[:num_classes]
            print(f"  Adjusted names: {class_names}")

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
    if save_checkpoints and checkpoint_every_n_epochs is not None:
        print(f"  Periodic checkpoint saving: Every {checkpoint_every_n_epochs} epochs")
    print(f"  Use class weighting: {use_class_weighting}")
    print()

    # Validate input data
    if len(raw_data) < val_volume_pool_size + 2:
        raise ValueError(
            f"Need at least {val_volume_pool_size + 2} volumes for training"
        )

    # Validate shapes with multi-resolution support
    if raw_data.shape[0] != gt_data.shape[0]:
        raise ValueError(
            f"Number of volumes must match: {raw_data.shape[0]} vs {gt_data.shape[0]}"
        )

    # For multi-resolution training, we allow different spatial dimensions
    # The raw data (high-res) and GT data (base-res) can have different sizes
    raw_shape = raw_data.shape[1:]  # (D, H, W)
    gt_shape = gt_data.shape[1:]  # (D, H, W)

    if raw_shape == gt_shape:
        print(f"✓ Same resolution: Raw and GT shapes match {raw_data.shape}")
        resolution_mode = "same"
    else:
        print(f"✓ Multi-resolution training: Raw {raw_shape} vs GT {gt_shape}")
        resolution_mode = "multi"

        # Validate that dimensions are reasonable multiples/factors
        # This helps catch obvious mistakes while allowing intentional multi-resolution
        for dim_name, (raw_dim, gt_dim) in zip(
            ["D", "H", "W"], zip(raw_shape, gt_shape)
        ):
            ratio = raw_dim / gt_dim
            if ratio < 0.25 or ratio > 8.0:  # Allow 4x smaller to 8x larger
                print(
                    f"Warning: {dim_name} dimension ratio ({ratio:.2f}) is quite extreme"
                )

        print(f"  - Raw data will be processed at {raw_shape} resolution")
        print(f"  - Features will be downsampled to match GT at {gt_shape} resolution")

    if len(raw_data.shape) != 4:
        raise ValueError(
            f"Expected 4D data (num_volumes, D, H, W), got {raw_data.shape}"
        )

    # Create 3D data loader
    print("Creating memory-efficient 3D data loader...")
    data_loader_3d = MemoryEfficientDataLoader3D(
        raw_data=raw_data,
        gt_data=gt_data,
        gt_masks=gt_masks,  # NEW: GT extension masks
        context_data=context_data,  # NEW: Context volumes for spatial context
        context_scale=context_scale,  # NEW: Context resolution in nm
        train_volume_pool_size=train_volume_pool_size,
        val_volume_pool_size=val_volume_pool_size,
        target_volume_size=target_volume_size,
        dinov3_slice_size=dinov3_slice_size,
        seed=seed,
        model_id=model_id,
        learn_upsampling=learn_upsampling,
        dinov3_stride=dinov3_stride,  # NEW: Sliding window parameter
        use_orthogonal_planes=use_orthogonal_planes,  # NEW: Orthogonal planes parameter
        verbose=verbose,  # NEW: Verbose output parameter
        output_type=output_type,  # NEW: Output type (labels or affinities or affinities_lsds)
        affinity_offsets=affinity_offsets,  # NEW: Affinity offsets
        lsds_sigma=lsds_sigma,  # NEW: LSDS sigma parameter
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
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        model_id=model_id,
        export_base_dir=export_base_dir,
        use_class_weighting=use_class_weighting,
        use_mixed_precision=use_mixed_precision,  # Pass through
        use_half_precision=use_half_precision,  # Pass through additional params
        use_gradient_checkpointing=use_gradient_checkpointing,
        memory_efficient_mode=memory_efficient_mode,
        learn_upsampling=learn_upsampling,  # Pass through new parameter
        enable_detailed_timing=enable_detailed_timing,  # Pass through new parameter
        min_resolution_for_raw=min_resolution_for_raw,  # Pass through data resolution parameters
        base_resolution=base_resolution,
        class_names=class_names,  # Pass through class names
        # Pass through loss function parameters
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        focal_weight=focal_weight,
        dice_weight=dice_weight,
        dice_smooth=dice_smooth,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        # Pass through boundary weighting parameters
        boundary_weight=boundary_weight,
        boundary_sigma=boundary_sigma,
        boundary_anisotropy=boundary_anisotropy,
        use_batchrenorm=use_batchrenorm,  # Pass through BatchRenorm option
        mask_clip_distance=mask_clip_distance,
    )

    print(f"\n3D UNet training completed!")

    # Print different metrics based on output type
    if output_type in ["affinities", "affinities_lsds"]:
        print(f"  Best validation loss: {results['best_val_loss']:.4f}")
    else:
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
