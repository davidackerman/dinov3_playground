"""
Inference Module for DINOv3 UNet Models

This module provides easy-to-use classes for loading trained models and running inference
on new images or volumes. It automatically handles:
- Loading the best model weights from training exports
- Reconstructing the correct DINOv3 model configuration
- Providing simple interfaces for 2D and 3D inference

Author: GitHub Copilot
Date: 2025-09-26
"""

import os
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import warnings

# Handle both relative and absolute imports
try:
    from .dinov3_core import initialize_dinov3, process, get_current_model_info
    from .models import DINOv3UNet, DINOv3UNet3D, DINOv3UNet3DPipeline
    from .memory_efficient_training import MemoryEfficientDataLoader3D
except ImportError:
    from dinov3_playground.dinov3_core import (
        initialize_dinov3,
        process,
        get_current_model_info,
    )
    from dinov3_playground.models import DINOv3UNet, DINOv3UNet3D, DINOv3UNet3DPipeline
    from dinov3_playground.memory_efficient_training import MemoryEfficientDataLoader3D


class DINOv3UNetInference:
    """
    Easy-to-use inference class for 2D DINOv3 UNet models.

    Automatically loads the best model weights and DINOv3 configuration
    from a training export directory.
    """

    def __init__(
        self,
        export_dir: str,
        device: Optional[str] = None,
        checkpoint_preference: str = "best",
    ):
        """
        Initialize the inference class.

        Args:
            export_dir: Path to the training export directory containing checkpoints
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.export_dir = Path(export_dir)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Model components
        self.unet = None
        self.dinov3_processor = None
        self.dinov3_model = None
        self.model_config = None
        self.training_config = None
        # checkpoint preference: "best" or "latest"
        self.checkpoint_preference = checkpoint_preference

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the best model weights and configuration."""
        # Find checkpoint directory - look for timestamped directories
        # Structure: export_dir/model_type_model_id/timestamp/model/
        timestamp_dirs = []
        for path in self.export_dir.iterdir():
            if path.is_dir():
                # Look for model subdirectories
                model_dir = path / "model"
                if model_dir.exists():
                    timestamp_dirs.append(model_dir)

        if not timestamp_dirs:
            # Fallback: look for any directories that might contain checkpoints
            checkpoint_dirs = list(self.export_dir.glob("**/model"))
            if not checkpoint_dirs:
                # Final fallback: look directly in export_dir
                checkpoint_dirs = [self.export_dir]
            timestamp_dirs = checkpoint_dirs

        if not timestamp_dirs:
            raise ValueError(f"No checkpoint directories found in {self.export_dir}")

        # Take the most recent directory (sort by name, which includes timestamp)
        checkpoint_dir = sorted(timestamp_dirs)[-1]

        # Select checkpoint file according to preference
        best_model_path = None
        if getattr(self, "checkpoint_preference", "best") == "latest":
            # look for epoch checkpoints highest-numbered
            epoch_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pkl"))
            max_file = None
            max_num = -1
            import re

            for f in epoch_files:
                m = re.search(r"checkpoint_epoch_(\d+)\.pkl$", f.name)
                if m:
                    try:
                        num = int(m.group(1))
                    except Exception:
                        continue
                    if num > max_num:
                        max_num = num
                        max_file = f

            if max_file is not None:
                best_model_path = max_file
            else:
                # fallback to best.pkl
                best_model_path = checkpoint_dir / "best.pkl"
        else:
            best_model_path = checkpoint_dir / "best.pkl"

        # If preferred file isn't present, fall back to any non-stats .pkl as before
        if not best_model_path.exists():
            pkl_files = list(checkpoint_dir.glob("*.pkl"))
            pkl_files = [f for f in pkl_files if not f.name.startswith("stats_epoch_")]
            if pkl_files:
                best_model_path = pkl_files[0]
                warnings.warn(
                    f"No preferred checkpoint found, using {best_model_path.name}"
                )
            else:
                raise ValueError(f"No model checkpoints found in {checkpoint_dir}")

        # Load checkpoint
        print(f"Loading model from: {best_model_path}")
        with open(best_model_path, "rb") as f:
            checkpoint = pickle.load(f)

        # Extract training configuration from checkpoint
        self.training_config = checkpoint.get("training_config", {})

        # Extract model configuration from checkpoint or training config
        self.model_config = checkpoint.get("model_config", {})
        # Ensure output_type and affinity_offsets are available in model_config
        if "output_type" not in self.model_config:
            if "output_type" in self.training_config:
                self.model_config["output_type"] = self.training_config["output_type"]
        if "affinity_offsets" not in self.model_config:
            if "affinity_offsets" in self.training_config:
                self.model_config["affinity_offsets"] = self.training_config[
                    "affinity_offsets"
                ]
        if not self.model_config and self.training_config:
            # Try to reconstruct from training config with better field mapping
            # Handle different field names between old and new configs
            target_size = self.training_config.get("target_size", 512)
            input_size = self.training_config.get(
                "input_size", (target_size, target_size)
            )

            self.model_config = {
                "num_classes": self.training_config.get("num_classes", 2),
                "base_channels": self.training_config.get("base_channels", 64),
                "input_size": input_size,
                "input_channels": self.training_config.get("input_channels", 384),
                "model_id": self.training_config.get(
                    "model_id", "facebook/dinov3-vitl16-pretrain-sat493m"
                ),
                "model_type": self.training_config.get("model_type", "dinov3_unet"),
            }
            print(f"Reconstructed model_config from training_config")

        # Initialize DINOv3
        model_id = self.model_config.get(
            "model_id", "facebook/dinov3-vitl16-pretrain-sat493m"
        )
        # Try multiple sources for image_size (2D case)
        image_size = (
            self.training_config.get("image_size")
            or self.training_config.get("target_size")
            or 512
        )

        print(f"Initializing DINOv3 model: {model_id}")
        self.dinov3_processor, self.dinov3_model, output_channels = initialize_dinov3(
            model_id=model_id, image_size=image_size
        )

        # Create UNet model
        num_classes = self.model_config.get("num_classes", 2)
        base_channels = self.model_config.get("base_channels", 64)
        input_size = self.model_config.get("input_size", (512, 512))

        print(
            f"Creating 2D UNet with {output_channels} input channels, {num_classes} classes"
        )
        self.unet = DINOv3UNet(
            input_channels=output_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            input_size=input_size,
        ).to(self.device)

        # Load model weights - the training saves UNet weights as "unet_state_dict"
        if "unet_state_dict" in checkpoint:
            self.unet.load_state_dict(checkpoint["unet_state_dict"])
        elif "model_state_dict" in checkpoint:
            self.unet.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.unet.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume the entire checkpoint is the state dict
            self.unet.load_state_dict(checkpoint)

        self.unet.train()

        print(f"✅ Model loaded successfully!")
        print(f"   - Device: {self.device}")
        print(f"   - Classes: {num_classes}")
        print(f"   - Input size: {input_size}")
        print(f"   - DINOv3 channels: {output_channels}")
        print(f"   - DINOv3 image size: {image_size}")
        print(f"   - Model config keys: {list(self.model_config.keys())}")
        print(f"   - Training config keys: {list(self.training_config.keys())}")

    def predict(
        self, image: np.ndarray, return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on a single 2D image.

        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            return_probabilities: If True, return both predictions and probabilities

        Returns:
            predictions: Predicted class labels (H, W)
            probabilities: Class probabilities (num_classes, H, W) if return_probabilities=True
        """
        # Ensure image is 2D
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        elif image.ndim == 3:
            # Convert to grayscale if RGB
            image = np.mean(image, axis=2)

        # Process through DINOv3
        features = process(image[np.newaxis, ...])  # Add batch dimension

        # Convert to tensor and move to device
        features_tensor = torch.from_numpy(features).float().to(self.device)

        # Run through UNet
        with torch.no_grad():
            logits = self.unet(features_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

        # Convert back to numpy
        predictions_np = predictions[0].cpu().numpy()
        probabilities_np = probabilities[0].cpu().numpy()

        if return_probabilities:
            return predictions_np, probabilities_np
        else:
            return predictions_np

    def predict_batch(
        self, images: np.ndarray, return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on a batch of 2D images.

        Args:
            images: Input images as numpy array (N, H, W) or (N, H, W, C)
            return_probabilities: If True, return both predictions and probabilities

        Returns:
            predictions: Predicted class labels (N, H, W)
            probabilities: Class probabilities (N, num_classes, H, W) if return_probabilities=True
        """
        batch_predictions = []
        batch_probabilities = []

        for i in range(images.shape[0]):
            if return_probabilities:
                pred, prob = self.predict(images[i], return_probabilities=True)
                batch_predictions.append(pred)
                batch_probabilities.append(prob)
            else:
                pred = self.predict(images[i], return_probabilities=False)
                batch_predictions.append(pred)

        predictions_np = np.stack(batch_predictions, axis=0)

        if return_probabilities:
            probabilities_np = np.stack(batch_probabilities, axis=0)
            return predictions_np, probabilities_np
        else:
            return predictions_np

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_type": "2D UNet",
            "num_classes": self.model_config.get("num_classes", 2),
            "base_channels": self.model_config.get("base_channels", 64),
            "input_size": self.model_config.get("input_size", (512, 512)),
            "dinov3_model": self.model_config.get("model_id", "unknown"),
            "device": str(self.device),
            "export_dir": str(self.export_dir),
        }


class DINOv3UNet3DInference:
    """
    Easy-to-use inference class for 3D DINOv3 UNet models.

    Automatically loads the best model weights and DINOv3 configuration
    from a training export directory.
    """

    def __init__(
        self,
        export_dir: str,
        device: Optional[str] = None,
        checkpoint_preference: str = "best",
    ):
        """
        Initialize the 3D inference class.

        Args:
            export_dir: Path to the training export directory containing checkpoints
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.export_dir = Path(export_dir)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Model components
        self.unet3d = None
        self.pipeline = None
        self.data_loader = None
        self.model_config = None
        self.training_config = None
        # checkpoint preference: "best" or "latest"
        self.checkpoint_preference = checkpoint_preference

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the best 3D model weights and configuration."""
        # Find checkpoint directory - look for timestamped directories
        # Structure: export_dir/model_type_model_id/timestamp/model/
        timestamp_dirs = []
        for path in self.export_dir.iterdir():
            if path.is_dir():
                # Look for model subdirectories
                model_dir = path / "model"
                if model_dir.exists():
                    timestamp_dirs.append(model_dir)

        if not timestamp_dirs:
            # Fallback: look for any directories that might contain checkpoints
            checkpoint_dirs = list(self.export_dir.glob("**/model"))
            if not checkpoint_dirs:
                # Final fallback: look directly in export_dir
                checkpoint_dirs = [self.export_dir]
            timestamp_dirs = checkpoint_dirs

        if not timestamp_dirs:
            raise ValueError(f"No checkpoint directories found in {self.export_dir}")

        # Take the most recent directory (sort by name, which includes timestamp)
        checkpoint_dir = sorted(timestamp_dirs)[-1]

        # Select checkpoint file according to preference (same logic as 2D loader)
        best_model_path = None
        if getattr(self, "checkpoint_preference", "best") == "latest":
            epoch_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pkl"))
            max_file = None
            max_num = -1
            import re

            for f in epoch_files:
                m = re.search(r"checkpoint_epoch_(\d+)\.pkl$", f.name)
                if m:
                    try:
                        num = int(m.group(1))
                    except Exception:
                        continue
                    if num > max_num:
                        max_num = num
                        max_file = f

            if max_file is not None:
                best_model_path = max_file
            else:
                best_model_path = checkpoint_dir / "best.pkl"
        else:
            best_model_path = checkpoint_dir / "best.pkl"

        if not best_model_path.exists():
            pkl_files = list(checkpoint_dir.glob("*.pkl"))
            pkl_files = [f for f in pkl_files if not f.name.startswith("stats_epoch_")]
            if pkl_files:
                best_model_path = pkl_files[0]
                warnings.warn(
                    f"No preferred checkpoint found, using {best_model_path.name}"
                )
            else:
                raise ValueError(f"No model checkpoints found in {checkpoint_dir}")

        # Load checkpoint
        print(f"Loading 3D model from: {best_model_path}")
        with open(best_model_path, "rb") as f:
            checkpoint = pickle.load(f)

        # Extract training configuration from checkpoint
        self.training_config = checkpoint.get("training_config", {})

        # Extract model configuration
        self.model_config = checkpoint.get("model_config", {})
        # Ensure output_type and affinity_offsets are available in model_config
        if "output_type" not in self.model_config:
            if "output_type" in self.training_config:
                self.model_config["output_type"] = self.training_config["output_type"]
        if "affinity_offsets" not in self.model_config:
            if "affinity_offsets" in self.training_config:
                self.model_config["affinity_offsets"] = self.training_config[
                    "affinity_offsets"
                ]
        if not self.model_config and self.training_config:
            # Better fallback reconstruction for 3D models
            target_volume_size = self.training_config.get(
                "target_volume_size", (128, 128, 128)
            )
            dinov3_slice_size = self.training_config.get("dinov3_slice_size", 512)

            self.model_config = {
                "num_classes": self.training_config.get("num_classes", 2),
                "base_channels": self.training_config.get("base_channels", 64),
                "input_size": target_volume_size,
                "input_channels": self.training_config.get("input_channels", 384),
                "dinov3_slice_size": dinov3_slice_size,
                "learn_upsampling": self.training_config.get(
                    "learn_upsampling", False
                ),  # NEW
                "dinov3_stride": self.training_config.get("dinov3_stride", None),  # NEW
                "use_context_fusion": self.training_config.get(
                    "use_context_fusion", False
                ),  # NEW - context support
                "context_channels": self.training_config.get(
                    "context_channels", None
                ),  # NEW - context channels
                "model_id": self.training_config.get(
                    "model_id", "facebook/dinov3-vitl16-pretrain-sat493m"
                ),
                "model_type": self.training_config.get("model_type", "dinov3_unet3d"),
            }
            print(f"Reconstructed 3D model_config from training_config")

        # Detect learned upsampling from state dict if not in config
        learn_upsampling = self.model_config.get("learn_upsampling", False)
        if not learn_upsampling:
            # Check if state dict contains learned upsampling layers
            state_dict_keys = []
            if "unet3d_state_dict" in checkpoint:
                state_dict_keys = list(checkpoint["unet3d_state_dict"].keys())
            elif "model_state_dict" in checkpoint:
                state_dict_keys = list(checkpoint["model_state_dict"].keys())
            elif "state_dict" in checkpoint:
                state_dict_keys = list(checkpoint["state_dict"].keys())

            # Check for learned upsampling layers
            if any("learned_upsample" in key for key in state_dict_keys):
                learn_upsampling = True
                self.model_config["learn_upsampling"] = True
                print("Detected learned upsampling from model state dict")

        # Detect context fusion from state dict if not in config
        use_context_fusion = self.model_config.get("use_context_fusion", False)
        if not use_context_fusion and state_dict_keys:
            # Check for context fusion layers
            if any("context" in key.lower() for key in state_dict_keys):
                use_context_fusion = True
                self.model_config["use_context_fusion"] = True
                print("Detected context fusion from model state dict")

        # Initialize DINOv3
        model_id = self.model_config.get(
            "model_id", "facebook/dinov3-vitl16-pretrain-sat493m"
        )
        # Try multiple sources for image_size (3D case)
        image_size = (
            self.training_config.get("image_size")
            or self.training_config.get("dinov3_slice_size")
            or self.model_config.get("dinov3_slice_size")
            or 512
        )

        print(f"Initializing DINOv3 model: {model_id}")
        processor, model, output_channels = initialize_dinov3(
            model_id=model_id, image_size=image_size
        )

        # Create 3D UNet model
        num_classes = self.model_config.get("num_classes", 2)
        base_channels = self.model_config.get("base_channels", 64)
        input_size = self.model_config.get("input_size", (128, 128, 128))
        dinov3_slice_size = self.model_config.get("dinov3_slice_size", 512)

        # Context fusion setup
        context_channels = None
        if use_context_fusion:
            # Context features come from DINOv3 at same resolution, so same channel count
            context_channels = output_channels
            print(f"Context fusion enabled with {context_channels} context channels")

        # Calculate DINOv3 feature size for learned upsampling
        dinov3_feature_size = None
        if learn_upsampling:
            # DINOv3 features are typically 1/16 of the slice size
            feature_spatial_size = dinov3_slice_size // 16
            dinov3_feature_size = (
                input_size[0],  # Keep depth dimension
                feature_spatial_size,
                feature_spatial_size,
            )

        upsampling_info = (
            "with learned upsampling"
            if learn_upsampling
            else "with interpolated upsampling"
        )
        context_info = (
            f"with context fusion ({context_channels} channels)"
            if use_context_fusion
            else "without context fusion"
        )
        print(
            f"Creating 3D UNet with {output_channels} input channels, {num_classes} classes {upsampling_info} {context_info}"
        )
        self.unet3d = DINOv3UNet3D(
            input_channels=output_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            input_size=input_size,
            learn_upsampling=learn_upsampling,
            dinov3_feature_size=dinov3_feature_size,
            use_context_fusion=use_context_fusion,  # Enable context fusion if detected
            context_channels=context_channels,  # Pass context channels
        ).to(self.device)

        # Load model weights - the training saves 3D UNet weights as "unet3d_state_dict"
        if "unet3d_state_dict" in checkpoint:
            self.unet3d.load_state_dict(checkpoint["unet3d_state_dict"])
        elif "model_state_dict" in checkpoint:
            self.unet3d.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.unet3d.load_state_dict(checkpoint["state_dict"])
        else:
            self.unet3d.load_state_dict(checkpoint)

        self.unet3d.eval()

        # Get orthogonal planes setting from training config
        # Default to False for backward compatibility with models trained before orthogonal planes were added
        use_orthogonal_planes = self.training_config.get("use_orthogonal_planes", False)
        print(f"Using orthogonal planes processing: {use_orthogonal_planes}")

        # Create pipeline for easier inference
        self.pipeline = DINOv3UNet3DPipeline(
            num_classes=num_classes,
            input_size=input_size,
            dinov3_slice_size=dinov3_slice_size,
            base_channels=base_channels,
            input_channels=output_channels,  # Pass the input channels explicitly
            use_orthogonal_planes=use_orthogonal_planes,  # Use training config setting
            device=self.device,
        )
        # Replace the pipeline's UNet with our trained model
        self.pipeline.unet3d = self.unet3d

        # Get stride parameter from training config (backward compatibility)
        dinov3_stride = self.training_config.get("dinov3_stride", None)
        if dinov3_stride is not None:
            print(f"Using sliding window stride: {dinov3_stride}")
        else:
            print("Using standard DINOv3 inference (stride=16)")

        # Create minimal data loader for feature extraction compatibility
        # Use dummy 4D data with correct shape
        dummy_volume = np.zeros((2, *input_size), dtype=np.float32)  # 2 dummy volumes
        self.data_loader = MemoryEfficientDataLoader3D(
            raw_data=dummy_volume,
            gt_data=dummy_volume,
            train_volume_pool_size=1,
            val_volume_pool_size=1,
            target_volume_size=input_size,
            dinov3_slice_size=dinov3_slice_size,
            seed=42,
            model_id=model_id,
            learn_upsampling=learn_upsampling,  # Match training mode
            dinov3_stride=dinov3_stride,  # Match training stride
            use_orthogonal_planes=use_orthogonal_planes,  # Match training orthogonal setting
        )

        print(f"✅ 3D Model loaded successfully!")
        print(f"   - Device: {self.device}")
        print(f"   - Classes: {num_classes}")
        print(f"   - Volume size: {input_size}")
        print(f"   - DINOv3 channels: {output_channels}")
        print(f"   - DINOv3 image size: {image_size}")
        print(f"   - DINOv3 slice size: {dinov3_slice_size}")
        print(f"   - DINOv3 stride: {dinov3_stride}")
        print(f"   - Learn upsampling: {learn_upsampling}")
        print(f"   - Use context fusion: {use_context_fusion}")
        if use_context_fusion:
            print(f"   - Context channels: {context_channels}")
        print(f"   - Model config keys: {list(self.model_config.keys())}")
        print(f"   - Training config keys: {list(self.training_config.keys())}")

    def predict(
        self,
        volume: np.ndarray,
        context_volume: Optional[np.ndarray] = None,
        return_probabilities: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on a single 3D volume.

        Args:
            volume: Input volume as numpy array (D, H, W)
            context_volume: Optional context volume at lower resolution for context fusion
            return_probabilities: If True, return both predictions and probabilities

        Returns:
            predictions: Predicted class labels (D, H, W)
            probabilities: Class probabilities (num_classes, D, H, W) if return_probabilities=True
        """
        # Ensure volume is 3D
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

        # Check if model uses context fusion
        use_context_fusion = self.model_config.get("use_context_fusion", False)

        # Add batch dimension and extract features
        volume_batch = volume[np.newaxis, ...]  # (1, D, H, W)

        # Extract DINOv3 features
        if use_context_fusion and context_volume is not None:
            # Extract both local and context features
            context_batch = context_volume[np.newaxis, ...]  # (1, D, H, W)

            # Use the multi-scale feature extraction
            local_features, context_features, timing = (
                self.data_loader.extract_multi_scale_dinov3_features_3d(
                    volume_batch, context_batch, enable_detailed_timing=True
                )
            )
            print(f"Feature extraction timing (with context): {timing}")
        elif use_context_fusion and context_volume is None:
            warnings.warn(
                "Model was trained with context fusion but no context_volume provided. "
                "Inference may produce suboptimal results."
            )
            # Extract just local features, pass None for context
            local_features, timing = self.data_loader.extract_dinov3_features_3d(
                volume_batch, enable_detailed_timing=True
            )
            context_features = None
            print(f"Feature extraction timing (without context): {timing}")
        else:
            # No context fusion - standard single-volume feature extraction
            local_features, timing = self.data_loader.extract_dinov3_features_3d(
                volume_batch, enable_detailed_timing=True
            )
            context_features = None
            print(f"Feature extraction timing: {timing}")

        # Run through 3D UNet
        with torch.no_grad():
            if use_context_fusion:
                # Pass context features separately for proper fusion
                logits = self.unet3d(local_features, context_features=context_features)
            else:
                # Standard inference without context
                logits = self.unet3d(local_features)

            # Check output type to determine post-processing
            output_type = self.model_config.get("output_type", "labels")

            if output_type in ("affinities", "affinities_lsds"):
                # For pure affinities: apply sigmoid to all channels and return
                # probabilities in [0,1]. For combined affinities+LSDS
                # (output_type == 'affinities_lsds') we want to keep the LSDS
                # channels as raw logits (first N channels) and only apply
                # sigmoid to the affinity channels (remaining channels).
                if output_type == "affinities":
                    # Output shape: (batch, num_offsets, D, H, W)
                    probabilities = torch.sigmoid(logits)
                    predictions = probabilities
                else:  # output_type == 'affinities_lsds'
                    # Number of LSDS channels (default 10)
                    num_lsds = int(self.model_config.get("num_lsds", 10))

                    # Safety: if model produced fewer channels than expected, fall back to full-sigmoid
                    # if logits.shape[1] <= num_lsds:
                    probabilities = torch.sigmoid(logits)
                    predictions = probabilities
                    # else:
                    #     lsds_logits = logits[:, :num_lsds, ...]
                    #     aff_logits = logits[:, num_lsds:, ...]
                    #
                    #     # Apply sigmoid only to affinity logits
                    #     aff_probs = torch.sigmoid(aff_logits)
                    #
                    #     # Compose predictions: LSDS raw logits first, then affinity probabilities
                    #     predictions = torch.cat([lsds_logits, aff_probs], dim=1)
                    #
                    #     # Keep a matching 'probabilities' variable for downstream return
                    #     probabilities = predictions
            else:
                # For classification: apply softmax and argmax
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

        # Convert back to numpy
        if output_type in ("affinities", "affinities_lsds"):
            # Return all channels for affinities (num_offsets, D, H, W)
            predictions_np = predictions[0].cpu().numpy()
            probabilities_np = probabilities[0].cpu().numpy()
        else:
            # Return class predictions for segmentation
            predictions_np = predictions[0].cpu().numpy()
            probabilities_np = probabilities[0].cpu().numpy()

        if return_probabilities:
            return predictions_np, probabilities_np
        else:
            return predictions_np

    def predict_large_volume(
        self,
        volume: np.ndarray,
        chunk_size: int = 64,
        overlap: int = 16,
        return_probabilities: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on a large 3D volume using sliding window approach.

        Args:
            volume: Input volume as numpy array (D, H, W)
            chunk_size: Size of chunks to process
            overlap: Overlap between chunks
            return_probabilities: If True, return both predictions and probabilities

        Returns:
            predictions: Predicted class labels (D, H, W)
            probabilities: Class probabilities (num_classes, D, H, W) if return_probabilities=True
        """
        if return_probabilities:
            predictions, probabilities = self.pipeline.predict_large_volume(
                volume,
                chunk_size=chunk_size,
                overlap=overlap,
                return_probabilities=True,
            )
            return predictions, probabilities
        else:
            predictions = self.pipeline.predict_large_volume(
                volume,
                chunk_size=chunk_size,
                overlap=overlap,
                return_probabilities=False,
            )
            return predictions

    def predict_batch(
        self, volumes: np.ndarray, return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on a batch of 3D volumes.

        Args:
            volumes: Input volumes as numpy array (N, D, H, W)
            return_probabilities: If True, return both predictions and probabilities

        Returns:
            predictions: Predicted class labels (N, D, H, W)
            probabilities: Class probabilities (N, num_classes, D, H, W) if return_probabilities=True
        """
        batch_predictions = []
        batch_probabilities = []

        for i in range(volumes.shape[0]):
            if return_probabilities:
                pred, prob = self.predict(volumes[i], return_probabilities=True)
                batch_predictions.append(pred)
                batch_probabilities.append(prob)
            else:
                pred = self.predict(volumes[i], return_probabilities=False)
                batch_predictions.append(pred)

        predictions_np = np.stack(batch_predictions, axis=0)

        if return_probabilities:
            probabilities_np = np.stack(batch_probabilities, axis=0)
            return predictions_np, probabilities_np
        else:
            return predictions_np

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded 3D model."""
        return {
            "model_type": "3D UNet",
            "num_classes": self.model_config.get("num_classes", 2),
            "base_channels": self.model_config.get("base_channels", 64),
            "input_size": self.model_config.get("input_size", (128, 128, 128)),
            "dinov3_slice_size": self.model_config.get("dinov3_slice_size", 512),
            "dinov3_model": self.model_config.get("model_id", "unknown"),
            "device": str(self.device),
            "export_dir": str(self.export_dir),
        }


def load_inference_model(
    export_dir: str,
    model_type: str = "auto",
    device: Optional[str] = None,
    checkpoint_preference: str = "best",
) -> Union[DINOv3UNetInference, DINOv3UNet3DInference]:
    """
    Convenience function to automatically load the appropriate inference model.

    Args:
        export_dir: Path to the training export directory. Can be either:
                   - Direct path to timestamp folder (e.g., .../run_20251003_110551)
                   - Path to parent folder (will auto-select most recent timestamp)
        model_type: Type of model to load ('2d', '3d', or 'auto' to detect)
        device: Device to run inference on

    Returns:
        Loaded inference model (2D or 3D)
    """
    export_path = Path(export_dir)

    # Check if the last folder in the path looks like a timestamp folder
    # Timestamp pattern: run_YYYYMMDD_HHMMSS or just YYYYMMDD_HHMMSS
    import re

    last_folder = export_path.name
    timestamp_pattern = r"(run_)?\d{8}_\d{6}"

    if not re.match(timestamp_pattern, last_folder):
        # Not a timestamp folder - look for timestamp subdirectories
        print(f"Path does not end with timestamp folder: {export_path}")
        print(f"Looking for most recent timestamp folder...")

        # Find all timestamp directories
        timestamp_dirs = []
        for item in export_path.iterdir():
            if item.is_dir() and re.match(timestamp_pattern, item.name):
                timestamp_dirs.append(item)

        if timestamp_dirs:
            # Sort by name (timestamp) and take the most recent
            most_recent = sorted(timestamp_dirs, key=lambda x: x.name)[-1]
            print(f"Found {len(timestamp_dirs)} timestamp folder(s)")
            print(f"Using most recent: {most_recent.name}")
            export_path = most_recent
        else:
            print(f"No timestamp folders found in {export_path}")
            print(f"Proceeding with provided path...")

    # Update export_dir to use the selected path (string version)
    export_dir = str(export_path)

    if model_type == "auto":
        # Try to detect model type from training config embedded in best.pkl
        # Find timestamp directories with model subdirectories
        timestamp_dirs = []
        for path in export_path.iterdir():
            if path.is_dir():
                model_dir = path / "model"
                if model_dir.exists():
                    timestamp_dirs.append(model_dir)

        if not timestamp_dirs:
            # Fallback: look for any directories that might contain checkpoints
            checkpoint_dirs = list(export_path.glob("**/model"))
            if not checkpoint_dirs:
                checkpoint_dirs = [export_path]
            timestamp_dirs = checkpoint_dirs

        if timestamp_dirs:
            # Take the most recent directory
            checkpoint_dir = sorted(timestamp_dirs)[-1]
            best_path = checkpoint_dir / "best.pkl"

            if best_path.exists():
                try:
                    with open(best_path, "rb") as f:
                        checkpoint = pickle.load(f)

                    config = checkpoint.get("training_config", {})

                    # Check for 3D-specific parameters
                    if (
                        any(
                            key in config
                            for key in [
                                "target_volume_size",
                                "volume_shape",
                                "dinov3_slice_size",
                                "volumes_per_batch",
                            ]
                        )
                        or config.get("model_type") == "dinov3_unet3d"
                    ):
                        model_type = "3d"
                    else:
                        model_type = "2d"
                except Exception as e:
                    warnings.warn(f"Could not read checkpoint for auto-detection: {e}")
                    model_type = "2d"
            else:
                # Default to 2D if can't determine
                model_type = "2d"
                warnings.warn("Could not determine model type, defaulting to 2D")
        else:
            model_type = "2d"
            warnings.warn("No checkpoint directories found, defaulting to 2D")

    if model_type == "3d":
        return DINOv3UNet3DInference(
            export_dir, device, checkpoint_preference=checkpoint_preference
        )
    else:
        return DINOv3UNetInference(
            export_dir, device, checkpoint_preference=checkpoint_preference
        )


# Example usage functions
def demo_2d_inference(export_dir: str):
    """Demonstrate how to use the 2D inference class."""
    print("Loading 2D inference model...")
    model = DINOv3UNetInference(export_dir)

    print("Model info:", model.get_model_info())

    # Create dummy test image
    test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

    # Run inference
    predictions = model.predict(test_image)
    predictions_with_prob, probabilities = model.predict(
        test_image, return_probabilities=True
    )

    print(f"Input shape: {test_image.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Unique predictions: {np.unique(predictions)}")


def demo_3d_inference(export_dir: str):
    """Demonstrate how to use the 3D inference class."""
    print("Loading 3D inference model...")
    model = DINOv3UNet3DInference(export_dir)

    print("Model info:", model.get_model_info())

    # Create dummy test volume
    test_volume = np.random.randint(0, 255, (128, 128, 128), dtype=np.uint8)

    # Run inference
    predictions = model.predict(test_volume)
    predictions_with_prob, probabilities = model.predict(
        test_volume, return_probabilities=True
    )

    print(f"Input shape: {test_volume.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Unique predictions: {np.unique(predictions)}")
