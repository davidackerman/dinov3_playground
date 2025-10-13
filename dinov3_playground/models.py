"""
Model Classes for DINOv3 Classification and Segmentation

This module contains the neural network model classes:
- ImprovedClassifier: Enhanced classifier with batch normalization and dropout
- SimpleClassifier: Basic classifier for comparison
- DINOv3UNet: UNet architecture for segmentation tasks using DINOv3 features
- DINOv3UNetPipeline: Complete pipeline for DINOv3 feature extraction and UNet segmentation

Author: GitHub Copilot
Date: 2025-09-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

from skimage.transform import resize
from .dinov3_core import (
    process,
    output_channels,
)  # Assuming output_channels is defined there


# ---------------------------
# Simple BatchRenorm implementation (1d/2d/3d)
# ---------------------------
class _BatchRenormNd(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        rmax=3.0,
        dmax=5.0,
        affine=True,
        track_running_stats=True,
        dims=2,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.rmax = rmax
        self.dmax = dmax
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.dims = dims  # 1,2,3 for BatchNorm1d/2d/3d

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x):
        # x shape: (N, C, *spatial)
        if x.dim() < 2:
            raise ValueError("Input for BatchRenorm must have >=2 dims")
        dims = [0] + list(range(2, x.dim()))
        if self.training:
            # compute batch mean/var over N and spatial dims
            batch_mean = x.mean(dim=dims, keepdim=False)
            batch_var = x.var(dim=dims, unbiased=False, keepdim=False)
            batch_std = torch.sqrt(batch_var + self.eps)

            if self.track_running_stats:
                # update running stats
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * batch_mean.detach()
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * batch_var.detach()
                self.num_batches_tracked += 1

                running_std = torch.sqrt(self.running_var + self.eps)
                # compute renorm params
                r = (batch_std / running_std).clamp(1.0 / self.rmax, self.rmax)
                d = ((batch_mean - self.running_mean) / running_std).clamp(
                    -self.dmax, self.dmax
                )
            else:
                r = torch.ones_like(batch_std)
                d = torch.zeros_like(batch_mean)

            # reshape for broadcasting
            shape = [1, -1] + [1] * (x.dim() - 2)
            batch_mean_b = batch_mean.view(shape)
            batch_std_b = batch_std.view(shape)
            r_b = r.view(shape)
            d_b = d.view(shape)

            x_hat = (x - batch_mean_b) / batch_std_b
            x_renorm = r_b * x_hat + d_b
            if self.affine:
                w = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
                b = self.bias.view(1, -1, *([1] * (x.dim() - 2)))
                return w * x_renorm + b
            else:
                return x_renorm
        else:
            # inference uses running stats
            if self.track_running_stats:
                running_std = torch.sqrt(self.running_var + self.eps)
                rm = self.running_mean.view([1, -1] + [1] * (x.dim() - 2))
                rs = running_std.view([1, -1] + [1] * (x.dim() - 2))
                x_hat = (x - rm) / rs
                if self.affine:
                    w = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
                    b = self.bias.view(1, -1, *([1] * (x.dim() - 2)))
                    return w * x_hat + b
                else:
                    return x_hat
            else:
                # fallback to identity if no running stats
                return x


class BatchRenorm1d(_BatchRenormNd):
    def __init__(self, num_features, **kwargs):
        super().__init__(num_features, dims=1, **kwargs)


class BatchRenorm2d(_BatchRenormNd):
    def __init__(self, num_features, **kwargs):
        super().__init__(num_features, dims=2, **kwargs)


class BatchRenorm3d(_BatchRenormNd):
    def __init__(self, num_features, **kwargs):
        super().__init__(num_features, dims=3, **kwargs)


# ---------------------------
# End BatchRenorm
# ---------------------------


class ImprovedClassifier(nn.Module):
    """
    Enhanced neural network classifier with batch normalization and dropout.

    Features:
    - Multiple hidden layers with configurable sizes
    - Batch normalization for training stability
    - Dropout for regularization
    - ReLU activation with optional LeakyReLU
    - Configurable architecture
    """

    def __init__(
        self,
        input_dim=384,
        hidden_dims=[256, 128, 64],
        num_classes=2,
        dropout_rate=0.3,
        use_batch_norm=True,
        use_batch_renorm=False,
        activation="relu",
    ):
        """
        Initialize the ImprovedClassifier.

        Parameters:
        -----------
        input_dim : int, default=384
            Input feature dimension (DINOv3 ViT-S/16 = 384)
        hidden_dims : list, default=[256, 128, 64]
            List of hidden layer dimensions
        num_classes : int, default=2
            Number of output classes
        dropout_rate : float, default=0.3
            Dropout probability
        use_batch_norm : bool, default=True
            Whether to use batch normalization
        activation : str, default='relu'
            Activation function ('relu' or 'leaky_relu')
        """
        super(ImprovedClassifier, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.use_batch_renorm = use_batch_renorm
        self.activation = activation

        # Build layers dynamically
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if use_batch_norm:
                if use_batch_renorm:
                    layers.append(BatchRenorm1d(dims[i + 1]))
                else:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))

            # Activation
            if activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.ReLU())

            layers.append(nn.Dropout(dropout_rate))

        # Final output layer
        layers.append(nn.Linear(dims[-1], num_classes))

        self.classifier = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == "relu":
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_out", nonlinearity="relu"
                    )
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim)

        Returns:
        --------
        torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.classifier(x)

    def get_feature_maps(self, x):
        """
        Get intermediate feature representations.

        Parameters:
        -----------
        x : torch.Tensor
            Input features

        Returns:
        --------
        list: Feature maps from each layer
        """
        features = []
        current_x = x

        for i, layer in enumerate(self.classifier):
            current_x = layer(current_x)
            if isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
                features.append(current_x.clone())

        return features


class SimpleClassifier(nn.Module):
    """
    Simple neural network classifier for baseline comparison.

    Features:
    - Single hidden layer
    - Basic ReLU activation
    - Minimal regularization
    """

    def __init__(self, input_dim=384, hidden_dim=128, num_classes=2):
        """
        Initialize the SimpleClassifier.

        Parameters:
        -----------
        input_dim : int, default=384
            Input feature dimension
        hidden_dim : int, default=128
            Hidden layer dimension
        num_classes : int, default=2
            Number of output classes
        """
        super(SimpleClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim)

        Returns:
        --------
        torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.classifier(x)


class DINOv3UNet(nn.Module):
    """
    2D UNet that takes DINOv3 features as input instead of raw images.

    Architecture:
    - Encoder: Downsampling path with skip connections
    - Decoder: Upsampling path with skip connections from encoder
    - Input: DINOv3 features (H, W, 384)
    - Output: Segmentation map (H, W, num_classes)
    """

    def __init__(
        self, input_channels=384, num_classes=2, base_channels=64, use_batchrenorm=False
    ):
        """
        Initialize DINOv3 UNet.

        Parameters:
        -----------
        input_channels : int, default=384
            Number of input channels (DINOv3 feature dimension)
        num_classes : int, default=2
            Number of output classes
        base_channels : int, default=64
            Base number of channels in the UNet
        """
        super(DINOv3UNet, self).__init__()
        self.use_batchrenorm = use_batchrenorm

        # choose normalization layer
        Norm2d = BatchRenorm2d if self.use_batchrenorm else nn.BatchNorm2d

        # Input projection to reduce channels
        self.input_conv = nn.Conv2d(input_channels, base_channels, kernel_size=1)

        # Encoder (downsampling path)
        self.enc1 = self._make_encoder_block(base_channels, base_channels, Norm2d)
        self.enc2 = self._make_encoder_block(base_channels, base_channels * 2, Norm2d)
        self.enc3 = self._make_encoder_block(
            base_channels * 2, base_channels * 4, Norm2d
        )
        self.enc4 = self._make_encoder_block(
            base_channels * 4, base_channels * 8, Norm2d
        )

        # Bottleneck
        self.bottleneck = self._make_encoder_block(
            base_channels * 8, base_channels * 16, Norm2d
        )

        # Decoder (upsampling path) - Fixed channel calculations
        # After concatenation: upsampled + skip connection
        self.dec4 = self._make_decoder_block(
            base_channels * 16 + base_channels * 8, base_channels * 8, Norm2d
        )
        self.dec3 = self._make_decoder_block(
            base_channels * 8 + base_channels * 4, base_channels * 4, Norm2d
        )
        self.dec2 = self._make_decoder_block(
            base_channels * 4 + base_channels * 2, base_channels * 2, Norm2d
        )
        self.dec1 = self._make_decoder_block(
            base_channels * 2 + base_channels, base_channels, Norm2d
        )

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(2)

    def _make_encoder_block(self, in_channels, out_channels, Norm):
        """Create an encoder block with two convolutions."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            Norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            Norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def _make_decoder_block(self, in_channels, out_channels, Norm):
        """Create a decoder block with two convolutions."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            Norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            Norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass through the UNet.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, H, W)

        Returns:
        --------
        torch.Tensor: Output segmentation map (batch_size, num_classes, H, W)
        """
        # Input projection
        x = self.input_conv(x)  # (B, base_channels, H, W)

        # Encoder path with skip connections
        enc1_out = self.enc1(x)  # (B, base_channels, H, W)
        enc1_pool = self.pool(enc1_out)  # (B, base_channels, H/2, W/2)

        enc2_out = self.enc2(enc1_pool)  # (B, base_channels*2, H/2, W/2)
        enc2_pool = self.pool(enc2_out)  # (B, base_channels*2, H/4, W/4)

        enc3_out = self.enc3(enc2_pool)  # (B, base_channels*4, H/4, W/4)
        enc3_pool = self.pool(enc3_out)  # (B, base_channels*4, H/8, W/8)

        enc4_out = self.enc4(enc3_pool)  # (B, base_channels*8, H/8, W/8)
        enc4_pool = self.pool(enc4_out)  # (B, base_channels*8, H/16, W/16)

        # Bottleneck
        bottleneck_out = self.bottleneck(enc4_pool)  # (B, base_channels*16, H/16, W/16)

        # Decoder path with skip connections
        # Dec4: upsample bottleneck and concatenate with enc4
        dec4_up = F.interpolate(
            bottleneck_out, scale_factor=2, mode="bilinear", align_corners=False
        )
        # dec4_up: (B, base_channels*16, H/8, W/8)
        # enc4_out: (B, base_channels*8, H/8, W/8)
        dec4_concat = torch.cat(
            [dec4_up, enc4_out], dim=1
        )  # (B, base_channels*24, H/8, W/8)
        dec4_out = self.dec4(dec4_concat)  # (B, base_channels*8, H/8, W/8)

        # Dec3: upsample dec4 and concatenate with enc3
        dec3_up = F.interpolate(
            dec4_out, scale_factor=2, mode="bilinear", align_corners=False
        )
        # dec3_up: (B, base_channels*8, H/4, W/4)
        # enc3_out: (B, base_channels*4, H/4, W/4)
        dec3_concat = torch.cat(
            [dec3_up, enc3_out], dim=1
        )  # (B, base_channels*12, H/4, W/4)
        dec3_out = self.dec3(dec3_concat)  # (B, base_channels*4, H/4, W/4)

        # Dec2: upsample dec3 and concatenate with enc2
        dec2_up = F.interpolate(
            dec3_out, scale_factor=2, mode="bilinear", align_corners=False
        )
        # dec2_up: (B, base_channels*4, H/2, W/2)
        # enc2_out: (B, base_channels*2, H/2, W/2)
        dec2_concat = torch.cat(
            [dec2_up, enc2_out], dim=1
        )  # (B, base_channels*6, H/2, W/2)
        dec2_out = self.dec2(dec2_concat)  # (B, base_channels*2, H/2, W/2)

        # Dec1: upsample dec2 and concatenate with enc1
        dec1_up = F.interpolate(
            dec2_out, scale_factor=2, mode="bilinear", align_corners=False
        )
        # dec1_up: (B, base_channels*2, H, W)
        # enc1_out: (B, base_channels, H, W)
        dec1_concat = torch.cat(
            [dec1_up, enc1_out], dim=1
        )  # (B, base_channels*3, H, W)
        dec1_out = self.dec1(dec1_concat)  # (B, base_channels, H, W)

        # Final output
        output = self.final_conv(dec1_out)  # (B, num_classes, H, W)

        return output


class DINOv3UNetPipeline(nn.Module):
    """
    Complete pipeline that combines DINOv3 feature extraction with UNet segmentation.
    Handles image upsampling, feature extraction, and segmentation in one model.
    """

    def __init__(
        self,
        num_classes=2,
        target_size=224,
        dinov3_input_size=896,
        base_channels=64,
        device=None,
    ):
        """
        Initialize the complete DINOv3-UNet pipeline.

        Parameters:
        -----------
        num_classes : int, default=2
            Number of output classes
        target_size : int, default=224
            Target resolution for input images
        dinov3_input_size : int, default=896
            Size for DINOv3 processing (should be multiple of 16)
        base_channels : int, default=64
            Base number of channels in UNet
        device : torch.device, optional
            Device for processing
        """
        super(DINOv3UNetPipeline, self).__init__()

        self.num_classes = num_classes
        self.target_size = target_size
        self.dinov3_input_size = dinov3_input_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # UNet for processing DINOv3 features
        self.unet = DINOv3UNet(
            input_channels=output_channels,  # from .dinov3_core
            num_classes=num_classes,
            base_channels=base_channels,
        )

    def extract_dinov3_features(self, images):
        """
        Extract DINOv3 features from a batch of images.

        Parameters:
        -----------
        images : numpy.ndarray or torch.Tensor
            Images of shape (batch_size, target_size, target_size)

        Returns:
        --------
        torch.Tensor: DINOv3 features (batch_size, output_channels, target_size, target_size)
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        batch_size = images.shape[0]

        # Step 1: Upsample to DINOv3 input size
        upsampled_images = np.zeros(
            (batch_size, self.dinov3_input_size, self.dinov3_input_size),
            dtype=images.dtype,
        )
        print(
            f"Upsampling images with shape {images.shape} to DINOv3 input size... {self.dinov3_input_size}x{self.dinov3_input_size}"
        )
        for i in range(batch_size):
            upsampled_images[i] = resize(
                images[i],
                (self.dinov3_input_size, self.dinov3_input_size),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(images.dtype)
        print(f"DINOv3 feature extraction in progress ...")
        # Step 2: Extract DINOv3 features
        dinov3_features = process(
            upsampled_images
        )  # (output_channels, batch_size, H_feat, W_feat)

        # Step 3: Rearrange to (batch_size, output_channels, H_feat, W_feat)
        if dinov3_features.shape[0] == output_channels:
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

        return features_upsampled.to(self.device)

    def forward(self, images):
        """
        Forward pass through the complete pipeline.

        Parameters:
        -----------
        images : numpy.ndarray or torch.Tensor
            Input images of shape (batch_size, target_size, target_size)

        Returns:
        --------
        torch.Tensor: Segmentation logits (batch_size, num_classes, target_size, target_size)
        """
        # Extract DINOv3 features
        features = self.extract_dinov3_features(images)

        # Pass through UNet
        segmentation_logits = self.unet(features)

        return segmentation_logits

    def predict(self, images, return_probabilities=False):
        """
        Generate predictions from input images.

        Parameters:
        -----------
        images : numpy.ndarray or torch.Tensor
            Input images
        return_probabilities : bool, default=False
            Whether to return class probabilities

        Returns:
        --------
        tuple: (predictions, probabilities) if return_probabilities else predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            if return_probabilities:
                return predictions, probabilities
            return predictions


class ContextAttentionFusion(nn.Module):
    """
    Context attention fusion module for multi-scale feature fusion.

    Uses context features to generate attention weights for raw features,
    allowing context to guide raw feature processing while preserving
    raw feature dominance.
    """

    def __init__(self, channels, use_batchrenorm=False):
        super().__init__()
        Norm3d = BatchRenorm3d if use_batchrenorm else nn.BatchNorm3d
        self.channels = channels
        self.context_proj = nn.Sequential(
            nn.Conv3d(channels, channels // 4, 1),
            Norm3d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )
        self.raw_proj = nn.Conv3d(channels, channels, 1)

    def forward(self, raw_features, context_features):
        """
        Fuse raw and context features using attention mechanism.

        Parameters:
        -----------
        raw_features : torch.Tensor
            Raw features of shape (B, C, D, H, W)
        context_features : torch.Tensor
            Context features of shape (B, C, D, H, W)

        Returns:
        --------
        torch.Tensor: Fused features of shape (B, C, D, H, W)
        """
        # Generate attention weights from context
        attention_weights = self.context_proj(context_features)

        # Apply attention to raw features
        raw_projected = self.raw_proj(raw_features)
        attended_features = raw_projected * attention_weights

        # Residual connection with raw features
        fused_features = raw_features + attended_features

        return fused_features


class DINOv3UNet3D(nn.Module):
    """
    3D UNet that takes DINOv3 features as input for volumetric segmentation.

    Architecture:
    - Encoder: 3D downsampling path with skip connections
    - Decoder: 3D upsampling path with skip connections from encoder
    - Input: DINOv3 features (D, H, W, feature_channels)
    - Output: 3D segmentation map (D, H, W, num_classes)
    """

    def __init__(
        self,
        input_channels=384,
        num_classes=2,
        base_channels=32,
        input_size=(112, 112, 112),
        use_half_precision=False,
        learn_upsampling=False,
        dinov3_feature_size=None,
        use_gradient_checkpointing=True,
        use_context_fusion=False,
        context_channels=384,
        use_batchrenorm=False,
    ):
        super(DINOv3UNet3D, self).__init__()
        self.use_batchrenorm = use_batchrenorm
        Norm3d = BatchRenorm3d if self.use_batchrenorm else nn.BatchNorm3d

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_size = input_size
        self.base_channels = base_channels
        self.use_half_precision = use_half_precision
        self.learn_upsampling = learn_upsampling
        self.dinov3_feature_size = dinov3_feature_size or input_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_context_fusion = use_context_fusion
        self.context_channels = context_channels

        # Input projection to reduce channels
        self.input_conv = nn.Conv3d(input_channels, self.base_channels, kernel_size=1)

        # Calculate upsampling factor if needed
        if self.learn_upsampling:
            self.upsampling_factor = tuple(
                int(target / feat)
                for target, feat in zip(input_size, self.dinov3_feature_size)
            )
            # Initial upsampling layers to bring features closer to target resolution
            self.learned_upsample = self._make_upsampling_layers()
        else:
            self.upsampling_factor = (1, 1, 1)
            self.learned_upsample = None

        # Encoder (downsampling path)
        self.enc1 = self._make_encoder_block(
            self.base_channels, self.base_channels, Norm3d
        )
        self.enc2 = self._make_encoder_block(
            self.base_channels, self.base_channels * 2, Norm3d
        )
        self.enc3 = self._make_encoder_block(
            self.base_channels * 2, self.base_channels * 4, Norm3d
        )
        self.enc4 = self._make_encoder_block(
            self.base_channels * 4, self.base_channels * 8, Norm3d
        )

        # Bottleneck
        self.bottleneck = self._make_encoder_block(
            self.base_channels * 8, self.base_channels * 16, Norm3d
        )

        # Decoder (upsampling path)
        self.dec4 = self._make_decoder_block(
            self.base_channels * 16 + self.base_channels * 8,
            self.base_channels * 8,
            Norm3d,
        )
        self.dec3 = self._make_decoder_block(
            self.base_channels * 8 + self.base_channels * 4,
            self.base_channels * 4,
            Norm3d,
        )
        self.dec2 = self._make_decoder_block(
            self.base_channels * 4 + self.base_channels * 2,
            self.base_channels * 2,
            Norm3d,
        )
        self.dec1 = self._make_decoder_block(
            self.base_channels * 2 + self.base_channels, self.base_channels, Norm3d
        )

        # Final output layer
        self.final_conv = nn.Conv3d(self.base_channels, num_classes, kernel_size=1)

        # 3D Pooling
        self.pool = nn.MaxPool3d(2)

        # Dropout for regularization (3D volumes have more parameters)
        self.dropout = nn.Dropout3d(0.2)

        # Context fusion modules (optional)
        if self.use_context_fusion:
            # Context input projection
            self.context_input_conv = nn.Conv3d(
                context_channels, self.base_channels, kernel_size=1
            )

            # Context projection layers for each encoder level
            self.context_proj1 = nn.Conv3d(
                self.base_channels, self.base_channels, kernel_size=1
            )  # Full resolution
            self.context_proj2 = nn.Conv3d(
                self.base_channels, self.base_channels * 2, kernel_size=1
            )  # 1/2 resolution
            self.context_proj3 = nn.Conv3d(
                self.base_channels, self.base_channels * 4, kernel_size=1
            )  # 1/4 resolution
            self.context_proj4 = nn.Conv3d(
                self.base_channels, self.base_channels * 8, kernel_size=1
            )  # 1/8 resolution

            # Context fusion modules for each skip connection level
            self.context_fusion1 = ContextAttentionFusion(
                self.base_channels, use_batchrenorm=self.use_batchrenorm
            )
            self.context_fusion2 = ContextAttentionFusion(
                self.base_channels * 2, use_batchrenorm=self.use_batchrenorm
            )
            self.context_fusion3 = ContextAttentionFusion(
                self.base_channels * 4, use_batchrenorm=self.use_batchrenorm
            )
            self.context_fusion4 = ContextAttentionFusion(
                self.base_channels * 8, use_batchrenorm=self.use_batchrenorm
            )

        # Convert to half precision if requested
        if use_half_precision:
            self.half()
            print("Model converted to half precision (float16)")

    def _make_encoder_block(self, in_channels, out_channels, Norm):
        """Create a 3D encoder block with two convolutions."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            Norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            Norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

    def _make_decoder_block(self, in_channels, out_channels, Norm):
        """Create a 3D decoder block with two convolutions."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            Norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            Norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

    def _make_upsampling_layers(self):
        """Create learned upsampling layers to bring DINOv3 features to target resolution."""
        layers = []
        current_channels = self.base_channels

        # Calculate how many upsampling stages we need
        max_factor = max(self.upsampling_factor)

        if max_factor <= 1:
            return None

        # Create progressive upsampling layers
        if max_factor >= 8:
            # Stage 1: 8x upsampling (if needed)
            layers.extend(
                [
                    nn.ConvTranspose3d(
                        current_channels,
                        current_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),  # 2x
                    nn.BatchNorm3d(current_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(
                        current_channels,
                        current_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),  # 4x
                    nn.BatchNorm3d(current_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(
                        current_channels,
                        current_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),  # 8x
                    nn.BatchNorm3d(current_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        elif max_factor >= 4:
            # Stage 2: 4x upsampling
            layers.extend(
                [
                    nn.ConvTranspose3d(
                        current_channels,
                        current_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),  # 2x
                    nn.BatchNorm3d(current_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(
                        current_channels,
                        current_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),  # 4x
                    nn.BatchNorm3d(current_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        elif max_factor >= 2:
            # Stage 3: 2x upsampling
            layers.extend(
                [
                    nn.ConvTranspose3d(
                        current_channels,
                        current_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),  # 2x
                    nn.BatchNorm3d(current_channels),
                    nn.ReLU(inplace=True),
                ]
            )

        # Final adjustment layer to match exact target size if needed
        layers.append(
            nn.Conv3d(current_channels, current_channels, kernel_size=3, padding=1)
        )

        return nn.Sequential(*layers)

    def forward(self, x, context_features=None):
        """
        Forward pass through the 3D UNet.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, D, H, W)
        context_features : torch.Tensor, optional
            Context features of shape (batch_size, context_channels, D, H, W)
            If provided and use_context_fusion=True, will be used for multi-scale fusion

        Returns:
        --------
        torch.Tensor: Output segmentation map (batch_size, num_classes, D, H, W)
        """
        # Convert input to half precision if model is in half precision
        if self.use_half_precision and x.dtype != torch.float16:
            x = x.half()

        # Input projection
        x = self.input_conv(x)  # (B, base_channels, D, H, W)

        # Process context features if provided
        context_features_processed = None
        if self.use_context_fusion and context_features is not None:
            # Convert context to half precision if needed
            if self.use_half_precision and context_features.dtype != torch.float16:
                context_features = context_features.half()

            # Project context features to base channels
            context_features_processed = self.context_input_conv(context_features)

            # Apply same upsampling to context if needed
            if self.learn_upsampling and self.learned_upsample is not None:
                context_features_processed = self.learned_upsample(
                    context_features_processed
                )

                # Match size to raw features
                if context_features_processed.shape[2:] != self.input_size:
                    context_features_processed = F.interpolate(
                        context_features_processed,
                        size=self.input_size,
                        mode="trilinear",
                        align_corners=False,
                    )

        # Apply learned upsampling if enabled
        if self.learn_upsampling and self.learned_upsample is not None:
            x = self.learned_upsample(x)

            # Fine-tune the size to exactly match target if needed
            current_size = x.shape[2:]  # (D, H, W)
            if current_size != self.input_size:
                x = F.interpolate(
                    x, size=self.input_size, mode="trilinear", align_corners=False
                )

        # Encoder path with skip connections (use gradient checkpointing if enabled)
        if self.use_gradient_checkpointing and self.training:
            enc1_out = checkpoint(self.enc1, x)  # (B, base_channels, D, H, W)
            enc1_pool = self.pool(enc1_out)  # (B, base_channels, D/2, H/2, W/2)

            enc2_out = checkpoint(
                self.enc2, enc1_pool
            )  # (B, base_channels*2, D/2, H/2, W/2)
            enc2_pool = self.pool(enc2_out)  # (B, base_channels*2, D/4, H/4, W/4)

            enc3_out = checkpoint(
                self.enc3, enc2_pool
            )  # (B, base_channels*4, D/4, H/4, W/4)
            enc3_pool = self.pool(enc3_out)  # (B, base_channels*4, D/8, H/8, W/8)

            enc4_out = checkpoint(
                self.enc4, enc3_pool
            )  # (B, base_channels*8, D/8, H/8, W/8)
            enc4_pool = self.pool(enc4_out)  # (B, base_channels*8, D/16, H/16, W/16)

            # Bottleneck
            bottleneck_out = checkpoint(
                self.bottleneck, enc4_pool
            )  # (B, base_channels*16, D/16, H/16, W/16)
        else:
            enc1_out = self.enc1(x)  # (B, base_channels, D, H, W)
            enc1_pool = self.pool(enc1_out)  # (B, base_channels, D/2, H/2, W/2)

            enc2_out = self.enc2(enc1_pool)  # (B, base_channels*2, D/2, H/2, W/2)
            enc2_pool = self.pool(enc2_out)  # (B, base_channels*2, D/4, H/4, W/4)

            enc3_out = self.enc3(enc2_pool)  # (B, base_channels*4, D/4, H/4, W/4)
            enc3_pool = self.pool(enc3_out)  # (B, base_channels*4, D/8, H/8, W/8)

            enc4_out = self.enc4(enc3_pool)  # (B, base_channels*8, D/8, H/8, W/8)
            enc4_pool = self.pool(enc4_out)  # (B, base_channels*8, D/16, H/16, W/16)

            # Bottleneck
            bottleneck_out = self.bottleneck(
                enc4_pool
            )  # (B, base_channels*16, D/16, H/16, W/16)
        bottleneck_out = self.dropout(
            bottleneck_out
        )  # Apply stronger dropout at bottleneck

        # Prepare context features at different scales if context fusion is enabled
        context_enc1, context_enc2, context_enc3, context_enc4 = None, None, None, None
        if self.use_context_fusion and context_features_processed is not None:
            # Create context features at encoder resolutions by downsampling
            context_base = context_features_processed  # Full resolution
            context_enc2_down = F.avg_pool3d(
                context_base, kernel_size=2, stride=2
            )  # 1/2 resolution
            context_enc3_down = F.avg_pool3d(
                context_enc2_down, kernel_size=2, stride=2
            )  # 1/4 resolution
            context_enc4_down = F.avg_pool3d(
                context_enc3_down, kernel_size=2, stride=2
            )  # 1/8 resolution

            # Project context features to match encoder channel dimensions
            context_enc1 = self.context_proj1(context_base)
            context_enc2 = self.context_proj2(context_enc2_down)
            context_enc3 = self.context_proj3(context_enc3_down)
            context_enc4 = self.context_proj4(context_enc4_down)

        # Decoder path with skip connections
        # Dec4: upsample bottleneck and concatenate with enc4
        dec4_up = F.interpolate(
            bottleneck_out, scale_factor=2, mode="trilinear", align_corners=False
        )

        # Apply context fusion to enc4 skip connection if available
        if context_enc4 is not None:
            enc4_fused = self.context_fusion4(enc4_out, context_enc4)
        else:
            enc4_fused = enc4_out

        dec4_concat = torch.cat([dec4_up, enc4_fused], dim=1)
        if self.use_gradient_checkpointing and self.training:
            dec4_out = checkpoint(self.dec4, dec4_concat)
        else:
            dec4_out = self.dec4(dec4_concat)

        # Dec3: upsample dec4 and concatenate with enc3
        dec3_up = F.interpolate(
            dec4_out, scale_factor=2, mode="trilinear", align_corners=False
        )

        # Apply context fusion to enc3 skip connection if available
        if context_enc3 is not None:
            enc3_fused = self.context_fusion3(enc3_out, context_enc3)
        else:
            enc3_fused = enc3_out

        dec3_concat = torch.cat([dec3_up, enc3_fused], dim=1)
        if self.use_gradient_checkpointing and self.training:
            dec3_out = checkpoint(self.dec3, dec3_concat)
        else:
            dec3_out = self.dec3(dec3_concat)

        # Dec2: upsample dec3 and concatenate with enc2
        dec2_up = F.interpolate(
            dec3_out, scale_factor=2, mode="trilinear", align_corners=False
        )

        # Apply context fusion to enc2 skip connection if available
        if context_enc2 is not None:
            enc2_fused = self.context_fusion2(enc2_out, context_enc2)
        else:
            enc2_fused = enc2_out

        dec2_concat = torch.cat([dec2_up, enc2_fused], dim=1)
        if self.use_gradient_checkpointing and self.training:
            dec2_out = checkpoint(self.dec2, dec2_concat)
        else:
            dec2_out = self.dec2(dec2_concat)

        # Dec1: upsample dec2 and concatenate with enc1
        dec1_up = F.interpolate(
            dec2_out, scale_factor=2, mode="trilinear", align_corners=False
        )

        # Apply context fusion to enc1 skip connection if available
        if context_enc1 is not None:
            enc1_fused = self.context_fusion1(enc1_out, context_enc1)
        else:
            enc1_fused = enc1_out

        dec1_concat = torch.cat([dec1_up, enc1_fused], dim=1)
        if self.use_gradient_checkpointing and self.training:
            dec1_out = checkpoint(self.dec1, dec1_concat)
        else:
            dec1_out = self.dec1(dec1_concat)

        # Final output
        output = self.final_conv(dec1_out)  # (B, num_classes, D, H, W)

        return output

    def get_memory_usage(self, batch_size=1):
        """
        Estimate memory usage for the model.

        Parameters:
        -----------
        batch_size : int, default=1
            Batch size for estimation

        Returns:
        --------
        dict: Memory usage estimates in MB
        """
        d, h, w = self.input_size

        # Calculate feature map sizes at each level
        sizes = {
            "input": (batch_size, self.input_channels, d, h, w),
            "enc1": (batch_size, self.base_channels, d, h, w),
            "enc2": (batch_size, self.base_channels * 2, d // 2, h // 2, w // 2),
            "enc3": (batch_size, self.base_channels * 4, d // 4, h // 4, w // 4),
            "enc4": (batch_size, self.base_channels * 8, d // 8, h // 8, w // 8),
            "bottleneck": (
                batch_size,
                self.base_channels * 16,
                d // 16,
                h // 16,
                w // 16,
            ),
        }

        memory_mb = {}
        for name, size in sizes.items():
            elements = np.prod(size)
            memory_mb[name] = elements * 4 / (1024 * 1024)  # 4 bytes per float32

        memory_mb["total_activations"] = sum(memory_mb.values())

        # Model parameters
        total_params = sum(p.numel() for p in self.parameters())
        memory_mb["parameters"] = total_params * 4 / (1024 * 1024)

        memory_mb["total_estimated"] = (
            memory_mb["total_activations"] + memory_mb["parameters"]
        )

        return memory_mb


class DINOv3UNet3DPipeline(nn.Module):
    """
    Complete pipeline for 3D volumetric segmentation using DINOv3 features.
    Handles 3D volume processing, feature extraction, and segmentation.
    """

    def __init__(
        self,
        num_classes=2,
        input_size=(112, 112, 112),
        dinov3_slice_size=896,
        base_channels=32,
        input_channels=1024,  # DINOv3 ViT-L/16 features = 1024 channels
        use_orthogonal_planes=True,  # Process all 3 orthogonal planes
        device=None,
        verbose=True,  # Control verbose output
    ):
        """
        Initialize the complete 3D DINOv3-UNet pipeline.

        Parameters:
        -----------
        num_classes : int, default=2
            Number of output classes
        input_size : tuple, default=(112, 112, 112)
            Input volume dimensions (D, H, W)
        dinov3_slice_size : int, default=896
            Size for DINOv3 processing per slice (should be multiple of 16)
        base_channels : int, default=32
            Base number of channels in 3D UNet
        input_channels : int, optional
            Number of input channels for the 3D UNet (DINOv3 feature channels)
        use_orthogonal_planes : bool, default=True
            If True, processes slices in all 3 orthogonal planes (XY, XZ, YZ) and averages them.
            If False, processes only XY planes (original behavior).
        device : torch.device, optional
            Device for processing
        """
        super(DINOv3UNet3DPipeline, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.dinov3_slice_size = dinov3_slice_size
        self.verbose = verbose
        if self.verbose:
            print(
                f"DINOv3UNet3DPipeline initialized with dinov3_slice_size: {self.dinov3_slice_size}"
            )
        self.use_orthogonal_planes = use_orthogonal_planes
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Determine input channels (DINOv3 feature channels)
        if input_channels is None:
            # Fall back to getting from dinov3_core if not provided
            from .dinov3_core import get_current_model_info

            model_info = get_current_model_info()
            input_channels = model_info["output_channels"]

        # 3D UNet for processing DINOv3 features
        self.unet3d = DINOv3UNet3D(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            input_size=input_size,
        )

    def extract_dinov3_features_3d(
        self,
        volume,
        use_orthogonal_planes=None,
        enable_timing=False,
        target_output_size=None,
    ):
        """
        Extract DINOv3 features from a 3D volume by processing slices in orthogonal planes.

        Parameters:
        -----------
        volume : numpy.ndarray or torch.Tensor
            Volume of shape (batch_size, D, H, W) or (D, H, W)
        use_orthogonal_planes : bool, optional
            If True, processes slices in all 3 orthogonal planes (XY, XZ, YZ) and averages them.
            If False, uses only XY planes (original behavior).
            If None (default), uses the instance's use_orthogonal_planes setting.
        target_output_size : tuple, optional
            Target size (D, H, W) for the output features. If provided, features will be
            processed at the input volume's native resolution but downsampled to this size.
            If None, uses self.input_size.

        Returns:
        --------
        torch.Tensor: DINOv3 features (batch_size, output_channels, D, H, W)
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().numpy()

        # Handle single volume vs batch
        if volume.ndim == 3:
            volume = volume[np.newaxis, ...]  # Add batch dimension

        batch_size, depth, height, width = volume.shape

        # Determine processing size (native resolution) and output size
        processing_d, processing_h, processing_w = depth, height, width

        if target_output_size is not None:
            output_d, output_h, output_w = target_output_size
        else:
            output_d, output_h, output_w = self.input_size

        # Initialize detailed timing and debugging
        import time

        detailed_timing = {}
        start_time = time.time()

        if enable_timing:
            print(f"\n=== DINOv3 Multi-Resolution Processing Debug ===")
            print(f"Input volume: {(depth, height, width)} (batch_size={batch_size})")
            print(
                f"Processing at: {(processing_d, processing_h, processing_w)} (native resolution)"
            )
            print(f"Output target: {(output_d, output_h, output_w)}")
            print(f"Input dtype: {volume.dtype}")
            print(f"Input memory: {volume.nbytes / 1e6:.2f} MB")
        else:
            if self.verbose:
                print(f"Multi-resolution DINOv3 processing:")
                print(f"  Input volume: {(depth, height, width)}")
                print(
                    f"  Processing at: {(processing_d, processing_h, processing_w)} (native resolution)"
                )
                print(f"  Output target: {(output_d, output_h, output_w)}")

        # Use volume at native resolution - selective downsampling will happen per plane
        processing_volume = volume

        if enable_timing:
            print(f"\n--- Plane-Specific Selective Downsampling Strategy ---")
            print(
                "Each orthogonal plane will selectively downsample its non-spatial dimension:"
            )
            print(
                f"  XY plane: downsample Z {processing_d}→{output_d}, keep XY {processing_h}×{processing_w}"
            )
            print(
                f"  XZ plane: downsample Y {processing_h}→{output_h}, keep XZ {processing_d}×{processing_w}"
            )
            print(
                f"  YZ plane: downsample X {processing_w}→{output_w}, keep YZ {processing_d}×{processing_h}"
            )
            print(
                "Reason: Preserve high spatial resolution for DINOv3 while reducing slice count"
            )

        if enable_timing:
            detailed_timing["setup_time"] = time.time() - start_time

        # Use instance variable if parameter not provided
        if use_orthogonal_planes is None:
            use_orthogonal_planes = self.use_orthogonal_planes

        # Clear previous timing info if enabling timing
        if enable_timing:
            self._plane_timing_info = {}

        if use_orthogonal_planes:
            # Process all 3 orthogonal planes and average them
            plane_features = []
            plane_timings = {}

            if enable_timing:
                print(f"\n--- Processing 3 Orthogonal Planes ---")

            # XY planes (slice along Z-axis) - original implementation
            plane_start = time.time()
            xy_features = self._extract_features_plane(
                processing_volume,
                "xy",
                processing_d,
                processing_h,
                processing_w,
                slice_batch_size=512,
                enable_timing=enable_timing,
                output_d=output_d,
                output_h=output_h,
                output_w=output_w,
            )
            plane_timings["xy_extraction"] = time.time() - plane_start

            if enable_timing:
                print(
                    f"XY features extracted and downsampled: {tuple(xy_features.shape)} in {plane_timings['xy_extraction']:.3f}s"
                )

            plane_features.append(xy_features)

            # XZ planes (slice along Y-axis)
            plane_start = time.time()
            xz_features = self._extract_features_plane(
                processing_volume,
                "xz",
                processing_d,
                processing_h,
                processing_w,
                slice_batch_size=512,
                enable_timing=enable_timing,
                output_d=output_d,
                output_h=output_h,
                output_w=output_w,
            )
            plane_timings["xz_extraction"] = time.time() - plane_start

            if enable_timing:
                print(
                    f"XZ features extracted and downsampled: {tuple(xz_features.shape)} in {plane_timings['xz_extraction']:.3f}s"
                )

            plane_features.append(xz_features)

            # YZ planes (slice along X-axis)
            plane_start = time.time()
            yz_features = self._extract_features_plane(
                processing_volume,
                "yz",
                processing_d,
                processing_h,
                processing_w,
                slice_batch_size=512,
                enable_timing=enable_timing,
                output_d=output_d,
                output_h=output_h,
                output_w=output_w,
            )
            plane_timings["yz_extraction"] = time.time() - plane_start

            if enable_timing:
                print(
                    f"YZ features extracted and downsampled: {tuple(yz_features.shape)} in {plane_timings['yz_extraction']:.3f}s"
                )

            plane_features.append(yz_features)

            # Enhanced debugging for orthogonal processing
            if enable_timing:
                print(f"\n--- Averaging Orthogonal Planes ---")
                print(f"XY features after downsampling: {tuple(xy_features.shape)}")
                print(f"XZ features after downsampling: {tuple(xz_features.shape)}")
                print(f"YZ features after downsampling: {tuple(yz_features.shape)}")
            else:
                if self.verbose:
                    print(f"Orthogonal plane feature shapes after downsampling:")
                    print(f"  XY features: {xy_features.shape}")
                    print(f"  XZ features: {xz_features.shape}")
                    print(f"  YZ features: {yz_features.shape}")

            # Ensure all plane features are on the same device before averaging
            if enable_timing:
                print("  Checking device consistency...")
                for i, feat in enumerate(["XY", "XZ", "YZ"]):
                    print(f"    {feat} features device: {plane_features[i].device}")

            # Move all features to the same device (preferably GPU if available)
            target_device = plane_features[0].device
            for i in range(len(plane_features)):
                if plane_features[i].device != target_device:
                    plane_features[i] = plane_features[i].to(target_device)
                    if enable_timing:
                        print(f"    Moved plane {i} to {target_device}")

            # Now we can safely average the features from all three planes
            averaging_start = time.time()
            batch_features = torch.stack(plane_features, dim=0).mean(dim=0)
            plane_timings["averaging"] = time.time() - averaging_start

            # Immediately free memory from individual plane features
            del plane_features, xy_features, xz_features, yz_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if enable_timing:
                print(
                    f"Averaged features: {tuple(batch_features.shape)} in {plane_timings['averaging']:.3f}s"
                )
                print("  Individual plane features freed to save memory")
                # Store plane timings in detailed timing
                detailed_timing.update(plane_timings)
            else:
                if self.verbose:
                    print(f"  Averaged features: {batch_features.shape}")
                    print("  Individual plane features freed to save memory")

        else:
            # Original behavior - only XY planes
            if enable_timing:
                print(f"\n--- Processing Single XY Plane ---")

            single_plane_start = time.time()
            batch_features = self._extract_features_plane(
                processing_volume,
                "xy",
                processing_d,
                processing_h,
                processing_w,
                slice_batch_size=512,
                enable_timing=enable_timing,
            )
            detailed_timing["xy_extraction"] = time.time() - single_plane_start

            if enable_timing:
                print(
                    f"XY features extracted: {tuple(batch_features.shape)} in {detailed_timing['xy_extraction']:.3f}s"
                )

            # Downsample features to target output size if different from processing size
            downsample_start = time.time()
            if (processing_d, processing_h, processing_w) != (
                output_d,
                output_h,
                output_w,
            ):
                if enable_timing:
                    print(
                        f"  Downsampling XY: {tuple(batch_features.shape)} → target {(batch_features.shape[0], batch_features.shape[1], output_d, output_h, output_w)}"
                    )
                else:
                    print(
                        f"Downsampling XY features from {(processing_d, processing_h, processing_w)} to {(output_d, output_h, output_w)}"
                    )

                batch_features = torch.nn.functional.interpolate(
                    batch_features,
                    size=(output_d, output_h, output_w),
                    mode="trilinear",
                    align_corners=False,
                )

                if enable_timing:
                    print(f"  XY downsampled to: {tuple(batch_features.shape)}")
            detailed_timing["single_plane_downsample"] = time.time() - downsample_start

        # Final device transfer and timing summary
        device_transfer_start = time.time()
        result = batch_features.to(self.device)
        detailed_timing["device_transfer"] = time.time() - device_transfer_start
        detailed_timing["total_time"] = time.time() - start_time

        if enable_timing:
            print(f"\n=== DINOv3 Processing Summary ===")
            print(f"Total time: {detailed_timing['total_time']:.3f}s")
            print(f"Final output shape: {tuple(result.shape)}")
            print(
                f"Output memory: {result.element_size() * result.nelement() / 1e6:.2f} MB"
            )
            print(f"Device: {result.device}")

            # Add detailed timing to the aggregated timing info
            if hasattr(self, "_get_aggregated_timing_info"):
                aggregated_timing = self._get_aggregated_timing_info()
                aggregated_timing.update(detailed_timing)
                return result, aggregated_timing
            else:
                return result, detailed_timing
        else:
            return result

    def extract_dinov3_features_3d_batch(
        self,
        volume_batch,
        use_orthogonal_planes=None,
        enable_timing=False,
        target_output_size=None,
    ):
        """
        GPU-optimized batch processing of multiple 3D volumes simultaneously.
        This method processes all volumes in the batch together for maximum efficiency.

        Parameters:
        -----------
        volume_batch : numpy.ndarray
            Batch of volumes of shape (batch_size, D, H, W)
        use_orthogonal_planes : bool, optional
            If True, processes slices in all 3 orthogonal planes (XY, XZ, YZ) and averages them.
        target_output_size : tuple, optional
            Target size (D, H, W) for the output features.

        Returns:
        --------
        torch.Tensor: DINOv3 features (batch_size, output_channels, D, H, W)
        """
        if isinstance(volume_batch, torch.Tensor):
            volume_batch = volume_batch.cpu().numpy()

        batch_size, depth, height, width = volume_batch.shape

        # Determine processing size and output size
        if target_output_size is not None:
            output_d, output_h, output_w = target_output_size
        else:
            output_d, output_h, output_w = self.input_size

        import time

        detailed_timing = {}
        start_time = time.time()

        if enable_timing:
            print(f"\n=== GPU-Optimized Batch DINOv3 Processing ===")
            print(f"Batch size: {batch_size}")
            print(f"Input volume per batch: {(depth, height, width)}")
            print(f"Output target: {(output_d, output_h, output_w)}")

        # Use instance variable if parameter not provided
        if use_orthogonal_planes is None:
            use_orthogonal_planes = self.use_orthogonal_planes

        # Pre-allocate output tensor on GPU for maximum efficiency
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from .dinov3_core import get_current_model_info

        model_info = get_current_model_info()
        output_channels = model_info["output_channels"]

        batch_output_shape = (batch_size, output_channels, output_d, output_h, output_w)
        batch_features = torch.zeros(
            batch_output_shape, device=device, dtype=torch.float32
        )

        if use_orthogonal_planes:
            # GPU-optimized batch processing for all 3 planes
            plane_features = []
            plane_timings = {}

            if enable_timing:
                print(f"Processing {batch_size} volumes across 3 orthogonal planes...")

            # Process all volumes for each plane type
            for plane_name in ["xy", "xz", "yz"]:
                plane_start = time.time()

                # Process all volumes in batch for this plane
                plane_batch_features = torch.zeros(
                    batch_output_shape, device=device, dtype=torch.float32
                )

                for b in range(batch_size):
                    single_volume_features = self._extract_features_plane(
                        volume_batch[b : b + 1],
                        plane_name,
                        depth,
                        height,
                        width,
                        slice_batch_size=512,
                        enable_timing=False,
                        output_d=output_d,
                        output_h=output_h,
                        output_w=output_w,
                    )

                    # Direct GPU assignment
                    if (
                        single_volume_features.dim() == 5
                        and single_volume_features.shape[0] == 1
                    ):
                        plane_batch_features[b] = single_volume_features.squeeze(0).to(
                            device
                        )
                    else:
                        plane_batch_features[b] = single_volume_features.to(device)

                plane_features.append(plane_batch_features)
                plane_timings[f"{plane_name}_extraction"] = time.time() - plane_start

                if enable_timing:
                    print(
                        f"  {plane_name.upper()} plane batch: {tuple(plane_batch_features.shape)} in {plane_timings[f'{plane_name}_extraction']:.3f}s"
                    )

            # GPU-accelerated averaging across all planes
            averaging_start = time.time()
            # Stack all plane features: (3, batch_size, channels, D, H, W)
            all_planes_stacked = torch.stack(plane_features, dim=0)
            # Average across planes: (batch_size, channels, D, H, W)
            batch_features = all_planes_stacked.mean(dim=0)
            plane_timings["averaging"] = time.time() - averaging_start

            # Free memory
            del plane_features, all_planes_stacked
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if enable_timing:
                print(
                    f"  Averaged all planes: {tuple(batch_features.shape)} in {plane_timings['averaging']:.3f}s"
                )
                detailed_timing.update(plane_timings)

        else:
            # Single XY plane batch processing
            if enable_timing:
                print(f"Processing {batch_size} volumes for XY plane only...")

            xy_start = time.time()
            for b in range(batch_size):
                single_features = self._extract_features_plane(
                    volume_batch[b : b + 1],
                    "xy",
                    depth,
                    height,
                    width,
                    slice_batch_size=512,
                    enable_timing=False,
                    output_d=output_d,
                    output_h=output_h,
                    output_w=output_w,
                )

                if single_features.dim() == 5 and single_features.shape[0] == 1:
                    batch_features[b] = single_features.squeeze(0).to(device)
                else:
                    batch_features[b] = single_features.to(device)

            detailed_timing["xy_batch_extraction"] = time.time() - xy_start

        # Final timing
        detailed_timing["total_batch_time"] = time.time() - start_time
        detailed_timing["batch_size"] = batch_size
        detailed_timing["processing_method"] = "gpu_batch_optimized"

        if enable_timing:
            print(f"\n=== Batch Processing Summary ===")
            print(f"Total batch time: {detailed_timing['total_batch_time']:.3f}s")
            print(
                f"Average time per volume: {detailed_timing['total_batch_time']/batch_size:.3f}s"
            )
            print(f"Final batch shape: {tuple(batch_features.shape)}")
            print(
                f"GPU memory used: {batch_features.element_size() * batch_features.nelement() / 1e6:.2f} MB"
            )

            return batch_features, detailed_timing
        else:
            return batch_features

    def extract_dinov3_features_3d(
        self,
        volume,
        use_orthogonal_planes=None,
        enable_timing=False,
        target_output_size=None,
    ):
        """
        Extract DINOv3 features from a 3D volume by processing slices in orthogonal planes.

        Parameters:
        -----------
        volume : numpy.ndarray or torch.Tensor
            Volume of shape (batch_size, D, H, W) or (D, H, W)
        use_orthogonal_planes : bool, optional
            If True, processes slices in all 3 orthogonal planes (XY, XZ, YZ) and averages them.
            If False, uses only XY planes (original behavior).
            If None (default), uses the instance's use_orthogonal_planes setting.
        target_output_size : tuple, optional
            Target size (D, H, W) for the output features. If provided, features will be
            processed at the input volume's native resolution but downsampled to this size.
            If None, uses self.input_size.

        Returns:
        --------
        torch.Tensor: DINOv3 features (batch_size, output_channels, D, H, W)
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().numpy()

        # Handle single volume vs batch
        if volume.ndim == 3:
            volume = volume[np.newaxis, ...]  # Add batch dimension

        batch_size, depth, height, width = volume.shape

        # Determine processing size (native resolution) and output size
        processing_d, processing_h, processing_w = depth, height, width

        if target_output_size is not None:
            output_d, output_h, output_w = target_output_size
        else:
            output_d, output_h, output_w = self.input_size

        # Initialize detailed timing and debugging
        import time

        detailed_timing = {}
        start_time = time.time()

        if enable_timing:
            print(f"\n=== DINOv3 Multi-Resolution Processing Debug ===")
            print(f"Input volume: {(depth, height, width)} (batch_size={batch_size})")
            print(
                f"Processing at: {(processing_d, processing_h, processing_w)} (native resolution)"
            )
            print(f"Output target: {(output_d, output_h, output_w)}")
            print(f"Input dtype: {volume.dtype}")
            print(f"Input memory: {volume.nbytes / 1e6:.2f} MB")
        else:
            if self.verbose:
                print(f"Multi-resolution DINOv3 processing:")
                print(f"  Input volume: {(depth, height, width)}")
                print(
                    f"  Processing at: {(processing_d, processing_h, processing_w)} (native resolution)"
                )
                print(f"  Output target: {(output_d, output_h, output_w)}")

        # Use volume at native resolution - selective downsampling will happen per plane
        processing_volume = volume

        if enable_timing:
            print(f"\n--- Plane-Specific Selective Downsampling Strategy ---")
            print(
                "Each orthogonal plane will selectively downsample its non-spatial dimension:"
            )
            print(
                f"  XY plane: downsample Z {processing_d}→{output_d}, keep XY {processing_h}×{processing_w}"
            )
            print(
                f"  XZ plane: downsample Y {processing_h}→{output_h}, keep XZ {processing_d}×{processing_w}"
            )
            print(
                f"  YZ plane: downsample X {processing_w}→{output_w}, keep YZ {processing_d}×{processing_h}"
            )
            print(
                "Reason: Preserve high spatial resolution for DINOv3 while reducing slice count"
            )

        if enable_timing:
            detailed_timing["setup_time"] = time.time() - start_time

        # Use instance variable if parameter not provided
        if use_orthogonal_planes is None:
            use_orthogonal_planes = self.use_orthogonal_planes

        # Clear previous timing info if enabling timing
        if enable_timing:
            self._plane_timing_info = {}

        if use_orthogonal_planes:
            # Process all 3 orthogonal planes and average them
            plane_features = []
            plane_timings = {}

            if enable_timing:
                print(f"\n--- Processing 3 Orthogonal Planes ---")

            # XY planes (slice along Z-axis) - original implementation
            plane_start = time.time()
            xy_features = self._extract_features_plane(
                processing_volume,
                "xy",
                processing_d,
                processing_h,
                processing_w,
                slice_batch_size=512,
                enable_timing=enable_timing,
                output_d=output_d,
                output_h=output_h,
                output_w=output_w,
            )
            plane_timings["xy_extraction"] = time.time() - plane_start

            if enable_timing:
                print(
                    f"XY features extracted and downsampled: {tuple(xy_features.shape)} in {plane_timings['xy_extraction']:.3f}s"
                )

            plane_features.append(xy_features)

            # XZ planes (slice along Y-axis)
            plane_start = time.time()
            xz_features = self._extract_features_plane(
                processing_volume,
                "xz",
                processing_d,
                processing_h,
                processing_w,
                slice_batch_size=512,
                enable_timing=enable_timing,
                output_d=output_d,
                output_h=output_h,
                output_w=output_w,
            )
            plane_timings["xz_extraction"] = time.time() - plane_start

            if enable_timing:
                print(
                    f"XZ features extracted and downsampled: {tuple(xz_features.shape)} in {plane_timings['xz_extraction']:.3f}s"
                )

            plane_features.append(xz_features)

            # YZ planes (slice along X-axis)
            plane_start = time.time()
            yz_features = self._extract_features_plane(
                processing_volume,
                "yz",
                processing_d,
                processing_h,
                processing_w,
                slice_batch_size=512,
                enable_timing=enable_timing,
                output_d=output_d,
                output_h=output_h,
                output_w=output_w,
            )
            plane_timings["yz_extraction"] = time.time() - plane_start

            if enable_timing:
                print(
                    f"YZ features extracted and downsampled: {tuple(yz_features.shape)} in {plane_timings['yz_extraction']:.3f}s"
                )

            plane_features.append(yz_features)

            # Enhanced debugging for orthogonal processing
            if enable_timing:
                print(f"\n--- Averaging Orthogonal Planes ---")
                print(f"XY features after downsampling: {tuple(xy_features.shape)}")
                print(f"XZ features after downsampling: {tuple(xz_features.shape)}")
                print(f"YZ features after downsampling: {tuple(yz_features.shape)}")
            else:
                if self.verbose:
                    print(f"Orthogonal plane feature shapes after downsampling:")
                    print(f"  XY features: {xy_features.shape}")
                    print(f"  XZ features: {xz_features.shape}")
                    print(f"  YZ features: {yz_features.shape}")

            # Ensure all plane features are on the same device before averaging
            if enable_timing:
                print("  Checking device consistency...")
                for i, feat in enumerate(["XY", "XZ", "YZ"]):
                    print(f"    {feat} features device: {plane_features[i].device}")

            # Move all features to the same device (preferably GPU if available)
            target_device = plane_features[0].device
            for i in range(len(plane_features)):
                if plane_features[i].device != target_device:
                    plane_features[i] = plane_features[i].to(target_device)
                    if enable_timing:
                        print(f"    Moved plane {i} to {target_device}")

            # Now we can safely average the features from all three planes
            averaging_start = time.time()
            batch_features = torch.stack(plane_features, dim=0).mean(dim=0)
            plane_timings["averaging"] = time.time() - averaging_start

            # Immediately free memory from individual plane features
            del plane_features, xy_features, xz_features, yz_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if enable_timing:
                print(
                    f"Averaged features: {tuple(batch_features.shape)} in {plane_timings['averaging']:.3f}s"
                )
                print("  Individual plane features freed to save memory")
                # Store plane timings in detailed timing
                detailed_timing.update(plane_timings)
            else:
                if self.verbose:
                    print(f"  Averaged features: {batch_features.shape}")
                    print("  Individual plane features freed to save memory")

        else:
            # Original behavior - only XY planes
            if enable_timing:
                print(f"\n--- Processing Single XY Plane ---")

            single_plane_start = time.time()
            batch_features = self._extract_features_plane(
                processing_volume,
                "xy",
                processing_d,
                processing_h,
                processing_w,
                slice_batch_size=512,
                enable_timing=enable_timing,
            )
            detailed_timing["xy_extraction"] = time.time() - single_plane_start

            if enable_timing:
                print(
                    f"XY features extracted: {tuple(batch_features.shape)} in {detailed_timing['xy_extraction']:.3f}s"
                )

            # Downsample features to target output size if different from processing size
            downsample_start = time.time()
            if (processing_d, processing_h, processing_w) != (
                output_d,
                output_h,
                output_w,
            ):
                if enable_timing:
                    print(
                        f"  Downsampling XY: {tuple(batch_features.shape)} → target {(batch_features.shape[0], batch_features.shape[1], output_d, output_h, output_w)}"
                    )
                else:
                    print(
                        f"Downsampling XY features from {(processing_d, processing_h, processing_w)} to {(output_d, output_h, output_w)}"
                    )

                batch_features = torch.nn.functional.interpolate(
                    batch_features,
                    size=(output_d, output_h, output_w),
                    mode="trilinear",
                    align_corners=False,
                )

                if enable_timing:
                    print(f"  XY downsampled to: {tuple(batch_features.shape)}")
            detailed_timing["single_plane_downsample"] = time.time() - downsample_start

        # Final device transfer and timing summary
        device_transfer_start = time.time()
        result = batch_features.to(self.device)
        detailed_timing["device_transfer"] = time.time() - device_transfer_start
        detailed_timing["total_time"] = time.time() - start_time

        if enable_timing:
            print(f"\n=== DINOv3 Processing Summary ===")
            print(f"Total time: {detailed_timing['total_time']:.3f}s")
            print(f"Final output shape: {tuple(result.shape)}")
            print(
                f"Output memory: {result.element_size() * result.nelement() / 1e6:.2f} MB"
            )
            print(f"Device: {result.device}")

            # Add detailed timing to the aggregated timing info
            if hasattr(self, "_get_aggregated_timing_info"):
                aggregated_timing = self._get_aggregated_timing_info()
                aggregated_timing.update(detailed_timing)
                return result, aggregated_timing
            else:
                return result, detailed_timing
        else:
            return result

    def extract_dinov3_features_3d_batch(
        self,
        volume_batch,
        use_orthogonal_planes=None,
        enable_timing=False,
        target_output_size=None,
    ):
        """
        GPU-optimized batch processing of multiple 3D volumes simultaneously.
        This method processes all volumes in the batch together for maximum efficiency.

        Parameters:
        -----------
        volume_batch : numpy.ndarray
            Batch of volumes of shape (batch_size, D, H, W)
        use_orthogonal_planes : bool, optional
            If True, processes slices in all 3 orthogonal planes (XY, XZ, YZ) and averages them.
        target_output_size : tuple, optional
            Target size (D, H, W) for the output features.

        Returns:
        --------
        torch.Tensor: DINOv3 features (batch_size, output_channels, D, H, W)
        """
        if isinstance(volume_batch, torch.Tensor):
            volume_batch = volume_batch.cpu().numpy()

        batch_size, depth, height, width = volume_batch.shape

        # Determine processing size and output size
        if target_output_size is not None:
            output_d, output_h, output_w = target_output_size
        else:
            output_d, output_h, output_w = self.input_size

        import time

        detailed_timing = {}
        start_time = time.time()

        if enable_timing:
            print(f"\n=== GPU-Optimized Batch DINOv3 Processing ===")
            print(f"Batch size: {batch_size}")
            print(f"Input volume per batch: {(depth, height, width)}")
            print(f"Output target: {(output_d, output_h, output_w)}")

        # Use instance variable if parameter not provided
        if use_orthogonal_planes is None:
            use_orthogonal_planes = self.use_orthogonal_planes

        # Pre-allocate output tensor on GPU for maximum efficiency
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from .dinov3_core import get_current_model_info

        model_info = get_current_model_info()
        output_channels = model_info["output_channels"]

        batch_output_shape = (batch_size, output_channels, output_d, output_h, output_w)
        batch_features = torch.zeros(
            batch_output_shape, device=device, dtype=torch.float32
        )

        if use_orthogonal_planes:
            # GPU-optimized batch processing for all 3 planes
            plane_features = []
            plane_timings = {}

            if enable_timing:
                print(f"Processing {batch_size} volumes across 3 orthogonal planes...")

            # Process all volumes for each plane type
            for plane_name in ["xy", "xz", "yz"]:
                plane_start = time.time()

                # Process all volumes in batch for this plane
                plane_batch_features = torch.zeros(
                    batch_output_shape, device=device, dtype=torch.float32
                )

                for b in range(batch_size):
                    single_volume_features = self._extract_features_plane(
                        volume_batch[b : b + 1],
                        plane_name,
                        depth,
                        height,
                        width,
                        slice_batch_size=512,
                        enable_timing=False,
                        output_d=output_d,
                        output_h=output_h,
                        output_w=output_w,
                    )

                    # Direct GPU assignment
                    if (
                        single_volume_features.dim() == 5
                        and single_volume_features.shape[0] == 1
                    ):
                        plane_batch_features[b] = single_volume_features.squeeze(0).to(
                            device
                        )
                    else:
                        plane_batch_features[b] = single_volume_features.to(device)

                plane_features.append(plane_batch_features)
                plane_timings[f"{plane_name}_extraction"] = time.time() - plane_start

                if enable_timing:
                    print(
                        f"  {plane_name.upper()} plane batch: {tuple(plane_batch_features.shape)} in {plane_timings[f'{plane_name}_extraction']:.3f}s"
                    )

            # GPU-accelerated averaging across all planes
            averaging_start = time.time()
            # Stack all plane features: (3, batch_size, channels, D, H, W)
            all_planes_stacked = torch.stack(plane_features, dim=0)
            # Average across planes: (batch_size, channels, D, H, W)
            batch_features = all_planes_stacked.mean(dim=0)
            plane_timings["averaging"] = time.time() - averaging_start

            # Free memory
            del plane_features, all_planes_stacked
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if enable_timing:
                print(
                    f"  Averaged all planes: {tuple(batch_features.shape)} in {plane_timings['averaging']:.3f}s"
                )
                detailed_timing.update(plane_timings)

        else:
            # Single XY plane batch processing
            if enable_timing:
                print(f"Processing {batch_size} volumes for XY plane only...")

            xy_start = time.time()
            for b in range(batch_size):
                single_features = self._extract_features_plane(
                    volume_batch[b : b + 1],
                    "xy",
                    depth,
                    height,
                    width,
                    slice_batch_size=512,
                    enable_timing=False,
                    output_d=output_d,
                    output_h=output_h,
                    output_w=output_w,
                )

                if single_features.dim() == 5 and single_features.shape[0] == 1:
                    batch_features[b] = single_features.squeeze(0).to(device)
                else:
                    batch_features[b] = single_features.to(device)

            detailed_timing["xy_batch_extraction"] = time.time() - xy_start

        # Final timing
        detailed_timing["total_batch_time"] = time.time() - start_time
        detailed_timing["batch_size"] = batch_size
        detailed_timing["processing_method"] = "gpu_batch_optimized"

        if enable_timing:
            print(f"\n=== Batch Processing Summary ===")
            print(f"Total batch time: {detailed_timing['total_batch_time']:.3f}s")
            print(
                f"Average time per volume: {detailed_timing['total_batch_time']/batch_size:.3f}s"
            )
            print(f"Final batch shape: {tuple(batch_features.shape)}")
            print(
                f"GPU memory used: {batch_features.element_size() * batch_features.nelement() / 1e6:.2f} MB"
            )

            return batch_features, detailed_timing
        else:
            return batch_features

    def extract_dinov3_features_3d(
        self,
        volume,
        use_orthogonal_planes=None,
        enable_timing=False,
        target_output_size=None,
    ):
        """
        Extract DINOv3 features from a 3D volume by processing slices in orthogonal planes.

        Parameters:
        -----------
        volume : numpy.ndarray or torch.Tensor
            Volume of shape (batch_size, D, H, W) or (D, H, W)
        use_orthogonal_planes : bool, optional
            If True, processes slices in all 3 orthogonal planes (XY, XZ, YZ) and averages them.
            If False, uses only XY planes (original behavior).
            If None (default), uses the instance's use_orthogonal_planes setting.
        target_output_size : tuple, optional
            Target size (D, H, W) for the output features. If provided, features will be
            processed at the input volume's native resolution but downsampled to this size.
            If None, uses self.input_size.

        Returns:
        --------
        torch.Tensor: DINOv3 features (batch_size, output_channels, D, H, W)
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().numpy()

        # Handle single volume vs batch
        if volume.ndim == 3:
            volume = volume[np.newaxis, ...]  # Add batch dimension

        batch_size, depth, height, width = volume.shape

        # Determine processing size (native resolution) and output size
        processing_d, processing_h, processing_w = depth, height, width

        if target_output_size is not None:
            output_d, output_h, output_w = target_output_size
        else:
            output_d, output_h, output_w = self.input_size

        # Initialize detailed timing and debugging
        import time

        detailed_timing = {}
        start_time = time.time()

        if enable_timing:
            print(f"\n=== DINOv3 Multi-Resolution Processing Debug ===")
            print(f"Input volume: {(depth, height, width)} (batch_size={batch_size})")
            print(
                f"Processing at: {(processing_d, processing_h, processing_w)} (native resolution)"
            )
            print(f"Output target: {(output_d, output_h, output_w)}")
            print(f"Input dtype: {volume.dtype}")
            print(f"Input memory: {volume.nbytes / 1e6:.2f} MB")
        else:
            if self.verbose:
                print(f"Multi-resolution DINOv3 processing:")
                print(f"  Input volume: {(depth, height, width)}")
                print(
                    f"  Processing at: {(processing_d, processing_h, processing_w)} (native resolution)"
                )
                print(f"  Output target: {(output_d, output_h, output_w)}")

        # Use volume at native resolution - selective downsampling will happen per plane
        processing_volume = volume

        if enable_timing:
            print(f"\n--- Plane-Specific Selective Downsampling Strategy ---")
            print(
                "Each orthogonal plane will selectively downsample its non-spatial dimension:"
            )
            print(
                f"  XY plane: downsample Z {processing_d}→{output_d}, keep XY {processing_h}×{processing_w}"
            )
            print(
                f"  XZ plane: downsample Y {processing_h}→{output_h}, keep XZ {processing_d}×{processing_w}"
            )
            print(
                f"  YZ plane: downsample X {processing_w}→{output_w}, keep YZ {processing_d}×{processing_h}"
            )
            print(
                "Reason: Preserve high spatial resolution for DINOv3 while reducing slice count"
            )

        if enable_timing:
            detailed_timing["setup_time"] = time.time() - start_time

        # Use instance variable if parameter not provided
        if use_orthogonal_planes is None:
            use_orthogonal_planes = self.use_orthogonal_planes

        # Clear previous timing info if enabling timing
        if enable_timing:
            self._plane_timing_info = {}

        if use_orthogonal_planes:
            # Process all 3 orthogonal planes and average them
            plane_features = []
            plane_timings = {}

            if enable_timing:
                print(f"\n--- Processing 3 Orthogonal Planes ---")

            # XY planes (slice along Z-axis) - original implementation
            plane_start = time.time()
            xy_features = self._extract_features_plane(
                processing_volume,
                "xy",
                processing_d,
                processing_h,
                processing_w,
                slice_batch_size=512,
                enable_timing=enable_timing,
                output_d=output_d,
                output_h=output_h,
                output_w=output_w,
            )
            plane_timings["xy_extraction"] = time.time() - plane_start

            if enable_timing:
                print(
                    f"XY features extracted and downsampled: {tuple(xy_features.shape)} in {plane_timings['xy_extraction']:.3f}s"
                )

            plane_features.append(xy_features)

            # XZ planes (slice along Y-axis)
            plane_start = time.time()
            xz_features = self._extract_features_plane(
                processing_volume,
                "xz",
                processing_d,
                processing_h,
                processing_w,
                slice_batch_size=512,
                enable_timing=enable_timing,
                output_d=output_d,
                output_h=output_h,
                output_w=output_w,
            )
            plane_timings["xz_extraction"] = time.time() - plane_start

            if enable_timing:
                print(
                    f"XZ features extracted and downsampled: {tuple(xz_features.shape)} in {plane_timings['xz_extraction']:.3f}s"
                )

            plane_features.append(xz_features)

            # YZ planes (slice along X-axis)
            plane_start = time.time()
            yz_features = self._extract_features_plane(
                processing_volume,
                "yz",
                processing_d,
                processing_h,
                processing_w,
                slice_batch_size=512,
                enable_timing=enable_timing,
                output_d=output_d,
                output_h=output_h,
                output_w=output_w,
            )
            plane_timings["yz_extraction"] = time.time() - plane_start

            if enable_timing:
                print(
                    f"YZ features extracted and downsampled: {tuple(yz_features.shape)} in {plane_timings['yz_extraction']:.3f}s"
                )

            plane_features.append(yz_features)

            # Enhanced debugging for orthogonal processing
            if enable_timing:
                print(f"\n--- Averaging Orthogonal Planes ---")
                print(f"XY features after downsampling: {tuple(xy_features.shape)}")
                print(f"XZ features after downsampling: {tuple(xz_features.shape)}")
                print(f"YZ features after downsampling: {tuple(yz_features.shape)}")
            else:
                if self.verbose:
                    print(f"Orthogonal plane feature shapes after downsampling:")
                    print(f"  XY features: {xy_features.shape}")
                    print(f"  XZ features: {xz_features.shape}")
                    print(f"  YZ features: {yz_features.shape}")

            # Ensure all plane features are on the same device before averaging
            if enable_timing:
                print("  Checking device consistency...")
                for i, feat in enumerate(["XY", "XZ", "YZ"]):
                    print(f"    {feat} features device: {plane_features[i].device}")

            # Move all features to the same device (preferably GPU if available)
            target_device = plane_features[0].device
            for i in range(len(plane_features)):
                if plane_features[i].device != target_device:
                    plane_features[i] = plane_features[i].to(target_device)
                    if enable_timing:
                        print(f"    Moved plane {i} to {target_device}")

            # Now we can safely average the features from all three planes
            averaging_start = time.time()
            batch_features = torch.stack(plane_features, dim=0).mean(dim=0)
            plane_timings["averaging"] = time.time() - averaging_start

            # Immediately free memory from individual plane features
            del plane_features, xy_features, xz_features, yz_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if enable_timing:
                print(
                    f"Averaged features: {tuple(batch_features.shape)} in {plane_timings['averaging']:.3f}s"
                )
                print("  Individual plane features freed to save memory")
                # Store plane timings in detailed timing
                detailed_timing.update(plane_timings)
            else:
                if self.verbose:
                    print(f"  Averaged features: {batch_features.shape}")
                    print("  Individual plane features freed to save memory")

        else:
            # Original behavior - only XY planes
            if enable_timing:
                print(f"\n--- Processing Single XY Plane ---")

            single_plane_start = time.time()
            batch_features = self._extract_features_plane(
                processing_volume,
                "xy",
                processing_d,
                processing_h,
                processing_w,
                slice_batch_size=512,
                enable_timing=enable_timing,
            )
            detailed_timing["xy_extraction"] = time.time() - single_plane_start

            if enable_timing:
                print(
                    f"XY features extracted: {tuple(batch_features.shape)} in {detailed_timing['xy_extraction']:.3f}s"
                )

            # Downsample features to target output size if different from processing size
            downsample_start = time.time()
            if (processing_d, processing_h, processing_w) != (
                output_d,
                output_h,
                output_w,
            ):
                if enable_timing:
                    print(
                        f"  Downsampling XY: {tuple(batch_features.shape)} → target {(batch_features.shape[0], batch_features.shape[1], output_d, output_h, output_w)}"
                    )
                else:
                    print(
                        f"Downsampling XY features from {(processing_d, processing_h, processing_w)} to {(output_d, output_h, output_w)}"
                    )

                batch_features = torch.nn.functional.interpolate(
                    batch_features,
                    size=(output_d, output_h, output_w),
                    mode="trilinear",
                    align_corners=False,
                )

                if enable_timing:
                    print(f"  XY downsampled to: {tuple(batch_features.shape)}")
            detailed_timing["single_plane_downsample"] = time.time() - downsample_start

        # Final device transfer and timing summary
        device_transfer_start = time.time()
        result = batch_features.to(self.device)
        detailed_timing["device_transfer"] = time.time() - device_transfer_start
        detailed_timing["total_time"] = time.time() - start_time

        if enable_timing:
            print(f"\n=== DINOv3 Processing Summary ===")
            print(f"Total time: {detailed_timing['total_time']:.3f}s")
            print(f"Final output shape: {tuple(result.shape)}")
            print(
                f"Output memory: {result.element_size() * result.nelement() / 1e6:.2f} MB"
            )
            print(f"Device: {result.device}")

            # Add detailed timing to the aggregated timing info
            if hasattr(self, "_get_aggregated_timing_info"):
                aggregated_timing = self._get_aggregated_timing_info()
                aggregated_timing.update(detailed_timing)
                return result, aggregated_timing
            else:
                return result, detailed_timing
        else:
            return result

    def _extract_features_plane(
        self,
        resized_volume,
        plane_type,
        target_d,
        target_h,
        target_w,
        slice_batch_size=512,
        enable_timing=False,
        output_d=None,
        output_h=None,
        output_w=None,
    ):
        """
        Extract DINOv3 features from a specific plane orientation with batching.
        Optionally downsamples immediately to save memory.

        Parameters:
        -----------
        resized_volume : numpy.ndarray
            Volume of shape (batch_size, D, H, W)
        plane_type : str
            'xy', 'xz', or 'yz' indicating which plane to slice
        target_d, target_h, target_w : int
            Processing dimensions for this plane
        slice_batch_size : int, default=1
            Number of slices to process simultaneously for efficiency
        output_d, output_h, output_w : int, optional
            Final output dimensions. If provided, downsamples immediately to save memory

        Returns:
        --------
        torch.Tensor: Features (batch_size, output_channels, D, H, W)
                     At processing size or output size if output dims provided
        """
        volume_batch_size = resized_volume.shape[0]

        # Plane-specific selective downsampling
        # Each plane downsamples the dimension it doesn't use for spatial processing
        downsample_threshold = 2.0

        if output_d is not None and output_h is not None and output_w is not None:
            import time
            from skimage.transform import resize

            # Determine which dimension to downsample for this plane
            if plane_type == "xy":
                # XY plane: downsample Z dimension, keep H×W spatial resolution
                should_downsample = target_d / output_d > downsample_threshold
                target_dims = (
                    (output_d, target_h, target_w)
                    if should_downsample
                    else (target_d, target_h, target_w)
                )
                downsample_dim = "Z"
            elif plane_type == "xz":
                # XZ plane: downsample Y dimension, keep D×W spatial resolution
                should_downsample = target_h / output_h > downsample_threshold
                target_dims = (
                    (target_d, output_h, target_w)
                    if should_downsample
                    else (target_d, target_h, target_w)
                )
                downsample_dim = "Y"
            elif plane_type == "yz":
                # YZ plane: downsample X dimension, keep D×H spatial resolution
                should_downsample = target_w / output_w > downsample_threshold
                target_dims = (
                    (target_d, target_h, output_w)
                    if should_downsample
                    else (target_d, target_h, target_w)
                )
                downsample_dim = "X"

            if should_downsample:
                if enable_timing:
                    print(
                        f"\n--- {plane_type.upper()} Plane Selective Downsampling ---"
                    )
                    print(
                        f"Downsampling {downsample_dim} dimension: {(target_d, target_h, target_w)} → {target_dims}"
                    )

                downsample_start = time.time()

                # Use GPU-accelerated torch interpolation instead of skimage.resize
                import torch.nn.functional as F

                # Convert to torch tensor and move to GPU if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                torch_volume = torch.from_numpy(resized_volume).float().to(device)

                if enable_timing:
                    conversion_time = time.time() - downsample_start
                    print(f"  NumPy → Torch conversion: {conversion_time:.3f}s")
                    torch_start = time.time()

                # Add channel dimension for interpolation: (B, 1, D, H, W)
                torch_volume = torch_volume.unsqueeze(1)

                # GPU-accelerated trilinear interpolation
                torch_downsampled = F.interpolate(
                    torch_volume,
                    size=target_dims,
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(
                    1
                )  # Remove channel dimension

                # Convert back to numpy
                resized_volume = (
                    torch_downsampled.cpu().numpy().astype(resized_volume.dtype)
                )
                target_d, target_h, target_w = target_dims

                if enable_timing:
                    torch_time = time.time() - torch_start
                    print(f"  GPU trilinear interpolation: {torch_time:.3f}s")

                if enable_timing:
                    downsample_time = time.time() - downsample_start
                    print(
                        f"{plane_type.upper()} selective downsampling: {downsample_time:.3f}s"
                    )
                    print(f"New dimensions: {target_dims}")

        all_features = []

        slice_collection_start = time.time()

        for b in range(volume_batch_size):
            # Collect all slices for this volume
            all_slices = []
            slice_dims = []

            # Determine final target dimensions for DINOv3 feature resizing
            if output_d is not None and output_h is not None and output_w is not None:
                # Use final output dimensions for direct DINOv3 feature resizing
                final_d, final_h, final_w = output_d, output_h, output_w
            else:
                # Fall back to processing dimensions
                final_d, final_h, final_w = target_d, target_h, target_w

            if plane_type == "xy":
                # Slice along Z-axis (XY planes) - need final_d slices
                slice_indices = np.linspace(0, target_d - 1, final_d, dtype=int)
                for z_idx in slice_indices:
                    slice_2d = resized_volume[b, z_idx]  # (H, W)
                    all_slices.append(slice_2d)
                    # Use final output dimensions for DINOv3 feature resizing
                    slice_dims.append((final_h, final_w))

            elif plane_type == "xz":
                # Slice along Y-axis (XZ planes) - need final_h slices
                slice_indices = np.linspace(0, target_h - 1, final_h, dtype=int)
                for y_idx in slice_indices:
                    slice_2d = resized_volume[b, :, y_idx, :]  # (D, W)
                    all_slices.append(slice_2d)
                    # Use final output dimensions for DINOv3 feature resizing
                    slice_dims.append((final_d, final_w))

            elif plane_type == "yz":
                # Slice along X-axis (YZ planes) - need final_w slices
                slice_indices = np.linspace(0, target_w - 1, final_w, dtype=int)
                for x_idx in slice_indices:
                    slice_2d = resized_volume[b, :, :, x_idx]  # (D, H)
                    all_slices.append(slice_2d)
                    # Use final output dimensions for DINOv3 feature resizing
                    slice_dims.append((final_d, final_h))

            if enable_timing:
                slice_collection_time = time.time() - slice_collection_start
                print(
                    f"  Slice collection for volume {b+1}: {slice_collection_time:.3f}s ({len(all_slices)} slices)"
                )

            # Process slices in batches
            if enable_timing:
                processing_start = time.time()
                volume_features, batch_timing = self._process_slice_batch(
                    all_slices, slice_dims, slice_batch_size, enable_timing=True
                )
                processing_time = time.time() - processing_start
                print(f"  DINOv3 processing for volume {b+1}: {processing_time:.3f}s")

                # Store timing info for this volume (we'll aggregate later)
                if not hasattr(self, "_plane_timing_info"):
                    self._plane_timing_info = {}
                if plane_type not in self._plane_timing_info:
                    self._plane_timing_info[plane_type] = []
                self._plane_timing_info[plane_type].append(batch_timing)
            else:
                volume_features = self._process_slice_batch(
                    all_slices, slice_dims, slice_batch_size
                )

            # Stack slices and reshape appropriately for each plane type
            # Features are now at final output dimensions (final_d, final_h, final_w)
            if enable_timing:
                stacking_start = time.time()

            if plane_type == "xy":
                # GPU-accelerated stacking for XY plane
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if volume_features[0].device != device:
                    volume_features = [f.to(device) for f in volume_features]
                    if enable_timing:
                        print(f"    {plane_type.upper()} moved to {device}")

                # (output_channels, final_d, final_h, final_w)
                volume_features_3d = torch.stack(volume_features, dim=1)

                if enable_timing:
                    print(
                        f"    {plane_type.upper()} stacking on device: {volume_features_3d.device}"
                    )

            elif plane_type == "xz":
                # GPU-accelerated stacking for XZ plane
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if volume_features[0].device != device:
                    volume_features = [f.to(device) for f in volume_features]
                    if enable_timing:
                        print(f"    {plane_type.upper()} moved to {device}")

                # Stack along H dimension, reshape to (output_channels, final_d, final_h, final_w)
                volume_features_3d = torch.stack(volume_features, dim=2)

                if enable_timing:
                    print(
                        f"    {plane_type.upper()} stacking on device: {volume_features_3d.device}"
                    )
            elif plane_type == "yz":
                # Try a fundamentally different approach: reduce tensor size first
                first_feature = volume_features[0]  # (1024, 128, 128)
                channels, height, width = first_feature.shape
                num_slices = len(volume_features)

                if enable_timing:
                    print(
                        f"    YZ stacking: {num_slices} features of shape {first_feature.shape}"
                    )
                    print(
                        f"    Total memory: {num_slices * channels * height * width * 4 / 1e9:.2f} GB"
                    )

                    # Try immediate GPU processing if available
                    gpu_start = time.time()

                # Move to GPU if available for faster operations
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if first_feature.device != device:
                    volume_features = [f.to(device) for f in volume_features]

                # Use the original torch.stack but on GPU
                volume_features_3d = torch.stack(volume_features, dim=3)

                if enable_timing:
                    gpu_time = time.time() - gpu_start
                    print(f"    GPU-accelerated stacking: {gpu_time:.3f}s")
                    print(f"    Device: {volume_features_3d.device}")

            if enable_timing:
                stacking_time = time.time() - stacking_start
                print(f"  Feature stacking for volume {b+1}: {stacking_time:.3f}s")

            all_features.append(volume_features_3d)

        # Stack batch
        if enable_timing:
            final_stacking_start = time.time()

        batch_features = torch.stack(
            all_features, dim=0
        )  # (batch_size, output_channels, final_d, final_h, final_w)

        if enable_timing:
            final_stacking_time = time.time() - final_stacking_start
            print(f"  Final batch stacking: {final_stacking_time:.3f}s")
            print(
                f"  {plane_type.upper()} features stacked: {tuple(batch_features.shape)} (already at target size)"
            )

        return batch_features

    def _process_slice(self, slice_2d, target_dim1, target_dim2):
        """
        Process a 2D slice through DINOv3 and resize back to target dimensions.

        Parameters:
        -----------
        slice_2d : numpy.ndarray
            2D slice of shape (dim1, dim2)
        target_dim1, target_dim2 : int
            Target dimensions for the output

        Returns:
        --------
        torch.Tensor: Features (output_channels, target_dim1, target_dim2)
        """
        # Upsample slice to DINOv3 input size
        slice_upsampled = resize(
            slice_2d,
            (self.dinov3_slice_size, self.dinov3_slice_size),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(slice_2d.dtype)

        # Extract DINOv3 features for this slice
        slice_batch = slice_upsampled[np.newaxis, ...]
        dinov3_features = process(slice_batch)  # (output_channels, 1, H_feat, W_feat)

        # Remove batch dimension
        slice_features = dinov3_features[
            :, 0, :, :
        ]  # (output_channels, H_feat, W_feat)

        # Resize features back to target spatial size
        slice_features_tensor = torch.tensor(
            slice_features, dtype=torch.float32
        ).unsqueeze(0)
        slice_features_resized = F.interpolate(
            slice_features_tensor,
            size=(target_dim1, target_dim2),
            mode="bilinear",
            align_corners=False,
        )

        return slice_features_resized.squeeze(
            0
        )  # (output_channels, target_dim1, target_dim2)

    def _process_slice_batch(
        self, all_slices, slice_dims, batch_size, enable_timing=False
    ):
        """
        Process multiple slices in batches for efficiency.

        Parameters:
        -----------
        all_slices : list
            List of 2D numpy arrays to process
        slice_dims : list
            List of (target_dim1, target_dim2) tuples for each slice
        batch_size : int
            Number of slices to process simultaneously
        enable_timing : bool, default=False
            Whether to enable detailed timing measurements

        Returns:
        --------
        tuple: (all_features, timing_info) if enable_timing else all_features
        """
        from dinov3_playground.dinov3_core import process
        from skimage.transform import resize
        import torch.nn.functional as F
        import time

        all_features = []
        num_slices = len(all_slices)

        # Initialize timing info
        timing_info = (
            {
                "total_upsampling_time": 0.0,
                "total_stacking_time": 0.0,
                "total_dinov3_inference_time": 0.0,
                "total_feature_extraction_time": 0.0,
                "total_downsampling_time": 0.0,
                "num_batches": 0,
                "avg_batch_size": 0.0,
            }
            if enable_timing
            else None
        )

        for start_idx in range(0, num_slices, batch_size):
            end_idx = min(start_idx + batch_size, num_slices)
            batch_slices = all_slices[start_idx:end_idx]
            batch_dims = slice_dims[start_idx:end_idx]

            if enable_timing:
                timing_info["num_batches"] += 1
                timing_info["avg_batch_size"] += len(batch_slices)

            # TIMING: Start upsampling
            upsample_start = time.time() if enable_timing else None

            # Prepare batch of slices - only resize if necessary
            batch_upsampled = []
            for slice_2d in batch_slices:
                # Check if slice is already at target size
                if slice_2d.shape == (self.dinov3_slice_size, self.dinov3_slice_size):
                    # No resizing needed - use native resolution
                    slice_upsampled = slice_2d.astype(slice_2d.dtype)
                    if (
                        enable_timing and len(batch_upsampled) == 0
                    ):  # Print once per batch
                        print(
                            f"      Using native resolution: {slice_2d.shape} (no upsampling)"
                        )
                else:
                    # Resize to DINOv3 input size
                    slice_upsampled = resize(
                        slice_2d,
                        (self.dinov3_slice_size, self.dinov3_slice_size),
                        preserve_range=True,
                        anti_aliasing=True,
                    ).astype(slice_2d.dtype)
                    if (
                        enable_timing and len(batch_upsampled) == 0
                    ):  # Print once per batch
                        print(
                            f"      Resizing: {slice_2d.shape} → {slice_upsampled.shape}"
                        )
                batch_upsampled.append(slice_upsampled)

            # TIMING: End upsampling, start stacking
            if enable_timing:
                upsample_end = time.time()
                batch_upsample_time = upsample_end - upsample_start
                timing_info["total_upsampling_time"] += batch_upsample_time
                print(
                    f"    Batch {timing_info['num_batches']}: Processed {len(batch_slices)} slices in {batch_upsample_time:.3f}s"
                )
                print(
                    f"      Input slice shapes: {[s.shape for s in batch_slices[:3]]}"
                    + ("..." if len(batch_slices) > 3 else "")
                )
                # Check if upsampling actually happened
                sample_shape = batch_slices[0].shape if batch_slices else (0, 0)
                if sample_shape == (self.dinov3_slice_size, self.dinov3_slice_size):
                    print(
                        f"      Using native resolution: {self.dinov3_slice_size}×{self.dinov3_slice_size} (no resize needed)"
                    )
                else:
                    print(
                        f"      Resized to: {self.dinov3_slice_size}×{self.dinov3_slice_size}"
                    )
                stack_start = time.time()

            # Stack into batch: (batch_size, H, W)
            batch_array = np.stack(batch_upsampled, axis=0)

            # TIMING: End stacking, start DINOv3 inference
            if enable_timing:
                stack_end = time.time()
                batch_stack_time = stack_end - stack_start
                timing_info["total_stacking_time"] += batch_stack_time
                print(
                    f"      Stacked batch: {batch_array.shape} in {batch_stack_time:.3f}s"
                )
                print(f"      Batch memory: {batch_array.nbytes / 1e6:.2f} MB")
                dinov3_start = time.time()

            # Process entire batch through DINOv3
            batch_features = process(
                batch_array
            )  # (output_channels, batch_size, H_feat, W_feat)

            # TIMING: End DINOv3 inference, start feature extraction
            if enable_timing:
                dinov3_end = time.time()
                batch_dinov3_time = dinov3_end - dinov3_start
                timing_info["total_dinov3_inference_time"] += batch_dinov3_time
                print(
                    f"      DINOv3 inference: {batch_array.shape} → {batch_features.shape} in {batch_dinov3_time:.3f}s"
                )
                print(f"      Features memory: {batch_features.nbytes / 1e6:.2f} MB")
                extraction_start = time.time()

            # OPTIMIZED: Convert entire batch_features to tensor at once
            batch_features_tensor = torch.tensor(batch_features, dtype=torch.float32)
            # batch_features_tensor shape: (output_channels, batch_size, H_feat, W_feat)
            # Rearrange to: (batch_size, output_channels, H_feat, W_feat)
            batch_features_tensor = batch_features_tensor.permute(1, 0, 2, 3)

            # TIMING: End feature extraction, start batched downsampling
            if enable_timing:
                extraction_end = time.time()
                batch_extraction_time = extraction_end - extraction_start
                timing_info["total_feature_extraction_time"] += batch_extraction_time
                print(
                    f"      Feature extraction: tensor conversion and permute in {batch_extraction_time:.3f}s"
                )
                print(f"      Tensor shape: {tuple(batch_features_tensor.shape)}")
                resize_start = time.time()

            # Group slices by target dimensions for efficient batch processing
            dim_groups = {}
            for i, (target_dim1, target_dim2) in enumerate(batch_dims):
                dim_key = (target_dim1, target_dim2)
                if dim_key not in dim_groups:
                    dim_groups[dim_key] = []
                dim_groups[dim_key].append(i)

            # Process each dimension group in batch
            slice_results = [None] * len(batch_slices)
            for (target_dim1, target_dim2), indices in dim_groups.items():
                if len(indices) > 1:
                    # Multiple slices with same target dimensions - batch process
                    group_features = batch_features_tensor[
                        indices
                    ]  # (group_size, output_channels, H_feat, W_feat)
                    group_resized = F.interpolate(
                        group_features,
                        size=(target_dim1, target_dim2),
                        mode="bilinear",
                        align_corners=False,
                    )
                    # Store results back to original positions
                    for j, idx in enumerate(indices):
                        slice_results[idx] = group_resized[j]
                else:
                    # Single slice - process individually (still more efficient than before)
                    idx = indices[0]
                    single_feature = batch_features_tensor[
                        idx : idx + 1
                    ]  # (1, output_channels, H_feat, W_feat)
                    single_resized = F.interpolate(
                        single_feature,
                        size=(target_dim1, target_dim2),
                        mode="bilinear",
                        align_corners=False,
                    )
                    slice_results[idx] = single_resized[0]

            # Add all processed slices to results
            for result in slice_results:
                all_features.append(result)

            # TIMING: End feature resizing
            if enable_timing:
                resize_end = time.time()
                batch_resize_time = resize_end - resize_start
                timing_info["total_downsampling_time"] += batch_resize_time
                print(
                    f"      DINOv3 feature resizing: {len(dim_groups)} dimension groups in {batch_resize_time:.3f}s"
                )
                for (dim1, dim2), indices in dim_groups.items():
                    print(
                        f"        {len(indices)} slices: DINOv3 32×32 → {dim1}×{dim2}"
                    )

        # Calculate average batch size
        if enable_timing and timing_info["num_batches"] > 0:
            timing_info["avg_batch_size"] = (
                timing_info["avg_batch_size"] / timing_info["num_batches"]
            )

        if enable_timing:
            return all_features, timing_info
        else:
            return all_features

    def _get_aggregated_timing_info(self):
        """
        Aggregate timing information from all planes and batches.

        Returns:
        --------
        dict: Aggregated timing information with detailed breakdown
        """
        if not hasattr(self, "_plane_timing_info") or not self._plane_timing_info:
            return {}

        aggregated = {
            "total_upsampling_time": 0.0,
            "total_stacking_time": 0.0,
            "total_dinov3_inference_time": 0.0,
            "total_feature_extraction_time": 0.0,
            "total_downsampling_time": 0.0,
            "total_batches": 0,
            "total_slices": 0,
            "plane_breakdown": {},
        }

        for plane_type, batch_timings in self._plane_timing_info.items():
            plane_total = {
                "upsampling_time": 0.0,
                "stacking_time": 0.0,
                "dinov3_inference_time": 0.0,
                "feature_extraction_time": 0.0,
                "downsampling_time": 0.0,
                "num_batches": 0,
                "total_slices": 0,
            }

            for timing in batch_timings:
                plane_total["upsampling_time"] += timing["total_upsampling_time"]
                plane_total["stacking_time"] += timing["total_stacking_time"]
                plane_total["dinov3_inference_time"] += timing[
                    "total_dinov3_inference_time"
                ]
                plane_total["feature_extraction_time"] += timing[
                    "total_feature_extraction_time"
                ]
                plane_total["downsampling_time"] += timing["total_downsampling_time"]
                plane_total["num_batches"] += timing["num_batches"]
                plane_total["total_slices"] += (
                    timing["avg_batch_size"] * timing["num_batches"]
                )

                # Add to overall totals
                aggregated["total_upsampling_time"] += timing["total_upsampling_time"]
                aggregated["total_stacking_time"] += timing["total_stacking_time"]
                aggregated["total_dinov3_inference_time"] += timing[
                    "total_dinov3_inference_time"
                ]
                aggregated["total_feature_extraction_time"] += timing[
                    "total_feature_extraction_time"
                ]
                aggregated["total_downsampling_time"] += timing[
                    "total_downsampling_time"
                ]
                aggregated["total_batches"] += timing["num_batches"]

            aggregated["total_slices"] += plane_total["total_slices"]
            aggregated["plane_breakdown"][plane_type] = plane_total

        # Calculate total time and percentages
        total_time = (
            aggregated["total_upsampling_time"]
            + aggregated["total_stacking_time"]
            + aggregated["total_dinov3_inference_time"]
            + aggregated["total_feature_extraction_time"]
            + aggregated["total_downsampling_time"]
        )

        aggregated["total_time"] = total_time

        if total_time > 0:
            aggregated["upsampling_percentage"] = (
                aggregated["total_upsampling_time"] / total_time
            ) * 100
            aggregated["stacking_percentage"] = (
                aggregated["total_stacking_time"] / total_time
            ) * 100
            aggregated["dinov3_inference_percentage"] = (
                aggregated["total_dinov3_inference_time"] / total_time
            ) * 100
            aggregated["feature_extraction_percentage"] = (
                aggregated["total_feature_extraction_time"] / total_time
            ) * 100
            aggregated["downsampling_percentage"] = (
                aggregated["total_downsampling_time"] / total_time
            ) * 100

        return aggregated

    def forward(self, volume):
        """
        Forward pass through the complete 3D pipeline.

        Parameters:
        -----------
        volume : numpy.ndarray or torch.Tensor
            Input volume of shape (batch_size, D, H, W) or (D, H, W)

        Returns:
        --------
        torch.Tensor: 3D segmentation logits (batch_size, num_classes, D, H, W)
        """
        # Extract DINOv3 features
        features = self.extract_dinov3_features_3d(
            volume, use_orthogonal_planes=self.use_orthogonal_planes
        )

        # Pass through 3D UNet
        segmentation_logits = self.unet3d(features)

        return segmentation_logits

    def predict(self, volume, return_probabilities=False):
        """
        Generate predictions from input volume.

        Parameters:
        -----------
        volume : numpy.ndarray or torch.Tensor
            Input volume
        return_probabilities : bool, default=False
            Whether to return class probabilities

        Returns:
        --------
        tuple: (predictions, probabilities) if return_probabilities else predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(volume)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            if return_probabilities:
                return predictions, probabilities
            return predictions

    def predict_large_volume(self, volume, chunk_size=64, overlap=16):
        """
        Predict on large volumes using sliding window approach.

        Parameters:
        -----------
        volume : numpy.ndarray
            Large input volume
        chunk_size : int, default=64
            Size of processing chunks
        overlap : int, default=16
            Overlap between chunks

        Returns:
        --------
        numpy.ndarray: Full volume predictions
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().numpy()

        if volume.ndim == 3:
            depth, height, width = volume.shape
        else:
            raise ValueError("Volume must be 3D")

        # Initialize output
        output_volume = np.zeros((depth, height, width), dtype=np.int32)
        count_volume = np.zeros((depth, height, width), dtype=np.int32)

        step = chunk_size - overlap

        # Process in chunks
        for z in range(0, depth, step):
            for y in range(0, height, step):
                for x in range(0, width, step):
                    # Extract chunk
                    z_end = min(z + chunk_size, depth)
                    y_end = min(y + chunk_size, height)
                    x_end = min(x + chunk_size, width)

                    chunk = volume[z:z_end, y:y_end, x:x_end]

                    # Pad chunk to expected size if needed
                    if chunk.shape != self.input_size:
                        padded_chunk = np.zeros(self.input_size, dtype=chunk.dtype)
                        padded_chunk[
                            : chunk.shape[0], : chunk.shape[1], : chunk.shape[2]
                        ] = chunk
                        chunk = padded_chunk

                    # Predict
                    chunk_pred = self.predict(chunk).cpu().numpy().squeeze()

                    # Add to output (handle overlaps by voting)
                    actual_chunk = chunk_pred[: z_end - z, : y_end - y, : x_end - x]
                    output_volume[z:z_end, y:y_end, x:x_end] += actual_chunk
                    count_volume[z:z_end, y:y_end, x:x_end] += 1

        # Average overlapping predictions
        output_volume = output_volume / np.maximum(count_volume, 1)

        return output_volume.astype(np.int32)


# Update the create_model function to include 3D UNet
def create_model(model_type, input_dim=None, num_classes=2, **kwargs):
    """
    Factory function to create different types of models.

    Parameters:
    -----------
    model_type : str
        Type of model ('simple', 'improved', 'dinov3_unet', 'dinov3_unet3d')
    input_dim : int, optional
        Input dimension (for non-UNet models)
    num_classes : int, default=2
        Number of output classes
    **kwargs : dict
        Additional parameters for model creation

    Returns:
    --------
    nn.Module: Created model
    """
    if model_type == "simple":
        if input_dim is None:
            raise ValueError("input_dim is required for simple model")
        return SimpleClassifier(input_dim=input_dim, num_classes=num_classes)

    elif model_type == "improved":
        if input_dim is None:
            raise ValueError("input_dim is required for improved model")
        return ImprovedClassifier(
            input_dim=input_dim, num_classes=num_classes, **kwargs
        )

    elif model_type == "dinov3_unet":
        input_channels = kwargs.get("input_channels", 384)
        base_channels = kwargs.get("base_channels", 64)
        return DINOv3UNet(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    elif model_type == "dinov3_unet3d":
        input_channels = kwargs.get("input_channels", 384)
        base_channels = kwargs.get("base_channels", 32)  # Lower default for 3D
        input_size = kwargs.get("input_size", (112, 112, 112))
        return DINOv3UNet3D(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            input_size=input_size,
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: 'simple', 'improved', 'dinov3_unet', 'dinov3_unet3d'"
        )


def print_model_summary(model, input_shape):
    """
    Print a summary of the model architecture.

    Parameters:
    -----------
    model : nn.Module
        The model to summarize
    input_shape : tuple
        Shape of input tensor (for 2D: channels, height, width; for 3D: channels, depth, height, width)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {input_shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")  # Assuming float32

    if hasattr(model, "num_classes"):
        print(f"Number of classes: {model.num_classes}")

    # For 3D models, show memory estimates
    if hasattr(model, "get_memory_usage"):
        memory_info = model.get_memory_usage()
        print(f"Estimated memory usage:")
        print(f"  Parameters: {memory_info['parameters']:.2f} MB")
        print(f"  Activations: {memory_info['total_activations']:.2f} MB")
        print(f"  Total: {memory_info['total_estimated']:.2f} MB")

    # Test forward pass to check dimensions
    try:
        dummy_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass test failed: {e}")
