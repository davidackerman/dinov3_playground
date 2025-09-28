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
import numpy as np

from skimage.transform import resize
from .dinov3_core import (
    process,
    output_channels,
)  # Assuming output_channels is defined there


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
        self.activation = activation

        # Build layers dynamically
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))

            # Activation
            if activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.ReLU())

            # Dropout
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

    def __init__(self, input_channels=384, num_classes=2, base_channels=64):
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

        self.input_channels = input_channels
        self.num_classes = num_classes

        # Input projection to reduce channels
        self.input_conv = nn.Conv2d(input_channels, base_channels, kernel_size=1)

        # Encoder (downsampling path)
        self.enc1 = self._make_encoder_block(base_channels, base_channels)
        self.enc2 = self._make_encoder_block(base_channels, base_channels * 2)
        self.enc3 = self._make_encoder_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._make_encoder_block(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = self._make_encoder_block(
            base_channels * 8, base_channels * 16
        )

        # Decoder (upsampling path) - Fixed channel calculations
        # After concatenation: upsampled + skip connection
        self.dec4 = self._make_decoder_block(
            base_channels * 16 + base_channels * 8, base_channels * 8
        )
        self.dec3 = self._make_decoder_block(
            base_channels * 8 + base_channels * 4, base_channels * 4
        )
        self.dec2 = self._make_decoder_block(
            base_channels * 4 + base_channels * 2, base_channels * 2
        )
        self.dec1 = self._make_decoder_block(
            base_channels * 2 + base_channels, base_channels
        )

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(2)

    def _make_encoder_block(self, in_channels, out_channels):
        """Create an encoder block with two convolutions."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder block with two convolutions."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
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

        for i in range(batch_size):
            upsampled_images[i] = resize(
                images[i],
                (self.dinov3_input_size, self.dinov3_input_size),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(images.dtype)

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
        learn_upsampling=False,  # NEW PARAMETER
        dinov3_feature_size=None,  # NEW PARAMETER
    ):
        """
        Initialize DINOv3 3D UNet.

        Parameters:
        -----------
        input_channels : int, default=384
            Number of input channels (DINOv3 feature dimension)
        num_classes : int, default=2
            Number of output classes
        base_channels : int, default=32
            Base number of channels in the UNet (lower than 2D due to memory)
        input_size : tuple, default=(112, 112, 112)
            Expected output spatial dimensions (D, H, W)
        use_half_precision : bool, default=False
            Whether to use half precision (float16)
        learn_upsampling : bool, default=False
            If True, UNet learns upsampling from lower-res DINOv3 features
            If False, DINOv3 features are pre-interpolated to full resolution
        dinov3_feature_size : tuple, optional
            Spatial size of DINOv3 features (D, H, W) when learn_upsampling=True
            If None, assumes same as input_size
        """
        super(DINOv3UNet3D, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_size = input_size
        self.base_channels = base_channels
        self.use_half_precision = use_half_precision
        self.learn_upsampling = learn_upsampling
        self.dinov3_feature_size = dinov3_feature_size or input_size

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
        self.enc1 = self._make_encoder_block(self.base_channels, self.base_channels)
        self.enc2 = self._make_encoder_block(self.base_channels, self.base_channels * 2)
        self.enc3 = self._make_encoder_block(
            self.base_channels * 2, self.base_channels * 4
        )
        self.enc4 = self._make_encoder_block(
            self.base_channels * 4, self.base_channels * 8
        )

        # Bottleneck
        self.bottleneck = self._make_encoder_block(
            self.base_channels * 8, self.base_channels * 16
        )

        # Decoder (upsampling path)
        self.dec4 = self._make_decoder_block(
            self.base_channels * 16 + self.base_channels * 8, self.base_channels * 8
        )
        self.dec3 = self._make_decoder_block(
            self.base_channels * 8 + self.base_channels * 4, self.base_channels * 4
        )
        self.dec2 = self._make_decoder_block(
            self.base_channels * 4 + self.base_channels * 2, self.base_channels * 2
        )
        self.dec1 = self._make_decoder_block(
            self.base_channels * 2 + self.base_channels, self.base_channels
        )

        # Final output layer
        self.final_conv = nn.Conv3d(self.base_channels, num_classes, kernel_size=1)

        # 3D Pooling
        self.pool = nn.MaxPool3d(2)

        # Dropout for regularization (3D volumes have more parameters)
        self.dropout = nn.Dropout3d(0.2)

        # Convert to half precision if requested
        if use_half_precision:
            self.half()
            print("Model converted to half precision (float16)")

    def _make_encoder_block(self, in_channels, out_channels):
        """Create a 3D encoder block with two convolutions."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),  # Light dropout in encoder
        )

    def _make_decoder_block(self, in_channels, out_channels):
        """Create a 3D decoder block with two convolutions."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),  # Light dropout in decoder
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

    def forward(self, x):
        """
        Forward pass through the 3D UNet.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, D, H, W)

        Returns:
        --------
        torch.Tensor: Output segmentation map (batch_size, num_classes, D, H, W)
        """
        # Convert input to half precision if model is in half precision
        if self.use_half_precision and x.dtype != torch.float16:
            x = x.half()

        # Input projection
        x = self.input_conv(x)  # (B, base_channels, D, H, W)

        # Apply learned upsampling if enabled
        if self.learn_upsampling and self.learned_upsample is not None:
            x = self.learned_upsample(x)

            # Fine-tune the size to exactly match target if needed
            current_size = x.shape[2:]  # (D, H, W)
            if current_size != self.input_size:
                x = F.interpolate(
                    x, size=self.input_size, mode="trilinear", align_corners=False
                )

        # Encoder path with skip connections
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

        # Decoder path with skip connections
        # Dec4: upsample bottleneck and concatenate with enc4
        dec4_up = F.interpolate(
            bottleneck_out, scale_factor=2, mode="trilinear", align_corners=False
        )
        dec4_concat = torch.cat([dec4_up, enc4_out], dim=1)
        dec4_out = self.dec4(dec4_concat)

        # Dec3: upsample dec4 and concatenate with enc3
        dec3_up = F.interpolate(
            dec4_out, scale_factor=2, mode="trilinear", align_corners=False
        )
        dec3_concat = torch.cat([dec3_up, enc3_out], dim=1)
        dec3_out = self.dec3(dec3_concat)

        # Dec2: upsample dec3 and concatenate with enc2
        dec2_up = F.interpolate(
            dec3_out, scale_factor=2, mode="trilinear", align_corners=False
        )
        dec2_concat = torch.cat([dec2_up, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_concat)

        # Dec1: upsample dec2 and concatenate with enc1
        dec1_up = F.interpolate(
            dec2_out, scale_factor=2, mode="trilinear", align_corners=False
        )
        dec1_concat = torch.cat([dec1_up, enc1_out], dim=1)
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
        device=None,
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
        device : torch.device, optional
            Device for processing
        """
        super(DINOv3UNet3DPipeline, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.dinov3_slice_size = dinov3_slice_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 3D UNet for processing DINOv3 features
        self.unet3d = DINOv3UNet3D(
            input_channels=output_channels,  # from .dinov3_core
            num_classes=num_classes,
            base_channels=base_channels,
            input_size=input_size,
        )

    def extract_dinov3_features_3d(self, volume):
        """
        Extract DINOv3 features from a 3D volume by processing each slice.

        Parameters:
        -----------
        volume : numpy.ndarray or torch.Tensor
            Volume of shape (batch_size, D, H, W) or (D, H, W)

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
        target_d, target_h, target_w = self.input_size

        # Resize volume to target size
        resized_volume = np.zeros(
            (batch_size, target_d, target_h, target_w), dtype=volume.dtype
        )

        for b in range(batch_size):
            resized_volume[b] = resize(
                volume[b],
                (target_d, target_h, target_w),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(volume.dtype)

        # Process each slice with DINOv3
        all_features = []

        for b in range(batch_size):
            volume_features = []

            for z in range(target_d):
                # Get slice and upsample for DINOv3
                slice_2d = resized_volume[b, z]

                # Upsample slice to DINOv3 input size
                slice_upsampled = resize(
                    slice_2d,
                    (self.dinov3_slice_size, self.dinov3_slice_size),
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype(volume.dtype)

                # Extract DINOv3 features for this slice
                # Add batch dimension for DINOv3 processing
                slice_batch = slice_upsampled[np.newaxis, ...]
                dinov3_features = process(
                    slice_batch
                )  # (output_channels, 1, H_feat, W_feat)

                # Remove batch dimension and rearrange
                slice_features = dinov3_features[
                    :, 0, :, :
                ]  # (output_channels, H_feat, W_feat)

                # Resize features back to target spatial size
                slice_features_tensor = torch.tensor(
                    slice_features, dtype=torch.float32
                ).unsqueeze(
                    0
                )  # (1, output_channels, H_feat, W_feat)
                slice_features_resized = F.interpolate(
                    slice_features_tensor,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )  # (1, output_channels, target_h, target_w)

                volume_features.append(
                    slice_features_resized.squeeze(0)
                )  # (output_channels, target_h, target_w)

            # Stack slices to form 3D volume
            volume_features_3d = torch.stack(
                volume_features, dim=1
            )  # (output_channels, target_d, target_h, target_w)
            all_features.append(volume_features_3d)

        # Stack batch
        batch_features = torch.stack(
            all_features, dim=0
        )  # (batch_size, output_channels, target_d, target_h, target_w)

        return batch_features.to(self.device)

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
        features = self.extract_dinov3_features_3d(volume)

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
