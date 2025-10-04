"""
Integration example for multi-scale context in existing training pipeline.

This shows how to modify the existing memory_efficient_training.py to incorporate
multi-scale contextual information.
"""

# Modifications to MemoryEfficientDataLoader3D class:


class MemoryEfficientDataLoader3D:
    def __init__(
        self,
        raw_data,
        gt_data,
        gt_masks=None,
        # Add these parameters:
        full_resolution_data=None,  # Full-res data for context
        context_scale=4,  # How much larger context should be
        context_resolution_factor=0.5,  # Resolution reduction for context
        **kwargs,
    ):

        # Existing initialization...

        # New context-related attributes
        self.full_resolution_data = full_resolution_data or raw_data
        self.context_scale = context_scale
        self.context_resolution_factor = context_resolution_factor

        # Store crop metadata for context extraction
        self.crop_centers = {}  # Will store center coordinates for each sampled crop

    def sample_training_batch_with_context(self, batch_size):
        """
        Enhanced version that also provides context information.
        """
        # Sample regular batch
        sampled_indices = self.rng.choice(
            self.train_pool_indices,
            size=batch_size,
            replace=len(self.train_pool_indices) < batch_size,
        )

        batch_volumes = self.raw_data[sampled_indices]
        batch_gt = self.gt_data[sampled_indices]
        batch_masks = self.gt_masks[sampled_indices]

        # Get corresponding full-resolution volumes for context
        batch_full_volumes = self.full_resolution_data[sampled_indices]

        # Calculate crop centers (you'd implement this based on your data structure)
        batch_crop_centers = self._calculate_crop_centers(sampled_indices)

        return (
            batch_volumes,
            batch_gt,
            batch_masks,
            batch_full_volumes,
            batch_crop_centers,
        )

    def extract_multi_scale_dinov3_features_3d(
        self, local_volumes, full_volumes, crop_centers
    ):
        """
        Extract both local high-res and contextual features.
        """
        from .multi_scale_context import MultiScaleContextExtractor

        # Initialize context extractor if not already done
        if not hasattr(self, "context_extractor"):
            self.context_extractor = MultiScaleContextExtractor(
                context_scale=self.context_scale,
                context_resolution_factor=self.context_resolution_factor,
            )

        # Extract local features (existing method)
        local_features = self.extract_dinov3_features_3d(local_volumes)

        # Extract context features
        batch_size = local_volumes.shape[0]
        context_features_list = []

        for i in range(batch_size):
            # Get context crop
            context_crop = self.context_extractor.get_context_crop(
                full_volumes[i], crop_centers[i], local_volumes[i].shape
            )

            # Process context through DINOv3
            context_feat = self.extract_dinov3_features_3d(context_crop[np.newaxis])

            # Resize to match local features
            if context_feat.shape[2:] != local_features.shape[2:]:
                context_feat = F.interpolate(
                    context_feat,
                    size=local_features.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )

            context_features_list.append(context_feat)

        # Combine context features
        context_features = torch.cat(context_features_list, dim=0)

        # Concatenate local and context features
        multi_scale_features = torch.cat([local_features, context_features], dim=1)

        return multi_scale_features

    def _calculate_crop_centers(self, sampled_indices):
        """
        Calculate center coordinates for sampled crops.
        You'd implement this based on how your crops are extracted.
        """
        # This is a placeholder - implement based on your data structure
        # For example, if you store crop metadata during data loading:
        centers = []
        for idx in sampled_indices:
            # Get center from stored metadata or calculate it
            # This depends on your specific data loading implementation
            center = self.crop_centers.get(idx, (32, 32, 32))  # Default center
            centers.append(center)
        return centers


# Modifications to training function:


def train_3d_unet_memory_efficient_v2_with_context(
    data_loader_3d,
    num_classes,
    device,
    use_multi_scale_context=True,  # New parameter
    **kwargs,
):
    """
    Training function with multi-scale context support.
    """

    # Get current model info
    from .dinov3_core import get_current_model_info

    model_info = get_current_model_info()
    current_output_channels = model_info["output_channels"]

    # Adjust input channels for multi-scale context
    if use_multi_scale_context:
        # Double the input channels (local + context features)
        model_input_channels = current_output_channels * 2
        print(f"Using multi-scale context: {model_input_channels} input channels")
    else:
        model_input_channels = current_output_channels

    # Initialize UNet with correct number of input channels
    from .models import DINOv3UNet3D

    unet3d = DINOv3UNet3D(
        input_channels=model_input_channels,  # Adjusted for context
        num_classes=num_classes,
        base_channels=base_channels,
        input_size=data_loader_3d.target_volume_size,
        **model_kwargs,
    ).to(device)

    # Training loop modifications
    for epoch in range(epochs):
        for batch_idx in range(batches_per_epoch):

            if use_multi_scale_context:
                # Sample batch with context
                (
                    train_volumes,
                    train_gt_volumes,
                    train_masks,
                    full_volumes,
                    crop_centers,
                ) = data_loader_3d.sample_training_batch_with_context(volumes_per_batch)

                # Extract multi-scale features
                train_features = data_loader_3d.extract_multi_scale_dinov3_features_3d(
                    train_volumes, full_volumes, crop_centers
                )
            else:
                # Regular sampling and feature extraction
                train_volumes, train_gt_volumes, train_masks = (
                    data_loader_3d.sample_training_batch(volumes_per_batch)
                )
                train_features = data_loader_3d.extract_dinov3_features_3d(
                    train_volumes
                )

            # Rest of training loop remains the same...
            train_labels = torch.tensor(train_gt_volumes, dtype=torch.long).to(device)
            # ... continue with existing training logic


# Example of how to initialize with context:


def create_context_aware_training():
    """
    Example initialization with multi-scale context.
    """

    # Load your data as usual
    raw, gt, gt_masks, dataset_sources, num_classes = load_random_3d_training_data(...)

    # If you have access to full-resolution data, pass it separately
    # Otherwise, the loader will use the same data for both local and context
    full_resolution_raw = load_full_resolution_data(...)  # Your implementation

    # Initialize data loader with context support
    data_loader_3d = MemoryEfficientDataLoader3D(
        raw_data=raw,
        gt_data=gt,
        gt_masks=gt_masks,
        full_resolution_data=full_resolution_raw,  # For context extraction
        context_scale=4,  # 4x larger spatial context
        context_resolution_factor=0.5,  # Half resolution for context
        # ... other parameters
    )

    # Train with context
    training_results = train_3d_unet_memory_efficient_v2_with_context(
        data_loader_3d=data_loader_3d,
        use_multi_scale_context=True,
        # ... other parameters
    )

    return training_results


if __name__ == "__main__":
    print("Multi-scale context integration example")
    print("This shows how to modify existing training to use contextual information")
