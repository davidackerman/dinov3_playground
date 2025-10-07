"""
Affinity Utilities for Instance Segmentation

This module provides functions to convert instance segmentations to affinity graphs
for training neural networks to predict pixel-to-pixel affinities.

Author: GitHub Copilot
Date: 2025-10-06
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    import edt

    HAS_EDT = True
except ImportError:
    HAS_EDT = False
    print("Warning: edt not available. Install with: pip install edt")
    print("Boundary-weighted loss will not be available.")


def compute_affinities_3d(instance_segmentation, offsets=None):
    """
    Convert 3D instance segmentation to affinity graph.

    Affinities indicate whether neighboring pixels belong to the same instance.
    For each pixel, we compute affinity to its neighbor at a given offset.

    Parameters:
    -----------
    instance_segmentation : numpy.ndarray or torch.Tensor
        3D instance segmentation of shape (D, H, W) where each unique non-zero
        value represents a different instance
    offsets : list of tuples, optional
        List of (z, y, x) offsets. Default is [(1,0,0), (0,1,0), (0,0,1)]
        which computes affinities in +z, +y, and +x directions

    Returns:
    --------
    numpy.ndarray or torch.Tensor
        Affinities of shape (num_offsets, D, H, W) where each channel represents
        affinity in the corresponding offset direction. Values are 1 if neighbors
        belong to same instance, 0 otherwise.
    """
    if offsets is None:
        # Default: affinities in +z, +y, +x directions
        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    is_torch = isinstance(instance_segmentation, torch.Tensor)

    if is_torch:
        device = instance_segmentation.device
        dtype = instance_segmentation.dtype
        affinities = torch.zeros(
            (len(offsets), *instance_segmentation.shape),
            dtype=torch.float32,
            device=device,
        )
    else:
        affinities = np.zeros(
            (len(offsets), *instance_segmentation.shape), dtype=np.float32
        )

    for idx, (dz, dy, dx) in enumerate(offsets):
        # For each offset, check if pixel and its neighbor belong to same instance
        if is_torch:
            affinity = _compute_single_affinity_torch(instance_segmentation, dz, dy, dx)
        else:
            affinity = _compute_single_affinity_numpy(instance_segmentation, dz, dy, dx)

        affinities[idx] = affinity

    return affinities


def _compute_single_affinity_numpy(instances, dz, dy, dx):
    """
    Compute affinity for a single offset direction using numpy.

    Parameters:
    -----------
    instances : numpy.ndarray
        Instance segmentation of shape (D, H, W)
    dz, dy, dx : int
        Offset in z, y, x directions

    Returns:
    --------
    numpy.ndarray
        Affinity map of shape (D, H, W)
    """
    D, H, W = instances.shape
    affinity = np.zeros((D, H, W), dtype=np.float32)

    # Calculate valid regions (where both pixel and neighbor exist)
    z_start = max(0, -dz)
    z_end = min(D, D - dz)
    y_start = max(0, -dy)
    y_end = min(H, H - dy)
    x_start = max(0, -dx)
    x_end = min(W, W - dx)

    # Get current pixels and their neighbors
    current = instances[z_start:z_end, y_start:y_end, x_start:x_end]
    neighbor = instances[
        z_start + dz : z_end + dz, y_start + dy : y_end + dy, x_start + dx : x_end + dx
    ]

    # Affinity is 1 if both pixels belong to the same instance (and not background)
    # Background is typically 0, so we need both to be non-zero AND equal
    same_instance = (current == neighbor) & (current > 0)

    affinity[z_start:z_end, y_start:y_end, x_start:x_end] = same_instance.astype(
        np.float32
    )

    return affinity


def _compute_single_affinity_torch(instances, dz, dy, dx):
    """
    Compute affinity for a single offset direction using PyTorch.

    Parameters:
    -----------
    instances : torch.Tensor
        Instance segmentation of shape (D, H, W)
    dz, dy, dx : int
        Offset in z, y, x directions

    Returns:
    --------
    torch.Tensor
        Affinity map of shape (D, H, W)
    """
    D, H, W = instances.shape
    affinity = torch.zeros((D, H, W), dtype=torch.float32, device=instances.device)

    # Calculate valid regions (where both pixel and neighbor exist)
    z_start = max(0, -dz)
    z_end = min(D, D - dz)
    y_start = max(0, -dy)
    y_end = min(H, H - dy)
    x_start = max(0, -dx)
    x_end = min(W, W - dx)

    # Get current pixels and their neighbors
    current = instances[z_start:z_end, y_start:y_end, x_start:x_end]
    neighbor = instances[
        z_start + dz : z_end + dz, y_start + dy : y_end + dy, x_start + dx : x_end + dx
    ]

    # Affinity is 1 if both pixels belong to the same instance (and not background)
    same_instance = (current == neighbor) & (current > 0)

    affinity[z_start:z_end, y_start:y_end, x_start:x_end] = same_instance.float()

    return affinity


def compute_affinities_2d(instance_segmentation, offsets=None):
    """
    Convert 2D instance segmentation to affinity graph.

    Parameters:
    -----------
    instance_segmentation : numpy.ndarray or torch.Tensor
        2D instance segmentation of shape (H, W) where each unique non-zero
        value represents a different instance
    offsets : list of tuples, optional
        List of (y, x) offsets. Default is [(1,0), (0,1)]
        which computes affinities in +y and +x directions

    Returns:
    --------
    numpy.ndarray or torch.Tensor
        Affinities of shape (num_offsets, H, W)
    """
    if offsets is None:
        # Default: affinities in +y, +x directions
        offsets = [(1, 0), (0, 1)]

    is_torch = isinstance(instance_segmentation, torch.Tensor)

    if is_torch:
        device = instance_segmentation.device
        affinities = torch.zeros(
            (len(offsets), *instance_segmentation.shape),
            dtype=torch.float32,
            device=device,
        )
    else:
        affinities = np.zeros(
            (len(offsets), *instance_segmentation.shape), dtype=np.float32
        )

    for idx, (dy, dx) in enumerate(offsets):
        if is_torch:
            affinity = _compute_single_affinity_2d_torch(instance_segmentation, dy, dx)
        else:
            affinity = _compute_single_affinity_2d_numpy(instance_segmentation, dy, dx)

        affinities[idx] = affinity

    return affinities


def _compute_single_affinity_2d_numpy(instances, dy, dx):
    """Compute 2D affinity for a single offset using numpy."""
    H, W = instances.shape
    affinity = np.zeros((H, W), dtype=np.float32)

    y_start = max(0, -dy)
    y_end = min(H, H - dy)
    x_start = max(0, -dx)
    x_end = min(W, W - dx)

    current = instances[y_start:y_end, x_start:x_end]
    neighbor = instances[y_start + dy : y_end + dy, x_start + dx : x_end + dx]

    same_instance = (current == neighbor) & (current > 0)
    affinity[y_start:y_end, x_start:x_end] = same_instance.astype(np.float32)

    return affinity


def _compute_single_affinity_2d_torch(instances, dy, dx):
    """Compute 2D affinity for a single offset using PyTorch."""
    H, W = instances.shape
    affinity = torch.zeros((H, W), dtype=torch.float32, device=instances.device)

    y_start = max(0, -dy)
    y_end = min(H, H - dy)
    x_start = max(0, -dx)
    x_end = min(W, W - dx)

    current = instances[y_start:y_end, x_start:x_end]
    neighbor = instances[y_start + dy : y_end + dy, x_start + dx : x_end + dx]

    same_instance = (current == neighbor) & (current > 0)
    affinity[y_start:y_end, x_start:x_end] = same_instance.float()

    return affinity


def compute_boundary_weights(
    instance_segmentation,
    boundary_weight=10.0,
    sigma=5.0,
    anisotropy=(1.0, 1.0, 1.0),
    black_border=True,
):
    """
    Compute per-pixel weights emphasizing instance boundaries.

    Uses Euclidean Distance Transform (EDT) per instance to compute distance
    to nearest boundary. Weights decay exponentially from boundaries.

    Parameters:
    -----------
    instance_segmentation : numpy.ndarray
        Instance segmentation of shape (D, H, W) or (H, W) where each unique
        non-zero value represents a different instance
    boundary_weight : float, default=10.0
        Maximum weight at boundaries. Interior pixels have weight 1.0.
        Final weight = 1 + (boundary_weight - 1) * exp(-dist^2 / (2*sigma^2))
    sigma : float, default=5.0
        Distance decay parameter in pixels. Controls how quickly weight
        decreases away from boundaries.
        - Small sigma (e.g., 2.0): Sharp falloff, only immediate boundary pixels weighted
        - Large sigma (e.g., 10.0): Smooth falloff, broader boundary region weighted
    anisotropy : tuple of float, default=(1.0, 1.0, 1.0)
        Anisotropy of the voxels for EDT computation (z, y, x) for 3D or (y, x) for 2D.
        Use physical voxel sizes for proper distance computation.
    black_border : bool, default=True
        Whether to treat volume borders as boundaries (penalize predictions at edges)

    Returns:
    --------
    numpy.ndarray
        Weight map of same shape as input. Values in range [1.0, boundary_weight].

    Notes:
    ------
    This implements per-instance distance transforms as recommended in:
    - Funke et al. "Large Scale Image Segmentation with Structured Loss" (2018)
    - Lee et al. "Superhuman Accuracy on the SNEMI3D Connectomics Challenge" (2017)

    The key insight: boundaries between instances are critical for instance
    segmentation, so we weight them more heavily in the loss to force the
    network to learn precise instance separation.
    """
    if not HAS_EDT:
        raise ImportError(
            "edt package required for boundary weighting. "
            "Install with: pip install edt"
        )

    is_3d = instance_segmentation.ndim == 3

    # Initialize weight map (all 1.0 initially)
    weights = np.ones_like(instance_segmentation, dtype=np.float32)

    # Get unique instances (excluding background=0)
    instance_ids = np.unique(instance_segmentation)
    instance_ids = instance_ids[instance_ids > 0]

    if len(instance_ids) == 0:
        # No instances, return uniform weights
        return weights

    # Compute distance transform for each instance separately
    for instance_id in instance_ids:
        # Create binary mask for this instance
        instance_mask = (instance_segmentation == instance_id).astype(np.uint8)

        # Compute EDT (distance to nearest boundary)
        # edt returns distance in physical units based on anisotropy
        if is_3d:
            dist = edt.edt(
                instance_mask,
                anisotropy=anisotropy,
                black_border=black_border,
            )
        else:
            dist = edt.edt(
                instance_mask,
                anisotropy=anisotropy[:2] if len(anisotropy) == 3 else anisotropy,
                black_border=black_border,
            )

        # Compute Gaussian falloff from boundaries
        # weight = 1 + (boundary_weight - 1) * exp(-dist^2 / (2*sigma^2))
        # At boundary (dist=0): weight = boundary_weight
        # Far from boundary (dist >> sigma): weight → 1.0
        gaussian_falloff = np.exp(-(dist**2) / (2 * sigma**2))
        instance_weights = 1.0 + (boundary_weight - 1.0) * gaussian_falloff

        # Update weight map where this instance exists
        weights[instance_mask > 0] = np.maximum(
            weights[instance_mask > 0], instance_weights[instance_mask > 0]
        )

    # Also weight background boundaries if black_border=True
    # This ensures boundaries at volume edges are also emphasized
    if black_border and len(instance_ids) > 0:
        # Create background mask
        background_mask = (instance_segmentation == 0).astype(np.uint8)
        if background_mask.sum() > 0:
            if is_3d:
                bg_dist = edt.edt(
                    background_mask,
                    anisotropy=anisotropy,
                    black_border=black_border,
                )
            else:
                bg_dist = edt.edt(
                    background_mask,
                    anisotropy=anisotropy[:2] if len(anisotropy) == 3 else anisotropy,
                    black_border=black_border,
                )

            # Only weight background pixels near instance boundaries
            bg_gaussian = np.exp(-(bg_dist**2) / (2 * sigma**2))
            bg_weights = 1.0 + (boundary_weight - 1.0) * bg_gaussian
            weights[background_mask > 0] = np.maximum(
                weights[background_mask > 0], bg_weights[background_mask > 0]
            )

    return weights


class AffinityLoss(torch.nn.Module):
    """
    Loss function for affinity prediction.

    Combines Binary Cross-Entropy for affinity classification with optional
    class balancing to handle foreground/background imbalance.
    """

    def __init__(self, use_class_weights=True, pos_weight=None):
        """
        Initialize affinity loss.

        Parameters:
        -----------
        use_class_weights : bool, default=True
            Whether to use class weights to balance positive/negative examples
        pos_weight : float, optional
            Weight for positive examples. If None and use_class_weights=True,
            will be computed from the target distribution
        """
        super(AffinityLoss, self).__init__()
        self.use_class_weights = use_class_weights
        self.pos_weight = pos_weight

    def forward(self, predictions, targets, mask=None):
        """
        Compute affinity loss.

        Parameters:
        -----------
        predictions : torch.Tensor
            Predicted affinities of shape (batch, num_offsets, D, H, W)
            Raw logits (not sigmoid applied)
        targets : torch.Tensor
            Target affinities of shape (batch, num_offsets, D, H, W)
            Binary values (0 or 1)
        mask : torch.Tensor, optional
            Valid region mask of shape (batch, D, H, W)

        Returns:
        --------
        torch.Tensor
            Scalar loss value
        """
        # Apply sigmoid to get probabilities
        pred_probs = torch.sigmoid(predictions)

        # Calculate pos_weight if needed
        pos_weight = self.pos_weight
        if self.use_class_weights and pos_weight is None:
            # Count positive and negative examples
            if mask is not None:
                # Expand mask to match affinity channels
                mask_expanded = mask.unsqueeze(1).expand_as(targets)
                valid_targets = targets[mask_expanded > 0]
            else:
                valid_targets = targets

            num_pos = (valid_targets > 0.5).sum().float()
            num_neg = (valid_targets < 0.5).sum().float()

            if num_pos > 0:
                pos_weight = num_neg / num_pos
            else:
                pos_weight = 1.0

        # Use BCEWithLogitsLoss for numerical stability
        if pos_weight is not None:
            # Create pos_weight tensor with same device as predictions
            if isinstance(pos_weight, torch.Tensor):
                pos_weight_tensor = pos_weight.detach().clone().to(predictions.device)
            else:
                pos_weight_tensor = torch.tensor(
                    pos_weight, device=predictions.device, dtype=predictions.dtype
                )
            criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=pos_weight_tensor, reduction="none"
            )
        else:
            criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

        # Compute loss
        loss = criterion(predictions, targets)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match affinity channels
            mask_expanded = mask.unsqueeze(1).expand_as(loss)
            loss = loss * mask_expanded

            # Average only over valid regions
            num_valid = mask_expanded.sum()
            if num_valid > 0:
                loss = loss.sum() / num_valid
            else:
                loss = loss.sum()
        else:
            loss = loss.mean()

        return loss


class BoundaryWeightedAffinityLoss(torch.nn.Module):
    """
    Boundary-weighted affinity loss for instance segmentation.

    Emphasizes correct predictions at instance boundaries where accurate
    affinities are critical for separating touching instances.

    Uses per-instance Euclidean Distance Transform (EDT) to compute weights
    that decay exponentially from boundaries into instance interiors.

    Reference:
    - Funke et al. "Large Scale Image Segmentation with Structured Loss" (2018)
    - Lee et al. "Superhuman Accuracy on the SNEMI3D Connectomics Challenge" (2017)
    """

    def __init__(
        self,
        boundary_weight=10.0,
        sigma=5.0,
        anisotropy=(1.0, 1.0, 1.0),
        use_class_weights=True,
        pos_weight=None,
    ):
        """
        Initialize boundary-weighted affinity loss.

        Parameters:
        -----------
        boundary_weight : float, default=10.0
            Maximum weight for boundary pixels. Interior pixels have weight 1.0.
            Recommended range: 5.0-20.0
            - Higher values: More emphasis on boundaries (may hurt interior accuracy)
            - Lower values: More balanced (may miss boundary details)
        sigma : float, default=5.0
            Distance decay parameter in pixels. Controls boundary region size.
            Recommended range: 2.0-10.0
            - Small (2-3): Only immediate boundary voxels weighted heavily
            - Medium (5-7): ~2-3 voxel boundary region
            - Large (8-10): Broader boundary region
        anisotropy : tuple of float, default=(1.0, 1.0, 1.0)
            Voxel anisotropy (z, y, x) for proper distance computation.
            Use physical voxel sizes (e.g., (2.0, 1.0, 1.0) for 2nm z, 1nm xy).
        use_class_weights : bool, default=True
            Whether to balance positive/negative affinity examples
        pos_weight : float, optional
            Manual weight for positive examples. Auto-computed if None.
        """
        super(BoundaryWeightedAffinityLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.sigma = sigma
        self.anisotropy = anisotropy
        self.use_class_weights = use_class_weights
        self.pos_weight = pos_weight

        if not HAS_EDT:
            raise ImportError(
                "edt package required for boundary-weighted loss. "
                "Install with: pip install edt"
            )

    def forward(self, predictions, targets, instance_seg=None, mask=None):
        """
        Compute boundary-weighted affinity loss.

        Parameters:
        -----------
        predictions : torch.Tensor
            Predicted affinities, shape (batch, num_offsets, D, H, W)
            Raw logits (sigmoid not applied)
        targets : torch.Tensor
            Target affinities, shape (batch, num_offsets, D, H, W)
            Binary values (0 or 1)
        instance_seg : numpy.ndarray or torch.Tensor, optional
            Instance segmentation used to compute boundary weights.
            Shape (batch, D, H, W) or (D, H, W) if batch=1.
            Each unique non-zero value is a different instance.
            If None, falls back to standard AffinityLoss.
        mask : torch.Tensor, optional
            Valid region mask, shape (batch, D, H, W)

        Returns:
        --------
        torch.Tensor
            Scalar loss value
        """
        batch_size = predictions.shape[0]

        # Fall back to standard affinity loss if no instance segmentation provided
        if instance_seg is None:
            print(
                "Warning: No instance segmentation provided, using standard AffinityLoss"
            )
            fallback_loss = AffinityLoss(
                use_class_weights=self.use_class_weights, pos_weight=self.pos_weight
            )
            return fallback_loss(predictions, targets, mask)

        # Convert instance_seg to numpy if needed
        if isinstance(instance_seg, torch.Tensor):
            instance_seg_np = instance_seg.cpu().numpy()
        else:
            instance_seg_np = instance_seg

        # Handle single volume case (no batch dimension)
        if instance_seg_np.ndim == 3:
            instance_seg_np = instance_seg_np[np.newaxis, ...]  # Add batch dim

        # Compute boundary weights for each volume in batch
        boundary_weights_list = []
        for b in range(batch_size):
            weights = compute_boundary_weights(
                instance_seg_np[b],
                boundary_weight=self.boundary_weight,
                sigma=self.sigma,
                anisotropy=self.anisotropy,
                black_border=True,
            )
            boundary_weights_list.append(weights)

        # Stack into tensor
        boundary_weights = np.stack(boundary_weights_list, axis=0)  # (batch, D, H, W)
        boundary_weights = torch.from_numpy(boundary_weights).to(
            predictions.device, dtype=predictions.dtype
        )

        # Expand boundary weights to match affinity dimensions
        # From (batch, D, H, W) to (batch, num_offsets, D, H, W)
        boundary_weights = boundary_weights.unsqueeze(1).expand_as(predictions)

        # Calculate pos_weight if needed (for class balancing)
        pos_weight = self.pos_weight
        if self.use_class_weights and pos_weight is None:
            if mask is not None:
                mask_expanded = mask.unsqueeze(1).expand_as(targets)
                valid_targets = targets[mask_expanded > 0]
            else:
                valid_targets = targets

            num_pos = (valid_targets > 0.5).sum().float()
            num_neg = (valid_targets < 0.5).sum().float()

            if num_pos > 0:
                pos_weight = num_neg / num_pos
            else:
                pos_weight = 1.0

        # Compute BCE loss with class weights
        if pos_weight is not None:
            if isinstance(pos_weight, torch.Tensor):
                pos_weight_tensor = pos_weight.detach().clone().to(predictions.device)
            else:
                pos_weight_tensor = torch.tensor(
                    pos_weight, device=predictions.device, dtype=predictions.dtype
                )
            criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=pos_weight_tensor, reduction="none"
            )
        else:
            criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

        # Compute per-pixel loss
        loss = criterion(predictions, targets)

        # Apply boundary weights
        loss = loss * boundary_weights

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(loss)
            loss = loss * mask_expanded

            # Normalize by sum of weights in valid region
            # This ensures we're averaging weighted losses correctly
            weighted_valid = (boundary_weights * mask_expanded).sum()
            if weighted_valid > 0:
                loss = loss.sum() / weighted_valid
            else:
                loss = loss.sum()
        else:
            # Normalize by sum of weights
            loss = loss.sum() / boundary_weights.sum()

        return loss


def affinities_to_instances(affinities, threshold=0.5):
    """
    Convert predicted affinities to instance segmentation.

    This is a simple implementation using connected components.
    For better results, consider using mutex watershed or other
    sophisticated segmentation algorithms.

    Parameters:
    -----------
    affinities : numpy.ndarray or torch.Tensor
        Predicted affinities of shape (num_offsets, D, H, W) or (num_offsets, H, W)
    threshold : float, default=0.5
        Threshold for binarizing affinities

    Returns:
    --------
    numpy.ndarray
        Instance segmentation of shape (D, H, W) or (H, W)
    """
    from scipy import ndimage

    # Convert to numpy if needed
    if isinstance(affinities, torch.Tensor):
        affinities = affinities.cpu().numpy()

    # Threshold affinities
    binary_affinities = affinities > threshold

    # Simple approach: combine all affinities with AND operation
    # This creates a foreground mask where all affinities are positive
    foreground = np.all(binary_affinities, axis=0)

    # Label connected components
    if foreground.ndim == 3:
        instances, num_instances = ndimage.label(foreground)
    else:
        instances, num_instances = ndimage.label(foreground)

    return instances


# Example usage and testing
if __name__ == "__main__":
    print("Testing affinity computation...")

    # Create a simple test instance segmentation
    instances = np.zeros((10, 10, 10), dtype=np.int32)
    instances[2:5, 2:5, 2:5] = 1  # First instance
    instances[6:9, 6:9, 6:9] = 2  # Second instance

    print(f"Instance segmentation shape: {instances.shape}")
    print(f"Number of instances: {len(np.unique(instances)) - 1}")  # -1 for background

    # Compute affinities
    affinities = compute_affinities_3d(instances)
    print(f"Affinities shape: {affinities.shape}")
    print(f"Affinity ranges: [{affinities.min()}, {affinities.max()}]")

    # Test with PyTorch
    instances_torch = torch.from_numpy(instances)
    affinities_torch = compute_affinities_3d(instances_torch)
    print(f"PyTorch affinities shape: {affinities_torch.shape}")

    # Test loss
    predictions = torch.randn(2, 3, 10, 10, 10)  # batch=2, offsets=3
    targets = affinities_torch.unsqueeze(0).repeat(2, 1, 1, 1, 1)

    loss_fn = AffinityLoss()
    loss = loss_fn(predictions, targets)
    print(f"Test loss: {loss.item():.4f}")

    print("\n✓ All tests passed!")
