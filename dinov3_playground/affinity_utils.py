"""
Affinity Utilities for Instance Segmentation

This module provides functions to convert instance segmentations to affinity graphs
for training neural networks to predict pixel-to-pixel affinities.

Author: GitHub Copilot
Date: 2025-10-06
"""

# %%
import numpy as np
import torch
import torch.nn.functional as F
import fastmorph

try:
    import edt

    HAS_EDT = True
except ImportError:
    HAS_EDT = False
    print("Warning: edt not available. Install with: pip install edt")
    print("Boundary-weighted loss will not be available.")

# Try to import lsd_lite's get_affs/get_lsds (C/NumPy implementation)
try:
    from lsd_lite import get_affs as _get_affs_lib, get_lsds as _get_lsds_lib

    HAS_LSDS_LIB = True
except Exception:
    _get_affs_lib = None
    _get_lsds_lib = None
    HAS_LSDS_LIB = False
    print("Warning: lsd_lite package not available. Install with: pip install lsd_lite")
    print("lsd_lite.get_affs/get_lsds will not be available.")

# Prefer our PyTorch implementation if available
try:
    from .lsd_utils import get_lsds_torch

    HAS_LSDS_TORCH = True
except Exception:
    get_lsds_torch = None
    HAS_LSDS_TORCH = False

# Presence of any LSDS backend
HAS_LSDS = HAS_LSDS_LIB or HAS_LSDS_TORCH
get_affs = _get_affs_lib
get_lsds_original = _get_lsds_lib


def per_sample_mean(x, mask, weights=None, eps=1e-8):
    """
    x:       [B, C, D, H, W] per-voxel loss (non-reduced)
    mask:    [B,   D, H, W]  0/1
    weights: [B, 1,D, H, W]  or [B, C, D, H, W] (optional)
    Returns scalar: mean over samples of (weighted mean over that sample's valid voxels)
    Also returns per-sample denominators for logging.
    """
    B = x.shape[0]
    m = mask.unsqueeze(1).expand_as(x)  # [B,C,D,H,W]
    xw = x * m

    if weights is not None:
        w = weights
        if w.dim() == 5 and w.shape[1] == 1:
            w = w.expand_as(x)
        xw = xw * w
        denom = (m * w).sum(dim=(1, 2, 3, 4)) + eps  # [B]
    else:
        denom = m.sum(dim=(1, 2, 3, 4)) + eps  # [B]
    per_samp = xw.sum(dim=(1, 2, 3, 4)) / denom  # [B]
    return per_samp.mean(), denom  # scalar, [B]


def safe_get_lsds(segmentation, sigma=20.0):
    """
    Safely compute LSDs, handling all-zero arrays.

    get_lsds() fails when the array contains only zeros (no instances/labels).
    This wrapper returns zeros with the correct shape in that case.

    Parameters:
    -----------
    segmentation : numpy.ndarray
        Instance segmentation where each unique non-zero value represents
        a different instance
    sigma : float, default=20.0
        Sigma parameter for LSDS computation (controls smoothing scale)

    Returns:
    --------
    numpy.ndarray
        LSDs of shape (10, *segmentation.shape)
    """
    if not HAS_LSDS:
        raise ImportError(
            "No LSDS backend available. Install lsd_lite or ensure get_lsds_torch is importable."
        )

    # If segmentation is empty (no labels) return zeros of correct shape
    if not np.any(segmentation > 0):
        return np.zeros((10, *segmentation.shape), dtype=np.float32)

    # Prefer the PyTorch implementation if available
    if HAS_LSDS_TORCH and get_lsds_torch is not None:
        return get_lsds_torch(segmentation, sigma=sigma)

    # Fallback to lsd_lite implementation
    if HAS_LSDS_LIB and get_lsds_original is not None:
        return get_lsds_original(segmentation, sigma=sigma)

    # Shouldn't reach here due to HAS_LSDS guard, but keep safe
    raise ImportError("LSDS computation failed: no available backend")


def compute_affinities_3d(instance_segmentation, offsets=None):
    """
    Convert 3D instance segmentation to affinity graph using lsd_lite.

    This is a wrapper around lsd_lite.get_affs for backward compatibility.
    Affinities indicate whether neighboring pixels belong to the same instance.

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
    if not HAS_LSDS:
        raise ImportError(
            "lsd_lite package required for affinity computation. "
            "Install with: pip install lsd_lite"
        )

    if offsets is None:
        # Default: affinities in +z, +y, +x directions
        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    is_torch = isinstance(instance_segmentation, torch.Tensor)

    # Convert to numpy if needed
    if is_torch:
        device = instance_segmentation.device
        instance_seg_np = instance_segmentation.cpu().numpy()
    else:
        instance_seg_np = instance_segmentation

    # Convert offsets to numpy array format expected by lsd_lite
    neighborhood = np.array(offsets, dtype=np.int32)

    # Compute affinities using lsd_lite (exclude background with dist="equality-no-bg")
    affinities = get_affs(instance_seg_np, neighborhood, dist="equality-no-bg")

    # Convert back to torch if needed
    if is_torch:
        affinities = torch.from_numpy(affinities).to(device)

    return affinities


def compute_affinities_2d(instance_segmentation, offsets=None):
    """
    Convert 2D instance segmentation to affinity graph using lsd_lite.

    This is a wrapper around lsd_lite.get_affs for backward compatibility.

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
    if not HAS_LSDS:
        raise ImportError(
            "lsd_lite package required for affinity computation. "
            "Install with: pip install lsd_lite"
        )

    if offsets is None:
        # Default: affinities in +y, +x directions
        offsets = [(1, 0), (0, 1)]

    is_torch = isinstance(instance_segmentation, torch.Tensor)

    # Convert to numpy if needed
    if is_torch:
        device = instance_segmentation.device
        instance_seg_np = instance_segmentation.cpu().numpy()
    else:
        instance_seg_np = instance_segmentation

    # Convert offsets to numpy array format expected by lsd_lite
    neighborhood = np.array(offsets, dtype=np.int32)

    # Compute affinities using lsd_lite (exclude background with dist="equality-no-bg")
    affinities = get_affs(instance_seg_np, neighborhood, dist="equality-no-bg")

    # Convert back to torch if needed
    if is_torch:
        affinities = torch.from_numpy(affinities).to(device)

    return affinities


def compute_boundary_weights(
    instance_segmentation: np.ndarray,
    boundary_weight: float = 5.0,
    sigma: float = 5.0,
    anisotropy=(1.0, 1.0, 1.0),
    mask: np.ndarray | None = None,
    black_border: bool = False,
):
    """
    Emphasize boundaries between:
      1) two different non-zero instances
      2) instance (>0) and background (0)
    ...but ONLY when both sides lie within `mask` (so mask edges are ignored).
    """
    seg = np.asarray(instance_segmentation)
    assert seg.ndim in (2, 3), "seg must be 2D or 3D"
    ndim = seg.ndim

    # Valid region; outside stays weight=1
    allow = np.ones_like(seg, dtype=bool) if mask is None else mask.astype(bool)

    if np.count_nonzero(seg) == 0:
        w = np.ones_like(seg, dtype=np.float32)
        w[~allow] = 1.0
        return w

    # Build boundary map inside the mask:
    # - instance≠instance
    # - instance vs background
    boundary = np.zeros_like(seg, dtype=bool)
    inside = allow  # alias

    for ax in range(ndim):
        # roll forward
        fwd = np.roll(seg, -1, axis=ax)
        in_fwd = np.roll(inside, -1, axis=ax)
        # avoid wrap-around at far edge
        slicer = [slice(None)] * ndim
        slicer[ax] = -1
        in_fwd[tuple(slicer)] = False

        # roll backward
        bwd = np.roll(seg, 1, axis=ax)
        in_bwd = np.roll(inside, 1, axis=ax)
        slicer[ax] = 0
        in_bwd[tuple(slicer)] = False

        # instance↔instance (different ids), both sides inside mask
        meet_inst_fwd = (seg > 0) & (fwd > 0) & (seg != fwd) & inside & in_fwd
        meet_inst_bwd = (seg > 0) & (bwd > 0) & (seg != bwd) & inside & in_bwd

        # instance↔background, both sides inside mask
        meet_bg_fwd = ((seg > 0) & (fwd == 0) & inside & in_fwd) | (
            (seg == 0) & (fwd > 0) & inside & in_fwd
        )
        meet_bg_bwd = ((seg > 0) & (bwd == 0) & inside & in_bwd) | (
            (seg == 0) & (bwd > 0) & inside & in_bwd
        )

        boundary |= meet_inst_fwd | meet_inst_bwd | meet_bg_fwd | meet_bg_bwd

    # Optional: treat image border as boundary, but only where inside==True
    if black_border:
        for ax in range(ndim):
            slicer0 = [slice(None)] * ndim
            slicerN = [slice(None)] * ndim
            slicer0[ax] = 0
            slicerN[ax] = -1
            boundary[tuple(slicer0)] |= inside[tuple(slicer0)]
            boundary[tuple(slicerN)] |= inside[tuple(slicerN)]

    # If nothing marked, return uniform weights
    if not boundary.any():
        w = np.ones_like(seg, dtype=np.float32)
        w[~allow] = 1.0
        return w

    # Distance to the nearest *true* boundary (edt on ~boundary)
    dist = edt.edt(~boundary, anisotropy=anisotropy, black_border=False).astype(
        np.float32
    )

    # Outside mask: make infinitely far so weight becomes 1.0
    dist[~allow] = np.inf

    # Gaussian falloff
    gaussian = np.exp(-(dist**2) / (2.0 * (sigma**2)))
    w = 1.0 + (boundary_weight - 1.0) * gaussian
    w[~allow] = 1.0
    return w


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
        loss = criterion(predictions, targets * 0.96 + 0.02)

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
        boundary_weight=5.0,
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
                black_border=False,
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


class AffinityFocalLoss(torch.nn.Module):
    """
    Focal loss for affinity prediction (binary per-pixel affinities).

    Parameters:
    -----------
    alpha : float
        Weighting factor for the positive class in focal loss (0-1). Default 0.25.
    gamma : float
        Focusing parameter. Default 2.0.
    use_class_weights : bool
        Unused for focal but kept for API compatibility.
    """

    def __init__(self, alpha=0.25, gamma=2.0, use_class_weights=True, pos_weight=None):
        super(AffinityFocalLoss, self).__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.use_class_weights = use_class_weights
        self.pos_weight = pos_weight

    def forward(self, predictions, targets, mask=None):
        """
        predictions: logits (batch, num_offsets, D, H, W)
        targets: binary {0,1} same shape
        mask: optional valid-region mask (batch, D, H, W)
        """
        # per-pixel BCE (numerically stable)
        bce = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        # p_t = exp(-bce) since bce = -log(p_t)
        p_t = torch.exp(-bce)
        focal_weight = (1.0 - p_t) ** self.gamma

        # alpha factor: alpha for positives, (1-alpha) for negatives
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        loss = alpha_factor * focal_weight * bce

        # apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(loss)
            loss = loss * mask_expanded
            num_valid = mask_expanded.sum()
            if num_valid > 0:
                loss = loss.sum() / num_valid
            else:
                loss = loss.sum()
        else:
            loss = loss.mean()

        return loss


class BoundaryWeightedAffinityFocalLoss(torch.nn.Module):
    """
    Boundary-weighted focal loss variant (applies focal loss per-pixel then weights by boundary map)
    """

    def __init__(
        self,
        boundary_weight=5.0,
        sigma=5.0,
        anisotropy=(1.0, 1.0, 1.0),
        alpha=0.25,
        gamma=2.0,
        use_class_weights=True,
        pos_weight=None,
    ):
        super(BoundaryWeightedAffinityFocalLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.sigma = sigma
        self.anisotropy = anisotropy
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.use_class_weights = use_class_weights
        self.pos_weight = pos_weight

        if not HAS_EDT:
            raise ImportError(
                "edt package required for boundary-weighted loss. "
                "Install with: pip install edt"
            )

    def forward(self, predictions, targets, instance_seg=None, mask=None):
        batch_size = predictions.shape[0]

        if instance_seg is None:
            # fallback to focal affinity without boundary weighting
            fallback = AffinityFocalLoss(alpha=self.alpha, gamma=self.gamma)
            return fallback(predictions, targets, mask)

        # Convert instance_seg to numpy if needed
        if isinstance(instance_seg, torch.Tensor):
            instance_seg_np = instance_seg.cpu().numpy()
        else:
            instance_seg_np = instance_seg

        if instance_seg_np.ndim == 3:
            instance_seg_np = instance_seg_np[np.newaxis, ...]

        boundary_weights_list = []
        for b in range(batch_size):
            weights = compute_boundary_weights(
                instance_seg_np[b],
                boundary_weight=self.boundary_weight,
                sigma=self.sigma,
                anisotropy=self.anisotropy,
                black_border=False,
            )
            boundary_weights_list.append(weights)

        boundary_weights = np.stack(boundary_weights_list, axis=0)
        boundary_weights = torch.from_numpy(boundary_weights).to(
            predictions.device, dtype=predictions.dtype
        )

        boundary_weights = boundary_weights.unsqueeze(1).expand_as(predictions)

        # compute focal per-pixel (without mask/weighting)
        bce = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = alpha_factor * focal_weight * bce

        # apply boundary weights
        loss = loss * boundary_weights

        # apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(loss)
            loss = loss * mask_expanded
            weighted_valid = (boundary_weights * mask_expanded).sum()
            if weighted_valid > 0:
                loss = loss.sum() / weighted_valid
            else:
                loss = loss.sum()
        else:
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


def compute_affinities_and_lsds_3d(
    instance_segmentation, offsets=None, lsds_sigma=20.0
):
    """
    Convert 3D instance segmentation to both affinities and LSDs.

    Uses the lsd_lite package to compute affinities and Local Shape Descriptors.
    LSDs provide 10 channels of shape information that complement affinities.

    Parameters:
    -----------
    instance_segmentation : numpy.ndarray
        3D instance segmentation of shape (D, H, W) where each unique non-zero
        value represents a different instance
    offsets : list of tuples, optional
        List of (z, y, x) offsets for affinities. Default is [(1,0,0), (0,1,0), (0,0,1)]
    lsds_sigma : float, default=20.0
        Sigma parameter for LSDS computation (controls smoothing scale)

    Returns:
    --------
    numpy.ndarray
        Combined array of shape (10 + num_offsets, D, H, W) where:
        - First 10 channels are LSDs
        - Remaining channels are affinities
    """
    # Ensure we have an affinity backend
    if get_affs is None:
        raise ImportError(
            "No affinity backend available. Install lsd_lite to compute affinities (get_affs)."
        )

    if offsets is None:
        # Default: affinities in +z, +y, +x directions
        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    # Convert to numpy if needed
    if isinstance(instance_segmentation, torch.Tensor):
        instance_segmentation = instance_segmentation.cpu().numpy()

    # Convert offsets to numpy array format expected by lsds
    neighborhood = np.array(offsets, dtype=np.int32)

    # Compute affinities using the available get_affs backend (exclude background with dist="equality-no-bg")
    # get_affs expects (gt_seg, neighborhood) and returns (num_offsets, D, H, W)
    affinities = get_affs(instance_segmentation, neighborhood, dist="equality-no-bg")

    # Compute LSDs using safe wrapper (handles all-zero arrays).
    # safe_get_lsds will prefer the PyTorch implementation if available, otherwise fall back to lsd_lite.
    lsds = safe_get_lsds(instance_segmentation, sigma=lsds_sigma)

    # Concatenate LSDs and affinities: (10 + num_offsets, D, H, W)
    combined = np.concatenate([lsds, affinities], axis=0)

    return combined


class AffinityLSDSLoss(torch.nn.Module):
    """
    Combined loss for affinity and LSDS prediction.

    Computes MSE loss for LSDS channels and BCE loss for affinity channels.
    The total loss is the sum of LSDS MSE loss and affinity BCE loss.

    Parameters:
    -----------
    num_lsds : int, default=10
        Number of LSDS channels (typically 10)
    use_class_weights : bool, default=True
        Whether to use class weights for affinity loss
    pos_weight : float, optional
        Weight for positive affinity examples
    lsds_weight : float, default=1.0
        Weight for LSDS loss component
    affinity_weight : float, default=1.0
        Weight for affinity loss component
    """

    def __init__(
        self,
        num_lsds=10,
        use_class_weights=True,
        pos_weight=None,
        lsds_weight=0.25,
        affinity_weight=1.0,
    ):
        super(AffinityLSDSLoss, self).__init__()
        self.num_lsds = num_lsds
        self.use_class_weights = use_class_weights
        self.pos_weight = pos_weight
        self.lsds_weight = lsds_weight
        self.affinity_weight = affinity_weight

        # MSE loss for LSDS
        self.mse_loss = torch.nn.MSELoss(reduction="none")

    def forward(self, predictions, targets, mask=None):
        """
        Compute combined affinity and LSDS loss.

        Parameters:
        -----------
        predictions : torch.Tensor
            Predicted affinities and LSDs of shape (batch, 10 + num_offsets, D, H, W)
            First 10 channels are LSDS predictions, remaining are affinity predictions
        targets : torch.Tensor
            Target affinities and LSDs of shape (batch, 10 + num_offsets, D, H, W)
            First 10 channels are LSDS targets, remaining are affinity targets
        mask : torch.Tensor, optional
            Valid region mask of shape (batch, D, H, W)

        Returns:
        --------
        torch.Tensor
            Scalar combined loss value
        """
        # Split predictions and targets into LSDS and affinities
        pred_lsds = predictions[:, : self.num_lsds, ...]  # (batch, 10, D, H, W)
        pred_affs = predictions[
            :, self.num_lsds :, ...
        ]  # (batch, num_offsets, D, H, W)

        target_lsds = targets[:, : self.num_lsds, ...]  # (batch, 10, D, H, W)
        target_affs = targets[:, self.num_lsds :, ...]  # (batch, num_offsets, D, H, W)

        # Compute LSDS loss (MSE on sigmoid(predictions) vs targets)
        lsds_loss_per_pixel = self.mse_loss(torch.sigmoid(pred_lsds), target_lsds)

        # print distribution of values for torch.sigmoid(pred_lsds) and target_lsds
        def _stats(tensor):
            try:
                arr = tensor.detach().cpu().numpy().ravel()
                if arr.size == 0:
                    return {
                        "mean": np.nan,
                        "median": np.nan,
                        "mode": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                    }
                mean = float(np.mean(arr))
                median = float(np.median(arr))
                mn = float(np.min(arr))
                mx = float(np.max(arr))
                counts, edges = np.histogram(arr, bins=100)
                centers = (edges[:-1] + edges[1:]) / 2.0
                mode = float(centers[np.argmax(counts)])
                return {
                    "mean": mean,
                    "median": median,
                    "mode": mode,
                    "min": mn,
                    "max": mx,
                }
            except Exception as e:
                print(f"Error computing stats: {e}")
                return {
                    "mean": np.nan,
                    "median": np.nan,
                    "mode": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }

        pred_lsds_sig = torch.sigmoid(pred_lsds)
        pred_affs_prob = torch.sigmoid(pred_affs)

        for name, tensor in [
            ("pred_lsds (sigmoid)", pred_lsds_sig),
            ("target_lsds", target_lsds),
            ("pred_affs (prob)", pred_affs_prob),
            ("target_affs", target_affs),
        ]:
            s = _stats(tensor)
            print(
                f"{name}: mean={s['mean']:.6f}, median={s['median']:.6f}, "
                f"mode≈{s['mode']:.6f}, min={s['min']:.6f}, max={s['max']:.6f}"
            )
        if mask is not None:
            # Expand mask to match LSDS channels
            mask_expanded = mask.unsqueeze(1).expand_as(lsds_loss_per_pixel)
            lsds_loss_per_pixel = lsds_loss_per_pixel * mask_expanded

            # Average only over valid regions
            num_valid = mask_expanded.sum()
            if num_valid > 0:
                lsds_loss = lsds_loss_per_pixel.sum() / num_valid
            else:
                lsds_loss = lsds_loss_per_pixel.sum()
        else:
            lsds_loss = lsds_loss_per_pixel.mean()

        # Compute affinity loss (BCE)
        # Calculate pos_weight if needed
        pos_weight = self.pos_weight
        if self.use_class_weights and pos_weight is None:
            # Count positive and negative examples
            if mask is not None:
                mask_expanded_aff = mask.unsqueeze(1).expand_as(target_affs)
                valid_targets = target_affs[mask_expanded_aff > 0]
            else:
                valid_targets = target_affs

            num_pos = (valid_targets > 0.5).sum().float()
            num_neg = (valid_targets < 0.5).sum().float()

            if num_pos > 0:
                pos_weight = num_neg / num_pos
            else:
                pos_weight = 1.0

        # Use BCEWithLogitsLoss for numerical stability
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

        # Compute affinity loss
        affinity_loss_per_pixel = criterion(
            pred_affs, target_affs * 0.96 + 0.02
        )  # smooth toward 0.5 a bit

        # Apply mask if provided
        if mask is not None:
            mask_expanded_aff = mask.unsqueeze(1).expand_as(affinity_loss_per_pixel)
            affinity_loss_per_pixel = affinity_loss_per_pixel * mask_expanded_aff

            num_valid = mask_expanded_aff.sum()
            if num_valid > 0:
                affinity_loss = affinity_loss_per_pixel.sum() / num_valid
            else:
                affinity_loss = affinity_loss_per_pixel.sum()
        else:
            affinity_loss = affinity_loss_per_pixel.mean()

        # Combine losses
        total_loss = self.lsds_weight * lsds_loss + self.affinity_weight * affinity_loss

        return total_loss


# --- Added: Boundary-weighted focal variant for combined LSDS + affinity loss ---
class BoundaryWeightedAffinityFocalLSDSLoss(torch.nn.Module):
    """
    Combined LSDS MSE + boundary-weighted focal affinity loss.

    This class mirrors AffinityLSDSLoss but replaces the affinity BCE term
    with a boundary-weighted focal loss on affinity channels. Accepts an
    additional `instance_seg` argument to compute boundary weights.

    Parameters:
    -----------
    num_lsds : int
        Number of LSDS channels (default 10)
    boundary_weight, sigma, anisotropy : for boundary weighting (see compute_boundary_weights)
    alpha, gamma : focal parameters
    use_class_weights, pos_weight : class balancing (kept for API consistency)
    lsds_weight, affinity_weight : component weights
    """

    def __init__(
        self,
        num_lsds=10,
        boundary_weight=5.0,
        sigma=5.0,
        anisotropy=(1.0, 1.0, 1.0),
        alpha=0.25,
        gamma=2.0,
        pos_weight=None,
        lsds_weight=0.25,
        affinity_weight=1.0,
        boundary_weight_power=1.0,
        mask_clip_distance=None,
    ):
        super(BoundaryWeightedAffinityFocalLSDSLoss, self).__init__()
        self.num_lsds = num_lsds
        self.boundary_weight = boundary_weight
        self.sigma = sigma
        self.anisotropy = anisotropy
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.pos_weight = pos_weight
        self.lsds_weight = lsds_weight
        self.affinity_weight = affinity_weight
        self.mask_clip_distance = mask_clip_distance
        self.boundary_weight_power = boundary_weight_power

        # MSE loss for LSDS
        self.mse_loss = torch.nn.MSELoss(reduction="none")

        if not HAS_EDT:
            raise ImportError(
                "edt package required for boundary-weighted loss. "
                "Install with: pip install edt"
            )

    def erode_mask(self, mask: torch.Tensor, voxels: int) -> torch.Tensor:
        """
        Erode a 3D binary mask by `voxels` in all directions.
        mask: (B, D, H, W) bool/0-1 tensor (on any device)
        returns: (B, D, H, W) same dtype/device
        """
        if voxels is None or voxels <= 0:
            return mask
        # avg_pool3d equals fraction of ones in neighborhood; ==1 means all ones
        k = 2 * voxels + 1
        m = mask.float().unsqueeze(1)  # (B,1,D,H,W)
        pooled = F.avg_pool3d(m, kernel_size=k, stride=1, padding=voxels)
        eroded = (pooled == 1).squeeze(1)  # (B,D,H,W) bool
        return eroded.to(mask.dtype)

    def forward(
        self,
        predictions,
        targets,
        instance_seg=None,
        boundary_weights=None,
        mask=None,
        return_components=False,
    ):
        B = predictions.shape[0]
        device = predictions.device
        dtype = predictions.dtype

        pred_lsds = predictions[:, : self.num_lsds, ...]
        pred_affs = predictions[:, self.num_lsds :, ...]
        target_lsds = targets[:, : self.num_lsds, ...]
        target_affs = targets[:, self.num_lsds :, ...]

        if mask is None:
            mask = torch.ones(
                (B, predictions.shape[2], predictions.shape[3], predictions.shape[4]),
                device=device,
                dtype=dtype,
            )

        og_mask_np = mask.detach().to("cpu").numpy()

        if self.mask_clip_distance and self.mask_clip_distance > 0:
            mask = self.erode_mask(mask, self.mask_clip_distance)

        # ✅ Check for empty mask
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        # -------------------------
        # LSDS loss
        # -------------------------
        pred_lsds_sig = torch.sigmoid(pred_lsds)

        lsds_per_voxel = torch.nn.functional.smooth_l1_loss(
            pred_lsds_sig, target_lsds, reduction="none", beta=1.0
        )

        lsds_loss, lsds_denoms = per_sample_mean(lsds_per_voxel, mask)

        # -------------------------
        # Boundary weights
        # -------------------------
        if boundary_weights is None:
            if isinstance(instance_seg, torch.Tensor):
                instance_seg_np = instance_seg.detach().to("cpu").numpy()
            else:
                instance_seg_np = instance_seg

            if instance_seg_np.ndim == 3:
                instance_seg_np = instance_seg_np[np.newaxis, ...]

            bweights_list = []
            for b in range(B):
                weights = compute_boundary_weights(
                    instance_seg_np[b],
                    boundary_weight=self.boundary_weight,
                    sigma=self.sigma,
                    anisotropy=self.anisotropy,
                    mask=og_mask_np[b],
                    black_border=False,
                )
                bweights_list.append(weights)

            boundary_weights = torch.from_numpy(np.stack(bweights_list, axis=0)).to(
                device=device, dtype=dtype
            )

        # ✅ Clamp boundary weights
        # boundary_weights = torch.clamp(boundary_weights, min=0.1, max=10.0)
        boundary_weights = boundary_weights.detach()  # ✅ Detach here
        boundary_weights = boundary_weights**self.boundary_weight_power
        boundary_weights = boundary_weights.unsqueeze(1).expand_as(pred_affs)
        # -------------------------
        # ✅ STABLE Affinity focal loss
        # -------------------------
        target_affs_smooth = target_affs * 0.96 + 0.02

        # ✅ Clamp logits to prevent extreme values
        pred_affs_clamped = torch.clamp(pred_affs, min=-15, max=15)

        # Get probabilities directly (more stable than exp(logpt))
        pred_probs = torch.sigmoid(pred_affs_clamped)  # p

        # Compute focal weight: (1 - p_t)^gamma where p_t is prob of true class
        # If target=1, p_t = p;  if target=0, p_t = 1-p
        p_t = target_affs_smooth * pred_probs + (1 - target_affs_smooth) * (
            1 - pred_probs
        )
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)  # Numerical stability

        focal_weight = (1 - p_t).pow(self.gamma)

        # Alpha balancing
        alpha_factor = target_affs_smooth * self.alpha + (1 - target_affs_smooth) * (
            1 - self.alpha
        )

        # BCE loss
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_affs_clamped, target_affs_smooth, reduction="none"
        )

        # Final focal loss
        aff_per_voxel = alpha_factor * focal_weight * bce_loss

        # ✅ Check for NaN/Inf before applying weights
        if torch.isnan(aff_per_voxel).any() or torch.isinf(aff_per_voxel).any():
            print("⚠️ NaN/Inf in affinity loss before weighting")
            print(
                f"  pred_probs: min={pred_probs.min():.4f}, max={pred_probs.max():.4f}"
            )
            print(f"  p_t: min={p_t.min():.4f}, max={p_t.max():.4f}")
            print(
                f"  focal_weight: min={focal_weight.min():.4f}, max={focal_weight.max():.4f}"
            )
            print(f"  bce_loss: min={bce_loss.min():.4f}, max={bce_loss.max():.4f}")
            # Replace NaN with 0
            aff_per_voxel = torch.where(
                torch.isnan(aff_per_voxel) | torch.isinf(aff_per_voxel),
                torch.zeros_like(aff_per_voxel),
                aff_per_voxel,
            )

        # Apply boundary weights through per_sample_mean
        affinity_loss, _ = per_sample_mean(
            aff_per_voxel,
            mask,
            weights=boundary_weights,  # Original weights (max=5)
        )

        # ✅ Final checks
        if torch.isnan(lsds_loss) or torch.isinf(lsds_loss):
            print(f"⚠️ NaN/Inf in LSDS loss: {lsds_loss}")
            lsds_loss = torch.zeros_like(lsds_loss)

        if torch.isnan(affinity_loss) or torch.isinf(affinity_loss):
            print(f"⚠️ NaN/Inf in affinity loss: {affinity_loss}")
            affinity_loss = torch.zeros_like(affinity_loss)

        total_loss = self.lsds_weight * lsds_loss + self.affinity_weight * affinity_loss

        if return_components:
            out = {
                "total": float(total_loss.detach().cpu()),
                "lsds": float(lsds_loss.detach().cpu()),
                "affinity": float(affinity_loss.detach().cpu()),
                "lsds_denoms_mean": float(lsds_denoms.mean().detach().cpu()),
                "aff_denoms_mean": float(aff_denoms.mean().detach().cpu()),
            }
            return total_loss, out

        return total_loss


# %%
if __name__ == "__main__":
    import numpy as np

    training_data = np.load(
        "/groups/cellmap/cellmap/ackermand/Programming/dinov3_playground/examples/training_data.npz"
    )
    instance_seg = training_data["instance_seg"]
    boundary_weights = training_data["boundary_weights"]
    masks = training_data["masks"]
    pred_affs = training_data["pred_affs"]
    target_affs = training_data["target_affs"]
    pred_lsds = training_data["pred_lsds"]
    target_lsds = training_data["target_lsds"]
    og_mask = training_data["instance_seg"]

    import matplotlib.pyplot as plt

    # Visualize a slice of every type above
    for batch in range(boundary_weights.shape[0]):
        for slice_idx in range(0, boundary_weights.shape[2], 8):
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes[0, 0].imshow(
                boundary_weights[batch, 0, slice_idx], cmap="hot", interpolation="none"
            )
            axes[0, 0].set_title("Boundary Weights")
            axes[0, 1].imshow(
                masks[batch, slice_idx], cmap="gray", interpolation="none"
            )
            axes[0, 1].set_title("Mask")
            axes[0, 2].imshow(
                og_mask[batch, slice_idx], cmap="gray", interpolation="none"
            )
            axes[0, 2].set_title("Original Mask")
            axes[1, 0].imshow(
                pred_affs[batch, 0, slice_idx],
                cmap="viridis",
                vmin=0,
                vmax=1,
                interpolation="none",
            )
            axes[1, 0].set_title("Predicted Affinities")
            axes[1, 1].imshow(
                target_affs[batch, 1, slice_idx],
                cmap="viridis",
                vmin=0,
                vmax=1,
                interpolation="none",
            )
            axes[1, 1].set_title("Target Affinities")
            axes[1, 2].imshow(
                pred_lsds[batch, 0, slice_idx],
                cmap="viridis",
                vmin=0,
                vmax=1,
                interpolation="none",
            )
            axes[1, 2].set_title("Predicted LSDs")
            axes[2, 0].imshow(
                target_lsds[batch, 1, slice_idx],
                cmap="viridis",
                vmin=0,
                vmax=1,
                interpolation="none",
            )
            axes[2, 0].set_title("Target LSDs")
            axes[2, 1].imshow(
                instance_seg[batch, slice_idx], cmap="tab20", interpolation="none"
            )
            axes[2, 1].set_title("Instance Segmentation")
            axes[2, 2].axis("off")
            for ax in axes.flatten():
                ax.axis("off")
            plt.tight_layout()

        # %%
        # %%
        failed = np.load("failed.npz", allow_pickle=True)
        # %%
        # get all_vol_data form failed
        # all_vol_data = failed["all_vol_data"][0]
        # raw = all_vol_data["vol_raw"]
        # gt = all_vol_data["vol_seg"]
        # targets = all_vol_data["vol_targets"]
        from lsd_lite import get_affs, get_lsds

        gt = instance_seg[0]
        # gt upsample  to 128x128x128 from 32x32x32
        gt = np.kron(gt, np.ones((4, 4, 4), dtype=gt.dtype))

        new_affs = get_affs(
            gt,
            neighborhood=[
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (3, 0, 0),
                (0, 3, 0),
                (0, 0, 3),
                (9, 0, 0),
                (0, 9, 0),
                (0, 0, 9),
            ],
            dist="equality-no-bg",
            pad=9,
        )
        new_lsds = get_lsds(gt, sigma=20.0)
        import time

        t0 = time.time()
        my_affs = compute_affinities_and_lsds_3d(
            gt,
            offsets=[
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (3, 0, 0),
                (0, 3, 0),
                (0, 0, 3),
                (9, 0, 0),
                (0, 9, 0),
                (0, 0, 9),
            ],
            lsds_sigma=20.0,
        )
        print(f"My computation time: {time.time() - t0:.2f} seconds")
        # swap x and z axes for gt

        for i in range(len(gt)):
            plt.figure()
            # create two columns
            plt.subplot(1, 4, 1)
            print(np.unique(gt[i]))
            # color unique values as red green blue
            color_map = {0: (0, 0, 0), 4: (255, 0, 0), 220: (0, 255, 0)}
            colored_gt = np.zeros((*gt[:, i, :].shape, 3), dtype=np.uint8)
            for label, color in color_map.items():
                plt.title(f"Slice {i} - label {label}")
                colored_gt[gt[i] == label] = color
            plt.imshow(colored_gt)
            plt.subplot(1, 4, 2)
            plt.title(f"Slice {i} - affs")
            plt.imshow(targets[0, 10, i] > 0, vmin=0, vmax=1)
            plt.subplot(1, 4, 3)
            plt.title(f"Slice {i} - new_affs")
            plt.imshow(new_affs[0, i] > 0, vmin=0, vmax=1)
            plt.subplot(1, 4, 4)
            plt.title(f"Slice {i} - my_affs")
            plt.imshow(my_affs[10, i] > 0, vmin=0, vmax=1)
# %%
