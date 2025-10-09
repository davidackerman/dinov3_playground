"""
Loss functions for segmentation tasks.

This module provides various loss functions optimized for class-imbalanced
segmentation, including Focal Loss and Dice Loss variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002

    The focal loss applies a modulating term to the cross entropy loss to focus
    learning on hard examples and down-weight easy examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
    - p_t is the model's estimated probability for the true class
    - alpha_t is the class weight
    - gamma is the focusing parameter (default: 2.0)

    Parameters:
    -----------
    alpha : torch.Tensor, optional
        Class weights. Shape: (num_classes,)
        If None, all classes are weighted equally.
    gamma : float
        Focusing parameter. Higher values put more focus on hard examples.
        - gamma = 0: Equivalent to CrossEntropyLoss
        - gamma = 2: Default from paper, works well in practice
        - gamma > 2: More aggressive focusing on hard examples
    reduction : str
        Specifies the reduction to apply: 'none', 'mean', or 'sum'

    Example:
    --------
    >>> # For 4-class segmentation with class weights
    >>> weights = torch.tensor([1.0, 5.0, 10.0, 20.0])
    >>> criterion = FocalLoss(alpha=weights, gamma=2.0)
    >>>
    >>> # logits: (batch, classes, depth, height, width)
    >>> # targets: (batch, depth, height, width)
    >>> loss = criterion(logits, targets)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute focal loss.

        Parameters:
        -----------
        inputs : torch.Tensor
            Model predictions (logits). Shape: (N, C, ...) where C is num_classes
        targets : torch.Tensor
            Ground truth labels. Shape: (N, ...)

        Returns:
        --------
        torch.Tensor
            Focal loss value
        """
        # Compute cross entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)

        # Get probability of true class
        # p_t = exp(-ce_loss) because ce_loss = -log(p_t)
        p_t = torch.exp(-ce_loss)

        # Apply focal term: (1 - p_t)^gamma
        # When p_t is high (easy example), (1-p_t)^gamma is small → loss is down-weighted
        # When p_t is low (hard example), (1-p_t)^gamma is large → loss is emphasized
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.

    The Dice coefficient measures overlap between predicted and ground truth
    segmentations. Dice Loss = 1 - Dice Coefficient.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)

    This is very similar to IoU (Jaccard index) and directly optimizes for
    segmentation overlap quality.

    Parameters:
    -----------
    smooth : float
        Smoothing constant to avoid division by zero and smooth gradients.
        Default: 1.0 (Laplace smoothing)
    per_class : bool
        If True, compute Dice loss per class and average.
        If False, compute global Dice across all classes.
        Default: True (recommended for multi-class)

    Example:
    --------
    >>> criterion = DiceLoss(smooth=1.0)
    >>> loss = criterion(logits, targets, num_classes=4)
    """

    def __init__(self, smooth=1.0, per_class=True):
        super().__init__()
        self.smooth = smooth
        self.per_class = per_class

    def forward(self, inputs, targets, num_classes):
        """
        Compute Dice loss.

        Parameters:
        -----------
        inputs : torch.Tensor
            Model predictions (logits). Shape: (N, C, D, H, W)
        targets : torch.Tensor
            Ground truth labels. Shape: (N, D, H, W)
        num_classes : int
            Number of classes

        Returns:
        --------
        torch.Tensor
            Dice loss value (1 - Dice coefficient)
        """
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)

        # One-hot encode targets
        # targets: (N, D, H, W) -> (N, C, D, H, W)
        targets_one_hot = F.one_hot(targets, num_classes)
        # Permute to (N, C, D, H, W)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        if self.per_class:
            # Compute Dice per class and average
            dice_loss = 0
            for c in range(num_classes):
                pred_c = inputs[:, c]
                target_c = targets_one_hot[:, c]

                # Flatten spatial dimensions
                pred_c = pred_c.reshape(-1)
                target_c = target_c.reshape(-1)

                # Compute intersection and union
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()

                # Dice coefficient with smoothing
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_loss += 1 - dice

            return dice_loss / num_classes
        else:
            # Global Dice across all classes
            inputs_flat = inputs.reshape(-1)
            targets_flat = targets_one_hot.reshape(-1)

            intersection = (inputs_flat * targets_flat).sum()
            union = inputs_flat.sum() + targets_flat.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice


class FocalDiceLoss(nn.Module):
    """
    Combined Focal Loss and Dice Loss for robust segmentation.

    This combines the benefits of both losses:
    - Focal Loss: Handles class imbalance by focusing on hard examples
    - Dice Loss: Directly optimizes IoU/overlap metric

    The combined loss provides stable training (from Focal) while optimizing
    the target metric (from Dice).

    Parameters:
    -----------
    alpha : torch.Tensor, optional
        Class weights for Focal Loss. Shape: (num_classes,)
    gamma : float
        Focusing parameter for Focal Loss. Default: 2.0
    dice_smooth : float
        Smoothing constant for Dice Loss. Default: 1.0
    focal_weight : float
        Weight for Focal Loss component. Default: 0.5
    dice_weight : float
        Weight for Dice Loss component. Default: 0.5

    Example:
    --------
    >>> weights = torch.tensor([1.0, 5.0, 10.0, 20.0])
    >>> criterion = FocalDiceLoss(
    ...     alpha=weights,
    ...     gamma=2.0,
    ...     focal_weight=0.5,
    ...     dice_weight=0.5
    ... )
    >>> loss = criterion(logits, targets, num_classes=4)
    """

    def __init__(
        self, alpha=None, gamma=2.0, dice_smooth=1.0, focal_weight=0.5, dice_weight=0.5
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(smooth=dice_smooth)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets, num_classes):
        """
        Compute combined Focal + Dice loss.

        Parameters:
        -----------
        inputs : torch.Tensor
            Model predictions (logits). Shape: (N, C, D, H, W)
        targets : torch.Tensor
            Ground truth labels. Shape: (N, D, H, W)
        num_classes : int
            Number of classes

        Returns:
        --------
        torch.Tensor
            Combined loss value
        """
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets, num_classes)

        combined_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss

        return combined_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - a generalization of Dice Loss with adjustable FP/FN tradeoff.

    The Tversky index allows you to control the balance between false positives
    and false negatives, which is useful when you want to emphasize precision
    or recall.

    Tversky = TP / (TP + alpha*FP + beta*FN)

    Parameters:
    -----------
    alpha : float
        Weight for false positives. Default: 0.5
        - Higher alpha → Penalize FP more → Emphasize precision
    beta : float
        Weight for false negatives. Default: 0.5
        - Higher beta → Penalize FN more → Emphasize recall
    smooth : float
        Smoothing constant. Default: 1.0

    Special cases:
    - alpha=beta=0.5: Equivalent to Dice Loss
    - alpha=beta=1.0: Equivalent to Tanimoto/Jaccard (IoU)
    - alpha<beta: Emphasize recall (reduce false negatives)
    - alpha>beta: Emphasize precision (reduce false positives)

    Example:
    --------
    >>> # For over-prediction problem, penalize FP more
    >>> criterion = TverskyLoss(alpha=0.7, beta=0.3)
    >>> loss = criterion(logits, targets, num_classes=4)
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets, num_classes):
        """
        Compute Tversky loss.

        Parameters:
        -----------
        inputs : torch.Tensor
            Model predictions (logits). Shape: (N, C, D, H, W)
        targets : torch.Tensor
            Ground truth labels. Shape: (N, D, H, W)
        num_classes : int
            Number of classes

        Returns:
        --------
        torch.Tensor
            Tversky loss value
        """
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        tversky_loss = 0
        for c in range(num_classes):
            pred_c = inputs[:, c]
            target_c = targets_one_hot[:, c]

            # Flatten spatial dimensions
            pred_c = pred_c.reshape(-1)
            target_c = target_c.reshape(-1)

            # Compute TP, FP, FN
            TP = (pred_c * target_c).sum()
            FP = (pred_c * (1 - target_c)).sum()
            FN = ((1 - pred_c) * target_c).sum()

            # Tversky index
            tversky = (TP + self.smooth) / (
                TP + self.alpha * FP + self.beta * FN + self.smooth
            )
            tversky_loss += 1 - tversky

        return tversky_loss / num_classes


def get_loss_function(loss_type, class_weights=None, **kwargs):
    """
    Factory function to get loss function by name.

    Parameters:
    -----------
    loss_type : str
        Type of loss function:
        - 'ce': Standard CrossEntropyLoss
        - 'weighted_ce': Weighted CrossEntropyLoss
        - 'focal': Focal Loss
        - 'dice': Dice Loss
        - 'focal_dice': Combined Focal + Dice Loss
        - 'tversky': Tversky Loss
        - 'affinity': Affinity Loss (BCE with class weighting)
        - 'boundary_affinity': Boundary-Weighted Affinity Loss (EDT-based)
        - 'affinity_lsds': Combined Affinity + LSDS Loss (BCE + MSE)
    class_weights : torch.Tensor, optional
        Class weights for weighted losses
    **kwargs : dict
        Additional arguments for specific losses:
        - gamma: Focal loss focusing parameter (default: 2.0)
        - focal_weight: Weight for focal component (default: 0.5)
        - dice_weight: Weight for dice component (default: 0.5)
        - dice_smooth: Smoothing for Dice loss (default: 1.0)
        - alpha: Tversky alpha parameter (default: 0.5)
        - beta: Tversky beta parameter (default: 0.5)
        - use_class_weights: For affinity losses (default: True)
        - pos_weight: Positive class weight for affinity losses (default: None)
        - boundary_weight: Max weight at boundaries for boundary_affinity (default: 10.0)
        - sigma: Distance decay for boundary_affinity (default: 5.0)
        - anisotropy: Voxel anisotropy tuple for boundary_affinity (default: (1.0, 1.0, 1.0))
        - num_lsds: Number of LSDS channels for affinity_lsds (default: 10)
        - lsds_weight: Weight for LSDS component in affinity_lsds (default: 1.0)
        - affinity_weight: Weight for affinity component in affinity_lsds (default: 1.0)

    Returns:
    --------
    nn.Module
        Loss function instance

    Example:
    --------
    >>> # Standard affinity loss
    >>> criterion = get_loss_function('affinity', use_class_weights=True)
    >>>
    >>> # Boundary-weighted affinity loss with custom parameters
    >>> criterion = get_loss_function(
    ...     'boundary_affinity',
    ...     boundary_weight=15.0,  # Strong boundary emphasis
    ...     sigma=3.0,              # Narrow boundary region
    ...     anisotropy=(2.0, 1.0, 1.0)  # 2nm z, 1nm xy
    ... )
    >>>
    >>> # Combined affinity + LSDS loss
    >>> criterion = get_loss_function(
    ...     'affinity_lsds',
    ...     lsds_weight=1.0,       # Weight for LSDS MSE loss
    ...     affinity_weight=1.0    # Weight for affinity BCE loss
    ... )
    """
    loss_type = loss_type.lower()

    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    elif loss_type == "weighted_ce":
        if class_weights is None:
            raise ValueError("class_weights required for weighted_ce loss")
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_type == "focal":
        gamma = kwargs.get("gamma", 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)

    elif loss_type == "dice":
        dice_smooth = kwargs.get("dice_smooth", 1.0)
        return DiceLoss(smooth=dice_smooth)

    elif loss_type == "focal_dice":
        gamma = kwargs.get("gamma", 2.0)
        dice_smooth = kwargs.get("dice_smooth", 1.0)
        focal_weight = kwargs.get("focal_weight", 0.5)
        dice_weight = kwargs.get("dice_weight", 0.5)
        return FocalDiceLoss(
            alpha=class_weights,
            gamma=gamma,
            dice_smooth=dice_smooth,
            focal_weight=focal_weight,
            dice_weight=dice_weight,
        )

    elif loss_type == "tversky":
        alpha = kwargs.get("alpha", 0.5)
        beta = kwargs.get("beta", 0.5)
        smooth = kwargs.get("dice_smooth", 1.0)
        return TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    elif loss_type == "affinity":
        # Import here to avoid circular imports
        from .affinity_utils import AffinityLoss

        use_class_weights = kwargs.get("use_class_weights", True)
        pos_weight = kwargs.get("pos_weight", None)
        return AffinityLoss(use_class_weights=use_class_weights, pos_weight=pos_weight)

    elif loss_type == "boundary_affinity":
        # Import here to avoid circular imports
        from .affinity_utils import BoundaryWeightedAffinityLoss

        boundary_weight = kwargs.get("boundary_weight", 10.0)
        sigma = kwargs.get("sigma", 5.0)
        anisotropy = kwargs.get("anisotropy", (1.0, 1.0, 1.0))
        use_class_weights = kwargs.get("use_class_weights", True)
        pos_weight = kwargs.get("pos_weight", None)
        return BoundaryWeightedAffinityLoss(
            boundary_weight=boundary_weight,
            sigma=sigma,
            anisotropy=anisotropy,
            use_class_weights=use_class_weights,
            pos_weight=pos_weight,
        )

    elif loss_type == "affinity_lsds":
        # Import here to avoid circular imports
        from .affinity_utils import AffinityLSDSLoss

        num_lsds = kwargs.get("num_lsds", 10)
        use_class_weights = kwargs.get("use_class_weights", True)
        pos_weight = kwargs.get("pos_weight", None)
        lsds_weight = kwargs.get("lsds_weight", 1.0)
        affinity_weight = kwargs.get("affinity_weight", 1.0)
        return AffinityLSDSLoss(
            num_lsds=num_lsds,
            use_class_weights=use_class_weights,
            pos_weight=pos_weight,
            lsds_weight=lsds_weight,
            affinity_weight=affinity_weight,
        )

    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Choose from: ce, weighted_ce, focal, dice, focal_dice, tversky, affinity, boundary_affinity, affinity_lsds"
        )
