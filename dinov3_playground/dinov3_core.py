"""
DINOv3 Core Processing Module

This module contains the core DINOv3 feature extraction functionality including:
- Model initialization and configuration
- Feature processing and normalization
- Utility functions for tensor operations

Author: GitHub Copilot
Date: 2025-09-11
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, DINOv3ViTModel
import functools


def enable_amp_inference(model, amp_dtype=torch.bfloat16):
    orig_forward = model.forward

    @functools.wraps(orig_forward)
    def amp_forward(*args, **kwargs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            return orig_forward(*args, **kwargs)

    model.forward = amp_forward
    return model


# Global variables to be initialized
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = None
model = None
output_channels = None
current_model_id = None


def initialize_dinov3(
    model_id="facebook/dinov3-vits16-pretrain-lvd1689m", image_size=896, device=None
):
    """
    Initialize DINOv3 model and processor with specified configuration.

    Parameters:
    -----------
    model_id : str, default="facebook/dinov3-vits16-pretrain-lvd1689m"
        HuggingFace model identifier
    image_size : int, default=896
        Input image size (height and width)
    device : torch.device, optional
        Device to load model on

    Returns:
    --------
    tuple: (processor, model, output_channels)
    """
    global processor, model, output_channels, current_model_id, DEVICE

    if device is not None:
        DEVICE = device

    ## Only reinitialize if model_id has changed
    # if current_model_id != model_id:
    print(f"Initializing DINOv3 with model: {model_id}")

    # Initialize processor
    processor = AutoImageProcessor.from_pretrained(model_id)

    # Override the default size to handle specified image size
    processor.size = {"height": image_size, "width": image_size}
    processor.crop_size = {"height": image_size, "width": image_size}

    # Initialize model
    model = DINOv3ViTModel.from_pretrained(model_id)
    model = enable_amp_inference(model, torch.bfloat16)
    model.to(DEVICE)
    model.eval()

    # Get output dimensions from model configuration
    output_channels = model.config.hidden_size
    current_model_id = model_id

    print(f"DINOv3 initialized:")
    print(f"  Model ID: {model_id}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Output channels: {output_channels}")
    print(f"  Device: {DEVICE}")

    return processor, model, output_channels


def get_current_model_info():
    """
    Get information about the currently loaded model.

    Returns:
    --------
    dict: Model information
    """
    return {
        "model_id": current_model_id,
        "output_channels": output_channels,
        "device": DEVICE,
        "is_initialized": model is not None,
    }


def ensure_initialized(model_id=None):
    """
    Ensure DINOv3 is initialized. If not, initialize with default or provided model_id.

    Parameters:
    -----------
    model_id : str, optional
        Model ID to use if not already initialized
    """
    global processor, model, output_channels

    if model is None or processor is None:
        if model_id is None:
            model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
        initialize_dinov3(model_id)


def get_img():
    """Load and return a sample image for testing."""
    import requests
    from io import BytesIO

    url = "https://github.com/facebookresearch/dinov2/blob/main/assets/dinov2_vitb14_pretrain.jpg?raw=true"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image


def round_to_multiple(x: int, k: int) -> int:
    """Round x to the nearest multiple of k."""
    return round(x / k) * k


def ensure_multiple(size: int, patch: int) -> int:
    """Ensure size is a multiple of patch size."""
    return round_to_multiple(size, patch)


def normalize_01(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] range."""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def normalize_features(features, method="standardize", eps=1e-6):
    """
    Normalize feature vectors using various methods.

    Parameters:
    -----------
    features : numpy.ndarray
        Feature array of shape (n_samples, n_features)
    method : str, default="standardize"
        Normalization method: "standardize", "minmax", "unit", or "robust"
    eps : float, default=1e-6
        Small epsilon for numerical stability

    Returns:
    --------
    tuple: (normalized_features, normalization_stats)
    """
    if method == "standardize":
        # Z-score normalization: (x - mean) / std
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std < eps, eps, std)  # Avoid division by zero
        normalized = (features - mean) / std
        stats = {"method": "standardize", "mean": mean, "std": std}

    elif method == "minmax":
        # Min-max normalization: (x - min) / (max - min)
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        range_val = max_val - min_val
        range_val = np.where(range_val < eps, eps, range_val)  # Avoid division by zero
        normalized = (features - min_val) / range_val
        stats = {"method": "minmax", "min": min_val, "max": max_val, "range": range_val}

    elif method == "unit":
        # Unit vector normalization: x / ||x||
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms < eps, eps, norms)  # Avoid division by zero
        normalized = features / norms
        stats = {"method": "unit", "eps": eps}

    elif method == "robust":
        # Robust normalization using median and IQR
        median = np.median(features, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        iqr = q75 - q25
        iqr = np.where(iqr < eps, eps, iqr)  # Avoid division by zero
        normalized = (features - median) / iqr
        stats = {
            "method": "robust",
            "median": median,
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
        }

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, stats


def apply_normalization_stats(features, stats):
    """
    Apply previously computed normalization statistics to new features.

    Parameters:
    -----------
    features : numpy.ndarray
        Feature array to normalize
    stats : dict
        Normalization statistics from normalize_features()

    Returns:
    --------
    numpy.ndarray: Normalized features
    """
    method = stats["method"]
    eps = 1e-6

    if method == "standardize":
        normalized = (features - stats["mean"]) / stats["std"]

    elif method == "minmax":
        normalized = (features - stats["min"]) / stats["range"]

    elif method == "unit":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms < eps, eps, norms)
        normalized = features / norms

    elif method == "robust":
        normalized = (features - stats["median"]) / stats["iqr"]

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def _combine_shifted_features(
    all_shift_features, stride, patch_size, output_channels, batch_size
):
    """
    Combine features from multiple shifted windows into a higher resolution grid.

    Parameters:
    -----------
    all_shift_features : list
        List of feature arrays from each shift
    stride : int
        Stride used for shifting
    patch_size : int
        DINOv3 patch size
    output_channels : int
        Number of feature channels
    batch_size : int
        Batch size

    Returns:
    --------
    numpy.ndarray: Combined high resolution features
    """
    # Get dimensions from first shift
    first_features = all_shift_features[0]  # (channels, batch, patch_h, patch_w)
    _, _, patch_h, patch_w = first_features.shape

    # Calculate high resolution dimensions
    scale_factor = patch_size // stride  # e.g., 16//8 = 2
    high_res_h = patch_h * scale_factor
    high_res_w = patch_w * scale_factor

    print(
        f"Combining features: {patch_h}x{patch_w} -> {high_res_h}x{high_res_w} (scale: {scale_factor}x)"
    )

    # Initialize high resolution feature map
    high_res_features = np.zeros((output_channels, batch_size, high_res_h, high_res_w))

    # Place each shifted feature set in the appropriate position
    shift_idx = 0
    for dy in range(0, patch_size, stride):
        for dx in range(0, patch_size, stride):
            if shift_idx < len(all_shift_features):
                features = all_shift_features[
                    shift_idx
                ]  # (channels, batch, patch_h, patch_w)

                # Calculate where to place these features in the high-res grid
                start_y = (dy * patch_h) // patch_size
                start_x = (dx * patch_w) // patch_size
                end_y = start_y + patch_h
                end_x = start_x + patch_w

                # Make sure we don't exceed bounds
                end_y = min(end_y, high_res_h)
                end_x = min(end_x, high_res_w)
                actual_h = end_y - start_y
                actual_w = end_x - start_x

                # Place the features
                high_res_features[:, :, start_y:end_y, start_x:end_x] = features[
                    :, :, :actual_h, :actual_w
                ]

                shift_idx += 1

    return high_res_features


def _process_sliding_window(data, stride, patch_size, image_size):
    """
    Process data using sliding window inference for higher resolution features.

    Parameters:
    -----------
    data : numpy.ndarray
        Input images of shape (batch_size, height, width)
    stride : int
        Stride for sliding window
    patch_size : int
        DINOv3 patch size (typically 16)
    image_size : int
        DINOv3 input image size

    Returns:
    --------
    numpy.ndarray: High resolution features
    """
    global model, processor, DEVICE

    batch_size, height, width = data.shape

    # Calculate how many shifts we need
    # For stride < patch_size, we need multiple shifts to cover all positions
    if stride >= patch_size:
        shifts = [(0, 0)]  # No sliding window needed
    else:
        shifts = []
        # Generate shifts from 0 to (patch_size - stride) in steps of stride
        for dy in range(0, patch_size, stride):
            for dx in range(0, patch_size, stride):
                shifts.append((dy, dx))

    print(f"Sliding window inference: stride={stride}, patch_size={patch_size}")
    print(f"Using {len(shifts)} shifts: {shifts}")

    # Pre-pad all images to handle maximum shifts
    max_shift = patch_size - stride
    padded_data = np.zeros(
        (batch_size, height + max_shift, width + max_shift), dtype=data.dtype
    )

    for b in range(batch_size):
        # Pad with reflection to avoid edge artifacts
        padded_data[b] = np.pad(
            data[b], ((0, max_shift), (0, max_shift)), mode="reflect"
        )

    print(f"Pre-padded data shape: {padded_data.shape}")

    # Get number of output channels from a test run
    test_features = _process_single_standard(data[:1], image_size)  # Just first image
    output_channels = test_features.shape[0]

    print(f"Output channels: {output_channels}")

    # Process each shift
    all_shift_features = []

    for shift_idx, (dy, dx) in enumerate(shifts):
        print(f"Processing shift {shift_idx + 1}/{len(shifts)}: ({dy}, {dx})")

        # Extract shifted windows from pre-padded data
        shifted_data = np.zeros((batch_size, height, width), dtype=data.dtype)
        for b in range(batch_size):
            # Extract the shifted region
            start_y = dy
            start_x = dx
            end_y = start_y + height
            end_x = start_x + width

            shifted_data[b] = padded_data[b, start_y:end_y, start_x:end_x]

        # Process the shifted images through standard DINOv3
        shift_features = _process_single_standard(shifted_data, image_size)
        all_shift_features.append(shift_features)

    # Now we need to combine the shifted features into a higher resolution grid
    # Each shift gives us features at positions that are stride apart
    if stride == patch_size:
        # No overlapping, just return the first (and only) shift
        high_res_features = all_shift_features[0]
    else:
        # Combine overlapping features by interleaving them
        high_res_features = _combine_shifted_features(
            all_shift_features, stride, patch_size, output_channels, batch_size
        )

    print(f"Final high resolution features shape: {high_res_features.shape}")
    return high_res_features


def _process_single_standard(data, image_size):
    """
    Process data through standard DINOv3 without sliding window.

    Parameters:
    -----------
    data : numpy.ndarray
        Input images of shape (batch_size, height, width)
    image_size : int
        DINOv3 input image size

    Returns:
    --------
    numpy.ndarray: Standard resolution features
    """
    global model, processor, DEVICE

    batch_size, height, width = data.shape

    # Convert to PIL Images for the processor
    pil_images = []
    for i in range(batch_size):
        # Normalize to 0-255 range for PIL
        img_normalized = (
            (data[i] - data[i].min()) / (data[i].max() - data[i].min() + 1e-8) * 255
        ).astype(np.uint8)
        pil_image = Image.fromarray(img_normalized).convert("RGB")
        pil_images.append(pil_image)

    # Process through DINOv3 processor
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
        # Use last_hidden_state which has shape (batch_size, num_tokens, hidden_size)
        features = outputs.last_hidden_state

        # Calculate expected number of patch tokens based on input size and patch size
        # Get the actual processed image size from the inputs
        processed_height, processed_width = inputs["pixel_values"].shape[-2:]
        patch_size = model.config.patch_size

        # Calculate expected patch grid dimensions
        expected_patch_h = processed_height // patch_size
        expected_patch_w = processed_width // patch_size
        expected_patch_tokens = expected_patch_h * expected_patch_w

        # Total tokens = patch_tokens + special_tokens (CLS + register tokens)
        total_tokens = features.shape[1]
        num_special_tokens = total_tokens - expected_patch_tokens

        # Extract only patch features (skip special tokens at the beginning)
        if num_special_tokens > 0:
            patch_features = features[
                :, num_special_tokens:, :
            ]  # Shape: (batch_size, num_patches, hidden_size)
        else:
            # Fallback: if calculation doesn't work, assume all tokens are patch tokens
            patch_features = features
            expected_patch_h = expected_patch_w = int(np.sqrt(total_tokens))
            print(
                f"Warning: Could not determine special tokens, using all {total_tokens} tokens as patches"
            )

        # Verify we have the expected number of patch tokens
        actual_patch_tokens = patch_features.shape[1]
        if actual_patch_tokens != expected_patch_tokens:
            print(
                f"Warning: Expected {expected_patch_tokens} patch tokens, got {actual_patch_tokens}"
            )
            # Recalculate patch dimensions based on actual tokens
            expected_patch_h = expected_patch_w = int(np.sqrt(actual_patch_tokens))
            if expected_patch_h * expected_patch_w != actual_patch_tokens:
                # If not a perfect square, find the closest factorization
                factors = []
                for i in range(1, int(np.sqrt(actual_patch_tokens)) + 1):
                    if actual_patch_tokens % i == 0:
                        factors.append((i, actual_patch_tokens // i))
                # Choose the factor pair closest to square
                expected_patch_h, expected_patch_w = min(
                    factors, key=lambda x: abs(x[0] - x[1])
                )
                print(
                    f"Using {expected_patch_h}x{expected_patch_w} patch grid for {actual_patch_tokens} tokens"
                )

        # Reshape to spatial format
        # Reshape: (batch_size, num_patches, hidden_size) -> (batch_size, patch_h, patch_w, hidden_size)
        spatial_features = patch_features.reshape(
            batch_size, expected_patch_h, expected_patch_w, -1
        )

        # Rearrange to (hidden_size, batch_size, patch_h, patch_w) to match expected output format
        output = spatial_features.permute(3, 0, 1, 2).cpu().numpy()

    return output


def process(data, model_id=None, image_size=896, stride=None):
    """
    Process raw image data through DINOv3 to extract features.

    Supports sliding window inference for higher resolution outputs by using
    overlapping patches with stride < patch_size.

    Parameters:
    -----------
    data : numpy.ndarray
        Input images of shape (batch_size, height, width) or (height, width)
    model_id : str, optional
        Model ID to use (will initialize if different from current)
    image_size : int, default=896
        Input image size for DINOv3
    stride : int, optional
        Stride for sliding window inference. If None, uses patch_size (no overlap).
        If stride < patch_size, creates overlapping windows for higher resolution.

    Returns:
    --------
    numpy.ndarray: Features of shape (output_channels, batch_size, patch_h, patch_w)
        If stride is provided, patch_h and patch_w will be higher than default.
    """
    global model, processor, DEVICE

    # Initialize or switch model if needed
    if model_id is not None and model_id != current_model_id:
        initialize_dinov3(model_id, image_size)
    else:
        ensure_initialized(model_id)

    # Handle single image
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # Add batch dimension

    batch_size, height, width = data.shape

    # If stride is specified and different from patch size, use sliding window inference
    patch_size = model.config.patch_size
    if stride is not None and stride != patch_size:
        return _process_sliding_window(data, stride, patch_size, image_size)

    # Otherwise, use standard processing
    return _process_single_standard(data, image_size)


# Initialize with default model for backward compatibility
# This can be overridden by calling initialize_dinov3() explicitly
initialize_dinov3()
