import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# cached AnyUp model
_ANYUP_MODEL = None


def _get_anyup_model(device: Optional[torch.device] = None):
    global _ANYUP_MODEL
    if _ANYUP_MODEL is None:
        # lazy load - may take a while on first call
        _ANYUP_MODEL = torch.hub.load("wimmerth/anyup", "anyup")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ANYUP_MODEL = _ANYUP_MODEL.to(device).eval()
    return _ANYUP_MODEL


def upsample_features_anyup(
    pil_images: Sequence[Union[Image.Image, np.ndarray, torch.Tensor]],
    lr_features: Union[np.ndarray, torch.Tensor],
    output_size: Tuple[int, int],
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    dtype: Optional[torch.dtype] = None,
) -> np.ndarray:
    """
    Upsample low-resolution DINO features to high-res feature maps using AnyUp.

    Args:
        pil_images: Sequence of PIL Images (RGB) or arrays/tensors convertible to ToTensor().
                    These provide the high-res guidance images used by AnyUp.
        lr_features: Low-resolution features, shape (B, C, h, w) as numpy or torch.Tensor.
        output_size: (H, W) output size to upsample to.
        device: torch.device to run AnyUp on (default cuda if available).
        batch_size: batch size to use when calling AnyUp (e.g., 16).
        dtype: optional torch.dtype to cast input tensors (defaults to float16 on CUDA autocast).

    Returns:
        hr_features: numpy array of upsampled features with shape (B, C, H, W), dtype float32.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _get_anyup_model(device)

    # Ensure lr_features is a torch tensor with shape (B, C, h, w)
    if isinstance(lr_features, np.ndarray):
        lr_features_t = torch.from_numpy(lr_features)
    else:
        lr_features_t = (
            lr_features.clone()
            if isinstance(lr_features, torch.Tensor)
            else torch.tensor(lr_features)
        )

    lr_features_t = lr_features_t.float()
    if lr_features_t.ndim != 4:
        raise ValueError("lr_features must have shape (B, C, h, w)")

    # Prepare guidance images tensors
    to_tensor = transforms.ToTensor()
    guidance_tensors = []
    for im in pil_images:
        if isinstance(im, Image.Image):
            guidance_tensors.append(to_tensor(im))
        else:
            # assume numpy array or torch tensor H,W,C or C,H,W
            t = torch.tensor(np.array(im)) if not isinstance(im, torch.Tensor) else im
            if t.ndim == 3 and t.shape[2] in (1, 3):  # H, W, C
                t = t.permute(2, 0, 1)
            guidance_tensors.append(t.float() / 255.0 if t.max() > 1.1 else t.float())

    # Stack guidance tensors and ensure correct dtype
    guidance_all = torch.stack(guidance_tensors, dim=0)  # (B, C, H, W)
    B = guidance_all.shape[0]
    if B != lr_features_t.shape[0]:
        raise ValueError(
            f"Batch size mismatch between guidance images ({B}) and lr_features ({lr_features_t.shape[0]})"
        )

    # Decide amp dtype for autocast if not provided
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    outputs = []
    model_device = device
    guidance_all = guidance_all.to(model_device)
    lr_features_t = lr_features_t.to(model_device)

    with torch.no_grad():
        # process in batches
        n_batches = math.ceil(B / batch_size)
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, B)
            g_batch = guidance_all[start:end].to(model_device)
            lr_batch = lr_features_t[start:end].to(model_device)

            # autocast only on CUDA
            if model_device.type == "cuda":
                with torch.cuda.amp.autocast(device_type="cuda", dtype=dtype):
                    hr = model(g_batch, lr_batch, output_size=output_size)
            else:
                # CPU fallback
                hr = model(g_batch, lr_batch, output_size=output_size)

            # ensure float32 on CPU numpy return
            outputs.append(hr.detach().cpu())

    hr_features = torch.cat(outputs, dim=0).numpy().astype(np.float32)
    return hr_features
