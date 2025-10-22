"""
DINOv3 Core Processing Module (HF-only, ViT or ConvNeXt)
- Robust to HF outputs for both backbones
- Removes CLS/special tokens for ViT
- Handles ConvNeXt outputs as (B, C, H, W) or (B, L, C)
- Sliding-window upsampling via stride < patch/effective reduction
- Device-aware AMP (CPU/CUDA; MPS runs without autocast)

Public API (unchanged):
  initialize_dinov3, get_current_model_info, ensure_initialized, get_img,
  round_to_multiple, ensure_multiple, normalize_01, normalize_features,
  apply_normalization_stats, process
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import functools


# -------------------- AMP wrapper (device-aware) --------------------
def enable_amp_inference(model, amp_dtype=torch.bfloat16, device_type=None):
    # device_type: "cuda", "cpu", "mps" (None -> infer)
    if device_type is None:
        try:
            device_type = next(model.parameters()).device.type
        except StopIteration:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"

    orig_forward = model.forward

    @functools.wraps(orig_forward)
    def amp_forward(*args, **kwargs):
        if device_type in ("cuda", "cpu"):
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                return orig_forward(*args, **kwargs)
        # e.g., MPS or unknown: no autocast
        return orig_forward(*args, **kwargs)

    model.forward = amp_forward
    return model


# -------------------- Globals (names unchanged) --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = None
model = None
output_channels = None
current_model_id = None

# Internals to support both ViT & ConvNeXt
_is_convnext = False  # True if loaded backbone is ConvNeXt
_effective_patch = None  # ViT: patch_size; ConvNeXt: inferred reduction
_vit_patch_size = None  # cached ViT patch size
_last_processed_hw = None  # (H, W) used to compute effective_patch for ConvNeXt


# -------------------- Init & Info --------------------
def initialize_dinov3(
    model_id="facebook/dinov3-vits16-pretrain-lvd1689m", image_size=896, device=None
):
    """
    Initialize DINOv3 model and processor with specified configuration.

    Returns:
        (processor, model, output_channels)
    """
    global processor, model, output_channels, current_model_id, DEVICE
    global _is_convnext, _effective_patch, _vit_patch_size, _last_processed_hw

    if device is not None:
        DEVICE = device

    # Detect ConvNeXt vs ViT from model_id (simple but effective)
    lid = (model_id or "").lower()
    _is_convnext = "convnext" in lid

    print("=" * 80)
    print(f"[init] Initializing DINOv3 with model: {model_id}")
    print(f"[init] Target image size: {image_size}x{image_size}")
    print(f"[init] Device: {DEVICE}")

    # HF processor (works for both backbones); force square resize/crop to image_size
    processor = AutoImageProcessor.from_pretrained(model_id)
    processor.size = {"height": image_size, "width": image_size}
    processor.crop_size = {"height": image_size, "width": image_size}

    # HF model
    model = AutoModel.from_pretrained(model_id)
    model = enable_amp_inference(model, torch.bfloat16, device_type=DEVICE.type)
    model.to(DEVICE)
    model.eval()

    # Output channels & "patch" bookkeeping
    if _is_convnext:
        output_channels = None
        _effective_patch = None
        _vit_patch_size = None
        print(
            "[init] Backbone: ConvNeXt (effective reduction will be inferred on first forward)"
        )
    else:
        # ViT: hidden dim + patch size from config
        hidden = getattr(model.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(model.config, "embed_dim", None)
        output_channels = int(hidden)
        _vit_patch_size = int(getattr(model.config, "patch_size", 16))
        _effective_patch = _vit_patch_size
        print(
            f"[init] Backbone: ViT (patch size = {_vit_patch_size}, hidden={output_channels})"
        )

    current_model_id = model_id
    _last_processed_hw = None

    print(
        f"[init] Output channels (now or later): {output_channels if output_channels is not None else '(lazy)'}"
    )
    print("=" * 80)
    return processor, model, output_channels


def get_current_model_info():
    return {
        "model_id": current_model_id,
        "output_channels": output_channels,
        "device": DEVICE,
        "is_initialized": model is not None,
        "is_convnext": _is_convnext,
        "effective_patch": _effective_patch,
        "vit_patch_size": _vit_patch_size,
        "last_processed_hw": _last_processed_hw,
    }


def ensure_initialized(model_id=None):
    global processor, model
    if model is None or processor is None:
        if model_id is None:
            raise RuntimeError(
                "DINOv3 model is not initialized. Call initialize_dinov3(model_id) first, "
                "or pass model_id to the function you're calling."
            )
        initialize_dinov3(model_id)


# -------------------- Small utilities (unchanged) --------------------
def get_img():
    """Simple sample image (requires internet)."""
    import requests
    from io import BytesIO

    url = "https://github.com/facebookresearch/dinov2/blob/main/assets/dinov2_vitb14_pretrain.jpg?raw=true"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image


def round_to_multiple(x: int, k: int) -> int:
    return round(x / k) * k


def ensure_multiple(size: int, patch: int) -> int:
    return round_to_multiple(size, patch)


def normalize_01(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def normalize_features(features, method="standardize", eps=1e-6):
    if method == "standardize":
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std < eps, eps, std)
        normalized = (features - mean) / std
        stats = {"method": "standardize", "mean": mean, "std": std}
    elif method == "minmax":
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        range_val = max_val - min_val
        range_val = np.where(range_val < eps, eps, range_val)
        normalized = (features - min_val) / range_val
        stats = {"method": "minmax", "min": min_val, "max": max_val, "range": range_val}
    elif method == "unit":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms < eps, eps, norms)
        normalized = features / norms
        stats = {"method": "unit", "eps": eps}
    elif method == "robust":
        median = np.median(features, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        iqr = q75 - q25
        iqr = np.where(iqr < eps, eps, iqr)
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
    method = stats["method"]
    eps = 1e-6
    if method == "standardize":
        return (features - stats["mean"]) / stats["std"]
    if method == "minmax":
        return (features - stats["min"]) / stats["range"]
    if method == "unit":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms < eps, eps, norms)
        return features / norms
    if method == "robust":
        return (features - stats["median"]) / stats["iqr"]
    raise ValueError(f"Unknown normalization method: {method}")


# -------------------- Internal helpers --------------------
def _extract_vit_patch_features(outputs, inputs, batch_size):
    """
    ViT path: outputs.last_hidden_state -> (B, tokens, C)
    Remove special tokens (e.g., CLS/distill), reshape to (C, B, ph, pw)
    """
    global model, _vit_patch_size, _effective_patch, _last_processed_hw, output_channels

    # Robust grab
    if hasattr(outputs, "last_hidden_state"):
        features = outputs.last_hidden_state  # (B, T, C)
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        features = outputs[0]
    else:
        raise RuntimeError("Unexpected ViT outputs: no last_hidden_state")

    B, T, C = features.shape
    ph, pw = inputs["pixel_values"].shape[-2:]
    patch_size = int(getattr(model.config, "patch_size", _vit_patch_size or 16))
    _vit_patch_size = patch_size
    _effective_patch = patch_size
    _last_processed_hw = (ph, pw)

    expected_ph = ph // patch_size
    expected_pw = pw // patch_size
    expected_patch_tokens = expected_ph * expected_pw

    # Remove all special tokens (CLS/distill/others) at the *front*
    num_special = T - expected_patch_tokens
    if num_special < 0:
        # Some models may not resize exactly; try to infer grid from T (no specials case)
        expected_ph = expected_pw = int(np.sqrt(T))
        expected_patch_tokens = expected_ph * expected_pw
        num_special = T - expected_patch_tokens
    if num_special > 0:
        patch_features = features[:, num_special:, :]  # drop CLS/distill/etc.
        # print(f"[vit] Dropped {num_special} special token(s).")
    else:
        patch_features = features

    # Output channels (hidden dim)
    if output_channels is None:
        output_channels = int(patch_features.shape[-1])

    # Reshape into grid (B, ph, pw, C)
    if patch_features.shape[1] != expected_patch_tokens:
        raise RuntimeError(
            f"[vit] Token count mismatch after dropping specials: "
            f"T'={patch_features.shape[1]} vs expected {expected_patch_tokens} "
            f"(ph={expected_ph}, pw={expected_pw}, patch={patch_size})"
        )

    spatial_features = patch_features.reshape(batch_size, expected_ph, expected_pw, -1)
    out = (
        spatial_features.permute(3, 0, 1, 2)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # print(
    #     f"[vit] last_hidden_state: (B={B}, T={T}, C={C}) -> grid (C={out.shape[0]}, B={out.shape[1]}, ph={expected_ph}, pw={expected_pw}) "
    #     f"| patch={patch_size}"
    # )
    return out


def _extract_convnext_map(outputs, inputs):
    """
    ConvNeXt path: accept either (B, C, h, w) or (B, L, C) where L = h*w (+ optional specials).
    - If 3D, will auto-drop up to a few leading special tokens to make L' factorable.
    Return (C, B, h, w). Also infer effective_patch and output_channels.
    """
    global _effective_patch, _last_processed_hw, output_channels

    # Pick a usable tensor from outputs
    feats = None
    if hasattr(outputs, "last_hidden_state"):
        feats = outputs.last_hidden_state
        src_tag = "last_hidden_state"
    elif hasattr(outputs, "logits"):
        feats = outputs.logits
        src_tag = "logits"
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        feats = outputs[0]
        src_tag = "tuple[0]"
    else:
        raise RuntimeError("Unexpected ConvNeXt outputs: no usable tensor.")

    ph, pw = inputs["pixel_values"].shape[-2:]  # processed H, W
    _last_processed_hw = (ph, pw)

    # 4D path is straightforward
    if feats.ndim == 4:
        B, C, h, w = feats.shape
        eff_h = max(1, ph // max(1, h))
        eff_w = max(1, pw // max(1, w))
        _effective_patch = eff_h if eff_h == eff_w else int(round((eff_h + eff_w) / 2))
        if output_channels is None:
            output_channels = int(C)
        out = feats.permute(1, 0, 2, 3).contiguous().cpu().numpy().astype(np.float32)
        # print(
        #     f"[convnext/4D] src={src_tag} feats: (B={B}, C={C}, h={h}, w={w}) "
        #     f"-> (C={out.shape[0]}, B={out.shape[1]}, h={h}, w={w}) | inferred_effective_patch={_effective_patch} "
        #     f"(from ph={ph}, pw={pw})"
        # )
        return out

    # 3D path: (B, L, C) possibly with extra special tokens at the front
    if feats.ndim == 3:
        B, L, C = feats.shape

        def try_factor(L_try, ph, pw):
            # infer reduction s from ph*pw and L_try
            s_float = np.sqrt((ph * pw) / max(1, L_try))
            s = int(round(s_float))
            if s < 1:
                return None
            h = ph // s
            w = pw // s
            if h * w == L_try:
                return s_float, s, h, w
            return None

        # First attempt: no specials
        guess = try_factor(L, ph, pw)
        dropped = 0

        # If that fails, drop up to a few leading tokens until it works
        if guess is None:
            for k in range(1, 5):  # try dropping 1..4 specials
                guess = try_factor(L - k, ph, pw)
                if guess is not None:
                    dropped = k
                    break

        if guess is None:
            # Final fallback: purely square grid from L (or L-1 if perfect square)
            side = int(round(np.sqrt(L)))
            if side * side == L:
                s = max(1, int(round(ph / side)))
                h, w = side, side
                s_float = ph / max(1, h)
            elif (side * side == (L - 1)) and L > 1:
                dropped = 1
                s = max(1, int(round(ph / side)))
                h, w = side, side
                s_float = ph / max(1, h)
            else:
                raise RuntimeError(
                    f"[convnext/3D] Could not infer spatial map: ph={ph}, pw={pw}, L={L}"
                )
        else:
            s_float, s, h, w = guess

        if dropped > 0:
            feats = feats[:, dropped:, :]
            L_after = feats.shape[1]
            # print(
            #     f"[convnext/3D] Dropped {dropped} leading special token(s) from sequence "
            #     f"(L {L} -> {L_after}) to form grid {h}x{w}."
            # )

        _effective_patch = int(s)
        if output_channels is None:
            output_channels = int(C)

        feats_hw = feats.reshape(B, h, w, C)
        out = feats_hw.permute(3, 0, 1, 2).contiguous().cpu().numpy().astype(np.float32)
        # print(
        #     f"[convnext/3D] src={src_tag} feats: (B={B}, L={L}, C={C}) -> grid (h={h}, w={w}) "
        #     f"-> (C={out.shape[0]}, B={out.shape[1]}, h={h}, w={w}) | inferred_effective_patch={_effective_patch} "
        #     f"(ph={ph}, pw={pw}, s≈{s_float:.3f}→{_effective_patch})"
        # )
        return out

    raise RuntimeError("Unexpected ConvNeXt tensor rank (expect 3D or 4D).")


def _process_single_standard(data, image_size):
    """
    Process data through the model without sliding window.
    Input:
        data: np.ndarray, (batch_size, height, width)
    Returns:
        np.ndarray: (C, B, patch_h, patch_w)  [ConvNeXt: patch_h/w are feature map size]
    """
    global model, processor, DEVICE, _is_convnext

    batch_size, height, width = data.shape

    # Convert to PIL Images for the processor
    pil_images = []
    for i in range(batch_size):
        img = data[i]
        # robust to constant slices
        denom = img.max() - img.min() + 1e-8
        img_normalized = ((img - img.min()) / denom * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_normalized).convert("RGB")
        pil_images.append(pil_image)

    # Process through HF processor
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

        # Prefer last_hidden_state; fall back to logits/first tuple
        if hasattr(outputs, "last_hidden_state"):
            lhs = outputs.last_hidden_state
            lhs_tag = "last_hidden_state"
        elif hasattr(outputs, "logits"):
            lhs = outputs.logits
            lhs_tag = "logits"
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            lhs = outputs[0]
            lhs_tag = "tuple[0]"
        else:
            raise RuntimeError("Unexpected model outputs: no usable tensor.")

        # print(
        #     f"[forward] is_convnext={_is_convnext} | pixel_values={tuple(inputs['pixel_values'].shape)} "
        #     f"| using='{lhs_tag}' with shape={tuple(lhs.shape)}"
        # )

        # Branching by rank; for 3D we choose by backbone type
        if lhs.ndim == 3:
            if _is_convnext:
                out = _extract_convnext_map(outputs, inputs)
            else:
                out = _extract_vit_patch_features(outputs, inputs, batch_size)
        elif lhs.ndim == 4:
            out = _extract_convnext_map(outputs, inputs)
        else:
            raise RuntimeError("Unknown output rank for vision backbone.")

    return out


def _combine_shifted_features(
    all_shift_features, stride, patch_size, output_channels_local, batch_size
):
    """
    Combine features from multiple shifted windows into a higher resolution grid.
    (C, B, ph, pw) -> interleaved (C, B, hi_h, hi_w)
    """
    first_features = all_shift_features[0]
    _, _, patch_h, patch_w = first_features.shape

    scale_factor = patch_size // stride  # e.g., 16//8 = 2
    high_res_h = patch_h * scale_factor
    high_res_w = patch_w * scale_factor

    high_res_features = np.zeros(
        (output_channels_local, batch_size, high_res_h, high_res_w), dtype=np.float32
    )

    shift_idx = 0
    for dy in range(0, patch_size, stride):
        for dx in range(0, patch_size, stride):
            if shift_idx < len(all_shift_features):
                features = all_shift_features[shift_idx]  # (C, B, ph, pw)
                for i in range(patch_h):
                    for j in range(patch_w):
                        hi_i = i * scale_factor + (dy // stride)
                        hi_j = j * scale_factor + (dx // stride)
                        if hi_i < high_res_h and hi_j < high_res_w:
                            high_res_features[:, :, hi_i, hi_j] = features[:, :, i, j]
                shift_idx += 1

    # print(
    #     f"[sliding] Combined {len(all_shift_features)} shifts -> high-res grid "
    #     f"(C={output_channels_local}, B={batch_size}, H={high_res_h}, W={high_res_w}) "
    #     f"| scale={scale_factor} (patch={patch_size}, stride={stride})"
    # )
    return high_res_features


def _process_sliding_window(data, stride, patch_size, image_size):
    """
    Sliding window inference for higher resolution features.
    'patch_size' is ViT.patch_size or ConvNeXt effective reduction.
    """
    global model, processor, DEVICE

    batch_size, height, width = data.shape

    if stride >= patch_size:
        shifts = [(0, 0)]
    else:
        shifts = [
            (dy, dx)
            for dy in range(0, patch_size, stride)
            for dx in range(0, patch_size, stride)
        ]

    max_shift = patch_size - stride if stride < patch_size else 0

    if max_shift > 0:
        padded_data = np.zeros(
            (batch_size, height + max_shift, width + max_shift), dtype=data.dtype
        )
        for b in range(batch_size):
            padded_data[b] = np.pad(
                data[b], ((0, max_shift), (0, max_shift)), mode="reflect"
            )
    else:
        padded_data = data

    # Determine output channels (and, for ConvNeXt, cache effective_patch) via a tiny run
    # print("[sliding] Probing a tiny forward to resolve channels/effective_patch...")
    test_features = _process_single_standard(data[:1], image_size)  # (C, 1, ph, pw)
    output_channels_local = test_features.shape[0]
    # print(
    #     f"[sliding] Probe result: C={output_channels_local}, effective_patch={_effective_patch}"
    # )

    # Collect shifted data
    all_shifted_data = []
    for dy, dx in shifts:
        shifted = np.zeros((batch_size, height, width), dtype=data.dtype)
        for b in range(batch_size):
            shifted[b] = padded_data[b, dy : dy + height, dx : dx + width]
        all_shifted_data.append(shifted)

    combined_shifted_data = np.concatenate(
        all_shifted_data, axis=0
    )  # (num_shifts*B, H, W)
    # print(
    #     f"[sliding] Running combined batch of {combined_shifted_data.shape[0]} slices for {len(shifts)} shift(s)"
    # )
    combined_features = _process_single_standard(
        combined_shifted_data, image_size
    )  # (C, num_shifts*B, ph, pw)

    # Split by shifts
    all_shift_features = []
    for i in range(len(shifts)):
        s = i * batch_size
        e = s + batch_size
        all_shift_features.append(combined_features[:, s:e, :, :])

    if stride == patch_size:
        return all_shift_features[0]
    else:
        return _combine_shifted_features(
            all_shift_features, stride, patch_size, output_channels_local, batch_size
        )


# -------------------- Public API --------------------
def process(data, model_id=None, image_size=896, stride=None):
    """
    Process raw image data through DINOv3 to extract features.

    Supports sliding window inference for higher resolution outputs by using
    overlapping patches with stride < patch_size (ViT) or < effective_patch (ConvNeXt).

    Args:
        data : np.ndarray (H, W) or (B, H, W)
        model_id : str (optional) -> initializes/switches model
        image_size : int (resize/crop target)
        stride : int (None -> no overlap; else < patch/effective_patch for HR)

    Returns:
        np.ndarray: (C, B, patch_h, patch_w)
    """
    global model, processor, DEVICE, current_model_id, output_channels
    global _is_convnext, _effective_patch, _vit_patch_size

    # Initialize or switch model if needed
    if model_id is not None and model_id != current_model_id:
        initialize_dinov3(model_id, image_size)
    else:
        ensure_initialized(model_id)

    # Ensure batch dim
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    # Decide the "patch size" concept for sliding:
    if not _is_convnext:
        patch_size = _vit_patch_size or int(getattr(model.config, "patch_size", 16))
    else:
        # If effective_patch not known yet, infer by a small run
        if _effective_patch is None:
            # print("[process] effective_patch unknown for ConvNeXt; probing...")
            _ = _process_single_standard(data[:1], image_size)
            # print(f"[process] effective_patch inferred: {_effective_patch}")
        patch_size = int(_effective_patch)

    if stride is not None and stride != patch_size:
        # print(f"[process] Sliding-window mode: patch={patch_size}, stride={stride}")
        return _process_sliding_window(data, stride, patch_size, image_size)

    # print(f"[process] Single-pass mode: patch/effective={patch_size}")
    return _process_single_standard(data, image_size)
