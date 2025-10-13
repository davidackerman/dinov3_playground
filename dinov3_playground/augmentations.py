import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    binary_dilation,
    maximum_filter,
    minimum_filter,
    uniform_filter,
)
import torch
import torch.nn.functional as F


def _random_choice(rng, seq):
    if rng is None:
        return np.random.choice(seq)
    return rng.choice(seq)


def _rotate_3d(arr, mode):
    # arr: (D,H,W)
    if mode == "none":
        return arr
    if mode == "xy":
        return np.transpose(arr, (0, 2, 1))
    if mode == "xz":
        return np.transpose(arr, (1, 0, 2))
    if mode == "yz":
        return np.transpose(arr, (2, 1, 0))
    # in-plane 90-degree rotations: apply to each slice
    if mode in ("rot90", "rot180", "rot270"):
        k = {"rot90": 1, "rot180": 2, "rot270": 3}[mode]
        # rotate each (H,W) slice
        return np.stack([np.rot90(s, k=k) for s in arr], axis=0)
    return arr


def apply_3d_augmentations(
    raw,
    context,
    gt,
    mask,
    *,
    augment=True,
    augment_prob=0.5,
    augment_params=None,
    rng=None,
    use_gpu=True,
    gpu_device=None,
):
    """
    Apply 3D augmentations to raw (D,H,W), context, gt and mask. Returns transformed copies.

    - Preserves raw dtype (casts back to original dtype, e.g., uint16).
    - Noise/contrast/brightness scales are proportional to the dynamic range of the input raw
      (raw.max() - raw.min()) to adapt across datasets.
    - Supports axis swaps and in-plane 90-degree rotations, applied consistently to gt and mask.
    """
    if not augment or (rng is None and np.random.rand() >= augment_prob):
        if rng is not None and rng.random() >= augment_prob:
            return raw, context, gt, mask
        return raw, context, gt, mask

    # Use rng if provided (np.random.RandomState or Generator), else numpy global
    rnd = rng if rng is not None else np.random

    # Keep original dtype for final casting
    orig_dtype = raw.dtype
    # convert to float for processing
    raw_f = raw.astype(np.float32)

    # dynamic range: compute robust percentiles (5th-95th) excluding zero pixels
    nz = raw_f[raw_f != 0]
    if nz.size > 0:
        try:
            p5 = float(np.percentile(nz, 5))
            p95 = float(np.percentile(nz, 95))
        except Exception:
            p5 = float(np.min(nz))
            p95 = float(np.max(nz))
        raw_min = p5
        raw_max = p95
        dyn = raw_max - raw_min
        # if degenerate (p95<=p5), fallback to robust std-based range
        if dyn <= 0:
            raw_min = float(np.mean(nz) - np.std(nz))
            raw_max = float(np.mean(nz) + np.std(nz))
            dyn = raw_max - raw_min
    else:
        # all zeros or no non-zero pixels -> fallback to full-range
        raw_max = float(raw_f.max())
        raw_min = float(raw_f.min())
        dyn = raw_max - raw_min
        if dyn <= 0:
            dyn = float(max(1.0, np.std(raw_f)))

    # rotation options: include axis-swaps and in-plane rotations
    rot_choices = ["none", "xy", "xz", "yz", "rot90", "rot180", "rot270"]
    if augment_params and "rotations" in augment_params:
        rot_choices = augment_params["rotations"]
    rot = rnd.choice(rot_choices)

    # Rotate GT and mask using numpy (cheap compared with blur/noise) so labels stay exact
    raw_f = _rotate_3d(raw_f, rot)
    if context is not None:
        try:
            context = _rotate_3d(context, rot)
        except Exception:
            pass
    gt = _rotate_3d(gt, rot)
    if mask is not None:
        mask = _rotate_3d(mask, rot)

    # --- Border-based morphological erosion/dilation of raw near GT borders ---
    # Apply optionally when gt is provided. Defaults: 50% chance, radius 3-5 voxels,
    # randomly choose erosion or dilation unless specified in augment_params.
    if gt is not None:
        bm_prob = (
            augment_params.get("border_morph_prob", 0.5) if augment_params else 0.5
        )

        if rnd.random() < bm_prob:
            print("radius ")
            # determine radius (allow scalar or (min,max))
            if augment_params and "border_morph_radius" in augment_params:
                rcfg = augment_params["border_morph_radius"]
                if isinstance(rcfg, (list, tuple)) and len(rcfg) == 2:
                    radius = int(rnd.randint(rcfg[0], rcfg[1] + 1))
                else:
                    radius = int(rcfg)
            else:
                radius = int(rnd.randint(0, 5))  # 0..5 inclusive

            btype = augment_params.get("border_morph_type") if augment_params else None
            if btype is None:
                btype = _random_choice(rnd, ["erode", "dilate", "mean"])

            # Ensure gt is the same shape as raw_f for accurate border ops.
            def _upsample_to_target(arr, target_shape):
                if arr is None:
                    return None
                if arr.shape == target_shape:
                    return arr
                # integer scale factors (rounded)
                scales = [
                    int(round(t / s)) if s > 0 else 1
                    for s, t in zip(arr.shape, target_shape)
                ]
                # try torch nearest on GPU if requested
                if use_gpu:
                    if torch.cuda.is_available():
                        try:
                            dev = (
                                torch.device(gpu_device)
                                if gpu_device is not None
                                else torch.device("cuda")
                            )
                            a = torch.from_numpy(arr).to(device=dev)
                            a = a.unsqueeze(0).unsqueeze(0).float()  # 1,1,D,H,W
                            sf = tuple(
                                float(t / s) for s, t in zip(arr.shape, target_shape)
                            )
                            a_up = F.interpolate(a, scale_factor=sf, mode="nearest")
                            return (
                                a_up.squeeze(0)
                                .squeeze(0)
                                .cpu()
                                .numpy()
                                .astype(arr.dtype)
                            )
                        except Exception:
                            pass
                # numpy repeat fallback
                up = arr
                for ax, sc in enumerate(scales):
                    if sc > 1:
                        up = np.repeat(up, sc, axis=ax)
                out = np.zeros(target_shape, dtype=arr.dtype)
                md = min(out.shape[0], up.shape[0])
                mh = min(out.shape[1], up.shape[1])
                mw = min(out.shape[2], up.shape[2])
                out[:md, :mh, :mw] = up[:md, :mh, :mw]
                return out

            gt_up = _upsample_to_target(gt, raw_f.shape)
            # compute border mask: prefer fastmorph if available (gt - erode(gt)),
            # otherwise fall back to 6-neighbor comparison (use gt_up)
            border = None
            try:
                import fastmorph

                # try a few common calling signatures; prefer operating on gt_up
                try:
                    eroded = fastmorph.erode(gt_up, radius=1)
                except TypeError:
                    try:
                        eroded = fastmorph.erode(gt_up, 1)
                    except TypeError:
                        eroded = fastmorph.erode(gt_up)

                border = (gt_up > 0) & (gt_up != eroded)
            except Exception:
                # fallback: 6-neighbor difference test
                g = gt_up
                p = np.pad(g, 1, mode="constant", constant_values=0)
                center = p[1:-1, 1:-1, 1:-1]
                nb = np.stack(
                    [
                        p[:-2, 1:-1, 1:-1],
                        p[2:, 1:-1, 1:-1],
                        p[1:-1, :-2, 1:-1],
                        p[1:-1, 2:, 1:-1],
                        p[1:-1, 1:-1, :-2],
                        p[1:-1, 1:-1, 2:],
                    ],
                    axis=0,
                )
                border = (center > 0) & np.any(nb != center, axis=0)

            # region is border dilated by `radius`.
            # Prefer GPU dilation (torch), then fastmorph.dilation, then scipy binary_dilation.
            if radius > 0:
                print("radius ", radius)
                region = None

                # 1) Try GPU dilation via max_pool3d if requested and available
                if use_gpu:

                    if torch.cuda.is_available():
                        try:
                            device = (
                                torch.device(gpu_device)
                                if gpu_device is not None
                                else torch.device("cuda")
                            )
                            b = torch.from_numpy(border.astype(np.uint8)).to(
                                device=device
                            )
                            # convert to float and pool
                            b_f = b.float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
                            k = 2 * radius + 1
                            pad = radius
                            out = F.max_pool3d(
                                b_f, kernel_size=k, stride=1, padding=pad
                            )
                            region = out.squeeze(0).squeeze(0).cpu().numpy() > 0.5
                        except Exception:
                            region = None

                # 2) Try fastmorph.dilation on CPU if available
                if region is None:
                    try:
                        import fastmorph

                        try:
                            dil = fastmorph.dilation(border, radius=radius)
                        except TypeError:
                            try:
                                dil = fastmorph.dilation(border, radius)
                            except TypeError:
                                dil = fastmorph.dilation(border)
                        region = dil.astype(bool)
                    except Exception:
                        region = None

                # 3) Fallback to scipy binary_dilation
                if region is None:
                    region = binary_dilation(border, iterations=radius)
            else:
                region = border

            # perform morph either on GPU (if available) or CPU
            if use_gpu:

                if torch.cuda.is_available():
                    device = (
                        torch.device(gpu_device)
                        if gpu_device is not None
                        else torch.device("cuda")
                    )
                    t_m = torch.from_numpy(raw_f).to(device=device, dtype=torch.float32)
                    # handle dilate/erode via max_pool3d, mean via avg_pool3d
                    if btype == "dilate" or btype == "erode":
                        t_in = t_m.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
                        k = 2 * radius + 1
                        pad = radius
                        if btype == "dilate":
                            out = F.max_pool3d(
                                t_in, kernel_size=k, stride=1, padding=pad
                            )
                        else:
                            out = -F.max_pool3d(
                                -t_in, kernel_size=k, stride=1, padding=pad
                            )
                        out = out.squeeze(0).squeeze(0)
                        region_t = torch.from_numpy(region.astype(np.bool_)).to(
                            device=device
                        )
                        t_m[region_t] = out[region_t]
                        raw_f = t_m.cpu().numpy()
                    elif btype == "mean":
                        t_in = t_m.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
                        k = 2 * radius + 1
                        pad = radius
                        out = F.avg_pool3d(t_in, kernel_size=k, stride=1, padding=pad)
                        out = out.squeeze(0).squeeze(0)
                        region_t = torch.from_numpy(region.astype(np.bool_)).to(
                            device=device
                        )
                        t_m[region_t] = out[region_t]
                        raw_f = t_m.cpu().numpy()
                else:
                    # fall through to CPU below
                    pass

            # CPU fallback: use minimum_filter / maximum_filter or uniform_filter for mean
            if not (use_gpu and torch.cuda.is_available()):
                size = (2 * radius + 1, 2 * radius + 1, 2 * radius + 1)
                if btype == "dilate":
                    morph = maximum_filter(raw_f, size=size)
                    raw_f[region] = morph[region]
                elif btype == "erode":
                    morph = minimum_filter(raw_f, size=size)
                    raw_f[region] = morph[region]
                else:  # mean
                    morph = uniform_filter(raw_f, size=size)
                    raw_f[region] = morph[region]

    # Intensity augmentations
    # Contrast multiplier (made stronger): ~ N(1, 0.15)
    if (
        rnd.random() < augment_params.get("contrast_prob", 0.8)
        if augment_params
        else rnd.random() < 0.8
    ):
        # stronger spread around 1.0 so contrast changes are more visible
        factor = float(rnd.normal(loc=1.0, scale=0.15))
        # Bias scaled to dyn (stronger): up to ~ +/- 0.08*dyn
        bias = float(rnd.normal(loc=0.0, scale=0.08 * dyn))
        raw_f = raw_f * factor + bias

    # Gamma correction occasionally
    if (
        rnd.random() < augment_params.get("gamma_prob", 0.25)
        if augment_params
        else rnd.random() < 0.25
    ):
        # slightly wider gamma variation to make midtone shifts more visible
        gamma = float(max(0.4, min(1.6, rnd.normal(1.0, 0.25))))
        # Map to [0,1] then back using dyn
        norm = (raw_f - raw_min) / (dyn + 1e-9)
        norm = np.clip(norm, 0, 1)
        norm = norm**gamma
        raw_f = norm * dyn + raw_min

    # Gaussian blur (per-slice blur in H,W) and additive noise
    # Increase default probabilities and means for blur/noise so effects are more apparent
    do_blur = rnd.random() < (
        augment_params.get("blur_prob", 0.7) if augment_params else 0.7
    )
    # stronger blur on average: mean ~1.2, std ~1.0, capped at 4.0
    sigma = float(max(0.0, min(4.0, rnd.normal(1.2, 1.0)))) if do_blur else 0.0
    do_noise = rnd.random() < (
        augment_params.get("noise_prob", 0.85) if augment_params else 0.85
    )
    # noise scaled up: mean 0.02*dyn, std ~0.015*dyn
    noise_sigma = (
        float(max(0.0, rnd.normal(0.02 * dyn, 0.015 * dyn))) if do_noise else 0.0
    )

    # If GPU requested and available, perform heavy ops on GPU via PyTorch
    if use_gpu:
        if torch.cuda.is_available():
            device = (
                torch.device(gpu_device)
                if gpu_device is not None
                else torch.device("cuda")
            )

            # Move numpy raw_f to GPU tensor
            t = torch.from_numpy(raw_f).to(device=device, dtype=torch.float32)

            # Apply blur via 2D conv per-slice if requested
            if do_blur and sigma > 1e-6:
                # build 2d gaussian kernel
                radius = max(1, int(round(4 * sigma)))
                ksize = radius * 2 + 1
                ax = np.arange(-radius, radius + 1, dtype=np.float32)
                gauss = np.exp(-(ax**2) / (2 * sigma * sigma))
                kernel2d = np.outer(gauss, gauss)
                kernel2d = kernel2d / kernel2d.sum()
                # to torch
                kernel = torch.from_numpy(kernel2d.astype(np.float32)).to(device=device)
                kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,ks,ks)

                D, H, W = t.shape
                t_in = t.unsqueeze(1)  # (D,1,H,W)
                # pad
                pad = ksize // 2
                t_in = F.pad(t_in, (pad, pad, pad, pad), mode="reflect")
                t_blur = F.conv2d(t_in, kernel, groups=1)
                t = t_blur.squeeze(1)

            # Add noise on GPU
            if do_noise and noise_sigma > 0:
                noise_t = torch.randn_like(t, device=device) * float(noise_sigma)
                t = t + noise_t

            # Streaks and blackout performed on GPU tensor
            if rnd.random() < (
                augment_params.get("streaks_prob", 0.35) if augment_params else 0.35
            ):
                # slightly more streaks and stronger magnitude by default
                num_streaks = int(rnd.randint(1, 5))
                D, H, W = t.shape
                for _ in range(num_streaks):
                    axis = int(rnd.choice([1, 2]))
                    thickness = max(
                        1,
                        int(round(0.01 * (H if axis == 1 else W) * rnd.uniform(1, 5))),
                    )
                    pos = int(rnd.randint(0, (H if axis == 1 else W)))
                    # stronger streak magnitude
                    val = float(rnd.normal(loc=0.0, scale=0.12 * dyn))
                    if axis == 1:
                        start = max(0, pos - thickness // 2)
                        end = min(H, start + thickness)
                        t[:, start:end, :] += float(val)
                    else:
                        start = max(0, pos - thickness // 2)
                        end = min(W, start + thickness)
                        t[:, :, start:end] += float(val)

            if rnd.random() < (
                augment_params.get("blackout_prob", 0.35) if augment_params else 0.35
            ):
                D, H, W = t.shape
                # slightly larger blackout fraction by default
                target_frac = float(np.clip(rnd.normal(0.08, 0.04), 0.0, 0.15))
                total_area = D * H * W
                remaining = int(total_area * target_frac)
                attempts = 0
                while remaining > 0 and attempts < 10:
                    box_d = int(rnd.randint(1, max(2, D // 4)))
                    box_h = int(rnd.randint(1, max(2, H // 4)))
                    box_w = int(rnd.randint(1, max(2, W // 4)))
                    sd = int(rnd.randint(0, D - box_d + 1))
                    sh = int(rnd.randint(0, H - box_h + 1))
                    sw = int(rnd.randint(0, W - box_w + 1))
                    t[sd : sd + box_d, sh : sh + box_h, sw : sw + box_w] = float(
                        raw_min
                    )
                    if mask is not None:
                        mask[sd : sd + box_d, sh : sh + box_h, sw : sw + box_w] = 0
                    remaining -= box_d * box_h * box_w
                    attempts += 1

            # Move back to CPU numpy
            t = t.cpu().numpy()
            raw_f = t
        else:
            # torch not available or no CUDA -> fall back to CPU path
            pass
    else:
        # CPU path: gaussian_filter and numpy-based noise
        if do_blur and sigma > 1e-6:
            raw_f = gaussian_filter(raw_f, sigma=(0, sigma, sigma))
        if do_noise and noise_sigma > 0:
            noise = rnd.normal(loc=0.0, scale=noise_sigma, size=raw_f.shape)
            raw_f = raw_f + noise

    # Streaks: bright/dark stripes along H or W axes
    if rnd.random() < (
        augment_params.get("streaks_prob", 0.35) if augment_params else 0.35
    ):
        num_streaks = int(rnd.randint(1, 5))
        D, H, W = raw_f.shape
        for _ in range(num_streaks):
            axis = int(rnd.choice([1, 2]))
            thickness = max(
                1, int(round(0.01 * (H if axis == 1 else W) * rnd.uniform(1, 5)))
            )
            pos = int(rnd.randint(0, (H if axis == 1 else W)))
            val = float(rnd.normal(loc=0.0, scale=0.12 * dyn))
            if axis == 1:
                start = max(0, pos - thickness // 2)
                end = min(H, start + thickness)
                raw_f[:, start:end, :] += val
            else:
                start = max(0, pos - thickness // 2)
                end = min(W, start + thickness)
                raw_f[:, :, start:end] += val

    # Blackout boxes (total area on average <= 10%)
    if rnd.random() < (
        augment_params.get("blackout_prob", 0.35) if augment_params else 0.35
    ):
        D, H, W = raw_f.shape
        target_frac = float(np.clip(rnd.normal(0.08, 0.04), 0.0, 0.15))
        total_area = D * H * W
        remaining = int(total_area * target_frac)
        attempts = 0
        while remaining > 0 and attempts < 10:
            box_d = int(rnd.randint(1, max(2, D // 4)))
            box_h = int(rnd.randint(1, max(2, H // 4)))
            box_w = int(rnd.randint(1, max(2, W // 4)))
            sd = int(rnd.randint(0, D - box_d + 1))
            sh = int(rnd.randint(0, H - box_h + 1))
            sw = int(rnd.randint(0, W - box_w + 1))
            raw_f[sd : sd + box_d, sh : sh + box_h, sw : sw + box_w] = (
                raw_min  # blackout to min
            )
            if mask is not None:
                mask[sd : sd + box_d, sh : sh + box_h, sw : sw + box_w] = 0
            remaining -= box_d * box_h * box_w
            attempts += 1

    # Final clipping and cast back to original dtype
    # Determine numeric bounds for dtype
    if np.issubdtype(orig_dtype, np.integer):
        dtype_info = np.iinfo(orig_dtype)
        lo, hi = dtype_info.min, dtype_info.max
    else:
        dtype_info = np.finfo(orig_dtype)
        lo, hi = dtype_info.min, dtype_info.max

    raw_out = np.clip(raw_f, lo, hi).astype(orig_dtype)

    return raw_out, context, gt, mask
