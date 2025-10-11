# %%
import functools
import logging
from collections.abc import Sequence
from math import ceil

import numpy as np

logger = logging.getLogger(__name__)


def get_lsds_torch(
    segmentation: np.ndarray,
    sigma: float | Sequence[float],
    voxel_size: Sequence[int] | None = None,
    labels: Sequence[int] | None = None,
    downsample: int = 1,
    use_paper_ordering: bool = True,
):
    """
    GPU-accelerated local shape descriptors using PyTorch (falls back to CPU if no CUDA).
    Returns a float32 array shaped (C, *segmentation.shape) in [0, 1], with channels:
      dims=3 -> 10 channels: mean(3) + var(3) + pearson(3) + mass(1)
      dims=2 ->  6 channels: mean(2) + var(2) + pearson(1) + mass(1)
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError(
            "This GPU version requires PyTorch: pip install torch"
        ) from e

    assert all(
        (s // downsample) * downsample == s for s in segmentation.shape
    ), f"Segmentation shape {segmentation.shape} must be divisible by downsample={downsample}."

    dims = segmentation.ndim
    assert dims in (2, 3), "Only 2D or 3D segmentations are supported."

    if isinstance(sigma, (int, float)):
        sigma = (float(sigma),) * dims
    sigma = tuple(float(s) for s in sigma)
    assert len(sigma) == dims

    if voxel_size is None:
        voxel_size = (1,) * dims
    else:
        assert len(voxel_size) == dims, "voxel_size must match segmentation dims"

    # Downsampled view (no copy)
    df = int(downsample)
    sub_seg = segmentation[tuple(slice(None, None, df) for _ in range(dims))]
    sub_shape = sub_seg.shape

    # World/voxel scales at sub-sampling
    voxel_size = np.array(voxel_size, dtype=np.float32)
    sub_vs = voxel_size * df
    sub_sigma_vox = np.array(sigma, dtype=np.float32) / sub_vs

    # Labels to process
    if labels is None:
        labels = np.unique(sub_seg)
    labels = [int(l) for l in labels if int(l) != 0]

    # Torch device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.tensor  # shorthand
    seg_t = torch.from_numpy(sub_seg).to(device)

    # Output
    C = 10 if dims == 3 else 6
    full_shape = segmentation.shape
    out = torch.zeros((C,) + full_shape, dtype=torch.float32, device=device)

    # Axis coordinates in world units at subscale (lazy slices later)
    axes_world = []
    for d in range(dims):
        n = sub_shape[d]
        aw = t(np.arange(n, dtype=np.float32) * sub_vs[d], device=device)
        axes_world.append(aw)

    # --- helpers ---
    def _gauss1d_kernel(sig, truncate=3.0):
        if sig <= 0:
            return torch.tensor([1.0], dtype=torch.float32, device=device)
        # SciPy: lw = int(truncate * sigma + 0.5)
        r = int(truncate * float(sig) + 0.5)
        x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        k = torch.exp(-(x * x) / (2.0 * sig * sig))
        k = k / k.sum()
        return k

    def _blur_separable(fields, sigmas_vox, truncate=3.0):
        if dims == 3:
            Cc, Z, Y, X = fields.shape
            x = fields.unsqueeze(0)  # (1,C,Z,Y,X)

            kz = _gauss1d_kernel(sigmas_vox[0], truncate)
            ky = _gauss1d_kernel(sigmas_vox[1], truncate)
            kx = _gauss1d_kernel(sigmas_vox[2], truncate)

            rz, ry, rx = kz.numel() // 2, ky.numel() // 2, kx.numel() // 2

            # Z pass
            w = kz.view(1, 1, -1, 1, 1).expand(Cc, 1, -1, 1, 1)
            x = F.conv3d(x, w, groups=Cc, padding=(rz, 0, 0))

            # Y pass
            w = ky.view(1, 1, 1, -1, 1).expand(Cc, 1, 1, -1, 1)
            x = F.conv3d(x, w, groups=Cc, padding=(0, ry, 0))

            # X pass
            w = kx.view(1, 1, 1, 1, -1).expand(Cc, 1, 1, 1, -1)
            x = F.conv3d(x, w, groups=Cc, padding=(0, 0, rx))

            return x.squeeze(0)
        else:
            Cc, Y, X = fields.shape
            x = fields.unsqueeze(0)  # (1,C,Y,X)

            ky = _gauss1d_kernel(sigmas_vox[0], truncate)
            kx = _gauss1d_kernel(sigmas_vox[1], truncate)
            ry, rx = ky.numel() // 2, kx.numel() // 2

            # Y pass
            w = ky.view(1, 1, -1, 1).expand(Cc, 1, -1, 1)
            x = F.conv2d(x, w, groups=Cc, padding=(ry, 0))

            # X pass
            w = kx.view(1, 1, 1, -1).expand(Cc, 1, 1, -1)
            x = F.conv2d(x, w, groups=Cc, padding=(0, rx))

            return x.squeeze(0)

    def _repeat_fullres(arr_sub):
        # arr_sub: (C, z,y,x) or (C, y,x) -> upsample by df along spatial dims
        a = arr_sub
        for _ in range(df - 1):
            pass  # cheap fast path when df==1
        for ax in range(1, a.ndim):
            a = torch.repeat_interleave(a, df, dim=ax)
        return a

    # precompute integer padding (in vox) for ROI (support 3*sigma)
    pad_vox = np.ceil(sub_sigma_vox * 3.0).astype(np.int64)

    # Process each label
    for lbl in labels:
        m = seg_t == lbl
        if not bool(m.any()):
            continue

        # ROI bounds in subscale
        idx = m.nonzero(as_tuple=False)
        lo = torch.maximum(
            idx.min(dim=0).values - t(pad_vox, device=device),
            torch.zeros(dims, device=device, dtype=torch.long),
        )
        hi = torch.minimum(
            idx.max(dim=0).values + 1 + t(pad_vox, device=device),
            t(list(sub_shape), device=device, dtype=torch.long),
        )

        sl = tuple(slice(int(lo[d].item()), int(hi[d].item())) for d in range(dims))
        M = m[sl].to(torch.float32)  # mask in ROI

        # Coordinate views in world units for ROI
        coords = []
        for d in range(dims):
            aw = axes_world[d][sl[d]]
            shape = [1] * dims
            shape[d] = aw.numel()
            coords.append(aw.view(*shape))  # broadcastable

        # Build fields (mass + first + second unique moments)
        fields = [M]  # mass
        # first moments
        for d in range(dims):
            fields.append(M * coords[d])

        # second moments (unique upper triangle i<=j)
        pair_idx = []
        for i in range(dims):
            for j in range(i, dims):
                pair_idx.append((i, j))
                fields.append(M * coords[i] * coords[j])

        Flds = torch.stack(fields, dim=0)  # (C_f, ROI...)

        # Blur all channels with Gaussian (separable)
        Flds_s = _blur_separable(Flds, sub_sigma_vox)

        # Unpack smoothed fields
        p = 0
        mass = Flds_s[p]
        p += 1
        mass_zero = mass.eq(0)
        mass_safe = torch.where(mass_zero, torch.ones_like(mass), mass)

        means = []
        for d in range(dims):
            means.append(Flds_s[p] / mass_safe)
            p += 1
        means = torch.stack(means, dim=0)  # (dims, ...)

        seconds = []
        for _ in pair_idx:
            seconds.append(Flds_s[p] / mass_safe)
            p += 1
        seconds = seconds  # list len = dims*(dims+1)/2

        # Covariance components: Cov[i,j] = E[x_i x_j] - E[x_i]E[x_j]
        # Build raw variances and off-diagonals in world units
        var_raw = []
        offdiag = []
        sec_idx = 0
        for i in range(dims):
            for j in range(i, dims):
                eij = seconds[sec_idx]
                sec_idx += 1
                if i == j:
                    var_raw.append(eij - means[i] * means[i])
                else:
                    offdiag.append((i, j, eij - means[i] * means[j]))
        var_raw = torch.stack(var_raw, dim=0)  # (dims, ...)

        # mean offset normalization: ((COM - coord)/sigma)*0.5 + 0.5
        center = torch.stack([c.expand_as(M) for c in coords], dim=0)
        sig_world_t = t(sigma, device=device, dtype=torch.float32).view(
            dims, *([1] * dims)
        )
        mean_offset = (means - center) / sig_world_t * 0.5 + 0.5  # (dims, ...)

        # Variance normalized by sigma^2 (world)
        variance = var_raw / (sig_world_t**2)
        variance = torch.clamp(variance, min=1e-3)

        # Pearson (use *un*normalized variances, to match your original)
        pearsons = []
        for i, j, cij in offdiag:
            denom = 2.0 * torch.sqrt(torch.clamp(var_raw[i] * var_raw[j], min=1e-6))
            pij = cij / denom + 0.5
            pearsons.append(pij)
        if pearsons:
            pearsons = torch.stack(pearsons, dim=0)  # (n_off, ...)
        else:
            pearsons = torch.empty((0,) + M.shape, device=device, dtype=torch.float32)

        # Paper-ordering for 3D: XY, XZ, YZ -> [0,2,1]
        # final desired order: [XY, XZ, YZ]
        if dims == 3 and pearsons.shape[0] == 3 and use_paper_ordering:
            pearsons = pearsons[[0, 1, 2], ...]  # i.e., no swap

        # Reset mass in empty regions (match SciPy trick)
        mass = mass * (~mass_zero)

        # Assemble descriptor in ROI (channel-first)
        desc_roi = torch.cat(
            [mean_offset, variance, pearsons, mass.unsqueeze(0)], dim=0
        )  # (C, ...)

        # Upsample to full resolution and paste masked into output
        if df > 1:
            desc_roi = _repeat_fullres(desc_roi)

        # Full-res ROI slices (convert to Python ints for indexing)
        lo_np = [int(lo[d].item()) for d in range(dims)]
        hi_np = [int(hi[d].item()) for d in range(dims)]

        sl_full = tuple(slice(lo_np[d] * df, hi_np[d] * df) for d in range(dims))
        mask_full = (
            torch.from_numpy(segmentation[sl_full]).to(device) == lbl
        ).unsqueeze(0)

        # Index the output tensor with regular slices
        out_view = out[(slice(None),) + sl_full]  # (C, spatial_dims...)
        out[(slice(None),) + sl_full] = out_view + desc_roi * mask_full

    out.clamp_(0.0, 1.0)
    return out.detach().cpu().numpy().astype(np.float32)


# # %%
#if name=="__main__":

# import numpy as np
# import matplotlib.pyplot as plt
# from lsd_lite import get_lsds as get_lsds_original
# from matplotlib import cm
# import time

# # Create a test segmentation
# np.random.seed(42)
# test_seg = np.zeros((128, 128, 128), dtype=np.uint8)
# # make sphere in center of gt filled with 1
# # z, y, x = np.ogrid[-64:64, -64:64, -64:64]
# # mask = x**2 + y**2 + z**2 <= 32**2
# # gt[mask] = 1
# test_seg[32:96, 32:96, 1:-1] = 1
# test_seg = gt[0]
# sigma = 20.0
# t0 = time.time()
# print(f"Computing LSDs with sigma={sigma}...")
# lsds_original = get_lsds_original(test_seg, sigma=sigma)
# t1 = time.time()
# print(f"  - Original (lsd_lite) took {t1 - t0:.2f} seconds")
# lsds_torch = get_lsds_torch(test_seg, sigma=sigma)
# t2 = time.time()
# print(f"  - PyTorch implementation took {t2 - t1:.2f} seconds")
# # Compute differences
# abs_diff = np.abs(lsds_original - lsds_torch)
# rel_diff = abs_diff / (np.abs(lsds_original) + 1e-8)

# channel_names = [
#     "Mean X",
#     "Mean Y",
#     "Mean Z",
#     "Var X",
#     "Var Y",
#     "Var Z",
#     "Pearson XY",
#     "Pearson XZ",
#     "Pearson YZ",
#     "Mass",
# ]

# # Print per-channel stats
# print(
#     f"{'Channel':<15} {'Mean Abs Diff':<15} {'Max Abs Diff':<15} {'Mean Rel Diff %':<20}"
# )
# print("-" * 70)
# for i in range(10):
#     ch_abs_diff = abs_diff[i]
#     ch_rel_diff = rel_diff[i]
#     print(
#         f"{channel_names[i]:<15} {ch_abs_diff.mean():<15.6f} {ch_abs_diff.max():<15.6f} {ch_rel_diff.mean() * 100:<20.4f}"
#     )

# # Find where differences are largest
# max_diff_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
# print(
#     f"\nMax difference at: channel {max_diff_idx[0]} ({channel_names[max_diff_idx[0]]}), position {max_diff_idx[1:]}"
# )
# print(f"  Original value: {lsds_original[max_diff_idx]:.6f}")
# print(f"  PyTorch value:  {lsds_torch[max_diff_idx]:.6f}")
# print(f"  Difference:     {abs_diff[max_diff_idx]:.6f}")

# # Interactive plotting
# # Plot 4 slices spread throughout the dataset and overlay channels as RGB,
# # add a delta row and extra spacer rows between slice groups, include colorbars.

# slices = np.linspace(0, test_seg.shape[0] - 1, 4, dtype=int)
# rgb_groups = [(0, 3), (3, 6), (6, 9)]  # (start, end) for RGB overlays
# n_cols = len(rgb_groups) + 1  # 3 RGB overlays + 1 mass channel

# # rows per slice: original, pytorch, delta, spacer -> bigger space between groups
# rows_per_slice = 4
# n_rows = len(slices) * rows_per_slice

# fig, axes = plt.subplots(
#     n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False
# )

# for i, z in enumerate(slices):
#     base_row = i * rows_per_slice

#     # Original and PyTorch rows
#     for rel_row, data, label in [
#         (0, lsds_original, "Original"),
#         (1, lsds_torch, "PyTorch"),
#     ]:
#         row = base_row + rel_row
#         for col, (ch_start, ch_end) in enumerate(rgb_groups):
#             # Use raw channel values without any normalization
#             rgb = np.stack(
#                 [data[ch, z].astype(np.float32) for ch in range(ch_start, ch_end)],
#                 axis=-1,
#             )
#             # imshow for RGB expects 0..1 floats or 0..255 ints; we intentionally do not normalize
#             axes[row, col].imshow(rgb)
#             axes[row, col].set_title(f"{label} Z={z} Ch[{ch_start}:{ch_end}]")
#             axes[row, col].axis("off")

#         # Mass channel (grayscale) - show raw values, scale by actual min/max for visibility
#         mass_img = data[9, z].astype(np.float32)
#         vmin = float(np.nanmin(mass_img))
#         vmax = float(np.nanmax(mass_img))
#         if vmax == vmin:
#             vmax = vmin + 1.0
#         im_mass = axes[row, -1].imshow(mass_img, cmap="gray", vmin=vmin, vmax=vmax)
#         axes[row, -1].set_title(f"{label} Z={z} Mass")
#         axes[row, -1].axis("off")
#         fig.colorbar(im_mass, ax=axes[row, -1], fraction=0.046, pad=0.01)

#     # Delta row: absolute differences (Original - PyTorch)
#     row_delta = base_row + 2
#     for col, (ch_start, ch_end) in enumerate(rgb_groups):
#         # compute per-channel abs diff WITHOUT per-channel normalization
#         chs = []
#         vmax_ch = 0.0
#         for ch in range(ch_start, ch_end):
#             diff = np.abs(
#                 lsds_original[ch, z].astype(np.float32)
#                 - lsds_torch[ch, z].astype(np.float32)
#             )
#             chs.append(diff)
#             vmax_ch = max(vmax_ch, float(np.nanmax(diff)))
#         rgb_delta = np.stack(chs, axis=-1)

#         # Display raw differences without normalizing channels; scale for display only if needed
#         if vmax_ch > 1.0:
#             rgb_disp = rgb_delta / vmax_ch
#         else:
#             rgb_disp = rgb_delta

#         axes[row_delta, col].imshow(rgb_disp)
#         axes[row_delta, col].set_title(f"Delta Z={z} Ch[{ch_start}:{ch_end}]")
#         axes[row_delta, col].axis("off")

#         # colorbar that reflects raw delta range for this slice (0..vmax_ch)
#         cmax = vmax_ch if vmax_ch > 0 else 1.0
#         m = cm.ScalarMappable(cmap="viridis")
#         m.set_array(np.linspace(0, cmax, 256))
#         m.set_clim(0, cmax)
#         fig.colorbar(m, ax=axes[row_delta, col], fraction=0.046, pad=0.01)

#     # Delta for mass - show raw absolute difference, use vmin=0 so not centered/normalized by min
#     mass_delta = np.abs(
#         lsds_original[9, z].astype(np.float32) - lsds_torch[9, z].astype(np.float32)
#     )
#     vmax = float(np.nanmax(mass_delta))
#     if vmax == 0:
#         vmax = 1.0
#     im_mass_delta = axes[row_delta, -1].imshow(
#         mass_delta, cmap="gray", vmin=0.0, vmax=vmax
#     )
#     axes[row_delta, -1].set_title(f"Delta Z={z} Mass")
#     axes[row_delta, -1].axis("off")
#     fig.colorbar(im_mass_delta, ax=axes[row_delta, -1], fraction=0.046, pad=0.01)

#     # Spacer row to increase separation
#     spacer_row = base_row + 3
#     for c in range(n_cols):
#         axes[spacer_row, c].axis("off")

# plt.tight_layout()
# plt.show()
# # %%
# Plot per-channel histograms of the PyTorch (lsds_torch) output, excluding zeros.
# lsds_torch = get_lsds_torch(gt[8], sigma=5)
# C = lsds_torch.shape[0]
# names = channel_names if "channel_names" in globals() else [f"Ch {i}" for i in range(C)]

# ncols = 5
# nrows = ceil(C / ncols)
# for current_lsd in [lsds_original, lsds_torch]:
#     fig, axes = plt.subplots(
#         nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
#     )
#     bins = 100
#     for i in range(C):
#         ax = axes[i // ncols][i % ncols]
#         vals = current_lsd[i].ravel().astype(np.float32)
#         vals_nz = vals[vals != 0]
#         if vals_nz.size == 0:
#             ax.text(0.5, 0.5, "no non-zero values", ha="center", va="center")
#             ax.set_title(names[i])
#             ax.set_xlabel("value")
#             ax.set_ylabel("count")
#             continue
#         ax.hist(vals_nz, bins=bins, range=(0.0, 1.0), color="C0", alpha=0.8)
#         ax.set_title(f"{names[i]} (n={vals_nz.size:,})")
#         ax.set_xlabel("value")
#         ax.set_ylabel("count")

#     # hide any empty subplots
#     for j in range(C, nrows * ncols):
#         axes[j // ncols][j % ncols].axis("off")

#     plt.tight_layout()
#     plt.show()

# %%
