import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import imageio


def create_and_log_validation_video_fast(
    writer,
    vol_idx,
    vol_raw,
    vol_seg,
    vol_mask,
    vol_targets,
    vol_predictions,
    epoch_step,
    tag_prefix="validation",
    fps=8,
    slice_axis=0,
    scale_up=1,
    gutter_px=10,
    header_px=26,
    left_label_px=60,
    save_gif_every=0,
    show_seg_column=True,
    show_mask_column=True,
):
    """Optimized version with vectorized operations and minimal conversions."""
    try:
        # ---------- Optimized helpers ----------
        def to_np(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        def resize3d_np(a, out_dhw, mode="trilinear"):
            """Batch resize - stays in torch until final conversion."""
            if a is None:
                return None
            is_four = a.ndim == 4
            if not is_four:
                a = a[None, ...]

            # Keep as torch tensor throughout
            if not isinstance(a, torch.Tensor):
                t = torch.from_numpy(a.astype(np.float32))
            else:
                t = a.float()

            t = F.interpolate(
                t.unsqueeze(0),
                size=out_dhw,
                mode="trilinear" if mode == "trilinear" else "nearest",
                align_corners=False if mode == "trilinear" else None,
            ).squeeze(0)

            out = t.numpy()
            return out if is_four else out[0]

        def norm_uint8_batch(img):
            """Vectorized normalization for entire volume."""
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            m, M = img.min(), img.max()
            if M <= m:
                return np.zeros_like(img, dtype=np.uint8)
            return ((img - m) / (M - m) * 255).astype(np.uint8)

        # Precompute color LUT for segmentation
        TAB20 = np.array(
            [
                [31, 119, 180],
                [255, 127, 14],
                [44, 160, 44],
                [214, 39, 40],
                [148, 103, 189],
                [140, 86, 75],
                [227, 119, 194],
                [127, 127, 127],
                [188, 189, 34],
                [23, 190, 207],
                [174, 199, 232],
                [255, 187, 120],
                [152, 223, 138],
                [255, 152, 150],
                [197, 176, 213],
                [196, 156, 148],
                [247, 182, 210],
                [199, 199, 199],
                [219, 219, 141],
                [158, 218, 229],
            ],
            dtype=np.uint8,
        )

        def label_to_rgb_batch(labels):
            """Vectorized label to RGB conversion for entire volume."""
            lab = np.nan_to_num(labels, nan=0).astype(np.int32) % len(TAB20)
            return TAB20[lab]

        def gray_to_rgb_batch(img):
            """Vectorized grayscale to RGB for entire volume."""
            g = norm_uint8_batch(img)
            return np.stack([g, g, g], axis=-1)

        def reorient(arr, hasC=False):
            """Move slice_axis to leading axis."""
            if arr is None:
                return None
            if hasC:
                arr = np.moveaxis(arr, 0, -1)
                axes = [slice_axis, (slice_axis + 1) % 3, (slice_axis + 2) % 3, 3]
            else:
                axes = [slice_axis, (slice_axis + 1) % 3, (slice_axis + 2) % 3]
            return np.transpose(arr, axes)

        def resize_tile_batch(imgs, scale):
            """Batch resize all slices at once using vectorized OpenCV."""
            if scale == 1:
                return imgs
            S, H, W = imgs.shape[:3]
            new_h, new_w = H * scale, W * scale
            # Preallocate output
            if imgs.ndim == 4:
                resized = np.empty((S, new_h, new_w, imgs.shape[3]), dtype=imgs.dtype)
            else:
                resized = np.empty((S, new_h, new_w), dtype=imgs.dtype)

            for i in range(S):
                resized[i] = cv2.resize(
                    imgs[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST
                )
            return resized

        def make_header(text, width, height):
            bar = np.zeros((height, width, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs, th = 0.5, 1
            (tw, th_pix), _ = cv2.getTextSize(text, font, fs, th)
            x = max(6, (width - tw) // 2)
            y = height - max(6, (height - th_pix) // 2)
            cv2.putText(
                bar, text, (x + 1, y + 1), font, fs, (0, 0, 0), th + 1, cv2.LINE_AA
            )
            cv2.putText(bar, text, (x, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
            return bar

        # ---------- Input processing ----------
        vol_raw = to_np(vol_raw)
        vol_seg = to_np(vol_seg)
        vol_mask = to_np(vol_mask)
        vol_targets = to_np(vol_targets)
        vol_predictions = to_np(vol_predictions)

        # Get target shape
        grids = []
        for x in [vol_raw, vol_seg, vol_mask, vol_targets, vol_predictions]:
            if x is not None:
                shape = x.shape[-3:] if x.ndim == 4 else x.shape
                grids.append(shape)

        if not grids:
            print(f"[WARN] No valid volumes for vol {vol_idx}")
            return

        target_DHW = tuple(np.min(grids, axis=0).tolist())

        # Batch resample all volumes
        raw_r = (
            resize3d_np(vol_raw, target_DHW, "trilinear")
            if vol_raw is not None
            else None
        )
        seg_r = (
            resize3d_np(vol_seg, target_DHW, "nearest") if vol_seg is not None else None
        )
        mask_r = (
            resize3d_np(vol_mask, target_DHW, "trilinear")
            if vol_mask is not None
            else None
        )
        targ_r = (
            resize3d_np(vol_targets, target_DHW, "nearest")
            if vol_targets is not None
            else None
        )
        pred_r = (
            resize3d_np(vol_predictions, target_DHW, "nearest")
            if vol_predictions is not None
            else None
        )

        # Reorient to slice-first
        raw_s = reorient(raw_r, hasC=False) if raw_r is not None else None
        seg_s = reorient(seg_r, hasC=False) if seg_r is not None else None
        mask_s = reorient(mask_r, hasC=False) if mask_r is not None else None
        targ_s = (
            reorient(targ_r, hasC=True)
            if (targ_r is not None and targ_r.ndim == 4)
            else None
        )
        pred_s = (
            reorient(pred_r, hasC=True)
            if (pred_r is not None and pred_r.ndim == 4)
            else None
        )

        # Get dimensions
        for arr in [raw_s, seg_s, mask_s, targ_s, pred_s]:
            if arr is not None:
                S, Hs, Ws = arr.shape[:3]
                break

        # ---------- Vectorized tile generation ----------
        columns_data = []  # (title, gt_tiles, pred_tiles) - already RGB

        # Raw
        if raw_s is not None:
            gt_tiles = gray_to_rgb_batch(raw_s)
            columns_data.append(("raw", gt_tiles, gt_tiles))

        # Seg
        if show_seg_column and (seg_s is not None or pred_s is not None):
            gt_tiles = (
                label_to_rgb_batch(seg_s)
                if seg_s is not None
                else gray_to_rgb_batch(np.zeros((S, Hs, Ws)))
            )

            if pred_s is not None and pred_s.shape[-1] > 1:
                pred_tiles = label_to_rgb_batch(np.argmax(pred_s, axis=-1))
            elif seg_s is not None:
                pred_tiles = label_to_rgb_batch(seg_s)
            else:
                pred_tiles = gray_to_rgb_batch(np.zeros((S, Hs, Ws)))

            columns_data.append(("seg", gt_tiles, pred_tiles))

        # Mask
        if show_mask_column and (mask_s is not None or pred_s is not None):
            gt_tiles = (
                gray_to_rgb_batch(mask_s)
                if mask_s is not None
                else gray_to_rgb_batch(np.zeros((S, Hs, Ws)))
            )

            if pred_s is None:
                pred_tiles = gt_tiles
            elif pred_s.shape[-1] == 1:
                pred_tiles = gray_to_rgb_batch(
                    (pred_s[:, :, :, 0] > 0.5).astype(np.float32)
                )
            else:
                pred_tiles = gray_to_rgb_batch(
                    (np.argmax(pred_s, axis=-1) > 0).astype(np.float32)
                )

            columns_data.append(("mask", gt_tiles, pred_tiles))

        # Per-channel columns
        Ct = targ_s.shape[-1] if targ_s is not None else 0
        Cp = pred_s.shape[-1] if pred_s is not None else 0
        C = max(Ct, Cp)

        for c in range(C):
            gt_tiles = (
                gray_to_rgb_batch(targ_s[:, :, :, c])
                if (targ_s is not None and c < Ct)
                else gray_to_rgb_batch(np.zeros((S, Hs, Ws)))
            )
            pred_tiles = (
                gray_to_rgb_batch(pred_s[:, :, :, c])
                if (pred_s is not None and c < Cp)
                else gray_to_rgb_batch(np.zeros((S, Hs, Ws)))
            )
            columns_data.append((f"target_{c}/pred_{c}", gt_tiles, pred_tiles))

        # ---------- Batch upscale ----------
        if scale_up > 1:
            columns_data = [
                (
                    title,
                    resize_tile_batch(gt, scale_up),
                    resize_tile_batch(pred, scale_up),
                )
                for title, gt, pred in columns_data
            ]

        # ---------- Compose frames (vectorized assembly) ----------
        tile_h = columns_data[0][1].shape[1]
        tile_w = columns_data[0][1].shape[2]

        # Precompute headers and gutters
        headers = [
            make_header(title, tile_w, header_px) for title, _, _ in columns_data
        ]
        v_gutter = np.zeros((gutter_px, tile_w, 3), dtype=np.uint8)
        h_gutter = np.zeros(
            (header_px + 2 * tile_h + gutter_px, gutter_px, 3), dtype=np.uint8
        )

        # Build left margin once
        total_h = header_px + 2 * tile_h + gutter_px
        left_margin = np.zeros((total_h, left_label_px, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.7, 2

        for label, y_offset in [
            ("GT", tile_h // 2),
            ("Pred", tile_h + gutter_px + tile_h // 2),
        ]:
            (tw, th_pix), _ = cv2.getTextSize(label, font, fs, th)
            x = max(6, (left_label_px - tw) // 2)
            y = header_px + y_offset + th_pix // 2
            cv2.putText(
                left_margin,
                label,
                (x + 1, y + 1),
                font,
                fs,
                (0, 0, 0),
                th + 1,
                cv2.LINE_AA,
            )
            cv2.putText(
                left_margin, label, (x, y), font, fs, (255, 255, 255), th, cv2.LINE_AA
            )

        # Assemble frames
        frames = []
        for s in range(S):
            cols = []
            for i, (_, gt_tiles, pred_tiles) in enumerate(columns_data):
                col = np.concatenate(
                    [headers[i], gt_tiles[s], v_gutter, pred_tiles[s]], axis=0
                )
                cols.append(col)

            row = cols[0]
            for col in cols[1:]:
                row = np.concatenate([row, h_gutter, col], axis=1)

            frame = np.concatenate([left_margin, row], axis=1)
            frames.append(frame)

        frames = np.stack(frames, axis=0)

        # ---------- TensorBoard logging ----------
        video = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
        )
        writer.add_video(f"{tag_prefix}/volume_{vol_idx}", video, epoch_step, fps=fps)
        writer.add_image(
            f"{tag_prefix}/volume_{vol_idx}_frame0",
            frames[0].transpose(2, 0, 1),
            epoch_step,
            dataformats="CHW",
        )

        # ---------- Optional GIF ----------
        if save_gif_every and (epoch_step % save_gif_every == 0):
            gif_dir = os.path.join(writer.log_dir, "validation_gifs")
            os.makedirs(gif_dir, exist_ok=True)
            gif_path = os.path.join(
                gif_dir, f"val_vol_{vol_idx:03d}_epoch_{epoch_step:04d}.gif"
            )
            imageio.mimsave(gif_path, frames, fps=fps)

    except Exception as e:
        print(f"[WARN] Failed to log validation video for vol {vol_idx}: {e}")
