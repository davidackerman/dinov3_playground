import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import imageio


def log_multi_volume_grid_to_tensorboard(
    writer,
    all_vol_data,
    epoch_step,
    tag_prefix="validation",
    fps=8,
    slice_axis=0,  # 0=Z, 1=Y, 2=X
    scale_up=3,  # tile upscaling for clarity
    gutter_px=10,  # gaps between columns/rows and between volumes
    header_px=26,  # per-column header height
    left_label_px=96,  # band for "GT/Pred"
    vol_label_px=110,  # vertical "vol id | loss" strip width
    save_gif_every=0,  # >0 to save GIF every N epochs
    show_seg_column=True,
    show_mask_column=True,
):
    """
    Multi-volume GT/Pred grid video + middle-slice image.
    Expects each item in all_vol_data to contain:
      vol_idx, vol_raw, vol_seg, vol_mask, vol_targets, vol_predictions, vol_loss
      (accepts (D,H,W), (N,D,H,W), (C,D,H,W), (N,C,D,H,W))
    """

    def to_np(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def dhw_of(a):
        if a is None:
            return None
        if a.ndim == 3:
            return tuple(a.shape)  # (D,H,W)
        if a.ndim == 4:
            return tuple(a.shape[1:])  # (C,D,H,W)
        return None

    def resize3d_np(a, out_dhw, mode="trilinear"):
        if a is None:
            return None
        a = np.asarray(a)
        out_dhw = tuple(int(x) for x in out_dhw)
        if a.ndim == 3:  # (D,H,W)
            t = torch.from_numpy(a.astype(np.float32))[None, None]  # (1,1,D,H,W)
            t = torch.nn.functional.interpolate(
                t,
                size=out_dhw,
                mode="nearest" if mode == "nearest" else "trilinear",
                align_corners=False if mode != "nearest" else None,
            )
            return t[0, 0].cpu().numpy()
        if a.ndim == 4:  # (C,D,H,W)
            t = torch.from_numpy(a.astype(np.float32))[None]  # (1,C,D,H,W)
            t = torch.nn.functional.interpolate(
                t,
                size=out_dhw,
                mode="nearest" if mode == "nearest" else "trilinear",
                align_corners=False if mode != "nearest" else None,
            )
            return t[0].cpu().numpy()
        raise ValueError(f"resize3d_np: unsupported ndim={a.ndim}; expected 3 or 4")

    def norm_uint8(img):
        img = np.nan_to_num(img)
        m, M = float(img.min()), float(img.max())
        if M <= m:
            return np.zeros_like(img, dtype=np.uint8)
        return np.clip((img - m) / (M - m) * 255, 0, 255).astype(np.uint8)

    def gray_to_rgb(img2d):
        g = norm_uint8(img2d)
        return np.stack([g, g, g], axis=-1)

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

    def label_to_rgb(label2d):
        lab = np.nan_to_num(label2d).astype(np.int32) % len(TAB20)
        return TAB20[lab]

    def reorient(arr, hasC=False):
        if arr is None:
            return None
        if hasC:
            arr = np.moveaxis(arr, 0, -1)  # (D,H,W,C)
            axes = [slice_axis, (slice_axis + 1) % 3, (slice_axis + 2) % 3, 3]
        else:
            axes = [slice_axis, (slice_axis + 1) % 3, (slice_axis + 2) % 3]
        return np.transpose(arr, axes)

    def resize_tile(img_rgb, scale):
        if scale == 1:
            return img_rgb
        h, w = img_rgb.shape[:2]
        return cv2.resize(
            img_rgb, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
        )

    def make_header(text, width, height):
        bar = np.zeros((height, width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.5, 1
        (tw, th_pix), _ = cv2.getTextSize(text, font, fs, th)
        x = max(6, (width - tw) // 2)
        y = height - max(6, (height - th_pix) // 2)
        cv2.putText(bar, text, (x + 1, y + 1), font, fs, (0, 0, 0), th + 1, cv2.LINE_AA)
        cv2.putText(bar, text, (x, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        return bar

    def vertical_text_strip(text, height, strip_w):
        """
        Create (height, strip_w, 3) with vertical centered text.
        """
        # draw on horizontal canvas (strip_w tall), then rotate
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.6, 1
        (tw, thp), _ = cv2.getTextSize(text, font, fs, th)
        # horizontal canvas big enough to hold text length
        canvas = np.zeros((strip_w, max(height, tw + 20), 3), dtype=np.uint8)
        x = 10
        y = strip_w // 2 + thp // 2
        cv2.putText(
            canvas, text, (x + 1, y + 1), font, fs, (0, 0, 0), th + 1, cv2.LINE_AA
        )
        cv2.putText(canvas, text, (x, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        rot = cv2.rotate(
            canvas, cv2.ROTATE_90_CLOCKWISE
        )  # (width, strip_w) -> (height_like, strip_w)
        # fit to exact height
        if rot.shape[0] > height:
            rot = rot[rot.shape[0] - height :, :, :]
        elif rot.shape[0] < height:
            pad = np.zeros((height - rot.shape[0], strip_w, 3), dtype=np.uint8)
            rot = np.concatenate([pad, rot], axis=0)
        return rot

    def left_margin_with_labels(total_h, tile_h, gutter_h, vol_text):
        vol_strip = vertical_text_strip(vol_text, total_h, vol_label_px)
        band = np.zeros((total_h, left_label_px, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.7, 2
        # GT centered in top tile
        (tw1, th1), _ = cv2.getTextSize("GT", font, fs, th)
        x1 = max(6, (left_label_px - tw1) // 2)
        y1 = header_px + tile_h // 2 + th1 // 2
        cv2.putText(
            band, "GT", (x1 + 1, y1 + 1), font, fs, (0, 0, 0), th + 1, cv2.LINE_AA
        )
        cv2.putText(band, "GT", (x1, y1), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        # Pred centered in bottom tile
        (tw2, th2), _ = cv2.getTextSize("Pred", font, fs, th)
        x2 = max(6, (left_label_px - tw2) // 2)
        y2 = header_px + tile_h + gutter_h + tile_h // 2 + th2 // 2
        cv2.putText(
            band, "Pred", (x2 + 1, y2 + 1), font, fs, (0, 0, 0), th + 1, cv2.LINE_AA
        )
        cv2.putText(band, "Pred", (x2, y2), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        return np.concatenate([vol_strip, band], axis=1)

    # ---------- preprocess each volume into a block builder ----------
    vol_blocks = []
    global_min_S = None

    for vd in all_vol_data:
        vol_id = int(vd.get("vol_idx", 0))
        # tensors/arrays
        raw_np = to_np(vd.get("vol_raw"))
        seg_np = to_np(vd.get("vol_seg"))
        mask_np = to_np(vd.get("vol_mask"))
        targ_np = to_np(vd.get("vol_targets"))
        pred_np = to_np(vd.get("vol_predictions"))
        vol_loss = vd.get("vol_loss", float("nan"))
        try:
            vol_loss = float(vol_loss)
        except Exception:
            vol_loss = float("nan")

        # If batched, pick first item (you can adapt this if you pass batch_idx)
        def pick_batch(arr):
            if arr is None:
                return None
            if arr.ndim == 5:  # (N,C,D,H,W)
                return arr[0]
            if arr.ndim == 4 and (arr.shape[0] > 1 and arr.shape[1] < 8):  # (N,D,H,W)
                return arr[0]
            return arr

        raw_np, seg_np, mask_np, targ_np, pred_np = map(
            pick_batch, [raw_np, seg_np, mask_np, targ_np, pred_np]
        )

        # target grid per-volume (smallest of provided)
        grids = [
            dhw_of(x)
            for x in [raw_np, seg_np, mask_np, targ_np, pred_np]
            if dhw_of(x) is not None
        ]
        if not grids:  # skip empty
            continue
        target_DHW = tuple(int(x) for x in np.min(np.array(grids), axis=0).tolist())

        raw_r = (
            resize3d_np(raw_np, target_DHW, "trilinear") if raw_np is not None else None
        )
        seg_r = (
            resize3d_np(seg_np, target_DHW, "nearest") if seg_np is not None else None
        )
        mask_r = (
            resize3d_np(mask_np, target_DHW, "trilinear")
            if mask_np is not None
            else None
        )
        targ_r = (
            resize3d_np(targ_np, target_DHW, "nearest") if targ_np is not None else None
        )
        pred_r = (
            resize3d_np(pred_np, target_DHW, "nearest") if pred_np is not None else None
        )

        # reorient to slice-first
        def reorient_any(a):
            if a is None:
                return None
            if a.ndim == 4:
                return reorient(a, hasC=True)  # (C,D,H,W) -> (S,H,W,C)
            if a.ndim == 3:
                return reorient(a, hasC=False)  # (D,H,W)   -> (S,H,W)
            return None

        raw_s, seg_s, mask_s, targ_s, pred_s = map(
            reorient_any, [raw_r, seg_r, mask_r, targ_r, pred_r]
        )

        # ADD THIS: Downsample for efficiency
        spatial_stride = 4  # downsample H, W by 4x
        slice_stride = 4  # take every 4th slice

        def downsample_volume(arr, slice_stride, spatial_stride):
            if arr is None:
                return None
            # Take every Nth slice
            arr = arr[::slice_stride]
            # Downsample spatial dimensions
            if arr.ndim == 3:  # (D, H, W)
                return arr[:, ::spatial_stride, ::spatial_stride]
            elif arr.ndim == 4:  # (D, H, W, C)
                return arr[:, ::spatial_stride, ::spatial_stride, :]
            return arr

        raw_s = downsample_volume(raw_s, slice_stride, spatial_stride)
        seg_s = downsample_volume(seg_s, slice_stride, spatial_stride)
        mask_s = downsample_volume(mask_s, slice_stride, spatial_stride)
        targ_s = downsample_volume(targ_s, slice_stride, spatial_stride)
        pred_s = downsample_volume(pred_s, slice_stride, spatial_stride)

        # sizes
        S = (
            raw_s.shape[0]
            if raw_s is not None
            else (
                seg_s.shape[0]
                if seg_s is not None
                else (
                    mask_s.shape[0]
                    if mask_s is not None
                    else targ_s.shape[0] if targ_s is not None else pred_s.shape[0]
                )
            )
        )

        Hs = (
            raw_s.shape[1]
            if raw_s is not None
            else (
                seg_s.shape[1]
                if seg_s is not None
                else (
                    mask_s.shape[1]
                    if mask_s is not None
                    else targ_s.shape[1] if targ_s is not None else pred_s.shape[1]
                )
            )
        )
        Ws = (
            raw_s.shape[2]
            if raw_s is not None
            else (
                seg_s.shape[2]
                if seg_s is not None
                else (
                    mask_s.shape[2]
                    if mask_s is not None
                    else targ_s.shape[2] if targ_s is not None else pred_s.shape[2]
                )
            )
        )

        # ----- IMPORTANT: bind arrays into lambdas to avoid late-binding bugs -----
        def make_gray_fn(arr):
            return (
                (lambda s, arr=arr: gray_to_rgb(arr[s]))
                if arr is not None
                else (lambda s, Hs=Hs, Ws=Ws: gray_to_rgb(np.zeros((Hs, Ws))))
            )

        def make_seg_gt_fn(seg_arr):
            return (
                (lambda s, seg_arr=seg_arr: label_to_rgb(seg_arr[s]))
                if seg_arr is not None
                else (lambda s, Hs=Hs, Ws=Ws: gray_to_rgb(np.zeros((Hs, Ws))))
            )

        def make_seg_pred_fn(seg_arr, pred_arr):
            if pred_arr is not None and pred_arr.ndim == 4 and pred_arr.shape[-1] > 1:
                return lambda s, pred_arr=pred_arr: label_to_rgb(
                    np.argmax(pred_arr[s], axis=-1)
                )
            if seg_arr is not None:
                return lambda s, seg_arr=seg_arr: label_to_rgb(seg_arr[s])
            return lambda s, Hs=Hs, Ws=Ws: gray_to_rgb(np.zeros((Hs, Ws)))

        def make_mask_gt_fn(mask_arr):
            return (
                (lambda s, mask_arr=mask_arr: gray_to_rgb(mask_arr[s]))
                if mask_arr is not None
                else (lambda s, Hs=Hs, Ws=Ws: gray_to_rgb(np.zeros((Hs, Ws))))
            )

        def make_mask_pred_fn(mask_arr, pred_arr):
            if pred_arr is None:
                return make_mask_gt_fn(mask_arr)
            if pred_arr.ndim == 4 and pred_arr.shape[-1] == 1:
                return lambda s, pred_arr=pred_arr: gray_to_rgb(
                    (pred_arr[s, :, :, 0] > 0.5).astype(np.float32)
                )
            if pred_arr.ndim == 4 and pred_arr.shape[-1] > 1:
                return lambda s, pred_arr=pred_arr: gray_to_rgb(
                    (np.argmax(pred_arr[s], axis=-1) > 0).astype(np.float32)
                )
            return lambda s, Hs=Hs, Ws=Ws: gray_to_rgb(np.zeros((Hs, Ws)))

        def make_ch_gt_fn(targ_arr, ci):
            """Return GT-channel function (s -> RGB), normalized 0–1 per GT channel."""
            if targ_arr is not None and targ_arr.ndim == 4 and ci < targ_arr.shape[-1]:
                gt_ch = targ_arr[..., ci]
                vmin, vmax = np.nanmin(gt_ch), np.nanmax(gt_ch)

                # handle constant-valued channels
                if np.isclose(vmin, vmax):
                    vmin, vmax = 0.0, 1.0

                def fn(s, targ_arr=targ_arr, ci=ci, vmin=vmin, vmax=vmax):
                    img = (targ_arr[s, :, :, ci] - vmin) / (vmax - vmin)
                    img = np.clip(img, 0, 1)
                    return gray_to_rgb(img)

                return fn

            return lambda s, Hs=Hs, Ws=Ws: gray_to_rgb(np.zeros((Hs, Ws)))

        def make_ch_pred_fn(pred_arr, targ_arr, ci):
            """Return Pred-channel function normalized by GT channel min/max, or 0–1 fallback."""
            if targ_arr is not None and targ_arr.ndim == 4 and ci < targ_arr.shape[-1]:
                vmin, vmax = np.nanmin(targ_arr[..., ci]), np.nanmax(targ_arr[..., ci])
            else:
                vmin, vmax = 0.0, 1.0

            # handle constant-valued target channels
            if np.isclose(vmin, vmax):
                vmin, vmax = 0.0, 1.0

            if pred_arr is not None and pred_arr.ndim == 4 and ci < pred_arr.shape[-1]:

                def fn(s, pred_arr=pred_arr, ci=ci, vmin=vmin, vmax=vmax):
                    img = (pred_arr[s, :, :, ci] - vmin) / (vmax - vmin)
                    img = np.clip(img, 0, 1)
                    return gray_to_rgb(img)

                return fn

            return lambda s, Hs=Hs, Ws=Ws: gray_to_rgb(np.zeros((Hs, Ws)))

        # column defs (each entry: title, gt_fn, pred_fn) — all fns are properly bound
        columns = []
        if raw_s is not None:
            columns.append(("raw", make_gray_fn(raw_s), make_gray_fn(raw_s)))

        if show_seg_column and (seg_s is not None or pred_s is not None):
            columns.append(
                ("seg", make_seg_gt_fn(seg_s), make_seg_pred_fn(seg_s, pred_s))
            )

        if show_mask_column and (mask_s is not None or pred_s is not None):
            columns.append(
                ("mask", make_mask_gt_fn(mask_s), make_mask_pred_fn(mask_s, pred_s))
            )

        Ct = targ_s.shape[-1] if (targ_s is not None and targ_s.ndim == 4) else 0
        Cp = pred_s.shape[-1] if (pred_s is not None and pred_s.ndim == 4) else 0
        C = max(Ct, Cp)
        for c in range(C):
            columns.append(
                (f"ch{c}", make_ch_gt_fn(targ_s, c), make_ch_pred_fn(pred_s, targ_s, c))
            )

        # per-volume block builder
        def build_block_for_slice(
            s, columns=columns, Hs=Hs, Ws=Ws, vol_id=vol_id, vol_loss=vol_loss
        ):
            col_imgs = []
            for title, gt_fn, pr_fn in columns:
                gt_t = resize_tile(gt_fn(s), scale_up)
                pr_t = resize_tile(pr_fn(s), scale_up)
                header = make_header(title, gt_t.shape[1], header_px)
                v_gut = np.zeros((gutter_px, gt_t.shape[1], 3), dtype=np.uint8)
                col = np.concatenate([header, gt_t, v_gut, pr_t], axis=0)
                col_imgs.append(col)

            v_gut = np.zeros((col_imgs[0].shape[0], gutter_px, 3), dtype=np.uint8)
            row_img = col_imgs[0]
            for k in range(1, len(col_imgs)):
                row_img = np.concatenate([row_img, v_gut, col_imgs[k]], axis=1)

            tile_h = resize_tile(np.zeros((Hs, Ws, 3), dtype=np.uint8), scale_up).shape[
                0
            ]
            vol_text = (
                f"vol {vol_id} | loss {vol_loss:.4f}"
                if not np.isnan(vol_loss)
                else f"vol {vol_id}"
            )
            left_margin = left_margin_with_labels(
                total_h=row_img.shape[0],
                tile_h=tile_h,
                gutter_h=gutter_px,
                vol_text=vol_text,
            )
            return np.concatenate([left_margin, row_img], axis=1)

        vol_blocks.append({"S": S, "build": build_block_for_slice})
        global_min_S = S if global_min_S is None else min(global_min_S, S)

    if not vol_blocks:
        print("[WARN] No valid volumes to render.")
        return

    # assemble frames (use min depth across volumes)
    frames = []
    for s in range(global_min_S):
        blocks = [vb["build"](s) for vb in vol_blocks]
        h_gut = np.zeros((gutter_px, blocks[0].shape[1], 3), dtype=np.uint8)
        big = blocks[0]
        for b in blocks[1:]:
            if b.shape[1] != big.shape[1]:
                W = max(b.shape[1], big.shape[1])

                def pad_right(img, W):
                    if img.shape[1] == W:
                        return img
                    pad = np.zeros((img.shape[0], W - img.shape[1], 3), dtype=np.uint8)
                    return np.concatenate([img, pad], axis=1)

                big = pad_right(big, W)
                b = pad_right(b, W)
                h_gut = np.zeros((gutter_px, W, 3), dtype=np.uint8)
            big = np.concatenate([big, h_gut, b], axis=0)
        frames.append(big)

    frames = np.stack(frames, axis=0)  # (T, Htot, Wtot, 3)

    # log video
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
    writer.add_video(f"{tag_prefix}/multi_volume_grid", video, epoch_step, fps=fps)

    # middle slice image
    mid = global_min_S // 2
    writer.add_image(
        f"{tag_prefix}/multi_volume_grid_middle",
        np.transpose(frames[mid], (2, 0, 1)),
        epoch_step,
        dataformats="CHW",
    )

    # optional GIF
    if save_gif_every and (epoch_step % save_gif_every == 0):
        gif_dir = os.path.join(writer.log_dir, "validation_gifs")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"val_multi_epoch_{epoch_step:04d}.gif")
        imageio.mimsave(gif_path, frames, fps=fps)


def log_validation_losses_to_tensorboard(
    writer, epoch, train_loss, val_loss_overall, val_loss_per_crop
):
    """
    Log training and validation losses to TensorBoard scalars.
    By using the same parent group 'loss/', TensorBoard will automatically
    plot all these metrics together on the same chart.

    Args:
        writer: TensorBoard writer
        epoch: Current epoch number
        train_loss: Training loss value
        val_loss_overall: Overall validation loss
        val_loss_per_crop: List of per-crop validation losses
    """
    # Log training loss (using consistent naming convention)
    writer.add_scalar("loss/train", train_loss, epoch)

    # Log overall validation loss
    writer.add_scalar("loss/val_overall", val_loss_overall, epoch)

    # Log per-crop validation losses (all under 'loss/' prefix so they appear together)
    for crop_idx, crop_loss in enumerate(val_loss_per_crop):
        writer.add_scalar(f"loss/val_crop_{crop_idx}", crop_loss, epoch)
