# %%
import zarr
from funlib.geometry import Coordinate
from pathlib import Path
import numpy as np


def get_scale_info(zarr_grp):
    attrs = zarr_grp.attrs
    resolutions = {}
    offsets = {}
    shapes = {}
    # making a ton of assumptions here, hopefully triggering KeyErrors though if they don't apply
    for scale in attrs["multiscales"][0]["datasets"]:
        resolutions[scale["path"]] = scale["coordinateTransformations"][0]["scale"]
        offsets[scale["path"]] = scale["coordinateTransformations"][1]["translation"]
        shapes[scale["path"]] = zarr_grp[scale["path"]].shape
    # offset = min(offsets.values())
    return offsets, resolutions, shapes


def find_target_scale(data_path, target_resolution):
    if type(target_resolution) is int or type(target_resolution) is float:
        target_resolution = Coordinate(3 * [target_resolution])
    if type(data_path) is str:
        data_path = Path(data_path)
    try:
        zarr_grp = zarr.open_group(data_path, mode="r")
    except Exception as e:
        msg = f"Could not open zarr group at {data_path}, error: {e}"
        raise ValueError(msg)
    offsets, resolutions, shapes = get_scale_info(zarr_grp)
    target_scale = None
    min_difference = np.inf
    use_approximate = True
    for scale, res in resolutions.items():
        # get closest scale
        if Coordinate(res) == Coordinate(target_resolution):
            target_scale = scale
            use_approximate = False
            break
        else:
            difference = np.linalg.norm(Coordinate(res) - Coordinate(target_resolution))
            if difference < min_difference:
                min_difference = difference
                target_scale = scale

    if target_scale is None:
        msg = f"Zarr {zarr_grp.store.path}, {zarr_grp.path} does not contain array with sampling {target_resolution}"
        raise ValueError(msg)
    if use_approximate:
        msg = f"Warning: Zarr {zarr_grp.store.path}, {zarr_grp.path} does not contain array with exact sampling {target_resolution}, using closest scale {target_scale} with resolution {resolutions[target_scale]}"
        # print warning
        print(msg)
    target_path = data_path / target_scale
    return (
        str(target_path),
        target_scale,
        offsets[target_scale],
        shapes[target_scale],
    )


def find_highest_resolution_scale(data_path, min_resolution=None):
    """
    Find the highest resolution (smallest voxel size) scale available in a zarr dataset.

    Parameters:
    -----------
    data_path : str or Path
        Path to the zarr dataset
    min_resolution : int, float, or Coordinate, optional
        Minimum allowed resolution (maximum voxel size). If specified, will not use
        scales with resolution smaller than this value. Useful to avoid extremely
        high resolution scales that may be computationally prohibitive.

    Returns:
    --------
    tuple: (target_path, target_scale, offset, shape)
    """
    if type(data_path) is str:
        data_path = Path(data_path)
    try:
        zarr_grp = zarr.open_group(data_path, mode="r")
    except Exception as e:
        msg = f"Could not open zarr group at {data_path}, error: {e}"
        raise ValueError(msg)

    offsets, resolutions, shapes = get_scale_info(zarr_grp)

    # Convert min_resolution to Coordinate if provided
    if min_resolution is not None:
        if type(min_resolution) is int or type(min_resolution) is float:
            min_resolution = Coordinate(3 * [min_resolution])
        else:
            min_resolution = Coordinate(min_resolution)

    # Find the scale with the smallest resolution (highest resolution)
    # that is still >= min_resolution
    best_scale = None
    best_resolution = None

    for scale, res in resolutions.items():
        res_coord = Coordinate(res)

        # Skip scales that are finer than min_resolution (if specified)
        if min_resolution is not None:
            if any(r < min_r for r, min_r in zip(res_coord, min_resolution)):
                print(
                    f"Skipping scale {scale} with resolution {res} (finer than min_resolution {min_resolution})"
                )
                continue

        # Calculate total resolution (product of all dimensions)
        total_res = np.prod(res)

        if best_resolution is None or total_res < best_resolution:
            best_resolution = total_res
            best_scale = scale

    if best_scale is None:
        if min_resolution is not None:
            msg = f"No scales found in zarr {zarr_grp.store.path}, {zarr_grp.path} that meet min_resolution requirement {min_resolution}"
        else:
            msg = f"No scales found in zarr {zarr_grp.store.path}, {zarr_grp.path}"
        raise ValueError(msg)

    resolution_info = f"Using highest resolution scale {best_scale} with resolution {resolutions[best_scale]} for {data_path}"
    if min_resolution is not None:
        resolution_info += f" (min_resolution: {min_resolution})"
    print(resolution_info)

    target_path = data_path / best_scale
    return (
        str(target_path),
        best_scale,
        offsets[best_scale],
        shapes[best_scale],
    )


def update_datapaths_with_target_scales(
    dataset_pairs,
    base_resolution,
    use_highest_res_for_raw=False,
    min_resolution_for_raw=None,
    context_scale=None,
):
    """
    Update dataset paths with target scales.

    Supports both legacy tuple format and new dictionary format:
    - Legacy: [(raw_path, gt_path), ...]
    - New: [{"raw": raw_path, "class_1": class1_path, ...}, ...]

    Parameters:
    -----------
    dataset_pairs : list
        List of dataset pairs in either tuple or dictionary format
    base_resolution : int or float
        Target resolution for segmentation labels
    use_highest_res_for_raw : bool, default=False
        If True, use the highest available resolution for raw data instead of base_resolution.
        This is useful for DINOv3 processing where higher resolution input can be beneficial.
    min_resolution_for_raw : int, float, or Coordinate, optional
        Minimum allowed resolution for raw data when use_highest_res_for_raw=True.
        Prevents using extremely high resolution scales that may be computationally prohibitive.
    context_scale : int, float, or Coordinate, optional
        If provided, updates context raw data paths (keys like "raw_64nm") to use the best
        available scale for the specified context resolution.

    Returns:
    --------
    list: Updated dataset pairs in the same format as input
    """
    updated_dataset_pairs = []

    for dataset_pair in dataset_pairs:
        print(f"Processing dataset: {dataset_pair}")

        if isinstance(dataset_pair, tuple):
            # Legacy tuple format: (raw_path, gt_path)
            if len(dataset_pair) != 2:
                raise ValueError(
                    f"Tuple format must have exactly 2 elements, got {len(dataset_pair)}"
                )

            # For legacy format, use the same resolution for both (maintain backward compatibility)
            if use_highest_res_for_raw:
                raw_path = find_highest_resolution_scale(
                    dataset_pair[0], min_resolution=min_resolution_for_raw
                )[0]
            elif min_resolution_for_raw is not None:
                raw_path = find_target_scale(dataset_pair[0], min_resolution_for_raw)[0]
            else:
                raw_path = find_target_scale(dataset_pair[0], base_resolution)[0]

            gt_path = find_target_scale(dataset_pair[1], base_resolution)[0]

            updated_dataset_pair = (raw_path, gt_path)
            updated_dataset_pairs.append(updated_dataset_pair)

        elif isinstance(dataset_pair, dict):
            # New dictionary format: {"raw": raw_path, "class_1": class1_path, ...}
            if "raw" not in dataset_pair:
                raise ValueError("Dictionary format must contain 'raw' key")

            updated_dataset_pair = {}

            # Update raw data path - use highest resolution if requested
            if use_highest_res_for_raw:
                updated_dataset_pair["raw"] = find_highest_resolution_scale(
                    dataset_pair["raw"], min_resolution=min_resolution_for_raw
                )[0]
                print(f"Using highest resolution for raw data")
            elif min_resolution_for_raw is not None:
                updated_dataset_pair["raw"] = find_target_scale(
                    dataset_pair["raw"], min_resolution_for_raw
                )[0]
            else:
                updated_dataset_pair["raw"] = find_target_scale(
                    dataset_pair["raw"], base_resolution
                )[0]

            # Update all class paths (any key that's not "raw" is treated as a class)
            # Always use base_resolution for segmentation labels
            for key, path in dataset_pair.items():
                if key == "raw":
                    continue  # Already handled above
                elif key.startswith("raw_") and "nm" in key:
                    # Context raw data - use context_scale if provided
                    if context_scale is not None:
                        updated_dataset_pair[key] = find_target_scale(
                            path, context_scale
                        )[0]
                        print(f"Using context resolution {context_scale} for {key}")
                    else:
                        # Keep original path if no context_scale specified
                        updated_dataset_pair[key] = path
                else:
                    # All other non-raw keys are class paths that use base_resolution
                    updated_dataset_pair[key] = find_target_scale(
                        path, base_resolution
                    )[0]
                    print(f"Using base resolution {base_resolution} for {key} labels")

            updated_dataset_pairs.append(updated_dataset_pair)

        else:
            raise ValueError(
                f"Dataset pair must be tuple or dict, got {type(dataset_pair)}"
            )

    return updated_dataset_pairs
