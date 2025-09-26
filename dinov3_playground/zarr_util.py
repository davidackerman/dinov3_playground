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


def update_datapaths_with_target_scales(dataset_pairs, base_resolution):
    updated_dataset_pairs = []
    for dataset_pair in dataset_pairs:
        updated_dataset_pair = (
            find_target_scale(dataset_pair[0], base_resolution)[0],
            find_target_scale(dataset_pair[1], base_resolution)[0],
        )
        updated_dataset_pairs.append(updated_dataset_pair)
    return updated_dataset_pairs
