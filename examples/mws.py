# %%
import numpy as np
import mwatershed as mws
from scipy import ndimage
import fastremap


def filter_fragments(
    affs_data: np.ndarray, fragments_data: np.ndarray, filter_val: float
) -> None:
    """Allows filtering of MWS fragments based on mean value of affinities & fragments. Will filter and update the fragment array in-place.

    Args:
        aff_data (``np.ndarray``):
            An array containing affinity data.

        fragments_data (``np.ndarray``):
            An array containing fragment data.

        filter_val (``float``):
            Threshold to filter if the average value falls below.
    """

    average_affs: float = np.mean(affs_data.data, axis=0)

    filtered_fragments: list = []

    fragment_ids: np.ndarray = np.unique(fragments_data)

    for fragment, mean in zip(
        fragment_ids, ndimage.mean(average_affs, fragments_data, fragment_ids)
    ):
        if mean < filter_val:
            filtered_fragments.append(fragment)

    filtered_fragments: np.ndarray = np.array(
        filtered_fragments, dtype=fragments_data.dtype
    )
    # replace: np.ndarray = np.zeros_like(filtered_fragments)
    fastremap.mask(fragments_data, filtered_fragments, in_place=True)
    return fragments_data


def mutex_watershed(affinities, neighborhood, adjacent_edge_bias, lr_bias, filter_val):
    if affinities.dtype == np.uint8:
        # logger.info("Assuming affinities are in [0,255]")
        max_affinity_value: float = 255.0
        affinities = affinities.astype(np.float64)
    else:
        affinities = affinities.astype(np.float64)
        max_affinity_value: float = 1.0

    affinities /= max_affinity_value

    if affinities.max() < 1e-4:
        segmentation = np.zeros(affinities.shape[1:], dtype=np.uint64)
        return segmentation

    # If affinities tensor contains extra channels (for example, LSDS
    # concatenated before affinities), try to select the last
    # `neighborhood.shape[0]` channels as the actual affinities. This handles
    # the convention where LSDS (10 channels) are prepended.
    expected_affs = neighborhood.shape[0]
    if affinities.ndim >= 1 and affinities.shape[0] > expected_affs:
        extra = affinities.shape[0] - expected_affs
        if extra == 10:
            # common case: 10 LSDS channels were prepended
            affinities = affinities[10:]
        else:
            # fallback: take the last `expected_affs` channels
            affinities = affinities[-expected_affs:]

    # Add tiny random noise to break ties
    random_noise = np.random.randn(*affinities.shape) * 0.0001

    # Compute gaussian blur sigma matching the spatial rank of affinities.
    # affinities usually has shape (C, Z, Y, X) or (C, Y, X). We want a sigma
    # tuple whose length equals affinities.ndim. For channel-first data we
    # include a zero sigma for the channel axis.
    dim = neighborhood.shape[1] if neighborhood.ndim > 1 else affinities.ndim - 1
    spatial_ndim = affinities.ndim - 1 if affinities.ndim > 1 else affinities.ndim
    max_offset = np.amax(neighborhood, axis=0)
    # Use the last `spatial_ndim` components of max_offset in case neighborhood
    # is 3D but affinities are 2D (e.g., (C,Y,X)).
    spatial_offsets = max_offset[-spatial_ndim:]
    spatial_sigmas = tuple((spatial_offsets / 3.0).tolist())
    if affinities.ndim == spatial_ndim:
        sigma = spatial_sigmas
    else:
        sigma = (0.0,) + spatial_sigmas

    smoothed_affs = (ndimage.gaussian_filter(affinities, sigma=sigma) - 0.5) * 0.001
    shift = []
    bias_idx = -1
    previous_offset = np.linalg.norm(neighborhood[0])
    for offset in neighborhood:
        current_offset = np.linalg.norm(offset)
        if current_offset <= 1:
            shift.append(adjacent_edge_bias)
        else:
            if type(lr_bias) is list:
                if current_offset > previous_offset:
                    bias_idx += 1
                shift.append(lr_bias[bias_idx])
            else:
                shift.append(current_offset * lr_bias)
        previous_offset = current_offset
    shift = np.array(shift).reshape((-1, *((1,) * (len(affinities.shape) - 1))))

    # filter fragments
    segmentation = mws.agglom(
        affinities + shift + random_noise + smoothed_affs,
        offsets=neighborhood,
    )

    if filter_val > 0.0:
        filter_fragments(affinities, segmentation, filter_val)
    # fragment_ids = fastremap.unique(segmentation[segmentation > 0])
    # fastremap.mask_except(segmentation, filtered_fragments, in_place=True)
    fastremap.renumber(segmentation, in_place=True)
    return segmentation


def process_mutex_watershed(
    affinities,
    neighborhood=np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [9, 0, 0],
            [0, 9, 0],
            [0, 0, 9],
        ]
    ),
    adjacent_edge_bias=-0.4,
    lr_bias=[-0.5, -0.5],
    filter_val=0.5,
):
    neighborhood = neighborhood[: affinities.shape[0]]
    segmentation = mutex_watershed(
        affinities=affinities,
        neighborhood=neighborhood,
        adjacent_edge_bias=adjacent_edge_bias,
        lr_bias=lr_bias,
        filter_val=filter_val,
    )
    return segmentation


# import zarr

# # read  "/groups/cellmap/cellmap/ackermand/Programming/dinov3_playground/examples/zarrs/jrc_mus-liver-zon-2/results_dinov3_finetune_3Dunet_cell_affinities_1_to_9_dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/output/s0" that has 9 channels and 3 spatial dimensions
# z = zarr.open(
#     "/groups/cellmap/cellmap/ackermand/Programming/dinov3_playground/examples/zarrs/jrc_mus-liver-zon-2/results_dinov3_finetune_3Dunet_cell_affinities_1_to_9_dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/output/s0",
#     mode="r",
# )
# affinities = z[:]
# # transpose last 3 channels
# affinities = affinities
# segs = process_mutex_watershed(affinities, adjacent_edge_bias=-0.5, filter_val=0.6)


# # %%
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

# unique_ids = np.unique(segs)
# unique_ids = unique_ids[unique_ids]  # optional: exclude background (0)

# # Assign a random color to each instance
# colors = np.random.rand(len(unique_ids), 3)
# colors = np.vstack(([0, 0, 0], colors))  # prepend black for background
# cmap = ListedColormap(colors)

# # Normalize the mask so IDs map to color indices
# normalized_mask = np.zeros_like(segs)
# for i, uid in enumerate(unique_ids):
#     normalized_mask[segs == uid] = i

# for slice in [0, 63, 127]:
#     plt.figure(figsize=(6, 6))
#     plt.imshow(normalized_mask[slice], cmap=cmap)
#     plt.title("Instance Segmentation (Random Colors)")
#     plt.axis("off")
#     plt.show()
