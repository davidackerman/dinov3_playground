# from dacapo_toolbox vis
# %%
import matplotlib.pyplot as plt
from funlib.geometry import Coordinate
from matplotlib import animation
from matplotlib.colors import ListedColormap
import numpy as np
from funlib.persistence import Array
from sklearn.decomposition import PCA

from pathlib import Path


from matplotlib import colors as mcolors
from matplotlib import cm

SKIP_PLOTS = True


def pca_nd(emb: Array, n_components: int = 3) -> Array:
    emb_data = emb[:]
    num_channels, *spatial_shape = emb_data.shape

    emb_data = emb_data - emb_data.mean(
        axis=tuple(range(1, len(emb_data.shape))), keepdims=True
    )  # center the data
    emb_data /= emb_data.std(
        axis=tuple(range(1, len(emb_data.shape))), keepdims=True
    )  # normalize the data

    emb_data = emb_data.reshape(num_channels, -1)  # flatten the spatial dimensions
    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(emb_data.T)
    principal_components = principal_components.T.reshape(n_components, *spatial_shape)

    principal_components -= principal_components.min(
        axis=tuple(range(1, n_components + 1)), keepdims=True
    )
    principal_components /= principal_components.max(
        axis=tuple(range(1, n_components + 1)), keepdims=True
    )
    return Array(
        principal_components,
        voxel_size=emb.voxel_size,
        offset=emb.offset,
        units=emb.units,
        axis_names=emb.axis_names,
        types=emb.types,
    )


def get_cmap(seed: int = 47) -> ListedColormap:
    np.random.seed(seed)
    colors = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]] + [
        list(np.random.choice(range(256), size=3) / 255.0) for _ in range(255)
    ]
    return ListedColormap(colors)


def gif_2d(
    arrays: dict[str, Array],
    array_types: dict[str, str],
    filename: str,
    title: str,
    fps: int = 10,
    overwrite: bool = False,
):
    """
    Create a 2D animated GIF from 3D arrays.

    Parameters:
    -----------
    arrays : dict[str, Array]
        Dictionary of arrays to visualize. For "combined" type, the value should be
        a tuple/list of (raw_array, labels_array).
    array_types : dict[str, str]
        Dictionary specifying the type of each array. Supported types:
        - "raw": Raw data displayed in grayscale
        - "labels": Label data with random colormap
        - "pca": PCA-transformed data
        - "affs": Affinity data
        - "combined": Overlay labels on raw data with transparency (requires tuple of arrays)
    filename : str
        Output filename for the GIF
    title : str
        Title for the visualization
    fps : int, default=10
        Frames per second for the animation
    overwrite : bool, default=False
        Whether to overwrite existing files
    """
    if Path(filename).exists() and not overwrite:
        return
    transformed_arrays = {}
    for key, arr in arrays.items():
        if array_types[key] == "combined":
            # For combined type, expect arr to be a tuple/list of (raw_array, labels_array)
            raw_arr, labels_arr = arr
            assert (
                raw_arr.voxel_size.dims == 3
            ), f"Raw array for {key} must be 3D, got {raw_arr.voxel_size.dims}D"
            assert (
                labels_arr.voxel_size.dims == 3
            ), f"Labels array for {key} must be 3D, got {labels_arr.voxel_size.dims}D"
            transformed_arrays[key] = (raw_arr, labels_arr)
        else:
            assert (
                arr.voxel_size.dims == 3
            ), f"Array {key} must be 3D, got {arr.voxel_size.dims}D"
            if array_types[key] == "pca":
                transformed_arrays[key] = pca_nd(arr)
            else:
                transformed_arrays[key] = arr
    arrays = transformed_arrays

    z_slices = None
    for key, arr in arrays.items():
        if array_types[key] == "combined":
            # For combined type, use the raw array (first element) for z_slices calculation
            raw_arr, _ = arr
            arr_z_slices = raw_arr.roi.shape[0] // raw_arr.voxel_size[0]
        else:
            arr_z_slices = arr.roi.shape[0] // arr.voxel_size[0]

        if z_slices is None:
            z_slices = arr_z_slices
        else:
            assert z_slices == arr_z_slices, (
                f"All arrays must have the same number of z slices, "
                f"got {z_slices} and {arr_z_slices}"
            )

    fig, axes = plt.subplots(1, len(arrays), figsize=(2 + 5 * len(arrays), 6))

    label_cmap = get_cmap()

    ims = []
    for ii in range(z_slices):
        slice_ims = []
        for jj, (key, arr) in enumerate(arrays.items()):
            if array_types[key] == "labels":
                roi = arr.roi.copy()
                roi.offset += Coordinate((ii,) + (0,) * (roi.dims - 1)) * arr.voxel_size
                roi.shape = Coordinate((arr.voxel_size[0], *roi.shape[1:]))
                # Show the raw data
                x = arr[roi].squeeze(-arr.voxel_size.dims)  # squeeze out z dim
                shape = x.shape
                scale_factor = shape[-2] // 256 if shape[-2] > 256 else 1
                # only show 256x256 pixels, more resolution not needed for gif
                if len(shape) == 2:
                    x = x[::scale_factor, ::scale_factor]
                elif len(shape) == 3:
                    x = x[:, ::scale_factor, ::scale_factor]
                else:
                    raise ValueError("Array must be 2D with or without channels")

                im = axes[jj].imshow(
                    x % 256,
                    vmin=0,
                    vmax=255,
                    cmap=label_cmap,
                    interpolation="none",
                    animated=ii != 0,
                )
            elif array_types[key] == "combined":
                # Expected format: arrays[key] should be a tuple/list of (raw_array, labels_array)
                raw_arr, labels_arr = arrays[key]

                # Get the raw and labels slices for this z position
                raw_roi = raw_arr.roi.copy()
                raw_roi.offset += (
                    Coordinate((ii,) + (0,) * (raw_roi.dims - 1)) * raw_arr.voxel_size
                )
                raw_roi.shape = Coordinate((raw_arr.voxel_size[0], *raw_roi.shape[1:]))
                raw_x = raw_arr[raw_roi].squeeze(-raw_arr.voxel_size.dims)

                labels_roi = labels_arr.roi.copy()
                labels_roi.offset += (
                    Coordinate((ii,) + (0,) * (labels_roi.dims - 1))
                    * labels_arr.voxel_size
                )
                labels_roi.shape = Coordinate(
                    (labels_arr.voxel_size[0], *labels_roi.shape[1:])
                )
                labels_x = labels_arr[labels_roi].squeeze(-labels_arr.voxel_size.dims)

                # Apply same scaling as done above
                shape = raw_x.shape
                scale_factor = shape[-2] // 256 if shape[-2] > 256 else 1
                if len(shape) == 2:
                    raw_x = raw_x[::scale_factor, ::scale_factor]
                    labels_x = labels_x[::scale_factor, ::scale_factor]
                elif len(shape) == 3:
                    raw_x = raw_x[:, ::scale_factor, ::scale_factor]
                    labels_x = labels_x[:, ::scale_factor, ::scale_factor]

                # Normalize raw data for display
                if raw_x.ndim == 2:
                    # Normalize and convert to RGB (not RGBA)
                    normalized = (raw_x - raw_x.min()) / (
                        raw_x.max() - raw_x.min() + 1e-8
                    )
                    raw_display = np.stack([normalized] * 3, axis=-1)  # Convert to RGB
                elif raw_x.ndim == 3:
                    raw_display = raw_x.transpose(1, 2, 0)
                    if raw_display.shape[2] == 1:
                        raw_display = np.repeat(raw_display, 3, axis=2)
                    raw_display = (raw_display - raw_display.min()) / (
                        raw_display.max() - raw_display.min() + 1e-8
                    )

                # Create labels overlay with transparency
                labels_colored = label_cmap((labels_x % 256) / 255.0)
                # Extract only RGB channels (drop alpha channel)
                labels_rgb = labels_colored[..., :3]

                # Create alpha mask (transparent where labels are 0/background)
                alpha_mask = (labels_x > 0).astype(
                    float
                ) * 0.4  # 40% opacity for labels

                # Ensure raw_display is RGB (3 channels)
                if raw_display.ndim == 3:
                    if raw_display.shape[2] == 4:  # RGBA -> RGB
                        raw_rgb = raw_display[..., :3]
                    elif raw_display.shape[2] == 3:  # Already RGB
                        raw_rgb = raw_display
                    elif raw_display.shape[2] == 1:  # Single channel -> RGB
                        raw_rgb = np.repeat(raw_display, 3, axis=2)
                    else:
                        raw_rgb = raw_display  # Hope for the best
                else:
                    # 2D grayscale -> RGB
                    raw_rgb = np.stack([raw_display] * 3, axis=-1)

                # Combine raw and labels
                combined = raw_rgb.copy()
                mask = alpha_mask > 0
                if mask.any():
                    combined[mask] = (1 - alpha_mask[mask][..., np.newaxis]) * raw_rgb[
                        mask
                    ] + alpha_mask[mask][..., np.newaxis] * labels_rgb[mask]

                im = axes[jj].imshow(
                    combined,
                    animated=ii != 0,
                )
            elif array_types[key] == "raw" or array_types[key] == "pca":
                roi = arr.roi.copy()
                roi.offset += Coordinate((ii,) + (0,) * (roi.dims - 1)) * arr.voxel_size
                roi.shape = Coordinate((arr.voxel_size[0], *roi.shape[1:]))
                # Show the raw data
                x = arr[roi].squeeze(-arr.voxel_size.dims)  # squeeze out z dim
                shape = x.shape
                scale_factor = shape[-2] // 256 if shape[-2] > 256 else 1
                # only show 256x256 pixels, more resolution not needed for gif
                if len(shape) == 2:
                    x = x[::scale_factor, ::scale_factor]
                elif len(shape) == 3:
                    x = x[:, ::scale_factor, ::scale_factor]
                else:
                    raise ValueError("Array must be 2D with or without channels")

                if x.ndim == 2:
                    im = axes[jj].imshow(
                        x,
                        cmap="grey",
                        animated=ii != 0,
                    )
                elif x.ndim == 3:
                    im = axes[jj].imshow(
                        x.transpose(1, 2, 0),
                        animated=ii != 0,
                    )
            elif array_types[key] == "affs":
                roi = arr.roi.copy()
                roi.offset += Coordinate((ii,) + (0,) * (roi.dims - 1)) * arr.voxel_size
                roi.shape = Coordinate((arr.voxel_size[0], *roi.shape[1:]))
                # Show the raw data
                x = arr[roi].squeeze(-arr.voxel_size.dims)  # squeeze out z dim
                shape = x.shape
                scale_factor = shape[-2] // 256 if shape[-2] > 256 else 1
                # only show 256x256 pixels, more resolution not needed for gif
                if len(shape) == 2:
                    x = x[::scale_factor, ::scale_factor]
                elif len(shape) == 3:
                    x = x[:, ::scale_factor, ::scale_factor]
                else:
                    raise ValueError("Array must be 2D with or without channels")

                # Show the affinities
                im = axes[jj].imshow(
                    x.transpose(1, 2, 0),
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="none",
                    animated=ii != 0,
                )
            axes[jj].set_title(key)
            slice_ims.append(im)
        ims.append(slice_ims)

    ims = ims + ims[::-1]
    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    fig.suptitle(title, fontsize=16)
    ani.save(filename, writer="pillow", fps=fps)
    plt.close()


def cube(
    arrays: dict[str, Array],
    array_types: dict[str, str],
    filename: str,
    title: str,
    elev: float = 30,
    azim: float = -60,
    light_azdeg: float = 205,
    light_altdeg: float = 20,
    overwrite: bool = False,
):
    """
    Create a 3D cube visualization from 3D arrays.

    Parameters:
    -----------
    arrays : dict[str, Array]
        Dictionary of arrays to visualize. For "combined" type, the value should be
        a tuple/list of (raw_array, labels_array).
    array_types : dict[str, str]
        Dictionary specifying the type of each array. Supported types:
        - "raw": Raw data displayed in grayscale
        - "labels": Label data with random colormap
        - "pca": PCA-transformed data
        - "affs": Affinity data
        - "combined": Overlay labels on raw data with transparency (requires tuple of arrays)
    filename : str
        Output filename for the image
    title : str
        Title for the visualization
    elev : float, default=30
        Elevation angle for 3D view
    azim : float, default=-60
        Azimuth angle for 3D view
    light_azdeg : float, default=205
        Light source azimuth angle
    light_altdeg : float, default=20
        Light source altitude angle
    overwrite : bool, default=False
        Whether to overwrite existing files
    """
    if Path(filename).exists() and not overwrite:
        return

    lightsource = mcolors.LightSource(azdeg=light_azdeg, altdeg=light_altdeg)

    transformed_arrays = {}
    for key, arr in arrays.items():
        if array_types[key] == "combined":
            # For combined type, expect arr to be a tuple/list of (raw_array, labels_array)
            raw_arr, labels_arr = arr
            assert (
                raw_arr.voxel_size.dims == 3
            ), f"Raw array for {key} must be 3D, got {raw_arr.voxel_size.dims}D"
            assert (
                labels_arr.voxel_size.dims == 3
            ), f"Labels array for {key} must be 3D, got {labels_arr.voxel_size.dims}D"

            # Normalize raw data
            raw_normalized = Array(
                (raw_arr.data - raw_arr.data.min())
                / (raw_arr.data.max() - raw_arr.data.min()),
                voxel_size=raw_arr.voxel_size,
                offset=raw_arr.offset,
                units=raw_arr.units,
                axis_names=raw_arr.axis_names,
                types=raw_arr.types,
            )

            # Normalize labels data
            labels_normalized = Array(
                labels_arr.data % 256 / 255.0,
                voxel_size=labels_arr.voxel_size,
                offset=labels_arr.offset,
                units=labels_arr.units,
                axis_names=labels_arr.axis_names,
                types=labels_arr.types,
            )

            transformed_arrays[key] = (raw_normalized, labels_normalized)
        else:
            assert (
                arr.voxel_size.dims == 3
            ), f"Array {key} must be 3D, got {arr.voxel_size.dims}D"
            if array_types[key] == "pca":
                transformed_arrays[key] = pca_nd(arr)
            elif array_types[key] == "labels":
                normalized = Array(
                    arr.data % 256 / 255.0,
                    voxel_size=arr.voxel_size,
                    offset=arr.offset,
                    units=arr.units,
                    axis_names=arr.axis_names,
                    types=arr.types,
                )
                transformed_arrays[key] = normalized
            elif array_types[key] == "raw":
                normalized = Array(
                    (arr.data - arr.data.min()) / (arr.data.max() - arr.data.min()),
                    voxel_size=arr.voxel_size,
                    offset=arr.offset,
                    units=arr.units,
                    axis_names=arr.axis_names,
                    types=arr.types,
                )
                transformed_arrays[key] = normalized
            else:
                transformed_arrays[key] = arr
    arrays = transformed_arrays

    fig, axes = plt.subplots(
        1,
        len(arrays),
        figsize=(2 + 5 * len(arrays), 6),
        subplot_kw={"projection": "3d"},
    )

    label_cmap = get_cmap()

    def draw_cube(ax, arr: Array, cmap=None, interpolation=None):
        assert (
            arr.voxel_size.dims == 3
        ), f"Array {arr.name} must be 3D, got {arr.voxel_size.dims}D"
        kwargs = {
            "interpolation": interpolation,
            "cmap": cmap,
        }

        z, y, x = tuple(
            np.linspace(start, stop, count)
            for start, stop, count in zip(
                arr.roi.begin, arr.roi.end, arr.roi.shape // arr.voxel_size
            )
        )
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

        face_colors = (
            cmap(arr[arr.roi])
            if cmap is not None
            else arr[arr.roi].transpose(1, 2, 3, 0)
        )

        kwargs = {
            "rcount": 256,
            "ccount": 256,
            "shade": True,
            "lightsource": lightsource,
        }

        _lz, ly, _lx = np.s_[0, :, :], np.s_[:, 0, :], np.s_[:, :, 0]
        uz, _uy, ux = np.s_[-1, :, :], np.s_[:, -1, :], np.s_[:, :, -1]
        # ax.plot_surface(xx[lx], yy[lx], zz[lx], facecolors=face_colors[lx], **kwargs)
        ax.plot_surface(xx[ux], yy[ux], zz[ux], facecolors=face_colors[ux], **kwargs)
        ax.plot_surface(xx[ly], yy[ly], zz[ly], facecolors=face_colors[ly], **kwargs)
        # ax.plot_surface(xx[uy], yy[uy], zz[uy], facecolors=face_colors[uy], **kwargs)
        # ax.plot_surface(xx[lz], yy[lz], zz[lz], facecolors=face_colors[lz], **kwargs)
        ax.plot_surface(xx[uz], yy[uz], zz[uz], facecolors=face_colors[uz], **kwargs)

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_zlim(z[0], z[-1])
        ax.set_box_aspect(arr.roi.shape[::-1])

        ax.axis("off")

    for jj, (key, arr) in enumerate(arrays.items()):
        ax = axes[jj]

        if array_types[key] == "combined":
            # For combined type, create overlay of raw and labels
            raw_arr, labels_arr = arr

            # Create combined visualization by blending raw and labels
            raw_data = raw_arr.data
            labels_data = labels_arr.data

            # Convert raw to RGB if it's grayscale
            if raw_data.ndim == 3:
                raw_rgb = np.stack([raw_data] * 3, axis=-1)
            else:
                raw_rgb = raw_data

            # Get colored labels
            labels_colored = label_cmap(labels_data)[..., :3]  # Remove alpha channel

            # Create alpha mask (transparent where labels are 0/background)
            alpha_mask = (labels_data > 0).astype(float) * 0.6  # 60% opacity

            # Blend raw and labels
            combined_data = raw_rgb.copy()
            mask = alpha_mask > 0
            if mask.any():
                combined_data[mask] = (1 - alpha_mask[mask][..., np.newaxis]) * raw_rgb[
                    mask
                ] + alpha_mask[mask][..., np.newaxis] * labels_colored[mask]

            # Create combined array for drawing
            combined_arr = Array(
                combined_data,
                voxel_size=raw_arr.voxel_size,
                offset=raw_arr.offset,
                units=raw_arr.units,
                axis_names=raw_arr.axis_names,
                types=raw_arr.types,
            )

            draw_cube(ax, combined_arr)
        elif array_types[key] == "labels":
            draw_cube(ax, arr, cmap=label_cmap, interpolation="none")
        elif array_types[key] == "raw" or array_types[key] == "pca":
            if arr.data.ndim == 3:
                draw_cube(ax, arr, cmap=cm.gray)
            elif arr.data.ndim == 4:
                draw_cube(ax, arr)
        elif array_types[key] == "affs":
            # Show the affinities
            draw_cube(ax, arr, interpolation="none")

        ax.set_title(key)
        # Without this line, the default cube view is elev = 30, azim = -60.
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# %%
get_cmap()
# %%
