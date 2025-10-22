# %%
from dinov3_playground.inference import load_inference_model
from cellmap_flow.image_data_interface import ImageDataInterface
from funlib.geometry import Roi
import numpy as np
from mws import process_mutex_watershed

# Define the source dataset and ROI
# These will be adjusted based on model config after loading
# dataset_path = "/groups/funceworm/funceworm/adult/Adult_Day1_DatasetA4/jrc_P3_E5_D1_N2_trimmed_align_v2.zarr/s2"
# dataset_path = "/nrs/cellmap/data/jrc_celegans_bw25113/jrc_celegans_bw25113.zarr/recon-1/em/fibsem-uint16/s0"
# roi_offset = (24326, 31571, 108576)
# roi_size_voxels = 512  # Size in voxels at the input resolution
for organelle in ["cell"]:  # "mito", "cell"]:
    if organelle == "mito":
        dataset_path = "/nrs/cellmap/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/em/fibsem-int16/s0"
        path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_mito_affinities_lsds_sigma_5_random_batchrenorm_updated_loss/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"
        roi_offset = (163, 18036, 17492)
    if organelle == "cell":
        path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_cell_affinities_lsds_larger_random_batchrenorm_updated_loss/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"
        dataset_path = "/nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/s2"
        roi_offset = (32826, 39186, 61592)

    roi_size_voxels = 512  # Size in voxels at the input resolution
    # Extract dataset name from path for file naming
    dataset_name = None
    for part in dataset_path.split("/"):
        if part.startswith("jrc_"):
            dataset_name = part.replace(".zarr", "")
            break
    if dataset_name is None:
        dataset_name = "unknown_dataset"

    # Note: input_voxel_size, context volumes, and actual ROI will be determined
    # automatically from the model configuration after loading

    # %%
    # Load the model first to get configuration
    # You can now provide either:
    # 1. Full path with timestamp: .../run_20251003_110551
    # 2. Parent path (auto-selects most recent): .../dinov3_unet3d_dinov3_vitl16_pretrain_sat493m
    # path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_mito_affinities_lsds_sigma_5_2nm_smaller_model/dinov3_unet3d_dinov3_vits16_pretrain_lvd1689m/run_20251011_152323"
    # path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_cell_affinities_lsds_larger_random_batchrenorm_updated_loss/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"
    # path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_cell_affinities_lsds_larger_random_batchrenorm/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"
    # path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_mito_affinities_1_to_9_random_batchrenorm/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"
    # path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_mito_affinities_lsds_sigma_5/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"
    # path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_mito_affinities_lsds_sigma_5/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/"
    # path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_cell_affinities_lsds_larger_only_full/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"  # /nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_cell_affinities/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"
    inference = load_inference_model(path, checkpoint_preference="best")
    output_name = "_".join(path.split("/")[-3:])

    # Extract model configuration
    model_config = inference.model_config
    training_config = (
        inference.training_config
    )  # Access training_config directly as an attribute

    # Get key parameters from model config
    uses_context = model_config.get("use_context_fusion", False)
    output_type = model_config.get("output_type", "labels")
    affinity_offsets = model_config.get("affinity_offsets", None)
    use_orthogonal_planes = model_config.get("use_orthogonal_planes", False)

    # Get resolution parameters
    min_resolution_for_raw = model_config.get("min_resolution_for_raw", 4)
    base_resolution = model_config.get("base_resolution", 16)
    context_scale = training_config.get("context_scale", None)

    print(f"\n{'='*60}")
    print(f"MODEL CONFIGURATION:")
    print(f"{'='*60}")
    print(f"Model uses context fusion: {uses_context}")
    if uses_context and context_scale:
        print(f"  Context resolution: {context_scale}nm")
    print(f"Model uses orthogonal planes: {use_orthogonal_planes}")
    print(f"Output type: {output_type}")
    if output_type == "affinities":
        num_offsets = len(affinity_offsets) if affinity_offsets else 0
        print(f"Number of affinity offsets: {num_offsets}")
        if affinity_offsets:
            for i, offset in enumerate(affinity_offsets):
                print(f"  Channel {i}: {offset}")
    print(f"\nRESOLUTION CONFIGURATION:")
    print(f"  Input (raw) resolution:  {min_resolution_for_raw}nm")
    print(f"  Output (GT) resolution:  {base_resolution}nm")
    if uses_context and context_scale:
        print(f"  Context resolution:      {context_scale}nm")
    print(f"{'='*60}\n")

    # Now load data with correct resolutions
    print(f"LOADING DATA:")
    print(f"{'='*60}")

    # Load main volume at input resolution
    input_voxel_size = min_resolution_for_raw
    roi = Roi(
        roi_offset,
        [roi_size_voxels * input_voxel_size] * 3,
    )

    print(f"Loading main volume:")
    print(f"  Dataset: {dataset_path}")
    print(f"  ROI offset: {roi_offset}")
    print(f"  ROI size: {roi.get_shape()}")
    print(f"  Voxel size: {input_voxel_size}nm isotropic")

    volume = ImageDataInterface(
        dataset_path,
        output_voxel_size=3 * [input_voxel_size],
    ).to_ndarray_ts(roi)

    print(f"  Loaded shape: {volume.shape}")

    # Load context volume if model expects it
    context_volume = None
    if uses_context and context_scale:
        # Context volume covers the same spatial region but at lower resolution
        # Calculate the ROI for context data
        context_voxel_size = context_scale
        context_roi = Roi(
            roi_offset,  # Same offset as main volume
            [roi_size_voxels * input_voxel_size] * 3,  # Same spatial extent
        )

        print(f"\nLoading context volume (model uses context fusion):")
        print(f"  Dataset: {dataset_path}")
        print(f"  ROI offset: {roi_offset}")
        print(f"  ROI size: {context_roi.get_shape()}")
        print(f"  Voxel size: {context_voxel_size}nm isotropic")
        print(
            f"  Resolution ratio: {context_voxel_size / input_voxel_size}x lower resolution"
        )

        context_volume = ImageDataInterface(
            dataset_path,
            output_voxel_size=3 * [context_voxel_size],
        ).to_ndarray_ts(context_roi)

        print(f"  Loaded shape: {context_volume.shape}")
        print(
            f"  Spatial coverage: same as main volume but {context_voxel_size / input_voxel_size}x fewer voxels per dimension"
        )
    elif uses_context:
        print(
            f"\n⚠️  WARNING: Model expects context volume but context_scale not found in config!"
        )
        print("   Proceeding without context - results may be suboptimal.")

    print(f"{'='*60}\n")

    # Run inference
    print(f"RUNNING INFERENCE:")
    print(f"{'='*60}")
    if uses_context and context_volume is not None:
        print(f"Running with context fusion...")
        prediction = inference.predict(volume, context_volume=context_volume)
    else:
        print(f"Running without context...")
        prediction = inference.predict(volume)
    print(f"{'='*60}\n")

    # output_type = "labels"
    # prediction = np.squeeze(prediction[0] > 0.5)
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction dtype: {prediction.dtype}")
    # If model predicts both LSDS + affinities (output_type == 'affinities_lsds'),
    # the convention is that the first 10 channels are LSDs and the remaining
    # channels are affinities. Split those here and ensure downstream processing
    # only uses the affinity channels.
    lsds = None
    affinities = None
    if output_type == "affinities_lsds":
        if prediction.shape[0] <= 10:
            raise RuntimeError(
                f"Expected >10 channels for affinities_lsds but got {prediction.shape[0]}"
            )
        lsds = prediction[:10]
        affinities = prediction[10:]
        print(
            f"Split prediction into lsds (shape={lsds.shape}) + affinities (shape={affinities.shape})"
        )

    if output_type in ("affinities", "affinities_lsds"):
        # choose the correct affinities array for processing
        use_affinities = affinities if affinities is not None else prediction

        print(
            f"Prediction range: [{use_affinities.min():.3f}, {use_affinities.max():.3f}]"
        )

        # --- CROPPING: remove 9 voxels from each side of spatial dims ---
        crop_margin = 9  # voxels to remove at each edge
        # Ensure channel-first for cropping
        if use_affinities.ndim == 3:
            use_affinities = use_affinities[np.newaxis, ...]  # (C, Z, Y, X)

        if any(s <= 2 * crop_margin for s in use_affinities.shape[1:]):
            raise RuntimeError(
                f"Cannot crop affinities by {crop_margin} on each side: spatial dims too small: {use_affinities.shape[1:]}"
            )

        # Crop spatial dimensions (assumes channel-first)
        use_affinities_cropped = use_affinities[
            :,
            crop_margin:-crop_margin,
            crop_margin:-crop_margin,
            crop_margin:-crop_margin,
        ]

        # If lsds present, crop them similarly
        if lsds is not None:
            if lsds.ndim == 3:
                lsds = lsds[np.newaxis, ...]
            lsds = lsds[
                :,
                crop_margin:-crop_margin,
                crop_margin:-crop_margin,
                crop_margin:-crop_margin,
            ]

        # Compute new output offset (nm) for cropped arrays
        # roi_offset is in nanometers in this script; add crop_margin * base_resolution
        cropped_offset_nm = [
            float(roi_offset[0] + crop_margin * base_resolution),
            float(roi_offset[1] + crop_margin * base_resolution),
            float(roi_offset[2] + crop_margin * base_resolution),
        ]

        print(
            f"\nProcessing affinities with mutex watershed (cropped by {crop_margin} voxels per side)..."
        )
        bias = -0.5
        segmentation = process_mutex_watershed(
            affinities=use_affinities_cropped,
            neighborhood=(
                np.array(affinity_offsets)
                if affinity_offsets
                else np.array(
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
                )
            ),
            adjacent_edge_bias=bias,
            lr_bias=[-0.5, -0.5],
            filter_val=0.5,
        )

        # replace affinities variable with cropped version for downstream saving/visualization
        affinities = use_affinities_cropped

        print(f"Segmentation shape: {segmentation.shape}")
        print(f"Segmentation dtype: {segmentation.dtype}")
        print(f"Number of segments: {len(np.unique(segmentation))}")
    else:
        print(f"Unique prediction values: {np.unique(prediction)}")

    # Determine prediction spatial shape and channel counts for metadata
    prediction_spatial_shape = prediction.shape[-3:]
    output_channels = None
    lsds_channels = None
    if output_type in ("affinities", "affinities_lsds"):
        if affinities is not None:
            prediction_spatial_shape = affinities.shape[1:]
            output_channels = int(affinities.shape[0])
        else:
            # prediction may already be channel-first
            if prediction.ndim >= 3:
                output_channels = int(prediction.shape[0])
                prediction_spatial_shape = prediction.shape[-3:]
    if lsds is not None:
        lsds_channels = int(lsds.shape[0])

    # %%
    # Optional: Export raw and prediction to zarr using tensorstore
    EXPORT_TO_ZARR = True  # Set to True to enable zarr export

    if EXPORT_TO_ZARR:
        import tensorstore as ts
        import os
        import json

        # Helper function to convert numpy dtype to zarr dtype string
        def numpy_to_zarr_dtype(np_dtype):
            """Convert numpy dtype to zarr-compatible dtype string."""
            # Handle both numpy dtype objects and dtype strings
            if hasattr(np_dtype, "name"):
                dtype_str = np_dtype.name
            else:
                dtype_str = str(np_dtype)

            # Map common numpy dtypes to zarr format
            dtype_map = {
                "uint8": "|u1",
                "uint16": "<u2",
                "uint32": "<u4",
                "uint64": "<u8",
                "int8": "|i1",
                "int16": "<i2",
                "int32": "<i4",
                "int64": "<i8",
                "float32": "<f4",
                "float64": "<f4",  # Use float32 for float64 (TensorStore doesn't support <f8)
            }
            return dtype_map.get(dtype_str, "<f4")  # Default to float32 if unknown

        # Get resolution information from model config
        base_resolution = inference.model_config.get(
            "base_resolution", 16
        )  # Output resolution
        min_resolution_for_raw = inference.model_config.get(
            "min_resolution_for_raw", 4
        )  # Input resolution

        # Create directory structure: ./zarrs/{dataset_name}/{model_name}/{input,output}
        dir_name = f"{dataset_name}/{output_name.replace('/', '_')}"
        zarr_base_dir = os.path.join("zarrs", dir_name)

        # Paths for input and output (base directories for multiscale zarrs)
        input_zarr_base = os.path.join(zarr_base_dir, "input")
        output_zarr_base = os.path.join(zarr_base_dir, "output")

        # Actual data will be saved in s0 subdirectories
        input_zarr_path = os.path.join(input_zarr_base, "s0")
        output_zarr_path = os.path.join(output_zarr_base, "s0")

        # Create directories
        os.makedirs(input_zarr_path, exist_ok=True)
        os.makedirs(output_zarr_path, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Exporting to zarr:")
        print(f"  Input:  {input_zarr_path}")
        print(f"  Output: {output_zarr_path}")
        print(f"  Input resolution:  {min_resolution_for_raw}nm")
        print(f"  Output resolution: {base_resolution}nm")
        print(f"{'='*60}")

        # Export raw volume (input)
        print("Writing raw volume (input)...")
        raw_spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": input_zarr_path,
            },
            "metadata": {
                "shape": volume.shape,
                "chunks": [min(128, s) for s in volume.shape],
                "dtype": numpy_to_zarr_dtype(volume.dtype),
                "compressor": {
                    "id": "blosc",
                    "cname": "zstd",
                    "clevel": 3,
                    "shuffle": 1,
                },
                "dimension_separator": "/",
            },
            "create": True,
            "delete_existing": True,
        }

        raw_dataset = ts.open(raw_spec).result()
        raw_dataset[:].write(volume).result()

        # Add .zgroup file to base directory (required for zarr group)
        raw_zgroup = {"zarr_format": 2}
        raw_zgroup_path = os.path.join(input_zarr_base, ".zgroup")
        with open(raw_zgroup_path, "w") as f:
            json.dump(raw_zgroup, f, indent=2)

        # Add neuroglancer-compatible metadata to raw (OME-NGFF multiscales format)
        # Write to the base directory (.zattrs goes in input/, not input/s0/)
        # Build raw_attrs axes dynamically: if volume has a leading channel axis include it
        raw_axes = []
        raw_scale = []
        if volume.ndim == 4:
            # channel-first expected: (C, Z, Y, X) or (C, D, H, W)
            raw_axes.append({"name": "c", "type": "channel"})
            raw_scale.append(1.0)
            # spatial dims follow
            raw_axes += [
                {"name": "z", "type": "space", "unit": "nanometer"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "x", "type": "space", "unit": "nanometer"},
            ]
            raw_scale += [
                float(min_resolution_for_raw),
                float(min_resolution_for_raw),
                float(min_resolution_for_raw),
            ]
        else:
            # standard (Z,Y,X)
            raw_axes = [
                {"name": "z", "type": "space", "unit": "nanometer"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "x", "type": "space", "unit": "nanometer"},
            ]
            raw_scale = [
                float(min_resolution_for_raw),
                float(min_resolution_for_raw),
                float(min_resolution_for_raw),
            ]

        raw_attrs = {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": "",
                    "axes": raw_axes,
                    "datasets": [
                        {
                            "path": "s0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": raw_scale},
                                {
                                    "type": "translation",
                                    "translation": [
                                        float(roi_offset[0]),
                                        float(roi_offset[1]),
                                        float(roi_offset[2]),
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        }
        raw_attrs_path = os.path.join(input_zarr_base, ".zattrs")
        with open(raw_attrs_path, "w") as f:
            json.dump(raw_attrs, f, indent=2)

        print(f"  ✓ Raw volume saved: shape={volume.shape}, dtype={volume.dtype}")
        print(f"    Resolution: {min_resolution_for_raw}nm isotropic")
        print(f"    Offset: {roi_offset} nm")
        # Export prediction (output)
        print("Writing prediction (output)...")

    def write_zarr_group(
        group_base,
        name,
        array,
        axes,
        scale_list,
        translation_list,
        dtype_str=None,
        compressor=None,
        chunks=None,
        extra_attrs=None,
    ):
        """Write a single-level multiscale zarr group with dataset 's0' and .zattrs.

        Args:
            group_base: base output directory (e.g., output_zarr_base)
            name: subfolder name for this dataset (e.g., 'affs', 'lsds', 'seg')
            array: numpy array to write (channel-first or spatial)
            axes: list of axis dicts for .zattrs (ordered)
            scale_list: list of scales matching axes (channel=1.0 if present)
            translation_list: list of translations matching axes (channel=0.0 if present)
            dtype_str: zarr dtype string (if None will infer from numpy)
            compressor: compressor dict for .zattrs (optional)
            chunks: chunks list for dataset (optional)
        """
        group_dir = os.path.join(group_base, name)
        os.makedirs(group_dir, exist_ok=True)
        s0_path = os.path.join(group_dir, "s0")

        # infer dtype string
        if dtype_str is None:
            dtype_str = numpy_to_zarr_dtype(array.dtype)

        # build chunks default
        if chunks is None:
            if array.ndim == 4:
                chunks = [array.shape[0]] + [min(128, s) for s in array.shape[1:]]
            else:
                chunks = [min(128, s) for s in array.shape]

        spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": s0_path},
            "metadata": {
                "shape": array.shape,
                "chunks": chunks,
                "dtype": dtype_str,
                "compressor": (
                    compressor
                    if compressor is not None
                    else {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1}
                ),
                "dimension_separator": "/",
            },
            "create": True,
            "delete_existing": True,
        }
        ds = ts.open(spec).result()
        # write preserving integer dtypes for labels; convert floats to float32
        if array.dtype.kind == "f":
            ds[:].write(array.astype(np.float32)).result()
        else:
            ds[:].write(array).result()

        # write zgroup at the top-level output dir
        zgroup = {"zarr_format": 2}
        zgroup_path = os.path.join(group_base, ".zgroup")
        with open(zgroup_path, "w") as f:
            json.dump(zgroup, f, indent=2)

        # write .zattrs for this group
        attrs = {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": "",
                    "axes": axes,
                    "datasets": [
                        {
                            "path": "s0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": scale_list},
                                {
                                    "type": "translation",
                                    "translation": translation_list,
                                },
                            ],
                        }
                    ],
                }
            ]
        }
        # Merge any extra attributes into the top-level attrs
        if extra_attrs:
            for k, v in extra_attrs.items():
                attrs[k] = v
        attrs_path = os.path.join(group_dir, ".zattrs")
        with open(attrs_path, "w") as f:
            json.dump(attrs, f, indent=2)

        print(f"  ✓ Wrote group '{name}': shape={array.shape}, path={group_dir}")

    # affinities and lsds handling
    if output_type in ("affinities", "affinities_lsds"):
        affs_array = affinities if affinities is not None else prediction
        # ensure affs are channel-first (C, Z, Y, X)
        if affs_array.ndim == 3:
            # assume (Z,Y,X) -> add channel dim
            affs_array = affs_array[np.newaxis, ...]

        # write affinities under 'affs'
        affs_axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "nanometer"},
            {"name": "y", "type": "space", "unit": "nanometer"},
            {"name": "x", "type": "space", "unit": "nanometer"},
        ]
        affs_scale = [
            1.0,
            float(base_resolution),
            float(base_resolution),
            float(base_resolution),
        ]
        affs_translation = [
            0.0,
            (
                float(cropped_offset_nm[0])
                if "cropped_offset_nm" in locals()
                else float(roi_offset[0])
            ),
            (
                float(cropped_offset_nm[1])
                if "cropped_offset_nm" in locals()
                else float(roi_offset[1])
            ),
            (
                float(cropped_offset_nm[2])
                if "cropped_offset_nm" in locals()
                else float(roi_offset[2])
            ),
        ]
        write_zarr_group(
            zarr_base_dir,
            "affs",
            affs_array,
            affs_axes,
            affs_scale,
            affs_translation,
            extra_attrs={
                "affinity_offsets": (affinity_offsets if affinity_offsets else [])
            },
        )

        # channel labels for affinities go in metadata.json below
        # write lsds if present
        if lsds is not None:
            lsds_array = lsds
            if lsds_array.ndim == 3:
                lsds_array = lsds_array[np.newaxis, ...]
            lsds_axes = affs_axes
            lsds_scale = [
                1.0,
                float(base_resolution),
                float(base_resolution),
                float(base_resolution),
            ]
            lsds_translation = affs_translation
            write_zarr_group(
                zarr_base_dir,
                "lsds",
                lsds_array,
                lsds_axes,
                lsds_scale,
                lsds_translation,
            )

        # write segmentation
        if "segmentation" in locals():
            max_id = segmentation.max()
            seg_dtype = np.min_scalar_type(max_id)
            segmentation_optimized = segmentation.astype(seg_dtype)
            seg_axes = [
                {"name": "z", "type": "space", "unit": "nanometer"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "x", "type": "space", "unit": "nanometer"},
            ]
            seg_scale = [
                float(base_resolution),
                float(base_resolution),
                float(base_resolution),
            ]
            seg_translation = (
                [
                    float(cropped_offset_nm[0]),
                    float(cropped_offset_nm[1]),
                    float(cropped_offset_nm[2]),
                ]
                if "cropped_offset_nm" in locals()
                else [float(roi_offset[0]), float(roi_offset[1]), float(roi_offset[2])]
            )
            write_zarr_group(
                zarr_base_dir,
                f"segs_{bias}",
                segmentation_optimized,
                seg_axes,
                seg_scale,
                seg_translation,
                dtype_str=numpy_to_zarr_dtype(seg_dtype),
            )
    else:
        # For labels/segmentation
        pred_spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": output_zarr_path,
            },
            "metadata": {
                "shape": prediction.shape,
                "chunks": [min(128, s) for s in prediction.shape],
                "dtype": numpy_to_zarr_dtype(prediction.dtype),
                "compressor": {
                    "id": "blosc",
                    "cname": "zstd",
                    "clevel": 3,
                    "shuffle": 1,
                },
                "dimension_separator": "/",
            },
            "create": True,
            "delete_existing": True,
        }
        pred_dataset = ts.open(pred_spec).result()
        pred_dataset[:].write(prediction).result()

        # Add .zgroup file to base directory (required for zarr group)
        pred_zgroup = {"zarr_format": 2}
        pred_zgroup_path = os.path.join(output_zarr_base, ".zgroup")
        with open(pred_zgroup_path, "w") as f:
            json.dump(pred_zgroup, f, indent=2)

        # Add neuroglancer-compatible metadata for labels (OME-NGFF multiscales format)
        pred_attrs = {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": "",
                    "axes": [
                        {"name": "z", "type": "space", "unit": "nanometer"},
                        {"name": "y", "type": "space", "unit": "nanometer"},
                        {"name": "x", "type": "space", "unit": "nanometer"},
                    ],
                    "datasets": [
                        {
                            "path": "s0",
                            "coordinateTransformations": [
                                {
                                    "type": "scale",
                                    "scale": [
                                        float(base_resolution),
                                        float(base_resolution),
                                        float(base_resolution),
                                    ],
                                },
                                {
                                    "type": "translation",
                                    "translation": [
                                        float(roi_offset[0]),
                                        float(roi_offset[1]),
                                        float(roi_offset[2]),
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        }
        # Write .zattrs to base directory (output/, not output/s0/)
        pred_attrs_path = os.path.join(output_zarr_base, ".zattrs")
        with open(pred_attrs_path, "w") as f:
            json.dump(pred_attrs, f, indent=2)

        # Report what was actually saved. If LSDS were split, show both shapes.
        if lsds is not None:
            print(
                f"  ✓ Affinities saved: shape={affinities.shape}, dtype={affinities.dtype}"
            )
            print(f"  ✓ LSDS saved: shape={lsds.shape}, dtype={lsds.dtype}")
        else:
            print(
                f"  ✓ Prediction saved: shape={prediction.shape}, dtype={prediction.dtype}"
            )
        print(f"    Resolution: {base_resolution}nm isotropic")
        print(f"    Offset: {roi_offset} nm")

    # Save additional metadata as JSON in the base directory
    metadata = {
        "model_path": path,
        "output_type": output_type,
        "uses_context_fusion": uses_context,
        "use_orthogonal_planes": use_orthogonal_planes,
        "raw_shape": list(volume.shape),
        "prediction_shape": list(prediction.shape),
        # Ensure spatial dims are explicitly recorded
        "prediction_spatial_shape": list(prediction_spatial_shape),
        "input_spatial_shape": list(volume.shape),
        "raw_dtype": str(volume.dtype),
        "prediction_dtype": str(prediction.dtype),
        "input_resolution_nm": min_resolution_for_raw,
        "output_resolution_nm": base_resolution,
        "roi_offset_nm": list(roi_offset),  # Spatial offset in nm
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
    }
    if output_type in ("affinities", "affinities_lsds"):
        metadata["affinity_offsets"] = affinity_offsets
    if output_channels is not None:
        metadata["output_channels"] = int(output_channels)
    if lsds_channels is not None:
        metadata["lsds_channels"] = int(lsds_channels)
    # Record whether lsds were saved (separate from output_type)
    if lsds is not None:
        metadata["lsds_saved"] = True
        metadata["lsds_shape"] = list(lsds.shape)
    else:
        metadata["lsds_saved"] = False

    # Write metadata to base directory
    metadata_path = os.path.join(zarr_base_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Export complete:")
    print(f"  Base directory: {zarr_base_dir}")
    print(f"  Input zarr:  {input_zarr_path}")
    print(f"  Output zarr base: {zarr_base_dir}")
    print(f"  Metadata:    {metadata_path}")
    print(f"{'='*60}\n")

# %%

# from dinov3_playground.vis import gif_2d_fast
# from funlib.persistence import Array

# print("Creating visualization...")

# # Handle different resolutions: downsample input volume to match prediction resolution
# # Determine spatial shape of prediction (works for labels, affinities, affinities_lsds)
# pred_spatial_shape = prediction.shape[-3:]
# if volume.shape != pred_spatial_shape:  # Compare with spatial dims of prediction
#     from skimage.transform import resize

#     # For affinities, prediction has shape (num_offsets, D, H, W)
#     target_shape = pred_spatial_shape

#     print(
#         f"Downsampling volume from {volume.shape} to {target_shape} for visualization"
#     )
#     volume_downsampled = resize(
#         volume,
#         target_shape,
#         preserve_range=True,
#         anti_aliasing=True,
#         order=1,  # Linear interpolation for raw data
#     ).astype(volume.dtype)

#     # Both arrays now at the same resolution
#     volume_array = Array(
#         volume_downsampled,
#         voxel_size=(base_resolution, base_resolution, base_resolution),
#         axis_names=["z", "y", "x"],
#     )
# else:
#     # Same resolution case
#     volume_array = Array(
#         volume,
#         voxel_size=(base_resolution, base_resolution, base_resolution),
#         axis_names=["z", "y", "x"],
#     )

# print(f"Volume shape: {volume_array.shape}")
# print(f"Prediction shape: {prediction.shape}")

# # Handle affinity predictions differently
# if output_type in ("affinities", "affinities_lsds"):
#     # Affinities have shape (num_offsets, D, H, W)
#     affinities_for_vis = affinities if affinities is not None else prediction
#     num_offsets = affinities_for_vis.shape[0]

#     print(f"Converting affinities to RGB for visualization...")
#     print(f"  Total affinity channels: {num_offsets}")

#     # For multi-scale affinities, visualize the first 3 channels as RGB
#     # (typically the finest scale: 1-voxel offsets in z, y, x)
#     if num_offsets >= 3:
#         print(f"  Using first 3 channels for RGB visualization:")
#         if affinity_offsets:
#             print(f"    Red:   {affinity_offsets[0]}")
#             print(f"    Green: {affinity_offsets[1]}")
#             print(f"    Blue:  {affinity_offsets[2]}")

#         # Use raw affinity values (0 to 1) directly - no thresholding
#         # This allows affinity strength to control transparency in the overlay
#         affinities_rgb = np.stack(
#             [
#                 affinities_for_vis[0],  # Red channel: first offset
#                 affinities_for_vis[1],  # Green channel: second offset
#                 affinities_for_vis[2],  # Blue channel: third offset
#             ],
#             axis=0,
#         )  # Shape: (3, D, H, W)
#     else:
#         # Fallback for fewer than 3 channels (shouldn't happen with proper models)
#         print(f"  WARNING: Only {num_offsets} channels available")
#         # Pad with zeros if needed
#         channels = [
#             (
#                 affinities_for_vis[i]
#                 if i < num_offsets
#                 else np.zeros_like(affinities_for_vis[0])
#             )
#             for i in range(3)
#         ]
#         affinities_rgb = np.stack(channels, axis=0)

#     print(f"Affinities RGB shape: {affinities_rgb.shape}")
#     print(f"RGB range: [{affinities_rgb.min():.3f}, {affinities_rgb.max():.3f}]")

#     # Create Array for affinities
#     affinities_array = Array(
#         affinities_rgb,
#         voxel_size=(base_resolution, base_resolution, base_resolution),
#         axis_names=["c^", "z", "y", "x"],
#         types=["channel", "space", "space", "space"],
#     )

#     gif_2d_fast(
#         arrays={
#             "raw": volume_array,
#             "affinities_rgb": affinities_array,
#             "combined": (volume_array, affinities_array),
#         },
#         array_types={
#             "raw": "raw",
#             "affinities_rgb": "rgb",  # RGB affinities
#             "combined": "combined",  # Raw with RGB overlay
#         },
#         filename=f"gifs/{dataset_name}_{output_name}_affinities.gif",
#         overwrite=True,
#         title=f"Prediction: {output_name} (first 3 channels)\nDataset: {dataset_name}",
#     )
# else:
#     # Standard label prediction
#     prediction_array = Array(
#         prediction, voxel_size=(base_resolution, base_resolution, base_resolution)
#     )

#     print(f"Unique prediction values: {np.unique(prediction_array[:])}")

#     gif_2d_fast(
#         arrays={
#             "raw": volume_array,
#             "pred": prediction_array,
#             "combined": (volume_array, prediction_array),
#         },
#         array_types={"raw": "raw", "pred": "labels", "combined": "combined"},
#         filename=f"gifs/{dataset_name}_{output_name}.gif",
#         overwrite=True,
#         title=f"Prediction: {output_name}\nDataset: {dataset_name}",
#     )

# print(
#     f"Visualization saved to gifs/{dataset_name}_{output_name}{'_affinities' if output_type == 'affinities' else ''}.gif"
# )
# # %%
