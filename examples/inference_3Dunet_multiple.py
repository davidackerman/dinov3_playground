# %%
from dinov3_playground.inference import load_inference_model
from cellmap_flow.image_data_interface import ImageDataInterface
from funlib.geometry import Roi
import numpy as np
from mws import process_mutex_watershed

# Define the source dataset and ROI
# These will be adjusted based on model config after loading
# dataset_path = "/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/recon-1/em/fibsem-uint8/s2"
dataset_path = "/nrs/cellmap/data/jrc_celegans_bw25113/jrc_celegans_bw25113.zarr/recon-1/em/fibsem-uint16/s0"
roi_offset = (14362, 16225, 20948)
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

# path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_cell_affinities_1_to_9/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"
path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_mito_affinities_1_to_9/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"  # /nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_cell_affinities/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"
inference = load_inference_model(path)
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
if output_type == "affinities":
    print(f"Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")

    # Run mutex watershed to get segmentation from affinities
    print("\nProcessing affinities with mutex watershed...")
    segmentation = process_mutex_watershed(
        affinities=prediction,
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
        adjacent_edge_bias=-0.7,
        lr_bias=[-0.7, -0.7],
        filter_val=0.7,
    )
    print(f"Segmentation shape: {segmentation.shape}")
    print(f"Segmentation dtype: {segmentation.dtype}")
    print(f"Number of segments: {len(np.unique(segmentation))}")
else:
    print(f"Unique prediction values: {np.unique(prediction)}")

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
            "chunks": [min(64, s) for s in volume.shape],
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
    raw_attrs = {
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
                                    float(min_resolution_for_raw),
                                    float(min_resolution_for_raw),
                                    float(min_resolution_for_raw),
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
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}
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
    if output_type == "affinities":

        # For affinities, save as a single multichannel array (C, Z, Y, X)
        # This is the preferred format for neuroglancer and other tools
        num_channels = prediction.shape[0]
        pred_spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": output_zarr_path,
            },
            "metadata": {
                "shape": prediction.shape,  # (num_offsets, D, H, W)
                "chunks": [prediction.shape[0]]
                + [min(64, s) for s in prediction.shape[1:]],
                "dtype": "<f4",  # float32 in zarr format
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
        pred_dataset[:].write(prediction.astype(np.float32)).result()

        # Generate channel names based on affinity offsets
        if affinity_offsets:
            channel_names = [f"{offset}" for offset in affinity_offsets]
        else:
            channel_names = [f"channel_{i}" for i in range(num_channels)]

        # Add .zgroup file to base directory (required for zarr group)
        pred_zgroup = {"zarr_format": 2}
        pred_zgroup_path = os.path.join(output_zarr_base, ".zgroup")
        with open(pred_zgroup_path, "w") as f:
            json.dump(pred_zgroup, f, indent=2)

        # Add neuroglancer-compatible metadata for affinities (OME-NGFF multiscales format)
        # For multichannel data, we need to include the channel axis and use omero metadata
        pred_attrs = {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": "",
                    "axes": [
                        {
                            "name": "c",
                            "type": "channel",
                        },
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
                                        1.0,  # Channel dimension has unit scale
                                        float(base_resolution),
                                        float(base_resolution),
                                        float(base_resolution),
                                    ],
                                },
                                {
                                    "type": "translation",
                                    "translation": [
                                        0.0,  # Channel dimension has no offset
                                        float(roi_offset[0]),
                                        float(roi_offset[1]),
                                        float(roi_offset[2]),
                                    ],
                                },
                            ],
                        }
                    ],
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]}
                    ],
                }
            ],
            "omero": {
                "channels": [
                    {
                        "label": name,
                        "color": "FFFFFF",  # White color for all channels
                        "window": {"start": 0.0, "end": 1.0, "min": 0.0, "max": 1.0},
                    }
                    for name in channel_names
                ]
            },
            "affinity_offsets": (affinity_offsets if affinity_offsets else []),
        }
        # Write .zattrs to base directory (output/, not output/s0/)
        pred_attrs_path = os.path.join(output_zarr_base, ".zattrs")
        with open(pred_attrs_path, "w") as f:
            json.dump(pred_attrs, f, indent=2)

        print(f"  ✓ Affinities saved as multichannel: shape={prediction.shape}")
        print(f"    Resolution: {base_resolution}nm isotropic")
        print(f"    Offset: {roi_offset} nm")
        print(f"    Channels: {num_channels} affinity offsets")
        for i, name in enumerate(channel_names):
            print(f"      [{i}] = {name}")

        # Export segmentation from mutex watershed
        print("\nWriting segmentation from mutex watershed...")
        seg_zarr_base = os.path.join(zarr_base_dir, "seg")
        seg_zarr_path = os.path.join(seg_zarr_base, "s0")
        os.makedirs(seg_zarr_path, exist_ok=True)

        # Determine minimum dtype to accommodate all IDs
        max_id = segmentation.max()
        seg_dtype = np.min_scalar_type(max_id)

        print(f"  Optimizing dtype: max_id={max_id} -> {seg_dtype}")
        segmentation_optimized = segmentation.astype(seg_dtype)

        seg_spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": seg_zarr_path,
            },
            "metadata": {
                "shape": segmentation_optimized.shape,
                "chunks": [min(64, s) for s in segmentation_optimized.shape],
                "dtype": numpy_to_zarr_dtype(seg_dtype),
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
        seg_dataset = ts.open(seg_spec).result()
        seg_dataset[:].write(segmentation_optimized).result()

        # Add .zgroup file to base directory
        seg_zgroup = {"zarr_format": 2}
        seg_zgroup_path = os.path.join(seg_zarr_base, ".zgroup")
        with open(seg_zgroup_path, "w") as f:
            json.dump(seg_zgroup, f, indent=2)

        # Add neuroglancer-compatible metadata for segmentation
        seg_attrs = {
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
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                    ],
                }
            ]
        }
        seg_attrs_path = os.path.join(seg_zarr_base, ".zattrs")
        with open(seg_attrs_path, "w") as f:
            json.dump(seg_attrs, f, indent=2)

        print(
            f"  ✓ Segmentation saved: shape={segmentation_optimized.shape}, dtype={segmentation_optimized.dtype}"
        )
        print(f"    Resolution: {base_resolution}nm isotropic")
        print(f"    Offset: {roi_offset} nm")
        print(f"    Number of segments: {len(np.unique(segmentation_optimized))}")
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
                "chunks": [min(64, s) for s in prediction.shape],
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
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                    ],
                }
            ]
        }
        # Write .zattrs to base directory (output/, not output/s0/)
        pred_attrs_path = os.path.join(output_zarr_base, ".zattrs")
        with open(pred_attrs_path, "w") as f:
            json.dump(pred_attrs, f, indent=2)

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
        "raw_dtype": str(volume.dtype),
        "prediction_dtype": str(prediction.dtype),
        "input_resolution_nm": min_resolution_for_raw,
        "output_resolution_nm": base_resolution,
        "roi_offset_nm": list(roi_offset),  # Spatial offset in nm
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
    }
    if output_type == "affinities":
        metadata["affinity_offsets"] = affinity_offsets

    # Write metadata to base directory
    metadata_path = os.path.join(zarr_base_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Export complete:")
    print(f"  Base directory: {zarr_base_dir}")
    print(f"  Input zarr:  {input_zarr_path}")
    print(f"  Output zarr: {output_zarr_path}")
    print(f"  Metadata:    {metadata_path}")
    print(f"{'='*60}\n")

# %%

from dinov3_playground.vis import gif_2d_fast
from funlib.persistence import Array

print("Creating visualization...")

# Handle different resolutions: downsample input volume to match prediction resolution
if volume.shape != prediction.shape[-3:]:  # Compare with spatial dims of prediction
    from skimage.transform import resize

    # For affinities, prediction has shape (num_offsets, D, H, W)
    target_shape = (
        prediction.shape[-3:] if output_type == "affinities" else prediction.shape
    )

    print(
        f"Downsampling volume from {volume.shape} to {target_shape} for visualization"
    )
    volume_downsampled = resize(
        volume,
        target_shape,
        preserve_range=True,
        anti_aliasing=True,
        order=1,  # Linear interpolation for raw data
    ).astype(volume.dtype)

    # Both arrays now at the same resolution
    volume_array = Array(
        volume_downsampled,
        voxel_size=(base_resolution, base_resolution, base_resolution),
        axis_names=["z", "y", "x"],
    )
else:
    # Same resolution case
    volume_array = Array(
        volume,
        voxel_size=(base_resolution, base_resolution, base_resolution),
        axis_names=["z", "y", "x"],
    )

print(f"Volume shape: {volume_array.shape}")
print(f"Prediction shape: {prediction.shape}")

# Handle affinity predictions differently
if output_type == "affinities":
    # Affinities have shape (num_offsets, D, H, W)
    num_offsets = prediction.shape[0]

    print(f"Converting affinities to RGB for visualization...")
    print(f"  Total affinity channels: {num_offsets}")

    # For multi-scale affinities, visualize the first 3 channels as RGB
    # (typically the finest scale: 1-voxel offsets in z, y, x)
    if num_offsets >= 3:
        print(f"  Using first 3 channels for RGB visualization:")
        if affinity_offsets:
            print(f"    Red:   {affinity_offsets[0]}")
            print(f"    Green: {affinity_offsets[1]}")
            print(f"    Blue:  {affinity_offsets[2]}")

        # Use raw affinity values (0 to 1) directly - no thresholding
        # This allows affinity strength to control transparency in the overlay
        affinities_rgb = np.stack(
            [
                prediction[0],  # Red channel: first offset
                prediction[1],  # Green channel: second offset
                prediction[2],  # Blue channel: third offset
            ],
            axis=0,
        )  # Shape: (3, D, H, W)
    else:
        # Fallback for fewer than 3 channels (shouldn't happen with proper models)
        print(f"  WARNING: Only {num_offsets} channels available")
        # Pad with zeros if needed
        channels = [
            prediction[i] if i < num_offsets else np.zeros_like(prediction[0])
            for i in range(3)
        ]
        affinities_rgb = np.stack(channels, axis=0)

    print(f"Affinities RGB shape: {affinities_rgb.shape}")
    print(f"RGB range: [{affinities_rgb.min():.3f}, {affinities_rgb.max():.3f}]")

    # Create Array for affinities
    affinities_array = Array(
        affinities_rgb,
        voxel_size=(base_resolution, base_resolution, base_resolution),
        axis_names=["c^", "z", "y", "x"],
        types=["channel", "space", "space", "space"],
    )

    gif_2d_fast(
        arrays={
            "raw": volume_array,
            "affinities_rgb": affinities_array,
            "combined": (volume_array, affinities_array),
        },
        array_types={
            "raw": "raw",
            "affinities_rgb": "rgb",  # RGB affinities
            "combined": "combined",  # Raw with RGB overlay
        },
        filename=f"gifs/{dataset_name}_{output_name}_affinities.gif",
        overwrite=True,
        title=f"Prediction: {output_name} (first 3 channels)\nDataset: {dataset_name}",
    )
else:
    # Standard label prediction
    prediction_array = Array(
        prediction, voxel_size=(base_resolution, base_resolution, base_resolution)
    )

    print(f"Unique prediction values: {np.unique(prediction_array[:])}")

    gif_2d_fast(
        arrays={
            "raw": volume_array,
            "pred": prediction_array,
            "combined": (volume_array, prediction_array),
        },
        array_types={"raw": "raw", "pred": "labels", "combined": "combined"},
        filename=f"gifs/{dataset_name}_{output_name}.gif",
        overwrite=True,
        title=f"Prediction: {output_name}\nDataset: {dataset_name}",
    )

print(
    f"Visualization saved to gifs/{dataset_name}_{output_name}{'_affinities' if output_type == 'affinities' else ''}.gif"
)
# %%
