# %%
from dinov3_playground.inference import load_inference_model
from cellmap_flow.image_data_interface import ImageDataInterface
from funlib.geometry import Roi
import numpy as np

input_voxel_size = 8
# volume = ImageDataInterface(
#     "/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr/recon-1/em/fibsem-uint8/s0",
#     output_voxel_size=3 * [input_voxel_size],
# ).to_ndarray_ts(
#     Roi(
#         (152236, 81035, 99548),
#         [512 * input_voxel_size, 512 * input_voxel_size, 512 * input_voxel_size],
#     )
# )
roi = Roi(
    (65150, 56187, 32626),
    [512 * input_voxel_size, 512 * input_voxel_size, 512 * input_voxel_size],
)
volume = ImageDataInterface(
    "/nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/s0",
    output_voxel_size=(input_voxel_size, input_voxel_size, input_voxel_size),
).to_ndarray_ts(roi)

# Load context volume at lower resolution if model uses context fusion
# Check model config after loading to determine if context is needed
context_volume = None
# Uncomment and adjust if your model was trained with context fusion:
context_voxel_size = 32  # Example: 32nm for context vs 8nm for local
context_roi = Roi(
    roi.get_center() - 256 * context_voxel_size, 3 * [512 * context_voxel_size]
)
context_volume = ImageDataInterface(
    "/nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/s2",
    output_voxel_size=(context_voxel_size, context_voxel_size, context_voxel_size),
).to_ndarray_ts(context_roi)

# %%
# Automatically load the best model from export directory
# You can now provide either:
# 1. Full path with timestamp: .../run_20251003_110551
# 2. Parent path (auto-selects most recent): .../dinov3_unet3d_dinov3_vitl16_pretrain_sat493m

# path = "/nrs/cellmap/ackermand/dinov3_training/results/multiple_3d_hybrid_stride_2/dinov3_unet3d_dinov3_vitl16_pretrain_lvd1689m/run_20250929_021253"
# path = "/nrs/cellmap/ackermand/dinov3_training/results/multiple_3d_orthogonal/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/run_20250930_020133"
# path = "/nrs/cellmap/ackermand/dinov3_training/results/multiple_3d_orthogonal_highest_raw_res/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/run_20250930_184141"
# path = "/nrs/cellmap/ackermand/dinov3_training/results/multiple_3d_orthogonal_highest_raw_res/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/run_20251001_005635"
# path = "/nrs/cellmap/ackermand/dinov3_training/results/challenge_base_res_16_extend/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/run_20251002_025525"
# path = "/nrs/cellmap/ackermand/dinov3_training/results/challenge_base_res_16_extend_more/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/run_20251002_164747"
# path = "/nrs/cellmap/ackermand/dinov3_training/results/challenge_base_res_16_extend_more_imagesize_64/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/run_20251002_225532"
# path = "/nrs/cellmap/ackermand/dinov3_training/results/challenge_base_res_32_extend_more/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/run_20251003_021038"

# Option 1: Full path with timestamp (explicit)
# path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_base_res_32_extend_more_mito/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/run_20251003_110551"

# Option 2: Parent path (auto-selects most recent timestamp) - RECOMMENDED
path = "/nrs/cellmap/ackermand/dinov3_training/results/dinov3_finetune_3Dunet_challenge_base_res_16_context_32_focal_dice_nuc/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m"

inference = load_inference_model(path)
output_name = "_".join(path.split("/")[-3:])

# Check if model uses context fusion
model_info = inference.get_model_info()
uses_context = inference.model_config.get("use_context_fusion", False)
print(f"\n{'='*60}")
print(f"Model uses context fusion: {uses_context}")
if uses_context and context_volume is None:
    print("⚠️  WARNING: Model expects context volume but none provided!")
    print("   Inference will run but may produce suboptimal results.")
print(f"{'='*60}\n")

# Run inference with or without context
if uses_context and context_volume is not None:
    prediction = inference.predict(volume, context_volume=context_volume)
else:
    prediction = inference.predict(volume)
from dinov3_playground.vis import gif_2d
from funlib.persistence import Array

print("Do gif...")

# Handle different resolutions: downsample input volume to match prediction resolution
if volume.shape != prediction.shape:
    from skimage.transform import resize

    print(
        f"Downsampling volume from {volume.shape} to {prediction.shape} for visualization"
    )
    volume_downsampled = resize(
        volume,
        prediction.shape,
        preserve_range=True,
        anti_aliasing=True,
        order=1,  # Linear interpolation for raw data
    ).astype(volume.dtype)

    # Both arrays now at the same resolution (64nm)
    volume_array = Array(volume_downsampled, voxel_size=(16, 16, 16))
    prediction_array = Array(prediction, voxel_size=(16, 16, 16))
else:
    # Same resolution case
    volume_array = Array(volume, voxel_size=(16, 16, 16))
    prediction_array = Array(prediction, voxel_size=(16, 16, 16))

print(f"Volume shape: {volume_array.shape}")
print(f"Prediction shape: {prediction_array.shape}")
print(f"Unique prediction values: {np.unique(prediction_array[:])}")

gif_2d(
    arrays={
        "raw": volume_array,
        "pred": prediction_array,
        "combined": (volume_array, prediction_array),
    },
    array_types={"raw": "raw", "pred": "labels", "combined": "combined"},
    filename=f"gifs/{output_name}.gif",
    title="3D UNet Mito Inference",
    overwrite=True,
)
# %%
