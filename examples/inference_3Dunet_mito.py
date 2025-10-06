# %%
from dinov3_playground.inference import load_inference_model
from cellmap_flow.image_data_interface import ImageDataInterface
from funlib.geometry import Roi

volume = ImageDataInterface(
    "/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr/recon-1/em/fibsem-uint8/s1"
).to_ndarray_ts(Roi((138484, 82455, 99548), [128 * 16, 128 * 16, 128 * 16]))
# volume = ImageDataInterface(
#     "/nrs/cellmap/data/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8/s2"
# ).to_ndarray_ts(Roi((20467, 1178, 26508), [128 * 5 * 2**2, 128 * 16, 128 * 16]))
# %%
# Automatically load the best model from export directory
inference = load_inference_model(
    # "/nrs/cellmap/ackermand/dinov3_training/results/mito_3d/dinov3_unet3d_dinov3_vitl16_pretrain_lvd1689m/run_20250925_150051"
    "/nrs/cellmap/ackermand/dinov3_training/results/mito_3d/dinov3_unet3d_dinov3_vitl16_pretrain_sat493m/run_20250926_132512"
)

prediction = inference.predict(volume)
from dinov3_playground.vis import gif_2d
from funlib.persistence import Array

print("Do gif...")
volume = Array(volume, voxel_size=(16, 16, 16))
prediction = Array(prediction, voxel_size=(16, 16, 16))
gif_2d(
    arrays={"raw": volume, "pred": prediction},
    array_types={"raw": "raw", "pred": "labels"},
    filename="inference_3dunet_mito_sat.gif",
    title="3D UNet Mito Inference",
    overwrite=True,
)
# %%
