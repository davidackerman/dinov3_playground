"""
Example: Using the DINOv3 3D UNet Inference with your trained model

This script shows how to load the model you trained in dinov3_finetune_3Dunet_mito_sat.py
and use it for inference on new volumes.

Author: GitHub Copilot
Date: 2025-09-26
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the inference classes
from dinov3_playground.inference import DINOv3UNet3DInference, load_inference_model


def example_with_your_trained_model():
    """
    Example using the model trained in your dinov3_finetune_3Dunet_mito_sat.py script.
    """

    # This should match the EXPORT_BASE_DIR from your training script
    # The actual structure will be: /nrs/cellmap/ackermand/dinov3_training/results/mito_3d/dinov3_unet3d_model_id/timestamp/model/
    export_dir = "/nrs/cellmap/ackermand/dinov3_training/results/mito_3d"

    if not Path(export_dir).exists():
        print(f"❌ Export directory not found: {export_dir}")
        print("Make sure you've run the training script and it completed successfully.")
        return

    print("🔄 Loading your trained 3D UNet model...")

    try:
        # Method 1: Explicit 3D model loading
        model = DINOv3UNet3DInference(export_dir)

        # Alternative Method 2: Auto-detection
        # model = load_inference_model(export_dir, model_type='auto')

        print("✅ Model loaded successfully!")

        # Show model information
        model_info = model.get_model_info()
        print("\n📋 Model Information:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")

        # Create a test volume with the same size as training
        volume_size = model_info["input_size"]  # Should be (128, 128, 128)
        print(f"\n🎲 Creating test volume with shape: {volume_size}")

        # Create synthetic test data (replace with your real data)
        test_volume = np.random.randint(0, 255, volume_size, dtype=np.uint8)

        # Add some structure to make it more realistic
        # Create some "mitochondria-like" structures
        center_z, center_y, center_x = np.array(volume_size) // 2
        z, y, x = np.ogrid[: volume_size[0], : volume_size[1], : volume_size[2]]

        # Add some bright spots (simulating mitochondria)
        for i in range(10):
            spot_z = np.random.randint(20, volume_size[0] - 20)
            spot_y = np.random.randint(20, volume_size[1] - 20)
            spot_x = np.random.randint(20, volume_size[2] - 20)
            spot_size = np.random.randint(5, 15)

            mask = (
                (z - spot_z) ** 2 + (y - spot_y) ** 2 + (x - spot_x) ** 2
            ) <= spot_size**2
            test_volume[mask] = np.clip(test_volume[mask] + 100, 0, 255)

        print("🔮 Running inference...")

        # Run inference - get both predictions and probabilities
        predictions, probabilities = model.predict(
            test_volume, return_probabilities=True
        )

        print("✅ Inference completed!")
        print(f"   Input shape: {test_volume.shape}")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Probabilities shape: {probabilities.shape}")
        print(f"   Unique predictions: {np.unique(predictions)}")

        # Calculate prediction statistics
        total_voxels = predictions.size
        background_voxels = np.sum(predictions == 0)
        mito_voxels = np.sum(predictions == 1)

        print(
            f"   Background voxels: {background_voxels} ({100*background_voxels/total_voxels:.1f}%)"
        )
        print(
            f"   Mitochondria voxels: {mito_voxels} ({100*mito_voxels/total_voxels:.1f}%)"
        )

        # Visualize results
        print("🎨 Creating visualizations...")

        # Show slices from different parts of the volume
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))

        D, H, W = test_volume.shape
        z_slices = [0, D // 4, D // 2, 3 * D // 4, D - 1]

        for i, z in enumerate(z_slices):
            # Raw volume slice
            axes[0, i].imshow(test_volume[z], cmap="gray")
            axes[0, i].set_title(f"Input Z={z}", fontsize=10)
            axes[0, i].axis("off")

            # Predictions slice
            axes[1, i].imshow(predictions[z], cmap="tab10", vmin=0, vmax=1)
            axes[1, i].set_title(f"Predictions Z={z}", fontsize=10)
            axes[1, i].axis("off")

            # Probability slice (mitochondria class)
            im = axes[2, i].imshow(probabilities[1, z], cmap="viridis", vmin=0, vmax=1)
            axes[2, i].set_title(f"Mito Probability Z={z}", fontsize=10)
            axes[2, i].axis("off")

        # Add colorbar for probabilities
        plt.colorbar(im, ax=axes[2, :], orientation="horizontal", pad=0.1, shrink=0.8)

        plt.suptitle("3D UNet Inference Results - Your Trained Model", fontsize=16)
        plt.tight_layout()
        plt.show()

        # Example of processing a larger volume with sliding window
        print("\n🔄 Testing sliding window inference on larger volume...")

        # Create a larger test volume
        large_volume = np.random.randint(0, 255, (256, 256, 256), dtype=np.uint8)
        print(f"   Large volume shape: {large_volume.shape}")

        # Use sliding window for memory-efficient processing
        large_predictions = model.predict_large_volume(
            large_volume,
            chunk_size=64,  # Process 64x64x64 chunks
            overlap=16,  # 16 voxel overlap between chunks
        )

        print(f"   Large predictions shape: {large_predictions.shape}")
        print(
            f"   Large volume mitochondria fraction: {np.mean(large_predictions):.3f}"
        )

        # Show a few slices from the large volume
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        D_large = large_volume.shape[0]
        z_slices_large = [0, D_large // 2, D_large - 1]

        for i, z in enumerate(z_slices_large):
            axes[0, i].imshow(large_volume[z], cmap="gray")
            axes[0, i].set_title(f"Large Volume Z={z}")
            axes[0, i].axis("off")

            axes[1, i].imshow(large_predictions[z], cmap="tab10")
            axes[1, i].set_title(f"Predictions Z={z}")
            axes[1, i].axis("off")

        plt.suptitle("Sliding Window Inference on Large Volume", fontsize=14)
        plt.tight_layout()
        plt.show()

        print("🎉 Example completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Replace the synthetic test data with your real volumes")
        print("   2. Load volumes using zarr, tiff, or your preferred format")
        print("   3. Process batches of volumes using model.predict_batch()")
        print("   4. Save results using zarr, numpy, or your preferred format")

        return model, test_volume, predictions, probabilities

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None, None


def example_with_real_data():
    """
    Template for processing real data - modify this with your actual data paths.
    """
    print("\n" + "=" * 60)
    print("REAL DATA PROCESSING TEMPLATE")
    print("=" * 60)

    print(
        """
This is a template for processing your real data. Modify the paths and 
loading code to match your data format:

```python
import zarr
from dinov3_playground.inference import DINOv3UNet3DInference

# Load your trained model
model = DINOv3UNet3DInference("/nrs/cellmap/ackermand/dinov3_training/results/mito_3d")

# Load your real volume data
# Example with zarr:
volume_path = "/path/to/your/volume.zarr"
volume = zarr.open(volume_path, mode='r')[:]

# Or with tiff:
# import tifffile
# volume = tifffile.imread("/path/to/your/volume.tif")

# Or with numpy:
# volume = np.load("/path/to/your/volume.npy")

# Run inference
if volume.shape[0] > 128:  # If larger than training volume
    predictions = model.predict_large_volume(volume, chunk_size=64, overlap=16)
else:
    predictions = model.predict(volume)

# Save results
# With zarr:
output_zarr = zarr.open("/path/to/output/predictions.zarr", 
                       mode='w', shape=predictions.shape, dtype=predictions.dtype)
output_zarr[:] = predictions

# With numpy:
# np.save("/path/to/output/predictions.npy", predictions)

# With tiff:
# import tifffile
# tifffile.imwrite("/path/to/output/predictions.tif", predictions.astype(np.uint8))
```
    """
    )


if __name__ == "__main__":
    print("DINOv3 3D UNet Inference Example")
    print("=" * 60)

    # Run the main example
    model, test_volume, predictions, probabilities = example_with_your_trained_model()

    # Show the real data template
    example_with_real_data()

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED")
    print("=" * 60)
