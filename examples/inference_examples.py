"""
Example usage of DINOv3 UNet Inference Classes

This script demonstrates how to use the inference classes to load trained models
and run inference on new images or volumes.

Author: GitHub Copilot
Date: 2025-09-26
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the package to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from dinov3_playground.inference import (
    DINOv3UNetInference,
    DINOv3UNet3DInference,
    load_inference_model,
)


def example_2d_inference():
    """Example of loading and using a 2D UNet model for inference."""

    # Path to your training export directory
    # Replace this with the actual path to your trained model
    export_dir = "/nrs/cellmap/ackermand/dinov3_training/results/example_2d"

    if not Path(export_dir).exists():
        print(f"Export directory {export_dir} does not exist. Please update the path.")
        return

    print("=" * 60)
    print("2D INFERENCE EXAMPLE")
    print("=" * 60)

    try:
        # Load the inference model
        print("Loading 2D inference model...")
        model = DINOv3UNetInference(export_dir)

        # Print model information
        model_info = model.get_model_info()
        print("\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")

        # Create a test image (replace with your actual image)
        print("\nCreating test image...")
        test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

        # Run inference
        print("Running inference...")
        predictions = model.predict(test_image)
        predictions_with_prob, probabilities = model.predict(
            test_image, return_probabilities=True
        )

        print(f"\nResults:")
        print(f"  Input shape: {test_image.shape}")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Probabilities shape: {probabilities.shape}")
        print(f"  Unique predictions: {np.unique(predictions)}")
        print(f"  Prediction distribution: {np.bincount(predictions.flatten())}")

        # Visualize results
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(test_image, cmap="gray")
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(predictions, cmap="tab10")
        axes[1].set_title("Predictions")
        axes[1].axis("off")

        axes[2].imshow(probabilities[0], cmap="viridis")
        axes[2].set_title("Class 0 Probability")
        axes[2].axis("off")

        axes[3].imshow(probabilities[1], cmap="viridis")
        axes[3].set_title("Class 1 Probability")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()

        # Test batch inference
        print("\nTesting batch inference...")
        batch_images = np.random.randint(0, 255, (3, 512, 512), dtype=np.uint8)
        batch_predictions = model.predict_batch(batch_images)
        print(f"  Batch input shape: {batch_images.shape}")
        print(f"  Batch predictions shape: {batch_predictions.shape}")

    except Exception as e:
        print(f"Error in 2D inference: {e}")
        import traceback

        traceback.print_exc()


def example_3d_inference():
    """Example of loading and using a 3D UNet model for inference."""

    # Path to your training export directory
    # Replace this with the actual path to your trained 3D model
    export_dir = "/nrs/cellmap/ackermand/dinov3_training/results/mito_3d"

    if not Path(export_dir).exists():
        print(f"Export directory {export_dir} does not exist. Please update the path.")
        return

    print("=" * 60)
    print("3D INFERENCE EXAMPLE")
    print("=" * 60)

    try:
        # Load the 3D inference model
        print("Loading 3D inference model...")
        model = DINOv3UNet3DInference(export_dir)

        # Print model information
        model_info = model.get_model_info()
        print("\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")

        # Create a test volume (replace with your actual volume)
        print("\nCreating test volume...")
        volume_size = model_info["input_size"]
        test_volume = np.random.randint(0, 255, volume_size, dtype=np.uint8)

        # Run inference on the full volume
        print("Running 3D inference...")
        predictions = model.predict(test_volume)
        predictions_with_prob, probabilities = model.predict(
            test_volume, return_probabilities=True
        )

        print(f"\nResults:")
        print(f"  Input shape: {test_volume.shape}")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Probabilities shape: {probabilities.shape}")
        print(f"  Unique predictions: {np.unique(predictions)}")
        print(f"  Prediction distribution: {np.bincount(predictions.flatten())}")

        # Visualize results - show middle slices
        D, H, W = test_volume.shape
        mid_slice = D // 2

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Top row: different Z slices
        z_slices = [0, D // 4, D // 2, D - 1]
        for i, z in enumerate(z_slices):
            axes[0, i].imshow(test_volume[z], cmap="gray")
            axes[0, i].set_title(f"Input Z={z}")
            axes[0, i].axis("off")

            axes[1, i].imshow(predictions[z], cmap="tab10")
            axes[1, i].set_title(f"Predictions Z={z}")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()

        # Test large volume inference (sliding window)
        print("\nTesting large volume inference with sliding window...")
        large_volume = np.random.randint(0, 255, (256, 256, 256), dtype=np.uint8)
        large_predictions = model.predict_large_volume(
            large_volume, chunk_size=64, overlap=16
        )
        print(f"  Large volume input shape: {large_volume.shape}")
        print(f"  Large volume predictions shape: {large_predictions.shape}")

        # Test batch inference
        print("\nTesting batch 3D inference...")
        batch_volumes = np.random.randint(0, 255, (2, *volume_size), dtype=np.uint8)
        batch_predictions = model.predict_batch(batch_volumes)
        print(f"  Batch input shape: {batch_volumes.shape}")
        print(f"  Batch predictions shape: {batch_predictions.shape}")

    except Exception as e:
        print(f"Error in 3D inference: {e}")
        import traceback

        traceback.print_exc()


def example_auto_inference():
    """Example of using the automatic model type detection."""

    # Path to your training export directory
    export_dir = "/nrs/cellmap/ackermand/dinov3_training/results/mito_3d"

    if not Path(export_dir).exists():
        print(f"Export directory {export_dir} does not exist. Please update the path.")
        return

    print("=" * 60)
    print("AUTO INFERENCE EXAMPLE")
    print("=" * 60)

    try:
        # Automatically detect and load the appropriate model
        print("Auto-loading inference model...")
        model = load_inference_model(export_dir, model_type="auto")

        # Print model information
        model_info = model.get_model_info()
        print(f"\nDetected model type: {model_info['model_type']}")
        print("\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")

        # Create appropriate test data based on model type
        if model_info["model_type"] == "3D UNet":
            # Create 3D test volume
            input_size = model_info["input_size"]
            test_data = np.random.randint(0, 255, input_size, dtype=np.uint8)
            print(f"\nCreated 3D test volume: {test_data.shape}")
        else:
            # Create 2D test image
            input_size = model_info["input_size"]
            test_data = np.random.randint(0, 255, input_size, dtype=np.uint8)
            print(f"\nCreated 2D test image: {test_data.shape}")

        # Run inference
        print("Running inference...")
        predictions = model.predict(test_data)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Unique predictions: {np.unique(predictions)}")

    except Exception as e:
        print(f"Error in auto inference: {e}")
        import traceback

        traceback.print_exc()


def example_real_data_inference():
    """Example showing how to load and process real image/volume data."""

    print("=" * 60)
    print("REAL DATA INFERENCE EXAMPLE")
    print("=" * 60)

    # This is a template - you would replace these with your actual data loading
    print("This is a template for real data inference.")
    print("Replace the following sections with your actual data loading code:")

    print(
        """
    # Example for 2D inference:
    
    from PIL import Image
    import numpy as np
    
    # Load your image
    image_path = "path/to/your/image.png"
    image = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale
    
    # Load inference model
    model = DINOv3UNetInference("path/to/export/directory")
    
    # Run inference
    predictions = model.predict(image)
    predictions_with_prob, probabilities = model.predict(image, return_probabilities=True)
    
    # Save results
    result_image = Image.fromarray((predictions * 255).astype(np.uint8))
    result_image.save("predictions.png")
    """
    )

    print(
        """
    # Example for 3D inference:
    
    import zarr
    import numpy as np
    
    # Load your volume (example with zarr)
    volume_path = "path/to/your/volume.zarr"
    volume = zarr.open(volume_path, mode='r')[:]
    
    # Load inference model  
    model = DINOv3UNet3DInference("path/to/export/directory")
    
    # Run inference
    predictions = model.predict(volume)
    
    # For large volumes, use sliding window
    if volume.shape[0] > 128:  # If larger than training size
        predictions = model.predict_large_volume(volume, chunk_size=64, overlap=16)
    
    # Save results
    output_zarr = zarr.open("predictions.zarr", mode='w', shape=predictions.shape, dtype=predictions.dtype)
    output_zarr[:] = predictions
    """
    )


if __name__ == "__main__":
    print("DINOv3 UNet Inference Examples")
    print("==============================")

    # You can uncomment and run the examples you're interested in:

    # example_2d_inference()
    # example_3d_inference()
    example_auto_inference()
    # example_real_data_inference()

    print("\n" + "=" * 60)
    print("INFERENCE EXAMPLES COMPLETED")
    print("=" * 60)
