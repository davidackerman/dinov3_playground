"""
Quick Start: DINOv3 UNet Inference

This script shows the simplest way to load a trained model and run inference.

Usage:
    python quick_inference.py /path/to/export/directory /path/to/image_or_volume.png

Author: GitHub Copilot
Date: 2025-09-26
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

# Import the inference module
from dinov3_playground.inference import load_inference_model


def load_image_or_volume(file_path):
    """Load image or volume from file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Try to load as image first
    try:
        if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
            image = Image.open(file_path).convert("L")  # Convert to grayscale
            return np.array(image)
        elif file_path.suffix.lower() == ".npy":
            return np.load(file_path)
        else:
            # Try generic numpy load
            return np.load(file_path)
    except Exception as e:
        raise ValueError(f"Could not load file {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run DINOv3 UNet inference on an image or volume"
    )
    parser.add_argument("export_dir", help="Path to the training export directory")
    parser.add_argument("input_file", help="Path to input image or volume file")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--probabilities", action="store_true", help="Also save probability maps"
    )

    args = parser.parse_args()

    print("DINOv3 UNet Quick Inference")
    print("=" * 40)

    try:
        # Load the model
        print(f"Loading model from: {args.export_dir}")
        device = None if args.device == "auto" else args.device
        model = load_inference_model(args.export_dir, device=device)

        # Print model info
        model_info = model.get_model_info()
        print(f"Model type: {model_info['model_type']}")
        print(f"Classes: {model_info['num_classes']}")
        print(f"Device: {model_info['device']}")

        # Load input data
        print(f"Loading input from: {args.input_file}")
        input_data = load_image_or_volume(args.input_file)
        print(f"Input shape: {input_data.shape}")

        # Run inference
        print("Running inference...")
        if args.probabilities:
            predictions, probabilities = model.predict(
                input_data, return_probabilities=True
            )
            print(f"Predictions shape: {predictions.shape}")
            print(f"Probabilities shape: {probabilities.shape}")
        else:
            predictions = model.predict(input_data)
            print(f"Predictions shape: {predictions.shape}")

        print(f"Unique predictions: {np.unique(predictions)}")

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            input_path = Path(args.input_file)
            output_path = (
                input_path.parent / f"{input_path.stem}_predictions{input_path.suffix}"
            )

        # Save predictions
        if output_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            # Save as image
            pred_image = Image.fromarray(
                (predictions * 255 / predictions.max()).astype(np.uint8)
            )
            pred_image.save(output_path)
            print(f"Predictions saved to: {output_path}")
        else:
            # Save as numpy array
            np.save(output_path.with_suffix(".npy"), predictions)
            print(f"Predictions saved to: {output_path.with_suffix('.npy')}")

        # Save probabilities if requested
        if args.probabilities:
            prob_path = output_path.parent / f"{output_path.stem}_probabilities.npy"
            np.save(prob_path, probabilities)
            print(f"Probabilities saved to: {prob_path}")

        print("✅ Inference completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
