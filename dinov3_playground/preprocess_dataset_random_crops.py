#!/usr/bin/env python
"""
Command-line script for preprocessing random crops from a dataset.

Simply samples random crops from a dataset's raw data and extracts
DINOv3 features - no ground truth, no organelles, just raw → features.

Usage:
    python preprocess_dataset_random_crops.py \
        --crop-index 0 \
        --output-dir /path/to/output \
        --dataset-path /path/to/raw.zarr \
        --output-image-dim 128 \
        --use-anyup
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path if running as script
sys.path.insert(0, str(Path(__file__).parent))

from dinov3_preprocessing_random_crops import preprocess_and_save_random_crop


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 features from a random crop"
    )

    # Required arguments
    parser.add_argument(
        "--crop-index",
        type=int,
        required=True,
        help="Crop index to process (also used as random seed for sampling location)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save preprocessed crops",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the raw dataset (e.g., path to zarr)",
    )

    # Model configuration
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-sat493m",
        help="DINOv3 model identifier",
    )
    parser.add_argument(
        "--input-resolution",
        type=int,
        default=32,
        help="Resolution of input raw data in nm",
    )
    parser.add_argument(
        "--output-resolution",
        type=int,
        default=128,
        help="Target resolution for features in nm",
    )
    parser.add_argument(
        "--output-image-dim",
        type=int,
        default=128,
        help="Output crop dimension (will be cube of this size)",
    )

    # Feature extraction configuration
    parser.add_argument(
        "--use-anyup",
        action="store_true",
        default=True,
        help="Use AnyUp for feature extraction (default: True)",
    )
    parser.add_argument(
        "--no-anyup",
        dest="use_anyup",
        action="store_false",
        help="Don't use AnyUp for feature extraction",
    )

    # Storage configuration
    parser.add_argument(
        "--compression",
        action="store_true",
        help="Use LZ4 compression (slower reads, smaller files)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Number of threads for TensorStore operations",
    )
    parser.add_argument(
        "--no-save-raw",
        action="store_true",
        help="Don't save raw data (only save features)",
    )
    parser.add_argument(
        "--no-use-orthogonal-planes",
        action="store_true",
        help="Don't use orthogonal planes for feature extraction (default: True)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("DINOv3 Random Crop Feature Extraction")
    print(f"{'='*60}")
    print(f"Crop index: {args.crop_index}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Model: {args.model_id}")
    print(f"Output dimension: {args.output_image_dim}")
    print(f"Input resolution: {args.input_resolution} nm")
    print(f"Output resolution: {args.output_resolution} nm")
    print(f"Use AnyUp: {args.use_anyup}")
    print(f"Compression: {'enabled' if args.compression else 'disabled'}")
    print(f"Save raw: {not args.no_save_raw}")

    try:
        metadata = preprocess_and_save_random_crop(
            crop_index=args.crop_index,
            output_dir=args.output_dir,
            dataset_path=args.dataset_path,
            model_id=args.model_id,
            input_resolution=args.input_resolution,
            output_resolution=args.output_resolution,
            output_image_dim=args.output_image_dim,
            use_compression=args.compression,
            use_anyup=args.use_anyup,
            num_threads=args.num_threads,
            save_raw=not args.no_save_raw,
            use_orthogonal_planes=not args.no_use_orthogonal_planes,
        )

        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"Crop: {metadata['crop_name']}")
        print(f"Features: {metadata['storage']['features_gb']:.3f} GB")
        print(f"Total time: {metadata['timing']['total_seconds']:.1f}s")
        print(f"Metadata: {metadata['crop_name']}_metadata.json")

        return 0

    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR!")
        print(f"{'='*60}")
        print(f"Failed to process crop {args.crop_index}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
