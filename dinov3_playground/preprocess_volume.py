#!/usr/bin/env python
"""
Command-line script for preprocessing individual volumes.
Usage: python preprocess_volume.py --volume-index 0 --output-dir /path/to/output
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path if running as script
sys.path.insert(0, str(Path(__file__).parent))

from dinov3_preprocessing import preprocess_and_save_volume


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess a single volume for DINOv3 training"
    )

    # Required arguments
    parser.add_argument(
        "--volume-index",
        type=int,
        required=True,
        help="Volume index to process (also used as random seed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save preprocessed volumes",
    )

    # Model configuration
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-sat493m",
        help="DINOv3 model identifier",
    )
    parser.add_argument(
        "--base-resolution",
        type=int,
        default=128,
        help="Base resolution in nm",
    )
    parser.add_argument(
        "--min-resolution-for-raw",
        type=int,
        default=32,
        help="Minimum resolution for raw data in nm",
    )
    parser.add_argument(
        "--output-image-dim",
        type=int,
        default=128,
        help="Output image dimension",
    )

    # Data loading configuration
    parser.add_argument(
        "--organelles",
        type=str,
        nargs="+",
        default=["cell"],
        help="List of organelles to load",
    )
    parser.add_argument(
        "--crop-filter",
        type=str,
        nargs="+",
        default=None,
        help="Dataset filters (e.g., jrc_mus-liver-zon-1)",
    )
    parser.add_argument(
        "--inference-filter",
        type=str,
        nargs="+",
        default=None,
        help="Dataset filters (e.g., jrc_mus-liver-zon-1)",
    )
    parser.add_argument(
        "--min-label-fraction",
        type=float,
        default=0.01,
        help="Minimum label fraction required",
    )
    parser.add_argument(
        "--min-unique-ids",
        type=int,
        default=2,
        help="Minimum unique instance IDs required",
    )
    parser.add_argument(
        "--min-ground-truth-fraction",
        type=float,
        default=0.05,
        help="Minimum ground truth fraction",
    )
    parser.add_argument(
        "--no-gt-extension",
        action="store_true",
        help="Disable GT extension (no masks)",
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

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("DINOv3 Volume Preprocessing")
    print(f"{'='*60}")
    print(f"Volume index: {args.volume_index}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_id}")
    print(f"Organelles: {args.organelles}")
    if args.inference_filter:
        print(f"Filters: {args.inference_filter}")
    if args.crop_filter:
        print(f"Crop filters: {args.crop_filter}")
    print(f"Compression: {'enabled' if args.compression else 'disabled'}")
    print(f"Threads: {args.num_threads}")

    try:
        metadata = preprocess_and_save_volume(
            volume_index=args.volume_index,
            output_dir=args.output_dir,
            model_id=args.model_id,
            base_resolution=args.base_resolution,
            min_resolution_for_raw=args.min_resolution_for_raw,
            output_image_dim=args.output_image_dim,
            organelle_list=args.organelles,
            crop_filter=args.crop_filter,
            inference_filter=args.inference_filter,
            min_label_fraction=args.min_label_fraction,
            min_unique_ids=args.min_unique_ids,
            min_ground_truth_fraction=args.min_ground_truth_fraction,
            allow_gt_extension=not args.no_gt_extension,
            use_compression=args.compression,
            num_threads=args.num_threads,
        )

        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"Volume: {metadata['volume_name']}")
        print(f"Features: {metadata['storage']['features_gb']:.3f} GB")
        print(f"Total time: {metadata['timing']['total_seconds']:.1f}s")
        print(f"Metadata: {metadata['volume_name']}_metadata.json")

        return 0

    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR!")
        print(f"{'='*60}")
        print(f"Failed to process volume {args.volume_index}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
