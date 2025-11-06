#!/usr/bin/env python
"""
Utility script to check preprocessing progress and verify outputs.

Usage:
    python check_preprocessing.py --output-dir /path/to/output
"""
import argparse
import json
from pathlib import Path
import numpy as np
from collections import defaultdict


def check_preprocessing_progress(output_dir):
    """
    Check the progress of preprocessing jobs and report statistics.
    
    Parameters:
    -----------
    output_dir : str or Path
        Directory containing preprocessed volumes
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        return
    
    # Find all metadata files
    metadata_files = sorted(output_dir.glob("volume_*_metadata.json"))
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Progress Report")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Completed volumes: {len(metadata_files)}")
    
    if not metadata_files:
        print("\nNo completed volumes found.")
        return
    
    # Load all metadata
    metadatas = []
    for mf in metadata_files:
        with open(mf, "r") as f:
            metadatas.append(json.load(f))
    
    # Extract volume indices
    volume_indices = sorted([m["volume_index"] for m in metadatas])
    print(f"Volume indices: {min(volume_indices)} to {max(volume_indices)}")
    
    # Check for gaps
    expected = set(range(min(volume_indices), max(volume_indices) + 1))
    actual = set(volume_indices)
    missing = expected - actual
    if missing:
        print(f"\nMissing volumes: {sorted(missing)}")
        print(f"Number missing: {len(missing)}")
    else:
        print(f"No gaps in volume indices")
    
    # Aggregate statistics
    total_time = sum(m["timing"]["total_seconds"] for m in metadatas)
    total_extraction = sum(m["timing"]["feature_extraction_seconds"] for m in metadatas)
    total_write = sum(m["timing"]["tensorstore_write_seconds"] for m in metadatas)
    total_storage = sum(m["storage"]["total_gb"] for m in metadatas)
    
    print(f"\n{'='*60}")
    print(f"Timing Statistics")
    print(f"{'='*60}")
    print(f"Total preprocessing time: {total_time/3600:.2f} hours")
    print(f"  Feature extraction: {total_extraction/3600:.2f} hours ({total_extraction/total_time*100:.1f}%)")
    print(f"  TensorStore write: {total_write/3600:.2f} hours ({total_write/total_time*100:.1f}%)")
    print(f"\nAverage per volume:")
    print(f"  Total: {total_time/len(metadatas):.1f}s")
    print(f"  Feature extraction: {total_extraction/len(metadatas):.1f}s")
    print(f"  TensorStore write: {total_write/len(metadatas):.1f}s")
    
    print(f"\n{'='*60}")
    print(f"Storage Statistics")
    print(f"{'='*60}")
    print(f"Total storage: {total_storage:.2f} GB")
    print(f"Average per volume: {total_storage/len(metadatas):.3f} GB")
    
    feature_storage = sum(m["storage"]["features_gb"] for m in metadatas)
    gt_storage = sum(m["storage"]["gt_gb"] for m in metadatas)
    mask_storage = sum(m["storage"]["mask_gb"] for m in metadatas)
    
    print(f"\nBreakdown:")
    print(f"  Features: {feature_storage:.2f} GB ({feature_storage/total_storage*100:.1f}%)")
    print(f"  GT: {gt_storage:.2f} GB ({gt_storage/total_storage*100:.1f}%)")
    print(f"  Masks: {mask_storage:.2f} GB ({mask_storage/total_storage*100:.1f}%)")
    
    # Dataset sources
    print(f"\n{'='*60}")
    print(f"Dataset Sources")
    print(f"{'='*60}")
    
    source_counts = defaultdict(int)
    for m in metadatas:
        dataset_name = str(m["source_dataset"]["paths"])
        source_counts[dataset_name] += 1
    
    print(f"Unique datasets: {len(source_counts)}")
    print(f"\nTop 5 most sampled datasets:")
    for dataset, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {count:4d} volumes: {dataset[:80]}...")
    
    # Configuration
    print(f"\n{'='*60}")
    print(f"Configuration")
    print(f"{'='*60}")
    config = metadatas[0]["configuration"]
    print(f"Model: {config['model_id']}")
    print(f"Base resolution: {config['base_resolution']} nm")
    print(f"Min resolution for raw: {config['min_resolution_for_raw']} nm")
    print(f"Output dimension: {config['output_image_dim']}")
    print(f"Compression: {config['compression']}")
    print(f"Chunks: {config['chunks']}")
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"Data Statistics")
    print(f"{'='*60}")
    
    all_classes = set()
    for m in metadatas:
        all_classes.update(m["statistics"]["unique_classes"])
    print(f"Unique classes found: {sorted(all_classes)}")
    
    avg_valid_fraction = np.mean([m["statistics"]["valid_gt_fraction"] for m in metadatas])
    print(f"Average valid GT fraction: {avg_valid_fraction:.3f}")
    
    print(f"\n{'='*60}")
    print(f"Files")
    print(f"{'='*60}")
    print(f"Feature files: {len(list(output_dir.glob('*_features.zarr')))}")
    print(f"GT files: {len(list(output_dir.glob('*_gt.zarr')))}")
    print(f"Mask files: {len(list(output_dir.glob('*_mask.zarr')))}")
    print(f"Metadata files: {len(metadata_files)}")
    
    # Check logs if available
    log_dir = output_dir / "logs"
    if log_dir.exists():
        out_files = list(log_dir.glob("*.out"))
        err_files = list(log_dir.glob("*.err"))
        
        print(f"\n{'='*60}")
        print(f"Job Logs")
        print(f"{'='*60}")
        print(f"Output logs: {len(out_files)}")
        print(f"Error logs: {len(err_files)}")
        
        # Check for errors
        errors = []
        for err_file in err_files:
            if err_file.stat().st_size > 0:  # Non-empty error file
                errors.append(err_file.name)
        
        if errors:
            print(f"\nJobs with errors: {len(errors)}")
            print(f"Error log files: {errors[:5]}")
            if len(errors) > 5:
                print(f"  ... and {len(errors)-5} more")


def verify_volume_integrity(output_dir, volume_indices=None, sample_size=5):
    """
    Verify that volumes can be loaded correctly.
    
    Parameters:
    -----------
    output_dir : str or Path
        Directory containing preprocessed volumes
    volume_indices : list or None
        Specific volume indices to verify, or None to sample randomly
    sample_size : int
        Number of volumes to verify if volume_indices is None
    """
    from dinov3_preprocessing import load_preprocessed_volume
    
    output_dir = Path(output_dir)
    metadata_files = sorted(output_dir.glob("volume_*_metadata.json"))
    
    if not metadata_files:
        print("No volumes found to verify")
        return
    
    if volume_indices is None:
        # Sample random volumes
        import random
        sample_size = min(sample_size, len(metadata_files))
        metadata_files = random.sample(metadata_files, sample_size)
    else:
        # Load specific volumes
        metadata_files = [
            output_dir / f"volume_{idx:06d}_metadata.json"
            for idx in volume_indices
        ]
    
    print(f"\n{'='*60}")
    print(f"Volume Integrity Verification")
    print(f"{'='*60}")
    print(f"Verifying {len(metadata_files)} volumes...")
    
    for i, mf in enumerate(metadata_files, 1):
        try:
            print(f"\n[{i}/{len(metadata_files)}] Loading {mf.stem}...")
            features, gt, mask, metadata = load_preprocessed_volume(mf)
            
            print(f"  ✓ Features shape: {features.shape}")
            print(f"  ✓ GT shape: {gt.shape}")
            if mask is not None:
                print(f"  ✓ Mask shape: {mask.shape}")
            print(f"  ✓ Classes: {np.unique(gt)}")
            print(f"  ✓ Feature range: [{features.min():.2f}, {features.max():.2f}]")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
    
    print(f"\n{'='*60}")
    print("Verification complete")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check preprocessing progress and verify outputs"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing preprocessed volumes",
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify volume integrity by loading samples",
    )
    
    parser.add_argument(
        "--verify-indices",
        type=int,
        nargs="+",
        help="Specific volume indices to verify",
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of random volumes to verify (default: 5)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check progress
    check_preprocessing_progress(args.output_dir)
    
    # Verify volumes if requested
    if args.verify:
        verify_volume_integrity(
            args.output_dir,
            volume_indices=args.verify_indices,
            sample_size=args.sample_size,
        )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
