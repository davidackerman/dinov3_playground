#!/usr/bin/env python
# >python submit_random_crop_jobs.py --dataset-path /nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/em/fibsem-int16 --input-resolution 16 --output-resolution 64 --num-crops 1 --output-dir /nrs/cellmap/to_delete/jrc_c-elegans-bw-1/smaller_anyup/ --queue gpu_h200
"""
Submit batch jobs to LSF for parallel random crop feature extraction.

Samples N random crops from a dataset and extracts DINOv3 features for each.
Much simpler than the standard preprocessing - just raw data → features.

Usage:
    python submit_random_crop_jobs.py \
        --output-dir /path/to/output \
        --dataset-path /path/to/raw.zarr \
        --num-crops 300 \
        --output-image-dim 128 \
        --use-anyup
"""
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import sys
import time


def submit_lsf_job(
    crop_index,
    output_dir,
    log_dir,
    script_path,
    dataset_path,
    model_id="facebook/dinov3-vitl16-pretrain-sat493m",
    input_resolution=32,
    output_resolution=128,
    output_image_dim=128,
    compression=False,
    use_anyup=True,
    save_raw=True,
    num_threads=8,
    num_processors=12,
    memory_gb=64,
    gpu=True,
    queue=None,
    use_orthogonal_planes=True,
    walltime="2:00",
):
    """
    Submit a single LSF job for random crop feature extraction.

    Parameters:
    -----------
    crop_index : int
        Crop index to process
    output_dir : str
        Output directory for preprocessed crops
    log_dir : Path
        Directory for log files (with timestamp)
    script_path : str
        Path to preprocess_dataset_random_crops.py script
    dataset_path : str
        Path to the dataset
    ... (other parameters passed to preprocessing script)

    Returns:
    --------
    str : Job ID
    """
    # Build command
    cmd = [
        "bsub",
        "-P",
        "cellmap",  # Chargeback group
        "-n",
        str(num_processors),
        "-o",
        str(log_dir / f"crop_{crop_index:06d}.out"),
        "-e",
        str(log_dir / f"crop_{crop_index:06d}.err"),
        "-J",
        f"dinov3_crop_{crop_index:06d}",
    ]

    # Add queue if specified
    if queue:
        cmd.extend(["-q", queue])

    # Add GPU request if needed
    if gpu:
        cmd.extend(["-gpu", "num=1"])
    print(cmd)
    # Build python command
    python_cmd = [
        sys.executable,  # Use same Python interpreter
        str(script_path),
        "--crop-index",
        str(crop_index),
        "--output-dir",
        str(output_dir),
        "--dataset-path",
        dataset_path,
        "--model-id",
        model_id,
        "--input-resolution",
        str(input_resolution),
        "--output-resolution",
        str(output_resolution),
        "--output-image-dim",
        str(output_image_dim),
        "--num-threads",
        str(num_threads),
    ]

    # Add flags
    if compression:
        python_cmd.append("--compression")
    if use_anyup:
        python_cmd.append("--use-anyup")
    else:
        python_cmd.append("--no-anyup")
    if not save_raw:
        python_cmd.append("--no-save-raw")
    if not use_orthogonal_planes:
        python_cmd.append("--no-use-orthogonal-planes")

    # Combine commands
    full_cmd = cmd + python_cmd

    # Submit job
    result = subprocess.run(full_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error submitting job for crop {crop_index}:")
        print(result.stderr)
        return None

    # Extract job ID from output (format: Job <12345> is submitted to default queue...)
    job_id = None
    for line in result.stdout.split("\n"):
        if "Job <" in line:
            job_id = line.split("<")[1].split(">")[0]
            break

    return job_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit batch jobs for parallel random crop feature extraction"
    )

    # Required arguments
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
    parser.add_argument(
        "--num-crops",
        type=int,
        required=True,
        help="Number of random crops to preprocess",
    )

    # Optional: specify start index
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting crop index (default: 0)",
    )

    # LSF configuration
    parser.add_argument(
        "--num-processors",
        type=int,
        default=12,
        help="Number of processors per job (-n flag)",
    )
    # parser.add_argument(
    #     "--memory-gb",
    #     type=int,
    #     default=64,
    #     help="Memory in GB per job",
    # )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Don't request GPU",
    )
    parser.add_argument(
        "--queue",
        type=str,
        default=None,
        help="Queue name (default: use default queue)",
    )
    parser.add_argument(
        "--walltime",
        type=str,
        default="2:00",
        help="Wall time limit (e.g., 2:00 for 2 hours)",
    )

    # Model configuration
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-sat493m",  # dinov3-vits16-pretrain-lvd1689m",
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

    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without submitting",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Find the preprocessing script
    script_path = Path(__file__).parent / "preprocess_dataset_random_crops.py"
    if not script_path.exists():
        print(
            f"Error: Could not find preprocess_dataset_random_crops.py at {script_path}"
        )
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / "logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("DINOv3 Random Crop Batch Feature Extraction")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Log directory: {log_dir}")
    print(f"Number of crops: {args.num_crops}")
    print(
        f"Crop indices: {args.start_index} to {args.start_index + args.num_crops - 1}"
    )
    print(f"Output image dim: {args.output_image_dim}")
    print(f"Input resolution: {args.input_resolution} nm")
    print(f"Output resolution: {args.output_resolution} nm")
    print(f"Processors per job: {args.num_processors}")
    # print(f"Memory per job: {args.memory_gb} GB")
    print(f"GPU: {'No' if args.no_gpu else 'Yes'}")
    print(f"Wall time: {args.walltime}")
    if args.queue:
        print(f"Queue: {args.queue}")
    print(f"Use AnyUp: {args.use_anyup}")
    print(f"Compression: {'enabled' if args.compression else 'disabled'}")
    print(f"Save raw: {not args.no_save_raw}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No jobs will be submitted ***")

    print(f"\n{'='*60}")
    print("Submitting jobs...")
    print(f"{'='*60}\n")

    submitted_jobs = []
    failed_submissions = []

    for i in range(args.num_crops):
        crop_index = args.start_index + i

        if args.dry_run:
            print(f"Would submit job for crop {crop_index}")
            continue

        job_id = submit_lsf_job(
            crop_index=crop_index,
            output_dir=output_dir,
            log_dir=log_dir,
            script_path=script_path,
            dataset_path=args.dataset_path,
            model_id=args.model_id,
            input_resolution=args.input_resolution,
            output_resolution=args.output_resolution,
            output_image_dim=args.output_image_dim,
            compression=args.compression,
            use_anyup=args.use_anyup,
            save_raw=not args.no_save_raw,
            num_threads=args.num_threads,
            num_processors=args.num_processors,
            # memory_gb=args.memory_gb,
            gpu=not args.no_gpu,
            queue=args.queue,
            walltime=args.walltime,
            use_orthogonal_planes=not args.no_use_orthogonal_planes,
        )
        time.sleep(1)  # Slight delay to avoid overwhelming LSF
        if job_id:
            submitted_jobs.append((crop_index, job_id))
            print(f"✓ Crop {crop_index:6d} -> Job {job_id}")
        else:
            failed_submissions.append(crop_index)
            print(f"✗ Crop {crop_index:6d} -> FAILED")

    # Summary
    print(f"\n{'='*60}")
    print("Submission Summary")
    print(f"{'='*60}")

    if args.dry_run:
        print(f"DRY RUN: Would submit {args.num_crops} jobs")
    else:
        print(f"Successfully submitted: {len(submitted_jobs)} jobs")
        if failed_submissions:
            print(f"Failed submissions: {len(failed_submissions)}")
            print(f"Failed crop indices: {failed_submissions}")

        print(f"\nJob logs will be written to: {log_dir}")
        print(f"\nMonitor jobs with: bjobs -P cellmap")
        print(f"Check job status: bjobs -P cellmap | grep dinov3_crop")
        print(f"\nWhen complete, verify with:")
        print(f"  ls {output_dir}/*_metadata.json | wc -l")

    return 0 if not failed_submissions else 1


if __name__ == "__main__":
    sys.exit(main())
