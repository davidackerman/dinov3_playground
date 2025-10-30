#!/usr/bin/env python
"""
Submit batch jobs to LSF for parallel volume preprocessing.

Usage:
    python submit_preprocessing_jobs.py \
        --output-dir /path/to/output \
        --num-volumes 300 \
        --organelles cell \
        --inference-filter jrc_mus-liver-zon-1 jrc_mus-liver-zon-2
"""
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import sys
import time


def submit_lsf_job(
    volume_index,
    output_dir,
    log_dir,
    script_path,
    model_id="facebook/dinov3-vitl16-pretrain-sat493m",
    base_resolution=128,
    min_resolution_for_raw=32,
    output_image_dim=128,
    organelles=None,
    inference_filter=None,
    crop_filter=None,
    min_label_fraction=0.01,
    min_unique_ids=2,
    min_ground_truth_fraction=0.05,
    no_gt_extension=False,
    compression=False,
    num_threads=8,
    num_processors=16,
    memory_gb=64,
    gpu=True,
    queue=None,
    walltime="2:00",
):
    """
    Submit a single LSF job for volume preprocessing.

    Parameters:
    -----------
    volume_index : int
        Volume index to process
    output_dir : str
        Output directory for preprocessed volumes
    log_dir : Path
        Directory for log files (with timestamp)
    script_path : str
        Path to preprocess_volume.py script
    num_processors : int
        Number of processors to request (-n flag)
    memory_gb : int
        Memory in GB to request
    gpu : bool
        Request GPU if True
    queue : str or None
        Queue name (if None, uses default)
    walltime : str
        Wall time limit (e.g., "2:00" for 2 hours)
    ... (other parameters passed to preprocessing script)

    Returns:
    --------
    str : Job ID
    """
    if organelles is None:
        organelles = ["cell"]

    # Build command
    cmd = [
        "bsub",
        "-P",
        "cellmap",  # Chargeback group
        "-n",
        str(num_processors),
        "-o",
        str(log_dir / f"volume_{volume_index:06d}.out"),
        "-e",
        str(log_dir / f"volume_{volume_index:06d}.err"),
        "-J",
        f"dinov3_preprocess_{volume_index:06d}",
    ]

    # Add queue if specified
    if queue:
        cmd.extend(["-q", queue])

    # Add GPU request if needed
    if gpu:
        cmd.extend(["-gpu", "num=1:gmem=24"])

    # Build python command
    python_cmd = [
        sys.executable,  # Use same Python interpreter
        str(script_path),
        "--volume-index",
        str(volume_index),
        "--output-dir",
        str(output_dir),
        "--model-id",
        model_id,
        "--base-resolution",
        str(base_resolution),
        "--min-resolution-for-raw",
        str(min_resolution_for_raw),
        "--output-image-dim",
        str(output_image_dim),
        "--min-label-fraction",
        str(min_label_fraction),
        "--min-unique-ids",
        str(min_unique_ids),
        "--min-ground-truth-fraction",
        str(min_ground_truth_fraction),
        "--num-threads",
        str(num_threads),
    ]

    # Add organelles
    python_cmd.extend(["--organelles"] + organelles)

    # Add inference filter if provided
    if inference_filter:
        python_cmd.extend(["--inference-filter"] + inference_filter)

    # Add crop filter if provided
    if crop_filter:
        python_cmd.extend(["--crop-filter"] + [str(c) for c in crop_filter])

    # Add flags
    if no_gt_extension:
        python_cmd.append("--no-gt-extension")
    if compression:
        python_cmd.append("--compression")

    # Combine commands
    full_cmd = cmd + python_cmd

    # Submit job
    result = subprocess.run(full_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error submitting job for volume {volume_index}:")
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
        description="Submit batch jobs for parallel volume preprocessing"
    )

    # Required arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save preprocessed volumes",
    )
    parser.add_argument(
        "--num-volumes",
        type=int,
        required=True,
        help="Number of volumes to preprocess",
    )

    # Optional: specify start index
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting volume index (default: 0)",
    )

    # LSF configuration
    parser.add_argument(
        "--num-processors",
        type=int,
        default=16,
        help="Number of processors per job (-n flag)",
    )
    parser.add_argument(
        "--memory-gb",
        type=int,
        default=64,
        help="Memory in GB per job",
    )
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
        "--inference-filter",
        type=str,
        nargs="+",
        default=None,
        help="Dataset filters (e.g., jrc_mus-liver-zon-1)",
    )
    parser.add_argument(
        "--crop-filter",
        type=int,
        nargs="+",
        default=None,
        help="Crop filters (dataset IDs to include)",
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
    script_path = Path(__file__).parent / "preprocess_volume.py"
    if not script_path.exists():
        print(f"Error: Could not find preprocess_volume.py at {script_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / "logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("DINOv3 Batch Preprocessing Submission")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Number of volumes: {args.num_volumes}")
    print(
        f"Volume indices: {args.start_index} to {args.start_index + args.num_volumes - 1}"
    )
    print(f"Processors per job: {args.num_processors}")
    print(f"Memory per job: {args.memory_gb} GB")
    print(f"GPU: {'No' if args.no_gpu else 'Yes'}")
    print(f"Wall time: {args.walltime}")
    if args.queue:
        print(f"Queue: {args.queue}")
    print(f"Organelles: {args.organelles}")
    if args.inference_filter:
        print(f"Inference filters: {args.inference_filter}")
    if args.crop_filter:
        print(f"Crop filters: {args.crop_filter}")
    print(f"Compression: {'enabled' if args.compression else 'disabled'}")
    print(f"Min label fraction: {args.min_label_fraction}")
    print(f"Min unique IDs: {args.min_unique_ids}")
    print(f"Min GT fraction: {args.min_ground_truth_fraction}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No jobs will be submitted ***")

    print(f"\n{'='*60}")
    print("Submitting jobs...")
    print(f"{'='*60}\n")

    submitted_jobs = []
    failed_submissions = []

    for i in range(args.num_volumes):
        volume_index = args.start_index + i

        if args.dry_run:
            print(f"Would submit job for volume {volume_index}")
            continue

        job_id = submit_lsf_job(
            volume_index=volume_index,
            output_dir=output_dir,
            log_dir=log_dir,
            script_path=script_path,
            model_id=args.model_id,
            base_resolution=args.base_resolution,
            min_resolution_for_raw=args.min_resolution_for_raw,
            output_image_dim=args.output_image_dim,
            organelles=args.organelles,
            inference_filter=args.inference_filter,
            crop_filter=args.crop_filter,
            min_label_fraction=args.min_label_fraction,
            min_unique_ids=args.min_unique_ids,
            min_ground_truth_fraction=args.min_ground_truth_fraction,
            no_gt_extension=args.no_gt_extension,
            compression=args.compression,
            num_threads=args.num_threads,
            num_processors=args.num_processors,
            memory_gb=args.memory_gb,
            gpu=not args.no_gpu,
            queue=args.queue,
            walltime=args.walltime,
        )
        time.sleep(1)  # Slight delay to avoid overwhelming LSF
        if job_id:
            submitted_jobs.append((volume_index, job_id))
            print(f"✓ Volume {volume_index:6d} -> Job {job_id}")
        else:
            failed_submissions.append(volume_index)
            print(f"✗ Volume {volume_index:6d} -> FAILED")

    # Summary
    print(f"\n{'='*60}")
    print("Submission Summary")
    print(f"{'='*60}")

    if args.dry_run:
        print(f"DRY RUN: Would submit {args.num_volumes} jobs")
    else:
        print(f"Successfully submitted: {len(submitted_jobs)} jobs")
        if failed_submissions:
            print(f"Failed submissions: {len(failed_submissions)}")
            print(f"Failed volume indices: {failed_submissions}")

        print(f"\nJob logs will be written to: {log_dir}")
        print(f"\nMonitor jobs with: bjobs -P cellmap")
        print(f"Check job status: bjobs -P cellmap | grep dinov3_preprocess")
        print(f"\nWhen complete, verify with:")
        print(f"  ls {output_dir}/*_metadata.json | wc -l")

    return 0 if not failed_submissions else 1


if __name__ == "__main__":
    sys.exit(main())
