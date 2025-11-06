"""
Test script to benchmark GIF generation speed improvements.

This script compares the performance of different GIF generation methods:
1. Fast imageio-based method (default, 10-50x faster)
2. ImageMagick-based method (2-5x faster than pillow)
3. Pillow-based method (original, slowest)

Run this to verify the speed improvements.
"""

import numpy as np
from funlib.persistence import Array
from funlib.geometry import Coordinate, Roi
import time
from dinov3_playground.vis import gif_2d
from pathlib import Path


# Create synthetic test data
def create_test_data(z_slices=50, size=512):
    """Create synthetic 3D data for testing."""
    # Create raw data
    raw_data = np.random.randint(0, 255, size=(z_slices, size, size), dtype=np.uint8)

    # Create label data
    labels_data = np.zeros((z_slices, size, size), dtype=np.uint16)
    for i in range(10):
        # Create random blob labels
        center_z = np.random.randint(5, z_slices - 5)
        center_y = np.random.randint(50, size - 50)
        center_x = np.random.randint(50, size - 50)
        radius = np.random.randint(20, 50)

        for z in range(
            max(0, center_z - radius // 2), min(z_slices, center_z + radius // 2)
        ):
            for y in range(max(0, center_y - radius), min(size, center_y + radius)):
                for x in range(max(0, center_x - radius), min(size, center_x + radius)):
                    if (y - center_y) ** 2 + (x - center_x) ** 2 < radius**2:
                        labels_data[z, y, x] = i + 1

    # Create Array objects
    voxel_size = Coordinate((1, 1, 1))
    roi = Roi((0, 0, 0), (z_slices, size, size))

    raw_array = Array(raw_data, roi=roi, voxel_size=voxel_size)
    labels_array = Array(labels_data, roi=roi, voxel_size=voxel_size)

    return raw_array, labels_array


def benchmark_gif_methods():
    """Benchmark different GIF creation methods."""
    print("Creating test data...")
    raw_array, labels_array = create_test_data(z_slices=30, size=512)

    output_dir = Path("benchmark_gifs")
    output_dir.mkdir(exist_ok=True)

    arrays = {
        "raw": raw_array,
        "labels": labels_array,
        "combined": (raw_array, labels_array),
    }
    array_types = {"raw": "raw", "labels": "labels", "combined": "combined"}

    results = {}

    # Test 1: Fast imageio method (default)
    print("\n1. Testing FAST IMAGEIO method...")
    start = time.time()
    gif_2d(
        arrays=arrays,
        array_types=array_types,
        filename=str(output_dir / "test_imageio.gif"),
        title="Fast ImageIO Method",
        fps=10,
        overwrite=True,
        writer="imageio",
    )
    imageio_time = time.time() - start
    results["imageio"] = imageio_time
    print(f"   ✓ Completed in {imageio_time:.2f} seconds")

    # Test 2: ImageMagick method
    print("\n2. Testing IMAGEMAGICK method...")
    start = time.time()
    gif_2d(
        arrays=arrays,
        array_types=array_types,
        filename=str(output_dir / "test_imagemagick.gif"),
        title="ImageMagick Method",
        fps=10,
        overwrite=True,
        writer="imagemagick",
        use_fast=False,  # Force matplotlib path
    )
    imagemagick_time = time.time() - start
    results["imagemagick"] = imagemagick_time
    print(f"   ✓ Completed in {imagemagick_time:.2f} seconds")

    # Test 3: Pillow method (slowest)
    print("\n3. Testing PILLOW method (original)...")
    start = time.time()
    gif_2d(
        arrays=arrays,
        array_types=array_types,
        filename=str(output_dir / "test_pillow.gif"),
        title="Pillow Method",
        fps=10,
        overwrite=True,
        writer="pillow",
        use_fast=False,  # Force matplotlib path
    )
    pillow_time = time.time() - start
    results["pillow"] = pillow_time
    print(f"   ✓ Completed in {pillow_time:.2f} seconds")

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    baseline = results["pillow"]
    for method, duration in results.items():
        speedup = baseline / duration
        print(f"{method:15s}: {duration:6.2f}s  (speedup: {speedup:.1f}x)")

    print("\n" + "=" * 60)
    print(f"Output files saved to: {output_dir}/")
    print("=" * 60)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 60)
    if "imageio" in results:
        print(f"✓ Use writer='imageio' (default) for MAXIMUM speed")
        print(
            f"  → {results['imageio']:.1f}s vs {results['pillow']:.1f}s = {baseline/results['imageio']:.1f}x faster!"
        )
    print(f"✓ Use writer='imagemagick' for good compatibility and speed")
    print(f"✓ Use writer='pillow' only if other options fail")


if __name__ == "__main__":
    try:
        benchmark_gif_methods()
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nTo use the fast imageio method, install imageio:")
        print("  conda install imageio")
        print("  # or")
        print("  pip install imageio")
