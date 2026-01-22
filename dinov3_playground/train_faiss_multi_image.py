# %%
# Train FAISS on multiple 3D images by sampling from each
import sys
import importlib.util
import faiss
import numpy as np
from pathlib import Path
from glob import glob

# python submit_random_crop_jobs.py --dataset-path /nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8 --input-resolution 8 --output-resolution 32 --num-crops 10 --output-dir /nrs/cellmap/to_delete/jrc_mus-liver-zon-1/smaller_anyup_noorthogonal/ --no-use-orthogonal-planes --queue gpu_h200 --model-id facebook/dinov3-vits16-pretrain-lvd1689m
# python submit_random_crop_jobs.py --dataset-path /nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8 --input-resolution 2 --output-resolution 32 --output-image-dim 32 --num-crops 10 --output-dir /nrs/cellmap/to_delete/jrc_hela-2/smaller_anyup_noorthogonal_noupsample/ --no-use-orthogonal-planes --queue gpu_h200 --model-id facebook/dinov3-vits16-pretrain-lvd1689m
# python submit_random_crop_jobs.py --dataset-path /nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8 --input-resolution 8 --output-resolution 128 --output-image-dim 32 --num-crops 10 --output-dir /nrs/cellmap/ackermand/to_delete/jrc_hela-2/smaller_anyup_noorthogonal_noupsample_inputres_8/ --no-use-orthogonal-planes --queue gpu_h200 --model-id facebook/dinov3-vits16-pretrain-lvd1689m
# Load the tensorstore_loader module directly without importing the package
spec = importlib.util.spec_from_file_location(
    "tensorstore_loader",
    "/groups/cellmap/cellmap/ackermand/Programming/dinov3_playground/dinov3_playground/tensorstore_loader.py",
)
tensorstore_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tensorstore_loader)

RandomCropLoader = tensorstore_loader.RandomCropLoader
# %%
# Find all crop zarr files
# base_path = "/nrs/cellmap/to_delete/jrc_mus-liver-zon-1/smaller_anyup"
base_path = "/nrs/cellmap/ackermand/to_delete/jrc_mus-liver-zon-1/anyup"  # _noorthogonal_noupsample_inputres_8/"
crop_pattern = f"{base_path}/crop_*.zarr"
crop_paths = sorted(glob(crop_pattern))[:10]

print(f"Found {len(crop_paths)} crop files")
print(f"First few: {crop_paths[:3]}")
print(f"Last few: {crop_paths[-3:]}")

# %%
# Sample features from each image
total_images = len(crop_paths)


# We'll collect all samples in a list first
all_samples = []
use_all_voxels = False
for i, zarr_path in enumerate(crop_paths):
    print(f"\n[{i+1}/{total_images}] Loading {Path(zarr_path).name}...")

    try:
        loader = RandomCropLoader(zarr_path)
        features = loader.load_features()  # Shape: (C, Z, Y, X)
        output_image_size = features.shape[1] * features.shape[2] * features.shape[3]
        if i == 0:
            if output_image_size < 20000:
                use_all_voxels = True
                samples_per_image = output_image_size
            else:
                samples_per_image = 20000
            total_samples = samples_per_image * total_images

            print(
                f"\nSampling {samples_per_image:,} points from each of {total_images} images"
            )
            print(f"Total training samples: {total_samples:,}")

        n_channels = features.shape[0]
        spatial_shape = features.shape[1:]
        n_voxels = np.prod(spatial_shape)

        print(f"  Features shape: {features.shape}")
        print(f"  Total voxels: {n_voxels:,}")

        # Reshape to (n_voxels, n_channels)
        features_flat = features.reshape(n_channels, -1).T.astype(np.float32)
        if not use_all_voxels:
            # Randomly sample indices
            sample_indices = np.random.choice(
                n_voxels, size=samples_per_image, replace=False
            )
            sampled_features = features_flat[sample_indices]
        else:
            sampled_features = features_flat

        all_samples.append(sampled_features)
        print(
            f"  Sampled {sampled_features.shape[0]:,} vectors of dimension {sampled_features.shape[1]}"
        )

    except Exception as e:
        print(f"  ERROR loading {zarr_path}: {e}")
        continue

# %%
# Combine all samples
print(f"\nCombining samples from {len(all_samples)} images...")
training_samples = np.vstack(all_samples)

print(f"Training samples shape: {training_samples.shape}")
print(f"Training samples dtype: {training_samples.dtype}")
print(f"Memory usage: {training_samples.nbytes / 1e9:.2f} GB")

# %%
# Train FAISS index with Product Quantization
n_channels = training_samples.shape[1]

print(f"\nTraining FAISS PQ index...")
print(f"Feature dimension: {n_channels}")

# Use 8 subquantizers, 8 bits per code = 8 bytes per vector
# This is a good balance between compression and quality
index_pq = faiss.IndexPQ(n_channels, 8, 8)

print("Training index...")
index_pq.train(training_samples)
print("FAISS index trained successfully!")

# Add all training samples to the index
print("Adding samples to index...")
index_pq.add(training_samples)
print(f"Index now contains {index_pq.ntotal:,} vectors")

# %%
# Save the trained index
output_path = f"{base_path}/faiss_index_multi_image_{total_images}crops_{total_samples//1000}k.bin"
faiss.write_index(index_pq, output_path)
print(f"\nSaved FAISS index to: {output_path}")

# Also save metadata about which samples came from which image
metadata = {
    "crop_paths": crop_paths,
    "samples_per_image": samples_per_image,
    "total_samples": total_samples,
    "n_channels": n_channels,
}

metadata_path = output_path.replace(".bin", "_metadata.npy")
np.save(metadata_path, metadata)
print(f"Saved metadata to: {metadata_path}")

# %%
# Test the index with a random query
print("\nTesting the index with a random query...")

query_idx = np.random.randint(0, training_samples.shape[0])
query_vector = training_samples[query_idx : query_idx + 1]

k = 10
distances, indices = index_pq.search(query_vector, k)

print(f"\nQuery vector index: {query_idx}")
print(f"Top {k} similar vectors:")
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    # Determine which image this sample came from
    image_idx = idx // samples_per_image
    local_idx = idx % samples_per_image
    print(
        f"  {rank}: Index {idx} (from {Path(crop_paths[image_idx]).name}, local idx {local_idx}), Distance={dist:.4f}"
    )

# %%
# Visualize similarity in a random image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Set distance threshold (None to disable, or a float value in pixels)
# If set, regions within this spatial distance will be masked out
distance_threshold = (
    10  # Example: 5.0 to mask regions within 5 pixels of the query point
)

print("\n" + "=" * 60)
print("Visualizing similarity map for a random image")
print("=" * 60)
if distance_threshold is not None:
    print(
        f"Spatial distance threshold: {distance_threshold} pixels (masking regions within this distance)"
    )
else:
    print("Spatial distance threshold: None (showing all regions)")

# Choose a random image
random_image_idx = np.random.randint(0, len(crop_paths))
selected_zarr_path = crop_paths[random_image_idx]
print(f"\nSelected image: {Path(selected_zarr_path).name} (index {random_image_idx})")

# Load the full features and raw for this image
loader = RandomCropLoader(selected_zarr_path)
features = loader.load_features()  # Shape: (C, Z, Y, X)
raw = loader.load_raw()

print(f"Features shape: {features.shape}")
print(f"Raw shape: {raw.shape}")

# Reshape features to (n_voxels, n_channels)
n_channels = features.shape[0]
spatial_shape = features.shape[1:]  # (Z, Y, X)
features_flat = features.reshape(n_channels, -1).T.astype(np.float32)

print(f"Features reshaped to: {features_flat.shape}")

# Choose a random query point in this image
query_z = np.random.randint(0, spatial_shape[0])
query_y = np.random.randint(0, spatial_shape[1])
query_x = np.random.randint(0, spatial_shape[2])

print(f"\nQuery point: Z={query_z}, Y={query_y}, X={query_x}")

# Get the query vector
query_flat_idx = (
    query_z * spatial_shape[1] * spatial_shape[2] + query_y * spatial_shape[2] + query_x
)
query_vector = features_flat[query_flat_idx : query_flat_idx + 1]

# Compute distances to ALL pixels in this image using numpy (faster than FAISS for this)
print("Computing distances to all pixels in the image...")
diff = features_flat - query_vector  # broadcasts: (N, d) - (1, d)
distances_flat = np.sum(diff**2, axis=1)  # squared L2 distance, shape (N,)

# Reshape distances back to 3D
distances_3d = distances_flat.reshape(spatial_shape)

print(f"Feature distance range: [{distances_3d.min():.4f}, {distances_3d.max():.4f}]")

# Apply spatial distance threshold if set
if distance_threshold is not None:
    # Create a spatial distance map from the query point
    z_coords, y_coords, x_coords = np.ogrid[
        0 : spatial_shape[0], 0 : spatial_shape[1], 0 : spatial_shape[2]
    ]
    spatial_distances = np.sqrt(
        (z_coords - query_z) ** 2
        + (y_coords - query_y) ** 2
        + (x_coords - query_x) ** 2
    )
    # Create mask for pixels within the spatial distance threshold
    distance_mask = spatial_distances < distance_threshold
    print(
        f"Masking {distance_mask.sum():,} / {distance_mask.size:,} pixels ({100*distance_mask.sum()/distance_mask.size:.1f}%) within {distance_threshold} pixels of query point"
    )
    print(
        f"Spatial distance range: [{spatial_distances.min():.2f}, {spatial_distances.max():.2f}] pixels"
    )
else:
    distance_mask = None

# Normalize distances to [0, 1] for color mapping (invert so similar = bright)
# If threshold is set, only use non-masked regions for percentile calculation
if distance_threshold is not None and distance_mask is not None:
    valid_distances = distances_3d[~distance_mask]
    if len(valid_distances) > 0:
        min_dist = np.percentile(valid_distances, 1)
        max_dist = np.percentile(valid_distances, 99)
    else:
        print("Warning: All regions masked by threshold, using full range")
        min_dist = np.percentile(distances_3d, 1)
        max_dist = np.percentile(distances_3d, 99)
else:
    min_dist = np.percentile(distances_3d, 1)
    max_dist = np.percentile(distances_3d, 99)

similarity_map = 1 - np.clip((distances_3d - min_dist) / (max_dist - min_dist), 0, 1)

print(f"Distance range for scaling: [{min_dist:.4f}, {max_dist:.4f}]")
print(f"Similarity map range: [{similarity_map.min():.4f}, {similarity_map.max():.4f}]")

# Visualize on the slice containing the query point
raw_scale = raw.shape[0] // features.shape[1]  # Calculate scale factor
print(f"Raw to features scale: {raw_scale}x")

slice_z = query_z
raw_slice_z = slice_z * raw_scale

# Get slices
raw_slice = raw[raw_slice_z]
similarity_slice = similarity_map[slice_z]

# Normalize raw for display
raw_normalized = (raw_slice - raw_slice.min()) / (raw_slice.max() - raw_slice.min())

# Create visualization with 4 panels (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel 1: Raw with selected point circled
ax = axes[0, 0]
ax.imshow(raw_normalized, cmap="gray")
ax.set_title(
    f"Raw Image (Z={raw_slice_z})\n{Path(selected_zarr_path).name}",
    fontsize=14,
    fontweight="bold",
)

# Circle the query point
raw_y = query_y * raw_scale
raw_x = query_x * raw_scale
circle = plt.Circle((raw_x, raw_y), radius=15, fill=False, color="red", linewidth=2)
ax.add_patch(circle)
ax.plot(raw_x, raw_y, "r+", markersize=12, markeredgewidth=2)
ax.text(
    10,
    20,
    f"Query: ({query_z}, {query_y}, {query_x})",
    color="white",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
)
ax.axis("off")

# Panel 2: Color-coded similarity map
ax = axes[0, 1]
# Apply masking if threshold is set
if distance_threshold is not None:
    similarity_slice_display = np.ma.masked_where(
        distance_mask[slice_z], similarity_slice
    )
    title_suffix = f"\n(excluding < {distance_threshold} pixels from query)"
else:
    similarity_slice_display = similarity_slice
    title_suffix = ""

im = ax.imshow(similarity_slice_display, cmap="hot", vmin=0.9, vmax=0.95)
ax.set_title(
    f"Similarity Map{title_suffix}\n(bright = similar to query)",
    fontsize=14,
    fontweight="bold",
)

# Mark the query point
circle = plt.Circle((query_x, query_y), radius=4, fill=False, color="cyan", linewidth=2)
ax.add_patch(circle)
ax.plot(query_x, query_y, "c+", markersize=10, markeredgewidth=2)
ax.axis("off")

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Similarity", rotation=270, labelpad=20)

# Panel 3: Overlay of color-coded map on raw
ax = axes[1, 0]
# Show raw in grayscale
ax.imshow(raw_normalized, cmap="gray", alpha=1.0)

# Upsample similarity map to match raw resolution
similarity_upsampled = zoom(similarity_slice, raw_scale, order=0)

# Create a masked version
# First, mask regions that don't meet the 0.3 similarity threshold
combined_mask = similarity_upsampled < 0.3
# Then, if distance threshold is set, also mask those regions
if distance_threshold is not None:
    distance_mask_upsampled = (
        zoom(distance_mask[slice_z].astype(float), raw_scale, order=0) > 0.5
    )
    combined_mask = combined_mask | distance_mask_upsampled

similarity_masked = np.ma.masked_where(combined_mask, similarity_upsampled)

# Overlay the similarity map with transparency
im = ax.imshow(similarity_masked, cmap="hot", alpha=0.4, vmin=0.9, vmax=1.0)

if distance_threshold is not None:
    overlay_title = f"Overlay: Similarity on Raw\n(similarity > 0.3, >= {distance_threshold} pixels from query)"
else:
    overlay_title = "Overlay: Similarity on Raw\n(showing pixels with similarity > 0.3)"

ax.set_title(
    overlay_title,
    fontsize=14,
    fontweight="bold",
)

# Circle the query point
circle = plt.Circle((raw_x, raw_y), radius=15, fill=False, color="cyan", linewidth=2)
ax.add_patch(circle)
ax.plot(raw_x, raw_y, "c+", markersize=12, markeredgewidth=2)
ax.axis("off")

# Panel 4: Raw feature distance per pixel (not normalized to similarity)
ax = axes[1, 1]
distance_slice = distances_3d[slice_z]

# Apply masking if threshold is set
if distance_threshold is not None:
    distance_slice_display = np.ma.masked_where(distance_mask[slice_z], distance_slice)
    # Calculate vmin/vmax based on non-masked regions only
    valid_distances_slice = distance_slice[~distance_mask[slice_z]]
    if len(valid_distances_slice) > 0:
        vmin_dist = np.percentile(valid_distances_slice, 1)
        vmax_dist = np.percentile(valid_distances_slice, 99)
    else:
        vmin_dist = distance_slice.min()
        vmax_dist = distance_slice.max()
    title_suffix = f"\n(excluding < {distance_threshold} pixels from query)"
else:
    distance_slice_display = distance_slice
    vmin_dist = 5
    vmax_dist = 10
    title_suffix = ""

im = ax.imshow(distance_slice_display, cmap="viridis", vmin=vmin_dist, vmax=vmax_dist)
ax.set_title(
    f"Feature Distance per Pixel{title_suffix}\n(dark = similar to query)",
    fontsize=14,
    fontweight="bold",
)

# Mark the query point
circle = plt.Circle((query_x, query_y), radius=4, fill=False, color="red", linewidth=2)
ax.add_patch(circle)
ax.plot(query_x, query_y, "r+", markersize=10, markeredgewidth=2)
ax.axis("off")

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("L2 Distance", rotation=270, labelpad=20)

plt.tight_layout()
output_path = f"{base_path}/similarity_map_{Path(selected_zarr_path).stem}.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved similarity visualization to: {output_path}")
plt.show()

# %%
# Find closest matching pixel across all other crops
print("\n" + "=" * 60)
print("Finding closest matching pixel across all crops")
print("=" * 60)

# Select a random crop and query point
query_crop_idx = np.random.randint(0, len(crop_paths))
query_crop_path = crop_paths[query_crop_idx]
print(f"\nQuery crop: {Path(query_crop_path).name} (index {query_crop_idx})")

# Load the query crop
query_loader = RandomCropLoader(query_crop_path)
query_features = query_loader.load_features()  # Shape: (C, Z, Y, X)
query_raw = query_loader.load_raw()

query_spatial_shape = query_features.shape[1:]  # (Z, Y, X)

# Choose a random query point
query_z = np.random.randint(0, query_spatial_shape[0])
query_y = np.random.randint(0, query_spatial_shape[1])
query_x = np.random.randint(0, query_spatial_shape[2])

print(f"Query point: Z={query_z}, Y={query_y}, X={query_x}")

# Get the query feature vector
query_vector = query_features[:, query_z, query_y, query_x].astype(np.float32)
print(f"Query vector shape: {query_vector.shape}")

# Search through all other crops to find the best match
best_distance = float("inf")
best_match = None

for i, crop_path in enumerate(crop_paths):
    if i == query_crop_idx:
        continue  # Skip the query crop itself

    print(f"\n[{i+1}/{len(crop_paths)}] Searching in {Path(crop_path).name}...")

    try:
        loader = RandomCropLoader(crop_path)
        features = loader.load_features()  # Shape: (C, Z, Y, X)

        n_channels = features.shape[0]
        spatial_shape = features.shape[1:]

        # Reshape to (n_voxels, n_channels)
        features_flat = features.reshape(n_channels, -1).T.astype(np.float32)

        # Compute distances to all pixels
        diff = features_flat - query_vector.reshape(1, -1)
        distances = np.sum(diff**2, axis=1)  # squared L2 distance

        # Find the minimum distance
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]

        print(f"  Min distance in this crop: {min_distance:.4f}")

        # Check if this is the best match so far
        if min_distance < best_distance:
            best_distance = min_distance
            # Convert flat index to 3D coordinates
            z = min_idx // (spatial_shape[1] * spatial_shape[2])
            y = (min_idx % (spatial_shape[1] * spatial_shape[2])) // spatial_shape[2]
            x = min_idx % spatial_shape[2]

            best_match = {
                "crop_idx": i,
                "crop_path": crop_path,
                "coords": (z, y, x),
                "distance": min_distance,
                "features": features,
                "raw": loader.load_raw(),
            }
            print(
                f"  *** New best match! Distance: {min_distance:.4f}, Coords: ({z}, {y}, {x})"
            )

    except Exception as e:
        print(f"  ERROR loading {crop_path}: {e}")
        continue

if best_match is None:
    print("\nNo matches found!")
else:
    print("\n" + "=" * 60)
    print("BEST MATCH FOUND")
    print("=" * 60)
    print(f"Query crop: {Path(query_crop_path).name}")
    print(f"Query coords: ({query_z}, {query_y}, {query_x})")
    print(f"\nBest match crop: {Path(best_match['crop_path']).name}")
    print(f"Best match coords: {best_match['coords']}")
    print(f"Distance: {best_match['distance']:.4f}")

    # Visualize the query and best match side by side
    match_z, match_y, match_x = best_match["coords"]

    # Calculate raw scale
    query_raw_scale = query_raw.shape[0] // query_features.shape[1]
    match_raw_scale = best_match["raw"].shape[0] // best_match["features"].shape[1]

    # Get raw slices
    query_raw_z = query_z * query_raw_scale
    match_raw_z = match_z * match_raw_scale

    query_raw_slice = query_raw[query_raw_z]
    match_raw_slice = best_match["raw"][match_raw_z]

    # Normalize for display
    query_raw_normalized = (query_raw_slice - query_raw_slice.min()) / (
        query_raw_slice.max() - query_raw_slice.min()
    )
    match_raw_normalized = (match_raw_slice - match_raw_slice.min()) / (
        match_raw_slice.max() - match_raw_slice.min()
    )

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Query point
    ax = axes[0]
    ax.imshow(query_raw_normalized, cmap="gray")
    ax.set_title(
        f"Query Point\n{Path(query_crop_path).name}\nZ={query_raw_z}, Coords=({query_z}, {query_y}, {query_x})",
        fontsize=14,
        fontweight="bold",
    )

    # Mark the query point
    query_raw_y = query_y * query_raw_scale
    query_raw_x = query_x * query_raw_scale
    circle = plt.Circle(
        (query_raw_x, query_raw_y), radius=15, fill=False, color="red", linewidth=2
    )
    ax.add_patch(circle)
    ax.plot(query_raw_x, query_raw_y, "r+", markersize=12, markeredgewidth=2)
    ax.axis("off")

    # Panel 2: Best match point
    ax = axes[1]
    ax.imshow(match_raw_normalized, cmap="gray")
    ax.set_title(
        f"Best Match (Distance={best_match['distance']:.4f})\n{Path(best_match['crop_path']).name}\nZ={match_raw_z}, Coords={best_match['coords']}",
        fontsize=14,
        fontweight="bold",
    )

    # Mark the match point
    match_raw_y = match_y * match_raw_scale
    match_raw_x = match_x * match_raw_scale
    circle = plt.Circle(
        (match_raw_x, match_raw_y), radius=15, fill=False, color="green", linewidth=2
    )
    ax.add_patch(circle)
    ax.plot(match_raw_x, match_raw_y, "g+", markersize=12, markeredgewidth=2)
    ax.axis("off")

    plt.tight_layout()
    output_path = f"{base_path}/best_match_{Path(query_crop_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved best match visualization to: {output_path}")
    plt.show()

# %%
