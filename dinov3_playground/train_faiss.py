# %%
import sys
import importlib.util

# Load the tensorstore_loader module directly without importing the package
spec = importlib.util.spec_from_file_location(
    "tensorstore_loader",
    "/groups/cellmap/cellmap/ackermand/Programming/dinov3_playground/dinov3_playground/tensorstore_loader.py",
)
tensorstore_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tensorstore_loader)

RandomCropLoader = tensorstore_loader.RandomCropLoader

import faiss
import numpy as np

# %%
# Train on a single image
# Import directly from the module file, bypassing __init__.py
import sys
import importlib.util

# Load the tensorstore_loader module directly without importing the package
spec = importlib.util.spec_from_file_location(
    "tensorstore_loader",
    "/groups/cellmap/cellmap/ackermand/Programming/dinov3_playground/dinov3_playground/tensorstore_loader.py",
)
tensorstore_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tensorstore_loader)

RandomCropLoader = tensorstore_loader.RandomCropLoader

import faiss
import numpy as np

zarr_path = "/nrs/cellmap/to_delete/jrc_c-elegans-bw-1/smaller_anyup/crop_000000.zarr"
loader = RandomCropLoader(zarr_path)

# Load features
features = loader.load_features()
raw = loader.load_raw()
print(f"Features shape: {features.shape}")
print(f"Features dtype: {features.dtype}")

# Reshape and convert to float32
n_channels = features.shape[0]
training_samples = features.reshape(n_channels, -1).T.astype(np.float32)

print(f"Training samples shape: {training_samples.shape}")

# Train FAISS
index_pq = faiss.IndexPQ(n_channels, 8, 8)
index_pq.train(training_samples)

print("FAISS index trained successfully!")

# %%
# Convert features to 8-byte quantized codes
print("\nQuantizing features to 8-byte codes...")

# Add the features to the index (this computes the codes)
index_pq.add(training_samples)

# Get the quantized codes (8 bytes per vector)
# The codes are stored internally, but we can retrieve them
codes = np.zeros((training_samples.shape[0], index_pq.code_size), dtype=np.uint8)
index_pq.sa_encode(training_samples, codes)

print(f"Quantized codes shape: {codes.shape}")
print(f"Codes dtype: {codes.dtype}")
print(f"Code size per vector: {index_pq.code_size} bytes")

# Reshape codes back to image dimensions (D, H, W, code_size)
codes_3d = codes.reshape(
    features.shape[1], features.shape[2], features.shape[3], index_pq.code_size
)
print(f"Quantized image shape: {codes_3d.shape}")

# Save the quantized codes
output_path = zarr_path.replace(".zarr", "_quantized_pq.npy")
np.save(output_path, codes_3d)
print(f"\nSaved quantized codes to: {output_path}")

# Also save the FAISS index for later use
index_path = zarr_path.replace(".zarr", "_faiss_index.bin")
faiss.write_index(index_pq, index_path)
print(f"Saved FAISS index to: {index_path}")

# Demonstrate reconstruction
print("\nDemonstrating reconstruction...")
reconstructed = np.ascontiguousarray(np.zeros_like(training_samples))
index_pq.sa_decode(codes, reconstructed)
reconstructed_3d = reconstructed.T.reshape(features.shape)

print(f"Original features range: [{features.min():.4f}, {features.max():.4f}]")
print(
    f"Reconstructed features range: [{reconstructed_3d.min():.4f}, {reconstructed_3d.max():.4f}]"
)
print(f"Mean absolute error: {np.mean(np.abs(features - reconstructed_3d)):.6f}")


# %%
import matplotlib.pyplot as plt

# Find similar pixels using FAISS with non-maximum suppression
print("\nFinding similar pixels using FAISS with non-maximum suppression...")

# Select a random query pixel in feature space
# np.random.seed(43)
query_z = np.random.randint(0, features.shape[1])
query_y = np.random.randint(0, features.shape[2])
query_x = np.random.randint(0, features.shape[3])

print(f"Query pixel coordinates (features): Z={query_z}, Y={query_y}, X={query_x}")

# Get the feature vector at this location
query_idx = (
    query_z * features.shape[2] * features.shape[3]
    + query_y * features.shape[3]
    + query_x
)
query_vector = training_samples[query_idx : query_idx + 1]

# Search for many more candidates than we need (to apply suppression)
k_candidates = 200
distances, indices = index_pq.search(query_vector, k_candidates)

print(f"\nFound {k_candidates} candidate matches, applying non-maximum suppression...")


# Convert all candidate indices to 3D coordinates
def idx_to_coords(idx, shape):
    """Convert flat index to 3D coordinates."""
    z = idx // (shape[2] * shape[3])
    remainder = idx % (shape[2] * shape[3])
    y = remainder // shape[3]
    x = remainder % shape[3]
    return (z, y, x)


# Non-maximum suppression parameters
suppression_radius = 10  # Don't allow matches within this radius
desired_matches = 5

# Apply non-maximum suppression
selected_coords = []
selected_indices = []
selected_distances = []

for idx, dist in zip(indices[0], distances[0]):
    coords = idx_to_coords(idx, features.shape)

    # Check if this candidate is too close to any already selected point
    too_close = False
    for prev_coords in selected_coords:
        # Calculate 3D Euclidean distance
        dist_3d = np.sqrt(
            (coords[0] - prev_coords[0]) ** 2
            + (coords[1] - prev_coords[1]) ** 2
            + (coords[2] - prev_coords[2]) ** 2
        )
        if dist_3d < suppression_radius:
            too_close = True
            break

    # If not too close to any previous point, add it
    if not too_close:
        selected_coords.append(coords)
        selected_indices.append(idx)
        selected_distances.append(dist)

        # Stop once we have enough matches
        if len(selected_coords) >= desired_matches:
            break

print(
    f"\nAfter non-maximum suppression (radius={suppression_radius}): {len(selected_coords)} matches"
)
print(f"\nTop {len(selected_coords)} spatially-separated similar pixels:")

similar_coords = selected_coords
for i, (coords, idx, dist) in enumerate(
    zip(selected_coords, selected_indices, selected_distances)
):
    z, y, x = coords
    print(f"  {i}: Index {idx}: Z={z}, Y={y}, X={x}, Distance={dist:.4f}")

# %%
# Visualize on 2D slices
print("\nVisualizing similar pixels on 2D slices...")

# Choose slice to visualize (use query z coordinate)
slice_z = query_z

# Raw image has 4x resolution of features
raw_scale = 4
raw_slice_z = slice_z * raw_scale

# Create figure with raw and first 3 feature channels
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Plot raw slice
ax = axes[0]
raw_slice = raw[raw_slice_z]
raw_normalized = (raw_slice - raw_slice.min()) / (raw_slice.max() - raw_slice.min())
ax.imshow(raw_normalized, cmap="gray")
ax.set_title(f"Raw (Z={raw_slice_z})")

# Mark query point and similar points on raw
for i, (z, y, x) in enumerate(similar_coords):
    if z == slice_z:  # Only show points on this slice
        raw_y = y * raw_scale
        raw_x = x * raw_scale
        if i == 0:
            # Query point - red star
            ax.plot(raw_x, raw_y, "r*", markersize=15, label="Query")
        else:
            # Similar points - yellow circles
            ax.plot(
                raw_x,
                raw_y,
                "yo",
                markersize=8,
                markeredgecolor="red",
                markeredgewidth=1,
            )
ax.legend()
ax.axis("off")

# Plot first 3 feature channels
for feat_idx in range(3):
    ax = axes[feat_idx + 1]
    feature_slice = features[feat_idx, slice_z, :, :]

    # Normalize for display
    vmin, vmax = np.percentile(feature_slice, [1, 99])
    feature_normalized = np.clip((feature_slice - vmin) / (vmax - vmin), 0, 1)

    ax.imshow(feature_normalized, cmap="viridis")
    ax.set_title(f"Feature {feat_idx} (Z={slice_z})")

    # Mark query point and similar points on features
    for i, (z, y, x) in enumerate(similar_coords):
        if z == slice_z:  # Only show points on this slice
            if i == 0:
                # Query point - red star
                ax.plot(x, y, "r*", markersize=15, label="Query")
            else:
                # Similar points - yellow circles
                ax.plot(
                    x, y, "yo", markersize=8, markeredgecolor="red", markeredgewidth=1
                )
    ax.legend()
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    zarr_path.replace(".zarr", "_similarity_visualization.png"),
    dpi=150,
    bbox_inches="tight",
)
print(
    f"Saved visualization to: {zarr_path.replace('.zarr', '_similarity_visualization.png')}"
)
plt.show()

# %%
# Show each match in its own row with raw + 3 features
print("\nVisualization of each match point...")

# Create figure with one row per match point
n_matches = len(similar_coords)
fig, axes = plt.subplots(n_matches, 4, figsize=(20, 4 * n_matches))
if n_matches == 1:
    axes = axes.reshape(1, -1)

for match_idx, (z, y, x) in enumerate(similar_coords):
    raw_slice_z = z * raw_scale

    # Determine if this is the query point
    is_query = match_idx == 0
    match_label = (
        f"Query (rank {match_idx})"
        if is_query
        else f"Match {match_idx} (rank {match_idx})"
    )

    # Column 0: Raw
    ax = axes[match_idx, 0]
    raw_slice = raw[raw_slice_z]
    raw_normalized = (raw_slice - raw_slice.min()) / (raw_slice.max() - raw_slice.min())
    ax.imshow(raw_normalized, cmap="gray")

    # Mark the point on raw with unfilled circle
    raw_y = y * raw_scale
    raw_x = x * raw_scale
    if is_query:
        # Query point - red unfilled circle
        ax.plot(
            raw_x,
            raw_y,
            "o",
            markersize=20,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=2,
            alpha=0.8,
        )
        ax.set_title(f"{match_label}\nRaw (Z={raw_slice_z})", fontweight="bold")
    else:
        # Match point - yellow unfilled circle with red edge
        ax.plot(
            raw_x,
            raw_y,
            "o",
            markersize=16,
            markerfacecolor="none",
            markeredgecolor="yellow",
            markeredgewidth=2,
            alpha=0.7,
        )
        ax.set_title(f"{match_label}\nRaw (Z={raw_slice_z})")

    # Add coordinate text
    ax.text(
        5,
        15,
        f"({z}, {y}, {x})",
        color="white",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
    )
    ax.axis("off")

    # Columns 1-3: First 3 feature channels
    for feat_idx in range(3):
        ax = axes[match_idx, feat_idx + 1]
        feature_slice = features[feat_idx, z, :, :]

        # Normalize for display
        vmin, vmax = np.percentile(feature_slice, [1, 99])
        feature_normalized = np.clip((feature_slice - vmin) / (vmax - vmin), 0, 1)

        ax.imshow(feature_normalized, cmap="viridis")

        # Mark the point on features with unfilled circle
        if is_query:
            # Query point - red unfilled circle
            ax.plot(
                x,
                y,
                "o",
                markersize=20,
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=2,
                alpha=0.8,
            )
            ax.set_title(f"Feature {feat_idx}", fontweight="bold")
        else:
            # Match point - yellow unfilled circle
            ax.plot(
                x,
                y,
                "o",
                markersize=16,
                markerfacecolor="none",
                markeredgecolor="yellow",
                markeredgewidth=2,
                alpha=0.7,
            )
            ax.set_title(f"Feature {feat_idx}")

        ax.axis("off")

plt.tight_layout()
plt.savefig(
    zarr_path.replace(".zarr", "_similarity_individual_matches.png"),
    dpi=150,
    bbox_inches="tight",
)
print(
    f"Saved individual matches visualization to: {zarr_path.replace('.zarr', '_similarity_individual_matches.png')}"
)
plt.show()

# %%
# chatgpt attempt:
import sys
import importlib.util

# Load the tensorstore_loader module directly without importing the package
spec = importlib.util.spec_from_file_location(
    "tensorstore_loader",
    "/groups/cellmap/cellmap/ackermand/Programming/dinov3_playground/dinov3_playground/tensorstore_loader.py",
)
tensorstore_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tensorstore_loader)

RandomCropLoader = tensorstore_loader.RandomCropLoader

import faiss
import numpy as np

# ---------------------------------------------------------------------
# 1) Load features and reshape to (N_voxels, C)
# ---------------------------------------------------------------------
zarr_path = "/nrs/cellmap/to_delete/jrc_c-elegans-bw-1/anyup/crop_000000.zarr"
loader = RandomCropLoader(zarr_path)

features = loader.load_features()  # expect (C, Z, Y, X)
raw = loader.load_raw()  # for visualization if you want

print("Features shape:", features.shape)  # e.g. (1024, 128, 128, 128)
print("Features dtype:", features.dtype)

n_channels = features.shape[0]  # C
spatial_shape = features.shape[1:]  # (Z, Y, X) or (H, W)
print("Spatial shape:", spatial_shape)

# Flatten spatial dims: (C, Z, Y, X) -> (N_voxels, C)
training_samples = features.reshape(n_channels, -1).T.astype(np.float32)
print("Training samples shape:", training_samples.shape)  # (N_voxels, C)

N, d = training_samples.shape


# ---------------------------------------------------------------------
# 2) (Recommended) L2-normalize if using cosine-like features
# ---------------------------------------------------------------------
def l2_normalize(x, axis=-1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


training_samples = l2_normalize(training_samples).astype(np.float32)

# ---------------------------------------------------------------------
# 3) Build an exact Flat index to check if neighbors make sense at all
# ---------------------------------------------------------------------
print("Building exact IndexFlatIP (for sanity-check)...")
index_flat = faiss.IndexFlatIP(d)  # inner-product on normalized vectors ≈ cosine
index_flat.add(training_samples)
print("IndexFlatIP ntotal:", index_flat.ntotal)

# ---------------------------------------------------------------------
# 4) Build a more reasonable PQ index (not as insanely compressed)
# ---------------------------------------------------------------------
# 1024 dims -> try 32 subquantizers => 32 dims/subvector, 8 bits/code => 32 bytes/vector
M = 8
nbits = 8

print("Building PQ index: d={}, M={}, nbits={}".format(d, M, nbits))
index_pq = faiss.IndexPQ(d, M, nbits, faiss.METRIC_INNER_PRODUCT)
# Optionally train on a subset to speed up:
n_train = min(500_000, N)
train_idx = np.random.choice(N, size=n_train, replace=False)
train_subset = training_samples[train_idx]

print("Training PQ on subset of size:", train_subset.shape[0])
index_pq.train(train_subset)
print("PQ index trained.")

print("Adding all vectors to PQ index...")
index_pq.add(training_samples)
print("PQ ntotal:", index_pq.ntotal)


# ---------------------------------------------------------------------
# 5) Helper: voxel coord -> flat index, and flat index -> coord
# ---------------------------------------------------------------------
def coord_to_flat(z, y, x):
    return np.ravel_multi_index((z, y, x), spatial_shape)


def flat_to_coord(idx):
    return tuple(np.unravel_index(idx, spatial_shape))


# ---------------------------------------------------------------------
# 6) Example query from a chosen voxel
# ---------------------------------------------------------------------
# pick some voxel to inspect
zq, yq, xq = 64, 64, 64  # center of 128^3 volume; change as you like
q_flat = coord_to_flat(zq, yq, xq)
query_vec = training_samples[q_flat : q_flat + 1]  # shape (1, d)

k = 10

print("\n=== Exact neighbors (IndexFlatIP) ===")
D_flat, I_flat = index_flat.search(query_vec, k)
for rank, (dist, idx) in enumerate(zip(D_flat[0], I_flat[0])):
    print(f"{rank:02d}: coord={flat_to_coord(idx)}, score={dist:.4f}")

print("\n=== PQ neighbors (IndexPQ) ===")
D_pq, I_pq = index_pq.search(query_vec, k)
for rank, (dist, idx) in enumerate(zip(D_pq[0], I_pq[0])):
    print(f"{rank:02d}: coord={flat_to_coord(idx)}, score={dist:.4f}")

# %%
import torch, numpy as np

torch.from_numpy(np.array([3, 3]))
# %%

# %%
import sys
import importlib.util
import faiss

# Load the tensorstore_loader module directly without importing the package
spec = importlib.util.spec_from_file_location(
    "tensorstore_loader",
    "/groups/cellmap/cellmap/ackermand/Programming/dinov3_playground/dinov3_playground/tensorstore_loader.py",
)
tensorstore_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tensorstore_loader)

RandomCropLoader = tensorstore_loader.RandomCropLoader
# Load quantized codes and create similarity-based color-coded visualization
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

print("Loading quantized codes and raw image...")

# Load the quantized codes
zarr_path = "/nrs/cellmap/to_delete/jrc_c-elegans-bw-1/smaller_anyup/crop_000000.zarr"
codes_path = zarr_path.replace(".zarr", "_quantized_pq.npy")
codes_3d = np.load(codes_path)
print(f"Loaded codes shape: {codes_3d.shape}")

# Load raw image
loader = RandomCropLoader(zarr_path)
raw = loader.load_raw()
print(f"Raw image shape: {raw.shape}")

# Load FAISS index
index_path = zarr_path.replace(".zarr", "_faiss_index.bin")
index_pq = faiss.read_index(index_path)
print(f"Loaded FAISS index with {index_pq.ntotal} vectors")

# Choose a random point
np.random.seed(42)  # For reproducibility
query_z = np.random.randint(0, codes_3d.shape[0])
query_y = np.random.randint(0, codes_3d.shape[1])
query_x = np.random.randint(0, codes_3d.shape[2])

print(f"\nQuery point: Z={query_z}, Y={query_y}, X={query_x}")

# Get the code at this location
query_code = codes_3d[query_z, query_y, query_x]

# Reshape codes to (N_voxels, code_size)
codes_flat = codes_3d.reshape(-1, codes_3d.shape[-1])

# Decode all codes to get feature vectors
print("Decoding all quantized codes to feature vectors...")
all_features = np.ascontiguousarray(
    np.zeros((codes_flat.shape[0], index_pq.d), dtype=np.float32)
)
index_pq.sa_decode(codes_flat, all_features)

# Get query vector
query_idx = (
    query_z * codes_3d.shape[1] * codes_3d.shape[2]
    + query_y * codes_3d.shape[2]
    + query_x
)
query_vector = all_features[query_idx : query_idx + 1]

# Compute distances to all pixels
print("Computing distances to all pixels...")
# Compute squared L2 distances to all pixels directly
diff = all_features - query_vector  # broadcasts: (N, d) - (1, d)
distances_flat = np.sum(diff**2, axis=1)  # shape (N,)

distances_3d = distances_flat.reshape(
    codes_3d.shape[0],
    codes_3d.shape[1],
    codes_3d.shape[2],
)
# Normalize distances to [0, 1] for color mapping (invert so similar = bright)
# Use percentile clipping to handle outliers
min_dist = np.percentile(distances_3d, 1)
max_dist = np.percentile(distances_3d, 99)
similarity_map = 1 - np.clip((distances_3d - min_dist) / (max_dist - min_dist), 0, 1)

print(f"Distance range: [{distances_3d.min():.4f}, {distances_3d.max():.4f}]")
print(f"Similarity map range: [{similarity_map.min():.4f}, {similarity_map.max():.4f}]")

# Visualize on the slice containing the query point
raw_scale = 4  # Raw has 4x resolution
slice_z = query_z
raw_slice_z = slice_z * raw_scale

# Get slices
raw_slice = raw[raw_slice_z]
similarity_slice = similarity_map[slice_z]

# Normalize raw for display
raw_normalized = (raw_slice - raw_slice.min()) / (raw_slice.max() - raw_slice.min())

# Create color-coded similarity map (hot colormap: black=dissimilar, yellow/white=similar)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Raw with selected point circled
ax = axes[0]
ax.imshow(raw_normalized, cmap="gray")
ax.set_title(f"Raw Image (Z={raw_slice_z})", fontsize=14, fontweight="bold")

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
ax = axes[1]
im = ax.imshow(similarity_slice, cmap="hot", vmin=0, vmax=1)
ax.set_title(
    "Similarity Map\n(bright = similar to query)", fontsize=14, fontweight="bold"
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
ax = axes[2]
# Show raw in grayscale
ax.imshow(raw_normalized, cmap="gray", alpha=1.0)

# Upsample similarity map to match raw resolution using nearest neighbor
from scipy.ndimage import zoom

similarity_upsampled = zoom(similarity_slice, raw_scale, order=0)

# Create a masked version where we only show similar regions (threshold at 0.5)
similarity_masked = np.ma.masked_where(similarity_upsampled < 0.3, similarity_upsampled)

# Overlay the similarity map with transparency
im = ax.imshow(similarity_masked, cmap="hot", alpha=0.4, vmin=0, vmax=1)
ax.set_title(
    "Overlay: Similarity on Raw\n(showing pixels with similarity > 0.3)",
    fontsize=14,
    fontweight="bold",
)

# Circle the query point
circle = plt.Circle((raw_x, raw_y), radius=15, fill=False, color="cyan", linewidth=2)
ax.add_patch(circle)
ax.plot(raw_x, raw_y, "c+", markersize=12, markeredgewidth=2)
ax.axis("off")

plt.tight_layout()
output_path = zarr_path.replace(".zarr", "_similarity_colormap.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved similarity visualization to: {output_path}")
plt.show()

# %%
