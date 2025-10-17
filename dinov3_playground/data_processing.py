"""
Data Processing Module for DINOv3 Training

This module contains functions for:
- Data sampling and augmentation
- Image preprocessing and resizing
- Data splitting and preparation

Author: GitHub Copilot
Date: 2025-09-11
"""

import numpy as np
from scipy import ndimage
from skimage import transform, exposure
import random
from funlib.geometry import Roi
from cellmap_flow.image_data_interface import ImageDataInterface
import warnings  # Add this import at the top
from funlib.geometry import Coordinate
import numpy as np
import gc


def apply_augmentation(raw_patch, gt_patch):
    """
    Apply data augmentation to raw and ground truth patches.

    Parameters:
    -----------
    raw_patch : numpy.ndarray
        Raw image patch (2D)
    gt_patch : numpy.ndarray
        Ground truth patch (2D)

    Returns:
    --------
    tuple: (augmented_raw, augmented_gt)
    """
    import numpy as np
    from scipy import ndimage
    from skimage import exposure

    # Random rotation (0, 90, 180, 270 degrees)
    if np.random.random() < 0.5:
        k = np.random.choice([1, 2, 3])  # 90, 180, 270 degrees
        raw_patch = np.rot90(raw_patch, k)
        gt_patch = np.rot90(gt_patch, k)

    # Random flipping
    if np.random.random() < 0.5:
        raw_patch = np.fliplr(raw_patch)  # Horizontal flip
        gt_patch = np.fliplr(gt_patch)

    if np.random.random() < 0.5:
        raw_patch = np.flipud(raw_patch)  # Vertical flip
        gt_patch = np.flipud(gt_patch)

    # Intensity augmentation (only for raw image)
    if np.random.random() < 0.7:
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        raw_patch = np.clip(raw_patch * brightness_factor, 0, 255)

        # Random contrast adjustment
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean_val = np.mean(raw_patch)
            raw_patch = np.clip(
                (raw_patch - mean_val) * contrast_factor + mean_val, 0, 255
            )

    # Gaussian noise (only for raw image)
    if np.random.random() < 0.3:
        noise_std = np.random.uniform(0, 5)
        noise = np.random.normal(0, noise_std, raw_patch.shape)
        raw_patch = np.clip(raw_patch + noise, 0, 255)

    return raw_patch.astype(raw_patch.dtype), gt_patch.astype(gt_patch.dtype)


def sample_training_data(
    raw_data,
    gt_data,
    target_size=224,
    num_samples=10,
    method="flexible",
    seed=None,
    use_augmentation=True,
    return_dataset_sources=False,
):
    """
    Sample training patches from raw and ground truth data.

    Parameters:
    -----------
    raw_data : numpy.ndarray
        3D raw image data (z, y, x)
    gt_data : numpy.ndarray
        3D ground truth data (z, y, x)
    target_size : int, default=224
        Size of output patches (target_size x target_size)
    num_samples : int, default=10
        Number of patches to sample
    method : str, default="flexible"
        Sampling method: "random", "grid", or "flexible"
    seed : int, optional
        Random seed for reproducibility
    use_augmentation : bool, default=True
        Whether to apply data augmentation
    return_dataset_sources : bool, default=False
        Whether to return dataset source indices (for compatibility)

    Returns:
    --------
    tuple: (sampled_images, sampled_gt) or (sampled_images, sampled_gt, dataset_sources)
           if return_dataset_sources=True
    """
    if seed is not None:
        np.random.seed(seed)

    z_max, y_max, x_max = raw_data.shape

    # Sample patches
    sampled_images = []
    sampled_gt = []

    for i in range(num_samples):
        if method == "random":
            # Random sampling from entire volume
            z = np.random.randint(0, z_max)
            y = np.random.randint(0, max(1, y_max - target_size))
            x = np.random.randint(0, max(1, x_max - target_size))

        elif method == "grid":
            # Grid-based sampling
            grid_size = int(np.ceil(np.sqrt(num_samples)))
            row = i // grid_size
            col = i % grid_size

            z = np.random.randint(0, z_max)
            y = min(row * (y_max // grid_size), y_max - target_size)
            x = min(col * (x_max // grid_size), x_max - target_size)

        elif method == "flexible":
            # Flexible sampling with boundary handling
            z = np.random.randint(0, z_max)

            if y_max >= target_size:
                y = np.random.randint(0, y_max - target_size + 1)
            else:
                y = 0

            if x_max >= target_size:
                x = np.random.randint(0, x_max - target_size + 1)
            else:
                x = 0

        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Extract patch
        y_end = min(y + target_size, y_max)
        x_end = min(x + target_size, x_max)

        raw_patch = raw_data[z, y:y_end, x:x_end]
        gt_patch = gt_data[z, y:y_end, x:x_end]

        # Resize if necessary
        if raw_patch.shape != (target_size, target_size):
            raw_patch = transform.resize(
                raw_patch,
                (target_size, target_size),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(raw_data.dtype)

            gt_patch = transform.resize(
                gt_patch,
                (target_size, target_size),
                preserve_range=True,
                anti_aliasing=False,
                order=0,
            ).astype(gt_data.dtype)

        # Apply augmentation if requested
        if use_augmentation:
            raw_patch, gt_patch = apply_augmentation(raw_patch, gt_patch)

        sampled_images.append(raw_patch)
        sampled_gt.append(gt_patch)

    sampled_images = np.array(sampled_images)
    sampled_gt = np.array(sampled_gt)

    if return_dataset_sources:
        # For single dataset, all samples come from dataset 0
        dataset_sources = [0] * num_samples
        return sampled_images, sampled_gt, dataset_sources
    else:
        return sampled_images, sampled_gt


def apply_intensity_augmentation(raw_slice, augment_prob=0.7):
    """
    Apply intensity-based data augmentation to raw image slice.

    Parameters:
    -----------
    raw_slice : numpy.ndarray
        Input image slice
    augment_prob : float, default=0.7
        Probability of applying each augmentation

    Returns:
    --------
    numpy.ndarray: Augmented image slice
    """
    augmented = raw_slice.copy()

    if np.random.random() < augment_prob:
        # Brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        augmented = augmented * brightness_factor

    if np.random.random() < augment_prob:
        # Contrast adjustment using histogram stretching
        p2, p98 = np.percentile(augmented, (2, 98))
        augmented = exposure.rescale_intensity(augmented, in_range=(p2, p98))

    if np.random.random() < augment_prob:
        # Gamma correction
        gamma = np.random.uniform(0.8, 1.2)
        augmented = exposure.adjust_gamma(augmented, gamma)

    if np.random.random() < augment_prob:
        # Add slight Gaussian noise
        noise_std = np.random.uniform(0.01, 0.05) * np.std(augmented)
        noise = np.random.normal(0, noise_std, augmented.shape)
        augmented = augmented + noise

    return augmented


def resize_and_crop_to_target(
    raw_slice, gt_slice, target_size, resize_method="crop_or_pad", use_augmentation=True
):
    """
    Resize and crop image slices to target size with optional augmentation.

    Parameters:
    -----------
    raw_slice : numpy.ndarray
        Raw image slice
    gt_slice : numpy.ndarray
        Ground truth slice
    target_size : int
        Target output size
    resize_method : str, default='crop_or_pad'
        Method: 'resize', 'crop_or_pad', or 'random_crop'
    use_augmentation : bool, default=True
        Whether to apply augmentation

    Returns:
    --------
    tuple: (processed_raw, processed_gt)
    """
    h, w = raw_slice.shape

    if resize_method == "resize":
        # Simple resize to target size
        raw_resized = transform.resize(
            raw_slice, (target_size, target_size), preserve_range=True
        )
        gt_resized = transform.resize(
            gt_slice, (target_size, target_size), preserve_range=True, order=0
        )  # Nearest neighbor for labels

    elif resize_method == "crop_or_pad":
        # Crop or pad to target size
        raw_resized = crop_or_pad_to_size(raw_slice, target_size)
        gt_resized = crop_or_pad_to_size(gt_slice, target_size)

    elif resize_method == "random_crop" and min(h, w) >= target_size:
        # Random crop if image is large enough
        top = np.random.randint(0, h - target_size + 1)
        left = np.random.randint(0, w - target_size + 1)
        raw_resized = raw_slice[top : top + target_size, left : left + target_size]
        gt_resized = gt_slice[top : top + target_size, left : left + target_size]

    else:
        # Fall back to crop_or_pad if random_crop can't be applied
        raw_resized = crop_or_pad_to_size(raw_slice, target_size)
        gt_resized = crop_or_pad_to_size(gt_slice, target_size)

    # Apply augmentation if requested
    if use_augmentation:
        # Intensity augmentation on raw data
        raw_resized = apply_intensity_augmentation(raw_resized)

        # Geometric augmentation on both raw and GT
        if np.random.random() < 0.5:
            # Random rotation
            angle = np.random.uniform(-15, 15)
            raw_resized = ndimage.rotate(raw_resized, angle, reshape=False, order=1)
            gt_resized = ndimage.rotate(gt_resized, angle, reshape=False, order=0)

        if np.random.random() < 0.5:
            # Random horizontal flip
            raw_resized = np.fliplr(raw_resized)
            gt_resized = np.fliplr(gt_resized)

        if np.random.random() < 0.5:
            # Random vertical flip
            raw_resized = np.flipud(raw_resized)
            gt_resized = np.flipud(gt_resized)

    return raw_resized.astype(np.float32), gt_resized.astype(np.int64)


def crop_or_pad_to_size(image, target_size):
    """
    Crop or pad image to target size.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    target_size : int
        Target size

    Returns:
    --------
    numpy.ndarray: Processed image of size (target_size, target_size)
    """
    h, w = image.shape

    # Calculate padding or cropping needed
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    crop_h = max(0, h - target_size)
    crop_w = max(0, w - target_size)

    # Apply padding if needed
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        image = np.pad(
            image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect"
        )

    # Apply cropping if needed
    if crop_h > 0 or crop_w > 0:
        crop_top = crop_h // 2
        crop_left = crop_w // 2
        image = image[
            crop_top : crop_top + target_size, crop_left : crop_left + target_size
        ]

    return image


def create_image_level_split_demo():
    """
    Create a demonstration of proper image-level train/validation splitting.

    This function shows how to split data at the image level to prevent data leakage.
    """
    print("=" * 60)
    print("IMAGE-LEVEL TRAIN/VALIDATION SPLITTING DEMONSTRATION")
    print("=" * 60)

    # Simulate image sampling with image indices
    n_images = 20
    pixels_per_image = 1000

    # Create mock image indices for all pixels
    image_indices = []
    for img_idx in range(n_images):
        image_indices.extend([img_idx] * pixels_per_image)
    image_indices = np.array(image_indices)

    print(f"Total images: {n_images}")
    print(f"Pixels per image: {pixels_per_image}")
    print(f"Total pixels: {len(image_indices)}")

    # Get unique images for splitting
    unique_images = np.unique(image_indices)
    np.random.shuffle(unique_images)

    # Split images 80/20
    split_idx = int(0.8 * len(unique_images))
    train_images = unique_images[:split_idx]
    val_images = unique_images[split_idx:]

    print(f"\nImage-level split:")
    print(f"Training images: {len(train_images)} ({train_images})")
    print(f"Validation images: {len(val_images)} ({val_images})")

    # Create pixel-level masks
    train_mask = np.isin(image_indices, train_images)
    val_mask = np.isin(image_indices, val_images)

    print(f"\nPixel-level results:")
    print(f"Training pixels: {train_mask.sum()}")
    print(f"Validation pixels: {val_mask.sum()}")
    print(f"Total: {train_mask.sum() + val_mask.sum()}")

    # Verify no overlap
    overlap = np.intersect1d(train_images, val_images)
    print(f"Image overlap: {len(overlap)} (should be 0)")

    return {
        "train_images": train_images,
        "val_images": val_images,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "image_indices": image_indices,
    }


def _sample_with_random_orientations(
    dataset_pairs, crop_shape, base_resolution, min_label_fraction, max_attempts, seed
):
    """
    Sample slices from random orientations and stack them to form the target shape.
    """
    target_z, target_y, target_x = crop_shape

    # Determine the main plane (the one with thickness > 1)
    if target_z > 1 and target_y == target_x:
        # Z is the thick dimension, Y and X are the plane dimensions
        plane_size = target_y
        num_slices = target_z
        output_shape = (target_z, target_y, target_x)
    elif target_y > 1 and target_z == target_x:
        # Y is the thick dimension
        plane_size = target_z
        num_slices = target_y
        output_shape = (target_z, target_y, target_x)
    elif target_x > 1 and target_z == target_y:
        # X is the thick dimension
        plane_size = target_z
        num_slices = target_x
        output_shape = (target_z, target_y, target_x)
    else:
        # All dimensions are similar, treat as a 3D block
        print("  Treating as 3D block (no dominant plane)")
        result = _sample_single_orientation(
            dataset_pairs,
            crop_shape,
            base_resolution,
            min_label_fraction,
            max_attempts,
            seed,
        )
        if result[0] is not None:
            return result[0], result[1], [result[2]]  # Add dataset info as list
        else:
            return None, None, []

    print(
        f"  Sampling {num_slices} slices of size {plane_size}x{plane_size} from random orientations"
    )

    # Collect slices from different orientations
    raw_slices = []
    gt_slices = []
    dataset_sources = []

    for slice_idx in range(num_slices):
        print(f"  Sampling slice {slice_idx + 1}/{num_slices}...")

        # Randomly choose orientation for this slice
        orientation = np.random.choice(["xy", "xz", "yz"])

        # Define slice shape based on orientation
        if orientation == "xy":
            slice_shape = (1, plane_size, plane_size)  # (z, y, x)
        elif orientation == "xz":
            slice_shape = (plane_size, 1, plane_size)  # (z, y, x)
        elif orientation == "yz":
            slice_shape = (plane_size, plane_size, 1)  # (z, y, x)

        print(f"    Orientation: {orientation}, shape: {slice_shape}")

        # Sample this slice
        result = _sample_single_orientation(
            dataset_pairs,
            slice_shape,
            base_resolution,
            min_label_fraction,
            max_attempts // num_slices,
            seed + slice_idx if seed else None,
        )

        raw_slice, gt_slice, dataset_idx = result

        if raw_slice is None or gt_slice is None:
            print(f"    Failed to find valid slice for orientation {orientation}")
            continue

        # Reshape slice to 2D for consistent handling
        raw_2d = _extract_2d_from_3d(raw_slice, orientation)
        gt_2d = _extract_2d_from_3d(gt_slice, orientation)

        raw_slices.append(raw_2d)
        gt_slices.append(gt_2d)
        dataset_sources.append(dataset_idx)

    if len(raw_slices) == 0:
        print("  Failed to sample any valid slices")
        return None, None, []

    # If we don't have enough slices, repeat some randomly
    while len(raw_slices) < num_slices:
        idx = np.random.randint(0, len(raw_slices))
        raw_slices.append(raw_slices[idx])
        gt_slices.append(gt_slices[idx])
        dataset_sources.append(dataset_sources[idx])

    # Stack slices into the target shape
    raw_stacked, gt_stacked = _stack_slices_to_target_shape(
        raw_slices[:num_slices], gt_slices[:num_slices], output_shape
    )

    print(
        f"  Successfully stacked {len(raw_slices)} slices into shape {raw_stacked.shape}"
    )

    return raw_stacked, gt_stacked, dataset_sources[:num_slices]


def _extract_2d_from_3d(data_3d, orientation):
    """
    Extract 2D slice from 3D data based on orientation.
    """
    if orientation == "xy":
        return data_3d[0, :, :]  # Remove z dimension
    elif orientation == "xz":
        return data_3d[:, 0, :]  # Remove y dimension
    elif orientation == "yz":
        return data_3d[:, :, 0]  # Remove x dimension
    else:
        raise ValueError(f"Unknown orientation: {orientation}")


def _stack_slices_to_target_shape(raw_slices, gt_slices, target_shape):
    """
    Stack 2D slices into the target 3D shape.
    """
    target_z, target_y, target_x = target_shape

    # Initialize output arrays
    raw_output = np.zeros(target_shape, dtype=raw_slices[0].dtype)
    gt_output = np.zeros(target_shape, dtype=gt_slices[0].dtype)

    # Determine stacking direction based on target shape
    if target_z > 1 and target_y == target_x:
        # Stack along Z dimension
        for i, (raw_slice, gt_slice) in enumerate(zip(raw_slices, gt_slices)):
            # Resize slice to match Y, X dimensions if needed
            if raw_slice.shape != (target_y, target_x):
                from skimage import transform

                raw_slice = transform.resize(
                    raw_slice, (target_y, target_x), preserve_range=True
                )
                gt_slice = transform.resize(
                    gt_slice, (target_y, target_x), preserve_range=True, order=0
                )

            raw_output[i, :, :] = raw_slice
            gt_output[i, :, :] = gt_slice

    elif target_y > 1 and target_z == target_x:
        # Stack along Y dimension
        for i, (raw_slice, gt_slice) in enumerate(zip(raw_slices, gt_slices)):
            if raw_slice.shape != (target_z, target_x):
                from skimage import transform

                raw_slice = transform.resize(
                    raw_slice, (target_z, target_x), preserve_range=True
                )
                gt_slice = transform.resize(
                    gt_slice, (target_z, target_x), preserve_range=True, order=0
                )

            raw_output[:, i, :] = raw_slice
            gt_output[:, i, :] = gt_slice

    elif target_x > 1 and target_z == target_y:
        # Stack along X dimension
        for i, (raw_slice, gt_slice) in enumerate(zip(raw_slices, gt_slices)):
            if raw_slice.shape != (target_z, target_y):
                from skimage import transform

                raw_slice = transform.resize(
                    raw_slice, (target_z, target_y), preserve_range=True
                )
                gt_slice = transform.resize(
                    gt_slice, (target_z, target_y), preserve_range=True, order=0
                )

            raw_output[:, :, i] = raw_slice
            gt_output[:, :, i] = gt_slice

    return raw_output, gt_output


def _sample_single_orientation(
    dataset_pairs, crop_shape, base_resolution, min_label_fraction, max_attempts, seed
):
    """
    Sample a single crop from datasets (original functionality).
    Now returns dataset index as well.
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert crop shape to nm
    crop_shape_nm = np.array(crop_shape) * base_resolution

    # Randomly shuffle dataset order
    dataset_indices = list(range(len(dataset_pairs)))
    np.random.shuffle(dataset_indices)

    for dataset_idx in dataset_indices:
        raw_path, gt_path = dataset_pairs[dataset_idx]

        # try:
        # Initialize data interfaces
        raw_idi = ImageDataInterface(
            raw_path,
            output_voxel_size=3 * [base_resolution],
            force_pseudo_isotropic=True,
        )
        gt_idi = ImageDataInterface(
            gt_path,
            output_voxel_size=3 * [base_resolution],
            force_pseudo_isotropic=True,
            is_segmentation=True,
        )

        # Get bounds
        raw_begin = np.array(raw_idi.roi.begin)
        raw_end = np.array(raw_idi.roi.end)
        gt_begin = np.array(gt_idi.roi.begin)
        gt_end = np.array(gt_idi.roi.end)

        # Find overlapping region
        overlap_begin = np.maximum(raw_begin, gt_begin)
        overlap_end = np.minimum(raw_end, gt_end)
        overlap_shape = overlap_end - overlap_begin

        # Check if overlap is large enough for our crop
        if np.any(overlap_shape < crop_shape_nm):
            warnings.warn(
                f"Dataset {dataset_idx} overlap {overlap_shape} is smaller than crop shape {crop_shape_nm}. "
                f"ROI may extend beyond data bounds and will be padded with zeros.",
                UserWarning,
            )
            # Don't skip - continue with sampling, allowing padding

        # Calculate valid offset range (use data bounds even if smaller than crop)
        # Allow offsets that may extend beyond the data

        data_begin = overlap_begin
        data_end = overlap_end

        # For offsets, we can start from the beginning of the overlap region
        # and extend beyond if needed
        min_offset = data_begin
        max_offset = data_end - 1  # Allow at least 1 voxel overlap

        # If the data is smaller than crop in any dimension, we may need negative offsets
        # or offsets that extend beyond the data

        allow_extension_beyond_roi = False
        for i in range(3):
            if overlap_shape[i] < crop_shape_nm[i]:
                allow_extension_beyond_roi = True
                # Data is smaller than crop - center the crop on the available data
                center_offset = (
                    data_begin[i] + overlap_shape[i] / 2 - crop_shape_nm[i] / 2
                )
                min_offset[i] = (
                    center_offset - overlap_shape[i] / 4
                )  # Allow some variation
                max_offset[i] = center_offset + overlap_shape[i] / 4

        # Round to base_resolution multiples
        min_offset_aligned = (
            np.floor(min_offset / base_resolution).astype(int) * base_resolution
        )
        max_offset_aligned = (
            np.ceil(max_offset / base_resolution).astype(int) * base_resolution
        )

        # Ensure we have at least one valid offset
        if np.any(max_offset_aligned < min_offset_aligned):
            warnings.warn(
                f"Dataset {dataset_idx} has no valid aligned offsets. Using center position.",
                UserWarning,
            )
            # Use center of overlap region
            center_offset = overlap_begin + overlap_shape / 2 - crop_shape_nm / 2
            min_offset_aligned = (
                np.floor(center_offset / base_resolution).astype(int) * base_resolution
            )
            max_offset_aligned = min_offset_aligned.copy()

        # Try to find a valid crop from this dataset
        for attempt in range(max_attempts):
            # Generate random offset (multiple of base_resolution)
            offset_multiples = []
            for i in range(3):
                min_mult = min_offset_aligned[i] // base_resolution
                max_mult = max_offset_aligned[i] // base_resolution
                if min_mult <= max_mult:
                    offset_multiples.append(np.random.randint(min_mult, max_mult + 1))
                else:
                    offset_multiples.append(min_mult)

            random_offset = np.array(offset_multiples) * base_resolution

            # Create ROI (may extend beyond data bounds - that's OK)
            roi = Roi(random_offset, crop_shape_nm)

            # Check if ROI extends significantly beyond data bounds and warn
            roi_begin = np.array(roi.begin)
            roi_end = np.array(roi.end)
            extends_before = np.any(roi_begin < overlap_begin)
            extends_after = np.any(roi_end > overlap_end)

            if extends_before or extends_after:
                if allow_extension_beyond_roi:
                    warnings.warn(
                        f"Dataset {dataset_idx}: ROI {roi} extends beyond data bounds "
                        f"[{overlap_begin} to {overlap_end}]. Will be padded with zeros.",
                        UserWarning,
                    )
                else:
                    continue  # Skip this ROI and try again
            # try:
            # First check GT to see if it has enough labels
            gt_crop = gt_idi.to_ndarray_ts(roi)

            # Convert to boolean if needed and calculate label fraction
            if gt_crop.dtype != bool:
                gt_crop = gt_crop > 0

            label_fraction = np.sum(gt_crop) / gt_crop.size

            if label_fraction >= min_label_fraction:
                # Valid GT, now get raw data
                raw_crop = raw_idi.to_ndarray_ts(roi)
                return raw_crop, gt_crop.astype(bool), dataset_idx
            # except Exception as e:
            #         warnings.warn(f"Error accessing ROI {roi} in dataset {dataset_idx}: {e}", UserWarning)
            #         continue

        # If we get here, we couldn't find a valid crop with enough labels
        warnings.warn(
            f"Dataset {dataset_idx}: Could not find crop with sufficient labels "
            f"(min_label_fraction={min_label_fraction}) after {max_attempts} attempts",
            UserWarning,
        )

        # except Exception as e:
        #     warnings.warn(f"Error accessing dataset {dataset_idx}: {e}", UserWarning)
        #     continue

    return None, None, -1


def sample_from_multiple_datasets(
    dataset_pairs,
    crop_shape=(224, 224, 10),
    base_resolution=32,
    min_label_fraction=0.05,
    max_attempts=1000,
    seed=None,
    random_orientations=True,
):
    """
    Randomly sample crops from multiple datasets with validation.

    Parameters:
    -----------
    dataset_pairs : list of tuples
        List of (raw_path, gt_path) pairs
    crop_shape : tuple, default=(224, 224, 10)
        Shape of the crop in voxels (will be multiplied by base_resolution)
    base_resolution : int, default=32
        Base resolution in nm (offsets must be multiples of this)
    min_label_fraction : float, default=0.05
        Minimum fraction of non-zero labels required for valid GT
    max_attempts : int, default=100
        Maximum attempts to find a valid crop per dataset
    seed : int, optional
        Random seed for reproducibility
    random_orientations : bool, default=True
        If True, sample slices from random orientations and stack them

    Returns:
    --------
    tuple: (raw_data, gt_data, dataset_sources) where dataset_sources indicates
           which dataset each slice came from
    """
    if seed is not None:
        np.random.seed(seed)

    print(f"Sampling from {len(dataset_pairs)} datasets...")
    print(f"Target crop shape: {crop_shape} voxels")
    print(f"Base resolution: {base_resolution} nm")
    print(f"Minimum label fraction: {min_label_fraction}")
    print(f"Random orientations: {random_orientations}")

    if random_orientations:
        return _sample_with_random_orientations(
            dataset_pairs,
            crop_shape,
            base_resolution,
            min_label_fraction,
            max_attempts,
            seed,
        )
    else:
        result = _sample_single_orientation(
            dataset_pairs,
            crop_shape,
            base_resolution,
            min_label_fraction,
            max_attempts,
            seed,
        )
        if result[0] is not None:
            return result[0], result[1], [result[2]]  # Return as list for consistency
        else:
            return None, None, []


def load_random_training_data(
    dataset_pairs,
    crop_shape=(224, 224, 10),
    base_resolution=32,
    min_label_fraction=0.05,
    seed=None,
    random_orientations=True,
):
    """
    Load random training data from multiple datasets.

    Parameters:
    -----------
    dataset_pairs : list of tuples
        List of (raw_path, gt_path) pairs
    crop_shape : tuple, default=(224, 224, 10)
        Shape of the crop in voxels
    base_resolution : int, default=32
        Base resolution in nm
    min_label_fraction : float, default=0.05
        Minimum fraction of non-zero labels required
    seed : int, optional
        Random seed for reproducibility
    random_orientations : bool, default=True
        If True, sample slices from random orientations and stack them

    Returns:
    --------
    tuple: (raw_data, gt_data, dataset_sources) where dataset_sources indicates
           which dataset each slice came from
    """
    raw, gt, dataset_sources = sample_from_multiple_datasets(
        dataset_pairs=dataset_pairs,
        crop_shape=crop_shape,
        base_resolution=base_resolution,
        min_label_fraction=min_label_fraction,
        seed=seed,
        random_orientations=random_orientations,
    )

    if raw is None or gt is None:
        raise ValueError("Could not find valid training data from any dataset")

    return raw, gt, dataset_sources


# Example usage function
def get_example_dataset_pairs():
    """
    Example dataset pairs for testing. Replace with your actual dataset paths.

    Supports both legacy tuple format and new dictionary format for multiple classes.

    Returns:
    --------
    list: List of dictionaries with keys "raw" and any class names (e.g., "nuc", "mito", "vesicles")
          OR list of (raw_path, gt_path) tuples for backward compatibility
    """
    # New format supporting multiple classes with descriptive names
    dataset_pairs = [
        # Dataset 1 - Multiple classes with descriptive names
        {
            "raw": "/nrs/cellmap/data/jrc_22ak351-leaf-3mb/jrc_22ak351-leaf-3mb.zarr/recon-1/em/fibsem-uint8/s3",
            "nuc": "/groups/cellmap/cellmap/parkg/for Aubrey/3mb_s3.zarr/jrc_22ak351-leaf-3mb_nuc/s0",
            "mito": "/groups/cellmap/cellmap/parkg/for Aubrey/3mb_s3.zarr/jrc_22ak351-leaf-3mb_mito/s0",
        },
        # Add more dataset pairs here
        # {
        #     "raw": "/path/to/raw2.zarr",
        #     "nuc": "/path/to/nuclei2.zarr",
        #     "mito": "/path/to/mito2.zarr",
        #     "vesicles": "/path/to/vesicles2.zarr",
        # },
    ]  # Legacy format (will be converted internally)
    # dataset_pairs = [
    #     (
    #         "/nrs/cellmap/data/jrc_22ak351-leaf-3mb/jrc_22ak351-leaf-3mb.zarr/recon-1/em/fibsem-uint8/s3",
    #         "/groups/cellmap/cellmap/parkg/for Aubrey/3mb_s3.zarr/jrc_22ak351-leaf-3mb_nuc/s0",
    #     ),
    # ]

    return dataset_pairs


def convert_dataset_pairs_format(dataset_pairs):
    """
    Convert dataset pairs to standardized dictionary format.

    Parameters:
    -----------
    dataset_pairs : list
        List of tuples (raw_path, gt_path) or dictionaries {"raw": path, "nuc": path, "mito": path, ...}
        Class keys can have any descriptive name (e.g., "nuc", "mito", "vesicles", "class_1", etc.)

    Returns:
    --------
    list: List of dictionaries with standardized format
    """
    converted_pairs = []

    for pair in dataset_pairs:
        if isinstance(pair, tuple):
            # Legacy format: (raw_path, gt_path)
            if len(pair) == 2:
                converted_pairs.append({"raw": pair[0], "class_1": pair[1]})
            else:
                raise ValueError(
                    f"Tuple format must have exactly 2 elements (raw, gt), got {len(pair)}"
                )
        elif isinstance(pair, dict):
            # New format: already a dictionary
            if "raw" not in pair:
                raise ValueError("Dictionary format must contain 'raw' key")

            # Ensure we have at least one class (any key that's not "raw")
            class_keys = [k for k in pair.keys() if k != "raw"]
            if not class_keys:
                raise ValueError(
                    "Dictionary format must contain at least one class key (any key other than 'raw')"
                )

            converted_pairs.append(pair.copy())
        else:
            raise ValueError(f"Dataset pair must be tuple or dict, got {type(pair)}")

    return converted_pairs


def get_num_classes_from_dataset_pairs(dataset_pairs):
    """
    Determine the number of classes from dataset pairs.

    Parameters:
    -----------
    dataset_pairs : list
        List of dataset dictionaries or tuples

    Returns:
    --------
    int: Number of classes (including background)
    """
    converted_pairs = convert_dataset_pairs_format(dataset_pairs)

    if not converted_pairs:
        return 2  # Default to binary classification

    # Find maximum number of classes across all datasets
    max_classes = 0
    for pair in converted_pairs:
        # Count only organelle/class keys, not raw or context keys
        class_keys = [
            k
            for k in pair.keys()
            if k != "raw" and not (k.startswith("raw_") and "nm" in k)
        ]
        max_classes = max(max_classes, len(class_keys))

    # Add 1 for background class (class 0)
    return max_classes + 1


def get_class_names_from_dataset_pairs(dataset_pairs):
    """
    Extract class names from dataset pairs.

    Parameters:
    -----------
    dataset_pairs : list
        List of dataset dictionaries or tuples

    Returns:
    --------
    list: List of class names, with "background" as first element
    """
    converted_pairs = convert_dataset_pairs_format(dataset_pairs)

    if not converted_pairs:
        return ["background", "foreground"]  # Default binary classification

    # Collect all unique class keys across all datasets
    all_class_keys = set()
    for pair in converted_pairs:
        # Get only organelle/class keys, not raw or context keys
        class_keys = [
            k
            for k in pair.keys()
            if k != "raw" and not (k.startswith("raw_") and "nm" in k)
        ]
        all_class_keys.update(class_keys)

    # Sort alphabetically for consistent ordering
    sorted_class_keys = sorted(all_class_keys)

    # Return with background as first class
    return ["background"] + sorted_class_keys


# %%
# Load random training data from multiple datasets (3D volumes)
def load_random_3d_training_data(
    dataset_pairs,
    volume_shape,
    base_resolution,
    min_label_fraction=0.05,
    num_volumes=10,
    seed=42,
    dinov3_stride=None,
    min_resolution_for_raw=None,
    allow_gt_extension=False,  # NEW PARAMETER
    context_scale=None,  # NEW: Load context data at this resolution
    min_unique_ids=None,  # NEW: Minimum number of unique instance IDs required
    allow_smaller_overlap=True,  # NEW: Allow overlap smaller than requested volume size
    augment=False,  # If True, apply random augmentations to raw (and context) data
    augment_prob=0.5,  # Per-volume probability to apply augmentation
    augment_params=None,  # Optional dict to control augmentation distributions
):
    """
    Load random 3D volumes from multiple datasets for 3D UNet training.
    Supports both legacy tuple format and new dictionary format for multiple classes.

    Parameters:
    -----------
    dataset_pairs : list
        List of dictionaries with keys {"raw": path, "class_1": path, "class_2": path, ...}
        OR list of (raw_path, gt_path) tuples for backward compatibility
    volume_shape : tuple
        3D shape of volumes to sample (D, H, W), e.g., (64, 64, 64)
    base_resolution : int
        Resolution in nm for sampling
    min_label_fraction : float
        Minimum fraction of positive labels required
    num_volumes : int
        Number of 3D volumes to sample
    seed : int
        Random seed
    dinov3_stride : int, optional
        If provided, will add ROI-level padding for sliding window inference
        to avoid boundary issues. Padding = 16 - stride pixels per spatial dimension.
    min_resolution_for_raw : int, default=16
        Minimum resolution (in nm) for raw data. If base_resolution is lower,
        raw data will be sampled at this minimum resolution to save memory.
    allow_gt_extension : bool, default=False
        If True, allows sampling of raw volumes that extend beyond ground truth regions.
        This provides more context for training but requires using masks for loss calculation.
        When True, returns gt_masks indicating valid ground truth regions.
    min_unique_ids : int, optional
        Minimum number of unique instance IDs required in the ground truth region.
        Useful for affinity training where you want boundary examples between multiple instances.
        For example, min_unique_ids=2 ensures at least 2 different cell IDs (excluding background),
        avoiding regions deep within a single cell that only have one ID.
        If None (default), no minimum ID check is performed.
    allow_smaller_overlap : bool, default=True
        If True, allows sampling from datasets where the overlap region is smaller than the
        requested volume size. The actual sampled volume will be adjusted to fit within the
        available overlap region. This is useful when working with datasets of varying sizes.
        If False, datasets with insufficient overlap will be skipped with a warning message.

    Returns:
    --------
    tuple: (raw_volumes, gt_volumes, dataset_sources, num_classes) or
           (raw_volumes, gt_volumes, gt_masks, dataset_sources, num_classes)
        - raw_volumes: np.array of shape (num_volumes, D, H, W) or (num_volumes, D+pad, H+pad, W+pad) if padded
        - gt_volumes: np.array of shape (num_volumes, D, H, W) with class indices (always unpadded)
        - gt_masks: np.array of shape (num_volumes, D, H, W) with 1 where GT is valid, 0 elsewhere (only if allow_gt_extension=True)
        - dataset_sources: list of dataset indices
        - num_classes: int, total number of classes including background
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate padding requirements for sliding window inference
    roi_padding = 0
    if dinov3_stride is not None and dinov3_stride < 16:
        roi_padding = 16 - dinov3_stride
        print(
            f"ROI-level padding enabled: {roi_padding} pixels per spatial dimension for stride={dinov3_stride}"
        )

    # Convert dataset pairs to standardized format
    converted_pairs = convert_dataset_pairs_format(dataset_pairs)
    num_classes = get_num_classes_from_dataset_pairs(dataset_pairs)

    print(
        f"Dataset format conversion complete. Found {num_classes} classes (including background)."
    )
    print(
        f"Note: Datasets with TensorStore/Zarr compatibility issues will be automatically skipped."
    )

    # Convert volume shape to nm
    volume_shape_nm = np.array(volume_shape) * base_resolution

    # Calculate padded volume shape for ROI sampling (only for raw data)
    padded_volume_shape_nm = volume_shape_nm.copy()
    if roi_padding > 0:
        # Add padding to spatial dimensions (H, W) but not depth (D)
        padded_volume_shape_nm[1:] += 2 * roi_padding * base_resolution
        print(f"Original volume shape (nm): {volume_shape_nm}")
        print(f"Padded ROI shape (nm): {padded_volume_shape_nm}")
        print(
            f"Padding per side: {roi_padding * base_resolution} nm ({roi_padding} pixels at {base_resolution}nm resolution)"
        )

    # initialize as numpy arrays
    raw_volume_shape = (
        int(base_resolution / min_resolution_for_raw)
        * np.array(volume_shape, dtype=int)
        if min_resolution_for_raw
        else volume_shape
    )
    raw_volumes = np.empty((num_volumes, *raw_volume_shape), dtype=np.uint16)
    gt_volumes = np.empty((num_volumes, *volume_shape), dtype=np.uint8)
    gt_masks = (
        np.empty((num_volumes, *volume_shape), dtype=np.uint8)
        if allow_gt_extension
        else None
    )
    # Context volumes at lower resolution (if context_scale provided).
    # Preallocate to avoid list growth and retain predictable memory usage.
    if context_scale is not None:
        # Use object dtype because some entries may be None when context is unavailable
        context_volumes = np.empty((num_volumes,), dtype=object)
        # Initialize to None explicitly
        for i in range(num_volumes):
            context_volumes[i] = None
    else:
        context_volumes = None

    # Preallocate dataset sources as integer array to avoid repeated appends
    dataset_sources = np.empty((num_volumes,), dtype=np.int32)

    print(
        f"Sampling {num_volumes} volumes of shape {volume_shape} from {len(converted_pairs)} datasets..."
    )

    volumes_collected = 0
    max_attempts = num_volumes * 10  # Allow multiple attempts

    # Add progress bar for data loading
    from tqdm import tqdm

    with tqdm(total=num_volumes, desc="Loading training volumes", unit="vol") as pbar:
        for attempt in range(max_attempts):
            if volumes_collected >= num_volumes:
                break

            # Randomly select a dataset
            dataset_idx = np.random.randint(0, len(converted_pairs))
            dataset_dict = converted_pairs[dataset_idx]
            raw_path = dataset_dict["raw"]

            # try:
            # Initialize raw data interface
            # Initialize raw data interface
            if min_resolution_for_raw:
                raw_idi = ImageDataInterface(
                    raw_path,
                    output_voxel_size=3 * [min_resolution_for_raw],
                    force_pseudo_isotropic=True,
                )
            else:
                raw_idi = ImageDataInterface(
                    raw_path, output_voxel_size=3 * [base_resolution]
                )

            # Initialize context data interface if requested
            context_idi = None
            if context_scale is not None:
                # Find the best matching raw data path for the context resolution
                # Look for keys like "raw_64nm", "raw_32nm", etc.
                context_keys = [
                    k for k in dataset_dict.keys() if k.startswith("raw_") and "nm" in k
                ]

                if context_keys:
                    # Find the context key that best matches our desired context_scale
                    best_context_key = None
                    min_difference = float("inf")

                    for key in context_keys:
                        # Extract resolution from key (e.g., "raw_64nm" -> 64)
                        try:
                            key_resolution = int(
                                key.replace("raw_", "").replace("nm", "")
                            )
                            difference = abs(key_resolution - context_scale)
                            if difference < min_difference:
                                min_difference = difference
                                best_context_key = key
                        except ValueError:
                            continue

                    if best_context_key is not None:
                        actual_resolution = int(
                            best_context_key.replace("raw_", "").replace("nm", "")
                        )
                        if actual_resolution != context_scale:
                            print(
                                f"    ℹ️  Context: requested {context_scale}nm, using closest available {actual_resolution}nm"
                            )

                        context_idi = ImageDataInterface(
                            dataset_dict[best_context_key],
                            output_voxel_size=3
                            * [context_scale],  # Resample to desired resolution
                            force_pseudo_isotropic=True,
                        )

            # Initialize all class data interfaces (any key that's not "raw" and not context)
            class_keys = [
                k
                for k in dataset_dict.keys()
                if k != "raw" and not (k.startswith("raw_") and "nm" in k)
            ]
            class_idis = {}
            for class_key in class_keys:
                class_idi = ImageDataInterface(
                    dataset_dict[class_key],
                    output_voxel_size=3 * [base_resolution],
                    force_pseudo_isotropic=True,
                    is_segmentation=True,
                )
                if allow_gt_extension:
                    dtype = class_idi._ds.dtype
                    custom_fill_value = np.iinfo(dtype).max
                    del class_idi
                    class_idis[class_key] = ImageDataInterface(
                        dataset_dict[class_key],
                        output_voxel_size=3 * [base_resolution],
                        force_pseudo_isotropic=True,
                        is_segmentation=True,
                        custom_fill_value=custom_fill_value,
                    )

            # Get bounds from raw data
            raw_begin = np.array(raw_idi.roi.begin)
            raw_end = np.array(raw_idi.roi.end)

            if allow_gt_extension:
                # When allowing GT extension, use raw data bounds for sampling
                # but ensure center of volume intersects with GT regions
                raw_overlap_begin = raw_begin.copy()
                raw_overlap_end = raw_end.copy()

                # Find GT overlap region (intersection of all GT datasets)
                gt_overlap_begin = raw_begin.copy()
                gt_overlap_end = raw_end.copy()
                for class_key, class_idi in class_idis.items():
                    class_begin = np.array(class_idi.roi.begin)
                    class_end = np.array(class_idi.roi.end)
                    gt_overlap_begin = np.maximum(gt_overlap_begin, class_begin)
                    gt_overlap_end = np.minimum(gt_overlap_end, class_end)

                gt_overlap_shape = gt_overlap_end - gt_overlap_begin

                # Check if GT overlap region exists (center constraint only needs a single voxel)
                if np.any(gt_overlap_shape <= 0):
                    print(
                        f"  Dataset {dataset_idx}: No GT overlap region found - GT regions don't intersect"
                    )
                    continue

                # Use raw bounds for sampling, but constrain center to GT region
                overlap_begin = raw_overlap_begin
                overlap_end = raw_overlap_end
                overlap_shape = overlap_end - overlap_begin

                # Store GT bounds for mask creation later
                dataset_gt_bounds = (gt_overlap_begin, gt_overlap_end)

            else:
                # Original behavior: find overlapping region across all datasets
                overlap_begin = raw_begin.copy()
                overlap_end = raw_end.copy()

                for class_key, class_idi in class_idis.items():
                    class_begin = np.array(class_idi.roi.begin)
                    class_end = np.array(class_idi.roi.end)
                    overlap_begin = np.maximum(overlap_begin, class_begin)
                    overlap_end = np.minimum(overlap_end, class_end)

                overlap_shape = overlap_end - overlap_begin
                dataset_gt_bounds = None

            # Always use the full padded volume shape - allow it to extend beyond overlap if needed
            required_shape = padded_volume_shape_nm

            # Calculate valid sampling region allowing volumes to extend beyond overlap
            # The constraint is that the volume should have SOME overlap with the data
            # but doesn't need to be fully contained within it

            # For minimum offset: allow volume to start before overlap_begin as long as it reaches into overlap
            # For maximum offset: allow volume to extend beyond overlap_end as long as it starts within overlap
            min_offset = (
                overlap_begin - required_shape + base_resolution
            )  # At least 1 voxel overlap
            max_offset = overlap_end - base_resolution  # At least 1 voxel overlap

            # Ensure we have a valid sampling region
            if np.any(max_offset < min_offset):
                print(
                    f"  Dataset {dataset_idx}: no valid sampling region even with extension allowed"
                )
                print(
                    f"    Overlap: {overlap_begin} to {overlap_end} (shape: {overlap_shape})"
                )
                print(f"    Required shape: {required_shape}")
                continue

            if allow_gt_extension and dataset_gt_bounds is not None:
                # Generate random offset ensuring volume center falls within GT bounds
                gt_begin, gt_end = dataset_gt_bounds
                volume_center_offset = (
                    np.array(required_shape) // 2
                )  # Use adjusted required_shape

                # Calculate constraints for center to be within GT region
                center_min = gt_begin
                center_max = gt_end

                # Calculate corresponding volume start constraints
                vol_min = center_min - volume_center_offset
                vol_max = center_max - volume_center_offset

                # Intersect with original sampling bounds
                vol_min = np.maximum(vol_min, min_offset)
                vol_max = np.minimum(vol_max, max_offset)

                # Generate random offset (aligned to base_resolution)
                random_offset = []
                for i in range(3):
                    min_mult = int(vol_min[i] // base_resolution)
                    max_mult = int(vol_max[i] // base_resolution)
                    if max_mult < min_mult:
                        # Fallback to original bounds if constraints are too tight
                        min_mult = int(min_offset[i] // base_resolution)
                        max_mult = int(max_offset[i] // base_resolution)
                    offset_mult = np.random.randint(min_mult, max_mult + 1)
                    random_offset.append(offset_mult * base_resolution)

                random_offset = np.array(random_offset)
            else:
                # Original behavior: generate random offset (aligned to base_resolution)
                random_offset = []
                for i in range(3):
                    min_mult = int(min_offset[i] // base_resolution)
                    max_mult = int(max_offset[i] // base_resolution)
                    offset_mult = np.random.randint(min_mult, max_mult + 1)
                    random_offset.append(offset_mult * base_resolution)

                random_offset = np.array(random_offset)

            # Create ROIs for 3D volume
            # Use adjusted required_shape for raw data (includes padding if specified)
            padded_roi = Roi(random_offset, required_shape)

            # Calculate the actual volume shape (without padding)
            # If required_shape was adjusted down from padded_volume_shape_nm, we need to compute
            # the corresponding unpadded GT shape
            if roi_padding > 0:
                # Calculate what the GT shape should be given the actual sampled size
                # required_shape includes padding, so subtract it to get GT shape
                actual_gt_shape = required_shape.copy()
                actual_gt_shape[1:] -= 2 * roi_padding * base_resolution

                # Center the GT ROI within the padded ROI
                gt_offset = random_offset + np.array(
                    [
                        0,
                        roi_padding * base_resolution,
                        roi_padding * base_resolution,
                    ]
                )
                gt_roi = Roi(gt_offset, actual_gt_shape)
            else:
                # No padding, so GT offset is the same as random offset
                gt_offset = random_offset
                gt_roi = padded_roi  # Same as padded when no padding

            # Sample all class volumes without padding (maintain target shape)
            class_volumes = {}
            failed_classes = []

            crop_failed = False
            # Print GT ROI information once
            if class_idis:
                print(
                    f"    📦 GT ROI: {gt_roi.begin} → {gt_roi.end} (shape: {gt_roi.shape} nm)"
                )
                print(
                    f"       Resolution: {base_resolution}nm, Expected voxels: {gt_roi.shape / base_resolution}"
                )

            for class_key, class_idi in class_idis.items():
                try:
                    class_volumes[class_key] = class_idi.to_ndarray_ts(gt_roi)
                except (ValueError, RuntimeError, OSError, IndexError) as e:
                    # Any error with any class should skip the entire crop attempt
                    # since we need all classes present per ROI
                    dataset_name = "unknown"
                    crop_info = "unknown"

                    if "raw" in dataset_dict:
                        raw_path = dataset_dict["raw"]
                        path_parts = raw_path.split("/")
                        for part in path_parts:
                            if part.startswith("jrc_") or "cellmap" in part:
                                dataset_name = part
                                break

                    if class_key in dataset_dict:
                        class_path = dataset_dict[class_key]
                        if "crop" in class_path:
                            crop_parts = class_path.split("crop")
                            if len(crop_parts) > 1:
                                crop_num = crop_parts[-1].split("/")[0]
                                crop_info = f"crop{crop_num}"

                    error_type = (
                        "Invalid ROI coordinates"
                        if isinstance(e, IndexError)
                        else "TensorStore/zarr compatibility error"
                    )
                    print(
                        f"  Dataset {dataset_idx}: {error_type} for '{class_key}' in '{dataset_name}', {crop_info} - skipping this crop"
                    )
                    print(f"    ROI: {gt_roi}")
                    print(f"    Error: {str(e)[:100]}...")
                    crop_failed = True
                    break

            # Skip this attempt if any class failed
            if crop_failed:
                continue

            # Skip this dataset entirely if no classes could be loaded
            if not class_volumes:
                print(
                    f"    ❌ All organelles failed for dataset {dataset_idx} - skipping entirely"
                )
                continue

            # Create multi-class ground truth volume
            # Initialize with background (class 0)
            gt_volume = np.zeros(volume_shape, dtype=np.uint8)

            # Assign class labels (assign sequential class numbers starting from 1)
            # Sort class keys alphabetically for consistent ordering - only use successfully loaded classes
            successfully_loaded_keys = [
                key for key in class_keys if key in class_volumes
            ]
            sorted_class_keys = sorted(successfully_loaded_keys)
            class_fractions = (
                {}
            )  # Store class masks for later label fraction calculation

            if failed_classes:
                print(
                    f"    ℹ️  Processing {len(sorted_class_keys)} organelles (skipped {len(failed_classes)}: {failed_classes})"
                )

            for class_idx, class_key in enumerate(sorted_class_keys, start=1):
                class_vol = class_volumes[class_key]

                # Ensure class_vol matches expected volume shape
                if class_vol.shape != gt_volume.shape:
                    # Extract dataset and crop information for warning
                    dataset_name = "unknown"
                    crop_info = "unknown"

                    # Try to extract dataset name from the dataset_dict
                    if "raw" in dataset_dict:
                        raw_path = dataset_dict["raw"]
                        path_parts = raw_path.split("/")
                        for part in path_parts:
                            if part.startswith("jrc_") or "cellmap" in part:
                                dataset_name = part
                                break

                    # Try to extract crop information from the class path
                    if class_key in dataset_dict:
                        class_path = dataset_dict[class_key]
                        if "crop" in class_path:
                            crop_parts = class_path.split("crop")
                            if len(crop_parts) > 1:
                                crop_num = crop_parts[-1].split("/")[0]
                                crop_info = f"crop{crop_num}"

                    print(
                        f"    ⚠️  WARNING: Shape mismatch for organelle '{class_key}' in dataset '{dataset_name}', {crop_info}:"
                    )
                    print(f"       Expected: {gt_volume.shape}, Got: {class_vol.shape}")
                    print(
                        f"       Likely due to asymmetric voxel dimensions in zarr data - resizing to match target shape"
                    )

                    # Resize to match target shape
                    from scipy.ndimage import zoom

                    print(np.unique(class_vol))
                    zoom_factors = np.array(gt_volume.shape) / np.array(class_vol.shape)

                    # For instance segmentation (integer dtype), preserve instance IDs
                    # For binary segmentation (boolean/float), threshold to binary
                    if class_vol.dtype in [
                        np.uint8,
                        np.uint16,
                        np.uint32,
                        np.uint64,
                        np.int32,
                        np.int64,
                    ]:
                        # Instance segmentation - use nearest neighbor and preserve IDs
                        class_vol = zoom(class_vol, zoom_factors, order=0)
                        class_vol = class_vol.astype(class_volumes[class_key].dtype)
                    else:
                        # Binary segmentation - zoom and threshold
                        class_vol = (
                            zoom(class_vol.astype(float), zoom_factors, order=0) > 0.5
                        )
                        class_vol = class_vol.astype(class_volumes[class_key].dtype)
                    print(np.unique(class_vol))
                # IMPORTANT: For single-class instance segmentation (e.g., for affinity training),
                # preserve the original instance IDs instead of converting to class labels
                if len(sorted_class_keys) == 1 and class_vol.dtype in [
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                    np.int32,
                    np.int64,
                ]:
                    # Single class with integer values - likely instance segmentation
                    # Preserve original instance IDs for affinity training
                    # Just copy the instance IDs directly (background=0 is preserved)
                    gt_volume = class_vol.copy()
                    if allow_gt_extension:
                        # Create GT mask where instance IDs are valid (not equal to custom fill value)
                        custom_fill_value = np.iinfo(gt_volume.dtype).max
                        gt_volume[gt_volume == custom_fill_value] = 0

                    class_mask = gt_volume > 0  # For label fraction calculation
                else:
                    current_class_vol = class_vol.copy()
                    if allow_gt_extension:
                        # Create GT mask where instance IDs are valid (not equal to custom fill value)
                        custom_fill_value = np.iinfo(current_class_vol.dtype).max
                        current_class_vol[current_class_vol == custom_fill_value] = 0

                    # Multi-class segmentation: convert to class labels
                    # Convert to boolean mask
                    if current_class_vol.dtype != bool:
                        class_mask = current_class_vol > 0
                    else:
                        class_mask = current_class_vol

                    # Assign class label where mask is True
                    # Later classes override earlier ones in overlapping regions
                    gt_volume[class_mask] = class_idx

                # Store class fraction for later calculation (after GT mask is created)
                class_fractions[class_key] = class_mask
            # Sample raw volume with padding (for sliding window context)
            try:
                raw_volume = raw_idi.to_ndarray_ts(padded_roi)

                # Print actual ROI coordinates being used
                print(
                    f"    📍 Raw ROI: {padded_roi.begin} → {padded_roi.end} (shape: {padded_roi.shape} nm)"
                )
                print(
                    f"       Resolution: {raw_idi.output_voxel_size[0]}nm, Expected voxels: {padded_roi.shape / raw_idi.output_voxel_size[0]}"
                )
                print(f"       Actual shape: {raw_volume.shape}")

            except (ValueError, RuntimeError, OSError, IndexError) as e:
                print(
                    f"  Dataset {dataset_idx}: Invalid ROI coordinates - skipping this crop"
                )
                print(f"    ROI: {padded_roi}")
                print(f"    Error: {str(e)[:100]}...")
                continue

            # Sample context volume if context data interface exists
            context_volume = None
            if context_idi is not None:
                try:
                    # Create expanded ROI for context to cover larger area at lower resolution
                    # We want same voxel dimensions but covering larger physical area
                    raw_resolution = raw_idi.output_voxel_size[0]  # e.g., 4nm
                    context_resolution = context_idi.output_voxel_size[0]  # e.g., 32nm
                    resolution_ratio = (
                        context_resolution / raw_resolution
                    )  # e.g., 32/4 = 8x

                    # Expand ROI by resolution ratio to cover larger area
                    roi_center = padded_roi.begin + padded_roi.shape // 2
                    expanded_shape = padded_roi.shape * resolution_ratio
                    expanded_begin = roi_center - expanded_shape // 2
                    context_roi = Roi(expanded_begin, expanded_shape)

                    # Sample from expanded ROI at context resolution
                    context_volume = context_idi.to_ndarray_ts(context_roi)

                    # Print context ROI coordinates
                    print(
                        f"    🌐 Context ROI: {context_roi.begin} → {context_roi.end} (shape: {context_roi.shape} nm)"
                    )
                    print(
                        f"       Resolution: {context_idi.output_voxel_size[0]}nm, Expected voxels: {context_roi.shape / context_idi.output_voxel_size[0]}"
                    )
                    print(
                        f"       Raw coverage ratio: {resolution_ratio:.1f}x larger spatial area"
                    )
                    print(f"       Actual shape: {context_volume.shape}")

                    # Context should naturally have same voxel dimensions as raw due to resolution difference
                    # If not, resample to match raw volume dimensions
                    if context_volume.shape != raw_volume.shape:
                        raise Exception(
                            f"Context shape {context_volume.shape} does not match raw shape {raw_volume.shape}"
                        )
                        # from scipy.ndimage import zoom

                        # zoom_factors = np.array(raw_volume.shape) / np.array(
                        #     context_volume.shape
                        # )
                        # context_volume = zoom(
                        #     context_volume.astype(float), zoom_factors, order=1
                        # context_volume = context_volume.astype(raw_volume.dtype)

                except (ValueError, RuntimeError, OSError, IndexError) as e:
                    print(
                        f"  Dataset {dataset_idx}: Failed to load context data - continuing without context"
                    )
                    print(
                        f"    Context ROI: {context_roi if 'context_roi' in locals() else 'undefined'}"
                    )
                    print(f"    Error: {str(e)[:100]}...")
                    context_volume = None

            # Validate shapes
            expected_raw_shape = (
                tuple(padded_volume_shape_nm // raw_idi.voxel_size[0])
                if roi_padding > 0
                else tuple(
                    np.array(volume_shape)
                    * base_resolution
                    // raw_idi.output_voxel_size[0]
                )
            )
            if gt_volume.shape != volume_shape:
                print(
                    f"  Dataset {dataset_idx}: GT shape mismatch - got: {gt_volume.shape}, expected: {volume_shape}"
                )
                continue

            if raw_volume.shape != expected_raw_shape:
                # Extract dataset info for warning
                dataset_name = "unknown"
                crop_info = "unknown"

                if "raw" in dataset_dict:
                    raw_path = dataset_dict["raw"]
                    path_parts = raw_path.split("/")
                    for part in path_parts:
                        if part.startswith("jrc_") or "cellmap" in part:
                            dataset_name = part
                            break

                print(
                    f"    ⚠️  WARNING: Raw data shape mismatch in dataset '{dataset_name}':"
                )
                print(f"       Expected: {expected_raw_shape}, Got: {raw_volume.shape}")
                print(
                    f"       Likely due to asymmetric voxel dimensions in zarr data - resizing to match target shape"
                )

                # Resize to match expected shape
                from scipy.ndimage import zoom

                zoom_factors = np.array(expected_raw_shape) / np.array(raw_volume.shape)
                raw_volume = zoom(raw_volume.astype(float), zoom_factors, order=1)
                raw_volume = raw_volume.astype(
                    np.uint8
                )  # Convert back to uint8 for raw data

            # Ensure gt_mask exists (may be created later if allow_gt_extension is True)
            gt_mask = gt_mask if "gt_mask" in locals() else None

            # Create GT mask if GT extension is allowed

            if allow_gt_extension:
                # Create GT-valid mask using the explicit custom fill value:
                # For each class volume, any voxel != fill_value is considered part of the GT coverage.
                gt_mask = np.ones(volume_shape, dtype=np.uint8)

                for class_key, vol in class_volumes.items():
                    if np.issubdtype(vol.dtype, np.integer):
                        fill_val = np.iinfo(vol.dtype).max
                        valid = vol != fill_val
                        vol[vol == fill_val] = 0  # Set fill values to background
                    elif np.issubdtype(vol.dtype, np.floating):
                        fill_val = np.finfo(vol.dtype).max
                        valid = vol != fill_val
                        vol[np.isclose(vol, fill_val)] = (
                            0  # Set fill values to background
                        )
                    elif vol.dtype == bool:
                        valid = vol.astype(bool)
                    else:
                        # Fallback: treat non-zero as valid if dtype unknown
                        valid = vol != 0

                    gt_mask &= valid.astype(np.uint8)

                if gt_mask.sum() == 0:
                    # Fallback to full mask if nothing flagged as valid (warn user)
                    print(
                        "    ⚠️  Warning: No valid GT voxels found via custom_fill_value; using full mask"
                    )
                    gt_mask = np.ones(volume_shape, dtype=np.uint8)

                # Don't append mask yet - wait until after label fraction check
                pass
            else:
                gt_mask = np.ones(volume_shape, dtype=np.uint8)

            # Now calculate label fraction within valid GT regions only
            total_label_fraction = 0.0
            valid_gt_voxels = np.sum(gt_mask)  # Number of valid GT voxels

            if valid_gt_voxels == 0:
                print(f"  Dataset {dataset_idx}: No valid GT region found - skipping")
                continue

            for class_key, class_mask in class_fractions.items():
                # Only count labels within valid GT regions
                valid_class_voxels = np.sum(class_mask & (gt_mask == 1))
                class_fraction = valid_class_voxels / valid_gt_voxels
                total_label_fraction += class_fraction

            # Check minimum label fraction within valid GT regions
            if total_label_fraction < min_label_fraction:
                print(
                    f"  Dataset {dataset_idx}: label fraction {total_label_fraction:.3f} too low within valid GT regions"
                )
                continue

            # Check minimum unique IDs if specified (for affinity training)
            if min_unique_ids is not None:
                # Count unique instance IDs in the valid GT region only
                valid_gt_volume = (
                    gt_volume[gt_mask == 1] if np.any(gt_mask == 1) else gt_volume
                )
                unique_ids = np.unique(valid_gt_volume)
                num_unique_ids = len(unique_ids)  # [unique_ids > 0])

                if num_unique_ids < min_unique_ids:
                    print(
                        f"  Dataset {dataset_idx}: only {num_unique_ids} unique instance IDs "
                        f"(need {min_unique_ids}) within valid GT regions - skipping"
                    )
                    continue
                print(
                    f"  Dataset {dataset_idx}: {num_unique_ids} unique instance IDs found "
                    f"(>= {min_unique_ids} required)"
                )

            print(
                f"  Dataset {dataset_idx}: label fraction {total_label_fraction:.3f} within valid GT regions, roi {gt_roi}"
            )

            # Use the centralized augmentation helpers (keeps code modular)
            if augment and (np.random.rand() < augment_prob):
                try:
                    from dinov3_playground.augmentations import apply_3d_augmentations
                except Exception:
                    # Fallback to relative import when running as package
                    from .augmentations import apply_3d_augmentations

                raw_volume, context_volume, gt_volume, gt_mask = apply_3d_augmentations(
                    raw_volume,
                    context_volume,
                    gt_volume,
                    gt_mask,
                    augment=augment,
                    augment_prob=augment_prob,
                    augment_params=augment_params,
                )
            raw_volumes[volumes_collected] = raw_volume
            gt_volumes[volumes_collected] = gt_volume
            # Store the volumes and mask (only after all checks pass)
            if gt_masks is not None and gt_mask is not None:
                gt_masks[volumes_collected] = gt_mask

            # Store into preallocated arrays/lists by index
            if context_volumes is not None:
                context_volumes[volumes_collected] = context_volume  # May be None
            dataset_sources[volumes_collected] = dataset_idx
            volumes_collected += 1
            # Print class distribution for this volume
            unique_classes, class_counts = np.unique(gt_volume, return_counts=True)
            class_info = ", ".join(
                [
                    f"class {c}: {cnt/gt_volume.size:.3f}"
                    for c, cnt in zip(unique_classes, class_counts)
                ]
            )

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(
                {
                    "dataset": dataset_idx,
                    "label_frac": f"{total_label_fraction:.3f}",
                    "classes": len(unique_classes),
                }
            )
            del gt_mask, raw_volume, gt_volume, class_volumes, raw_idi, class_idis
            gc.collect()
            # except Exception as e:
            #     error_msg = str(e)
            #     if "FAILED_PRECONDITION" in error_msg and "checksum" in error_msg:
            #         print(
            #             f"  Dataset {dataset_idx}: TensorStore/Zarr compatibility issue - skipping"
            #         )
            #         print(
            #             f"    Issue: Zarr file contains unsupported 'checksum' field in compressor"
            #         )
            #         print(f"    Path: {dataset_dict}")
            #         print(
            #             f"    Suggestion: This dataset may need to be re-saved with compatible Zarr format"
            #         )
            #     elif "Error opening" in error_msg and "zarr" in error_msg.lower():
            #         print(f"  Dataset {dataset_idx}: Zarr format issue - skipping")
            #         print(f"    Path: {dataset_dict}")
            #         print(f"    Error: {error_msg[:200]}...")
            #     else:
            #         print(f"  Dataset {dataset_idx}: error - {e}")
            #     continue

    if volumes_collected < num_volumes:
        print(f"Warning: Only collected {volumes_collected}/{num_volumes} volumes")

    # Handle context volumes
    # Determine whether any context data was captured
    if context_volumes is None:
        has_context = False
    else:
        # Check if at least one context entry is not None
        has_context = any(cv is not None for cv in context_volumes)
        if has_context:
            # If all entries are non-None, convert to a regular numpy array
            if all(cv is not None for cv in context_volumes):
                context_volumes = np.stack(context_volumes, axis=0)
            else:
                # Mixed None/non-None -> keep as object array for caller to handle
                valid_count = sum(1 for cv in context_volumes if cv is not None)
                if valid_count < len(context_volumes):
                    print(
                        f"Warning: Only {valid_count}/{len(context_volumes)} volumes have context data"
                    )
                # keep as object array
        else:
            # No context data found
            context_volumes = None

    print(f"Final dataset summary:")
    print(f"  Raw volumes shape: {raw_volumes.shape}")
    print(f"  GT volumes shape: {gt_volumes.shape}")
    if gt_masks is not None:
        print(f"  GT masks shape: {gt_masks.shape}")
    if has_context and context_volumes is not None:
        if context_volumes.dtype == object:
            print(f"  Context volumes: Mixed (some None)")
        else:
            print(f"  Context volumes shape: {context_volumes.shape}")
        print(f"  Context scale: {context_scale}nm")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes found in data: {np.unique(gt_volumes)}")
    if allow_gt_extension:
        print(f"  GT extension enabled - masks indicate valid GT regions")
        valid_mask_fraction = np.mean(gt_masks)
        print(f"  Average valid GT fraction: {valid_mask_fraction:.3f}")

    # Convert dataset_sources to a Python list trimmed to collected volumes for API compatibility
    dataset_sources = list(dataset_sources[:volumes_collected])

    if allow_gt_extension:
        if has_context:
            return (
                raw_volumes,
                gt_volumes,
                gt_masks,
                context_volumes,
                dataset_sources,
                num_classes,
            )
        else:
            return raw_volumes, gt_volumes, gt_masks, dataset_sources, num_classes
    else:
        if has_context:
            return (
                raw_volumes,
                gt_volumes,
                context_volumes,
                dataset_sources,
                num_classes,
            )
        else:
            return raw_volumes, gt_volumes, dataset_sources, num_classes


def extract_organelle_directories(
    base_path="/nrs/cellmap/data", organelle_list=None, annotation_type="gt"
):
    """
    Extract all organelle directories from the cellmap data structure.

    Searches for directories matching the pattern based on annotation_type:
    - "gt": /nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/labels/groundtruth/crop{number}/{organelle}
    - "inference": /nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/labels/inference/segmentations/{organelle}

    Parameters:
    -----------
    base_path : str
        Base path to search (default: "/nrs/cellmap/data")
    organelle_list : list of str, optional
        List of specific organelle names to filter for. If provided, only these
        organelles will be included in the results.
    annotation_type : str, default="gt"
        Type of annotations to search for:
        - "gt": Ground truth annotations in crop directories
        - "inference": Inference segmentations (no crop structure)

    Returns:
    --------
    dict: Dictionary with structure:
        For "gt":
        {
            'dataset_name': {
                'crop_number': ['organelle1', 'organelle2', ...],
                ...
            },
            ...
        }
        For "inference":
        {
            'dataset_name': {
                'inference': ['organelle1', 'organelle2', ...],
            },
            ...
        }
    list: Flat list of all unique organelles found across all datasets
    """
    import os
    import glob
    from pathlib import Path

    # Validate annotation_type
    if annotation_type not in ["gt", "inference"]:
        raise ValueError(
            f"annotation_type must be 'gt' or 'inference', got '{annotation_type}'"
        )

    print(f"Scanning for {annotation_type} organelle directories in: {base_path}")

    organelle_data = {}
    all_organelles = set()

    # Find all dataset directories
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} does not exist")
        return {}, []

    dataset_dirs = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]

    print(f"Found {len(dataset_dirs)} potential dataset directories")

    for dataset in dataset_dirs:
        dataset_path = os.path.join(base_path, dataset)
        zarr_path = os.path.join(dataset_path, f"{dataset}.zarr")

        # Check if .zarr directory exists
        if not os.path.exists(zarr_path):
            continue

        if annotation_type == "gt":
            # Ground truth path with crop structure
            labels_base = os.path.join(zarr_path, "recon-1", "labels", "groundtruth")

            # Check if groundtruth path exists
            if not os.path.exists(labels_base):
                continue

            print(f"  Processing dataset: {dataset}")
            organelle_data[dataset] = {}

            # Find all crop directories
            crop_pattern = os.path.join(labels_base, "crop*")
            crop_dirs = glob.glob(crop_pattern)

            for crop_dir in crop_dirs:
                crop_name = os.path.basename(crop_dir)  # e.g., "crop001", "crop002"

                # Extract crop number
                crop_number = crop_name.replace("crop", "")

                # Find all organelle directories in this crop
                if os.path.isdir(crop_dir):
                    organelles = [
                        d
                        for d in os.listdir(crop_dir)
                        if os.path.isdir(os.path.join(crop_dir, d))
                    ]

                    # Filter by organelle_list if provided
                    if organelle_list is not None:
                        organelles = [
                            org for org in organelles if org in organelle_list
                        ]

                    if organelles:
                        organelle_data[dataset][crop_number] = sorted(organelles)
                        all_organelles.update(organelles)
                        print(
                            f"    {crop_name}: {len(organelles)} organelles - {', '.join(sorted(organelles))}"
                        )

        elif annotation_type == "inference":
            # Inference path without crop structure
            labels_base = os.path.join(
                zarr_path, "recon-1", "labels", "inference", "segmentations"
            )

            # Check if inference path exists
            if not os.path.exists(labels_base):
                continue

            print(f"  Processing dataset: {dataset}")
            organelle_data[dataset] = {}

            # Find all organelle directories directly in segmentations
            if os.path.isdir(labels_base):
                organelles = [
                    d
                    for d in os.listdir(labels_base)
                    if os.path.isdir(os.path.join(labels_base, d))
                ]

                # Filter by organelle_list if provided
                if organelle_list is not None:
                    organelles = [org for org in organelles if org in organelle_list]

                if organelles:
                    # Use 'inference' as a pseudo-crop identifier for consistency
                    organelle_data[dataset]["inference"] = sorted(organelles)
                    all_organelles.update(organelles)
                    print(
                        f"    inference: {len(organelles)} organelles - {', '.join(sorted(organelles))}"
                    )

    # Convert set to sorted list
    all_organelles = sorted(list(all_organelles))

    print(f"\nSummary:")
    print(f"  Total datasets with organelles: {len(organelle_data)}")
    print(f"  Unique organelles found: {len(all_organelles)}")
    print(f"  All organelles: {', '.join(all_organelles)}")

    return organelle_data, all_organelles


def get_organelle_paths(
    dataset_name,
    crop_number=None,
    organelle=None,
    organelle_list=None,
    base_path="/nrs/cellmap/data",
):
    """
    Get full paths to specific organelle directories.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    crop_number : str or int, optional
        Specific crop number (e.g., "001" or 1). If None, returns all crops.
    organelle : str, optional
        Specific organelle name. If None, returns all organelles.
    organelle_list : list of str, optional
        List of specific organelle names to filter for. If provided, only paths
        for these organelles will be returned. Takes precedence over 'organelle' parameter.
    base_path : str
        Base path to search (default: "/nrs/cellmap/data")

    Returns:
    --------
    list: List of full paths matching the criteria
    """
    import os

    paths = []
    zarr_path = os.path.join(base_path, dataset_name, f"{dataset_name}.zarr")
    groundtruth_base = os.path.join(zarr_path, "recon-1", "labels", "groundtruth")

    if not os.path.exists(groundtruth_base):
        print(f"Warning: Groundtruth path does not exist: {groundtruth_base}")
        return paths

    # Handle crop number formatting
    if crop_number is not None:
        if isinstance(crop_number, int):
            crop_number = f"{crop_number:03d}"  # Convert to "001" format
        crop_dirs = [os.path.join(groundtruth_base, f"crop{crop_number}")]
    else:
        import glob

        crop_dirs = glob.glob(os.path.join(groundtruth_base, "crop*"))

    for crop_dir in crop_dirs:
        if not os.path.exists(crop_dir):
            continue

        if organelle_list is not None:
            # Filter by list of specific organelles
            for org in organelle_list:
                organelle_path = os.path.join(crop_dir, org)
                if os.path.exists(organelle_path):
                    paths.append(organelle_path)
        elif organelle is not None:
            # Specific organelle
            organelle_path = os.path.join(crop_dir, organelle)
            if os.path.exists(organelle_path):
                paths.append(organelle_path)
        else:
            # All organelles in this crop
            organelles = [
                d
                for d in os.listdir(crop_dir)
                if os.path.isdir(os.path.join(crop_dir, d))
            ]
            for org in organelles:
                paths.append(os.path.join(crop_dir, org))

    return sorted(paths)


def generate_dataset_pairs_for_organelles(
    organelle_list,
    max_pairs_per_organelle=10,
    base_path="/nrs/cellmap/data",
    base_resolution=None,
    use_highest_res_for_raw=False,
    min_resolution_for_raw=None,
    apply_scale_updates=True,
):
    """
    Generate dataset pairs for specific organelles for training with optional scale handling.

    Parameters:
    -----------
    organelle_list : list of str
        List of organelle names to generate pairs for
    max_pairs_per_organelle : int
        Maximum number of dataset pairs to generate per organelle
    base_path : str
        Base path to search (default: "/nrs/cellmap/data")
    base_resolution : int or float, optional
        Target resolution for labels. If provided with apply_scale_updates=True,
        will update paths with appropriate scale information.
    use_highest_res_for_raw : bool, default=False
        If True, use highest available resolution for raw data instead of base_resolution.
        Only used when apply_scale_updates=True.
    min_resolution_for_raw : int, float, or Coordinate, optional
        Minimum allowed resolution for raw data when use_highest_res_for_raw=True.
        Only used when apply_scale_updates=True.
    apply_scale_updates : bool, default=True
        If True and base_resolution is provided, applies scale updates to dataset paths
        similar to update_datapaths_with_target_scales().

    Returns:
    --------
    dict: Dictionary with organelle names as keys and lists of dataset pairs as values
        {
            'organelle_name': [
                {'raw': 'path_to_raw', 'organelle_name': 'path_to_gt'},
                ...
            ],
            ...
        }
    """
    import os

    result = {}

    # Get all organelle data
    organelle_data, _ = extract_organelle_directories(
        base_path=base_path, organelle_list=organelle_list
    )

    for target_organelle in organelle_list:
        pairs = []
        pair_count = 0

        for dataset, crops in organelle_data.items():
            if pair_count >= max_pairs_per_organelle:
                break

            for crop_num, organelles in crops.items():
                if pair_count >= max_pairs_per_organelle:
                    break

                if target_organelle in organelles:
                    raw_path = (
                        f"{base_path}/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8"
                    )
                    gt_path = f"{base_path}/{dataset}/{dataset}.zarr/recon-1/labels/groundtruth/crop{crop_num}/{target_organelle}"

                    # Verify paths exist
                    if os.path.exists(raw_path) and os.path.exists(gt_path):
                        pairs.append({"raw": raw_path, target_organelle: gt_path})
                        pair_count += 1

        result[target_organelle] = pairs
        print(f"Generated {len(pairs)} dataset pairs for '{target_organelle}'")

    # Apply scale updates if requested and base_resolution is provided
    if apply_scale_updates and base_resolution is not None:
        print(f"Applying scale updates with base_resolution={base_resolution}...")
        from dinov3_playground.zarr_util import update_datapaths_with_target_scales

        for organelle in result:
            if result[organelle]:  # Only process if we have pairs
                updated_pairs = update_datapaths_with_target_scales(
                    result[organelle],
                    base_resolution,
                    use_highest_res_for_raw=use_highest_res_for_raw,
                    min_resolution_for_raw=min_resolution_for_raw,
                )
                result[organelle] = updated_pairs
                print(
                    f"Updated {len(updated_pairs)} dataset pairs for '{organelle}' with scale information"
                )

    return result


def generate_multi_organelle_dataset_pairs(
    organelle_list,
    max_pairs=np.inf,
    base_path="/nrs/cellmap/data",
    base_resolution=None,
    use_highest_res_for_raw=False,
    min_resolution_for_raw=None,
    min_resolution_for_gt=None,  # NEW: Minimum resolution for GT (filter out finer resolutions)
    apply_scale_updates=True,
    require_all_organelles=False,
    context_scale=None,  # NEW: Add context data at this resolution (e.g., 8 for 8nm)
    crop_filter=None,  # NEW: Filter for specific crop numbers (e.g., [115, 203, 315])
    annotation_type="gt",  # NEW: "gt" for ground truth, "inference" for inference segmentations
    inference_filter=None,  # NEW: Filter for specific dataset names when annotation_type="inference"
):
    """
    Generate multi-organelle dataset pairs where each pair contains multiple organelles from the same dataset/crop.

    Parameters:
    -----------
    organelle_list : list of str
        List of organelle names to include in each dataset pair
    max_pairs : int
        Maximum number of dataset pairs to generate
    base_path : str
        Base path to search (default: "/nrs/cellmap/data")
    base_resolution : int or float, optional
        Target resolution for labels. If provided with apply_scale_updates=True,
        will update paths with appropriate scale information.
    use_highest_res_for_raw : bool, default=False
        If True, use highest available resolution for raw data instead of base_resolution.
    min_resolution_for_raw : int, float, or Coordinate, optional
        Minimum allowed resolution for raw data when use_highest_res_for_raw=True.
        Resolutions finer (smaller values) than this will be skipped.
    min_resolution_for_gt : int, float, or Coordinate, optional
        Minimum allowed resolution for ground truth data. GT datasets with resolutions
        finer (smaller values) than this will be excluded. Useful for focusing on
        large-scale segmentations (e.g., full cells) while ignoring fine-resolution
        organelle data. For example, set to 8 to exclude 4nm GT data.
    apply_scale_updates : bool, default=True
        If True and base_resolution is provided, applies scale updates to dataset paths.
    require_all_organelles : bool, default=False
        If True, only include dataset pairs that have ALL specified organelles.
        If False, include pairs that have at least one of the specified organelles.
    context_scale : int or float, optional
        If provided, adds a context raw data source at this resolution (e.g., 8 for 8nm).
        The context data will be named "raw_context" or "raw_{context_scale}nm" in the dataset pair.
    crop_filter : list of int, optional
        If provided, only include crops with these specific crop numbers from ground truth.
        For example, crop_filter=[115, 203, 315] will only include crop115, crop203, and crop315.
        If None (default), all crops are included (when searching ground truth).
        Can be combined with inference_filter to search both GT crops and inference datasets.
    annotation_type : str, default="gt"
        Default type of annotations to use when no filters are provided:
        - "gt": Use ground truth annotations from recon-1/labels/groundtruth/crop{num}/{organelle}
        - "inference": Use inference segmentations from recon-1/labels/inference/segmentations/{organelle}
        Note: crop_filter and inference_filter override this setting.
    inference_filter : list of str, optional
        If provided, only include datasets with these specific names from inference segmentations.
        For example, inference_filter=["jrc_mus-kidney", "jrc_mus-liver"] will only search those datasets.
        If None (default), all datasets are searched (when searching inference).
        Can be combined with crop_filter to search both GT crops and inference datasets.

    Returns:
    --------
    list: List of dataset pairs in multi-class format:
        [
            {
                'raw': 'path_to_raw',
                'organelle1': 'path_to_organelle1_gt_or_inference',
                'organelle2': 'path_to_organelle2_gt_or_inference',
                ...
            },
            ...
        ]
    """
    import os

    result = []

    # Validate annotation_type
    if annotation_type not in ["gt", "inference"]:
        raise ValueError(
            f"annotation_type must be 'gt' or 'inference', got '{annotation_type}'"
        )

    pair_count = 0

    # Prepare filters and determine which annotation types to search
    search_gt = False
    search_inference = False

    # Process crop_filter if provided
    if crop_filter is not None:
        search_gt = True
        # Convert integers to zero-padded strings (e.g., 115 -> "115", 1 -> "001")
        crop_filter_strs = set()
        for crop in crop_filter:
            # Try both zero-padded and non-padded versions
            crop_str = str(crop)
            crop_filter_strs.add(crop_str)  # "115"
            crop_filter_strs.add(crop_str.zfill(3))  # "115" (already 3 digits)
            crop_filter_strs.add(str(int(crop_str)))  # Remove leading zeros if any
        crop_filter = crop_filter_strs

    # Process inference_filter if provided
    if inference_filter is not None:
        search_inference = True
        inference_filter = set(inference_filter)

    # If neither filter is provided, use annotation_type
    if not search_gt and not search_inference:
        if annotation_type == "gt":
            search_gt = True
        else:
            search_inference = True

    # Collect organelle data from both sources if needed
    organelle_data_gt = {}
    organelle_data_inference = {}

    if search_gt:
        print(f"  Searching ground truth crops...")
        organelle_data_gt, _ = extract_organelle_directories(
            base_path=base_path,
            organelle_list=organelle_list,
            annotation_type="gt",
        )

    if search_inference:
        print(f"  Searching inference segmentations...")
        organelle_data_inference, _ = extract_organelle_directories(
            base_path=base_path,
            organelle_list=organelle_list,
            annotation_type="inference",
        )

    result = []

    # Process ground truth crops if requested
    if search_gt and organelle_data_gt:
        for dataset, crops in organelle_data_gt.items():
            if pair_count >= max_pairs:
                break

            for crop_num, available_organelles in crops.items():
                if pair_count >= max_pairs:
                    break

                # Filter by crop number if specified
                if crop_filter is not None:
                    if crop_num not in crop_filter:
                        continue

                # Check if this crop has the required organelles
                if require_all_organelles:
                    if not all(org in available_organelles for org in organelle_list):
                        continue
                    organelles_to_include = organelle_list
                else:
                    # Include any organelles that are available
                    organelles_to_include = [
                        org for org in organelle_list if org in available_organelles
                    ]
                    if not organelles_to_include:  # Skip if no organelles available
                        continue

                raw_path = (
                    f"{base_path}/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8"
                )

                # Verify raw path exists
                if not os.path.exists(raw_path):
                    continue

                # Build the dataset pair
                pair = {"raw": raw_path}
                all_paths_exist = True

                # Add context raw data if requested
                if context_scale is not None:
                    context_raw_path = (
                        f"{base_path}/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8"
                    )
                    context_key = f"raw_{int(context_scale)}nm"
                    pair[context_key] = context_raw_path
                    # Note: We'll handle the actual scale selection during data loading

                for organelle in organelles_to_include:
                    # Construct path for ground truth
                    label_path = f"{base_path}/{dataset}/{dataset}.zarr/recon-1/labels/groundtruth/crop{crop_num}/{organelle}"

                    if os.path.exists(label_path):
                        # Check GT resolution if min_resolution_for_gt is specified
                        if min_resolution_for_gt is not None:
                            import zarr
                            from funlib.geometry import Coordinate

                            try:
                                zarr_grp = zarr.open_group(label_path, mode="r")
                                from dinov3_playground.zarr_util import get_scale_info

                                _, resolutions, _ = get_scale_info(zarr_grp)

                                # Convert min_resolution_for_gt to Coordinate
                                if (
                                    type(min_resolution_for_gt) is int
                                    or type(min_resolution_for_gt) is float
                                ):
                                    min_res = Coordinate(3 * [min_resolution_for_gt])
                                else:
                                    min_res = Coordinate(min_resolution_for_gt)

                                # Find the finest resolution available
                                finest_resolution = None

                                for scale, res in resolutions.items():
                                    res_coord = Coordinate(res)

                                    # Track the finest resolution available
                                    if finest_resolution is None or all(
                                        r <= fr
                                        for r, fr in zip(res_coord, finest_resolution)
                                    ):
                                        finest_resolution = res_coord

                                # Skip if finest resolution is finer than min_resolution_for_gt
                                if finest_resolution is not None and any(
                                    r < min_r
                                    for r, min_r in zip(finest_resolution, min_res)
                                ):
                                    print(
                                        f"Skipping {dataset}/crop{crop_num}/{organelle}: finest resolution {finest_resolution} is finer than min_resolution_for_gt {min_res}"
                                    )
                                    all_paths_exist = False
                                    break
                            except Exception as e:
                                print(
                                    f"Warning: Could not check resolution for {label_path}: {e}"
                                )
                                # Continue anyway if we can't check resolution

                        pair[organelle] = label_path
                    else:
                        all_paths_exist = False
                        break

                if (
                    all_paths_exist and len(pair) > 1
                ):  # Must have at least raw + one organelle
                    result.append(pair)
                    pair_count += 1

    # Process inference segmentations if requested
    if search_inference and organelle_data_inference:
        for dataset, crops in organelle_data_inference.items():
            if pair_count >= max_pairs:
                break

            # Filter by dataset name if specified
            if inference_filter is not None:
                if dataset not in inference_filter:
                    continue

            for crop_num, available_organelles in crops.items():
                if pair_count >= max_pairs:
                    break

                # Check if this dataset has the required organelles
                if require_all_organelles:
                    if not all(org in available_organelles for org in organelle_list):
                        continue
                    organelles_to_include = organelle_list
                else:
                    # Include any organelles that are available
                    organelles_to_include = [
                        org for org in organelle_list if org in available_organelles
                    ]
                    if not organelles_to_include:  # Skip if no organelles available
                        continue

                raw_path = (
                    f"{base_path}/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8"
                )

                # Verify raw path exists
                if not os.path.exists(raw_path):
                    continue

                # Build the dataset pair
                pair = {"raw": raw_path}
                all_paths_exist = True

                # Add context raw data if requested
                if context_scale is not None:
                    context_raw_path = (
                        f"{base_path}/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8"
                    )
                    context_key = f"raw_{int(context_scale)}nm"
                    pair[context_key] = context_raw_path
                    # Note: We'll handle the actual scale selection during data loading

                for organelle in organelles_to_include:
                    # Construct path for inference segmentation
                    label_path = f"{base_path}/{dataset}/{dataset}.zarr/recon-1/labels/inference/segmentations/{organelle}"

                    if os.path.exists(label_path):
                        # Check GT resolution if min_resolution_for_gt is specified
                        if min_resolution_for_gt is not None:
                            import zarr
                            from funlib.geometry import Coordinate

                            try:
                                zarr_grp = zarr.open_group(label_path, mode="r")
                                from dinov3_playground.zarr_util import get_scale_info

                                _, resolutions, _ = get_scale_info(zarr_grp)

                                # Convert min_resolution_for_gt to Coordinate
                                if (
                                    type(min_resolution_for_gt) is int
                                    or type(min_resolution_for_gt) is float
                                ):
                                    min_res = Coordinate(3 * [min_resolution_for_gt])
                                else:
                                    min_res = Coordinate(min_resolution_for_gt)

                                # Find the finest resolution available
                                finest_resolution = None

                                for scale, res in resolutions.items():
                                    res_coord = Coordinate(res)

                                    # Track the finest resolution available
                                    if finest_resolution is None or all(
                                        r <= fr
                                        for r, fr in zip(res_coord, finest_resolution)
                                    ):
                                        finest_resolution = res_coord

                                # Skip if finest resolution is finer than min_resolution_for_gt
                                if finest_resolution is not None and any(
                                    r < min_r
                                    for r, min_r in zip(finest_resolution, min_res)
                                ):
                                    print(
                                        f"Skipping {dataset}/{organelle}: finest resolution {finest_resolution} is finer than min_resolution_for_gt {min_res}"
                                    )
                                    all_paths_exist = False
                                    break
                            except Exception as e:
                                print(
                                    f"Warning: Could not check resolution for {label_path}: {e}"
                                )
                                # Continue anyway if we can't check resolution

                        pair[organelle] = label_path
                    else:
                        all_paths_exist = False
                        break

                if (
                    all_paths_exist and len(pair) > 1
                ):  # Must have at least raw + one organelle
                    result.append(pair)
                    pair_count += 1

    print(f"Generated {len(result)} multi-organelle dataset pairs")

    # Report which sources were searched
    sources_searched = []
    if search_gt:
        sources_searched.append("ground truth crops")
    if search_inference:
        sources_searched.append("inference segmentations")
    print(f"  Sources searched: {', '.join(sources_searched)}")

    if crop_filter is not None:
        print(f"  Crop filter active: only including crops {crop_filter}")
    if inference_filter is not None:
        print(f"  Inference filter active: only including datasets {inference_filter}")
    if min_resolution_for_gt is not None:
        print(
            f"  GT resolution filter active: excluding GT data finer than {min_resolution_for_gt}nm"
        )
    if result:
        example_organelles = list(result[0].keys())
        example_organelles.remove("raw")
        print(f"  Example organelles in pairs: {example_organelles}")

    # Apply scale updates if requested and base_resolution is provided
    if apply_scale_updates and base_resolution is not None and result:
        print(f"Applying scale updates with base_resolution={base_resolution}...")
        from dinov3_playground.zarr_util import update_datapaths_with_target_scales

        result = update_datapaths_with_target_scales(
            result,
            base_resolution,
            use_highest_res_for_raw=use_highest_res_for_raw,
            min_resolution_for_raw=min_resolution_for_raw,
            context_scale=context_scale,  # Pass context_scale for context data path updates
        )
        print(
            f"Updated {len(result)} multi-organelle dataset pairs with scale information"
        )

    return result
