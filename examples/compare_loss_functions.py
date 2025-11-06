# %%
import numpy as np

data1 = np.load("debug_batch_epoch1_batch1.npz")
loss_1 = 0.08952406048774719

data2_boundary = np.load("boundary_weights_sample0_old.npz")
data2 = np.load("debug_batch_epoch1_batch1_old.npz")
loss_2 = 0.10007143020629883

# %%
import matplotlib.pyplot as plt
from dinov3_playground.affinity_utils import compute_boundary_weights

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(data1["target"][0, 0, 79, :, :])
plt.subplot(1, 2, 2)
plt.imshow(data2["target"][0, 0, 80, :, :], vmin=0)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(data1["boundary_weights"][0, 79, :, :], vmin=0, vmax=11)
plt.subplot(1, 3, 2)
plt.imshow(data1["boundary_weights"][0, 79, :, :] ** 1.5, vmin=0, vmax=11)
weights_10 = compute_boundary_weights(
    instance_segmentation=data1["gt"][0], boundary_weight=10, mask=data1["mask"][0]
)
plt.subplot(1, 3, 3)
plt.imshow(weights_10[79, :, :], vmin=0, vmax=11)
plt.figure()
plt.imshow(
    data1["boundary_weights"][0, 79, :, :]**1.5 - weights_10[79, :, :],
    cmap="bwr",
)
plt.colorbar()

# %%
print(np.allclose(data1["target"], data2["target"]))
print(np.allclose(data1["boundary_weights"], data2_boundary["weights"]))
print(np.allclose(data1["mask"], data2["mask"]))

print(np.allclose(data1["features"], data2["features"]))
print(np.allclose(data1["outputs"], data2["outputs"]))
# %%
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(data1["features"][0, 3, 79, :, :], vmin=0)
plt.subplot(1, 2, 2)
plt.imshow(data2["features"][0, 3, 79, :, :], vmin=0)
plt.imshow(
    data2["features"][0, 3, 79, :, :] - data1["features"][0, 3, 79, :, :],
    vmin=-1,
    vmax=1,
    cmap="bwr",
)
# %%
# plot outputs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(data1["outputs"][0, 3, 79, :, :], vmin=-1, vmax=1)
plt.subplot(1, 2, 2)
plt.imshow(data2["outputs"][0, 3, 79, :, :], vmin=-1, vmax=1)
plt.imshow(
    data2["outputs"][0, 3, 79, :, :] - data1["outputs"][0, 3, 79, :, :],
    vmin=-1,
    vmax=1,
    cmap="bwr",
)
# %%
import torch
from dinov3_playground.losses import get_loss_function

loss_fn = get_loss_function(
    loss_type="boundary_affinity_focal_lsds",
    sigma=5.0,
    alpha=0.50,
    gamma=2.0,
    boundary_weight_power=1,
    mask_clip_distance=3,
)
loss1 = loss_fn(
    torch.tensor(data1["outputs"]),
    torch.tensor(data1["target"]),
    boundary_weights=torch.tensor(data1["boundary_weights"]),
    mask=torch.tensor(data1["mask"]),
)
loss2 = loss_fn(
    torch.tensor(data2["outputs"]),
    torch.tensor(data2["target"]),
    boundary_weights=torch.tensor(data1["boundary_weights"]),
    mask=torch.tensor(data2["mask"]),
)
# print 7 digits of loss values
print(f"{loss1:.7f}", f"{loss2:.7f}", f"{loss1 - loss2:.7f}")

# %%
# Match training conditions exactly
loss_fn = get_loss_function(
    loss_type="boundary_affinity_focal_lsds",
    gamma=2.0,
    alpha=0.5,
    tversky_beta=0.5,
    boundary_sigma=5.0,
    mask_clip_distance=3,
)

# Match device and dtype from training
loss1 = loss_fn(
    torch.tensor(data1["outputs"], dtype=torch.float32),  # ← explicit dtype
    torch.tensor(data1["target"], dtype=torch.float32),
    boundary_weights=torch.tensor(data1["boundary_weights"], dtype=torch.float32),
    mask=torch.tensor(data1["mask"], dtype=torch.float32),
)
print(loss1)
# %%
