# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # abcdmicro: Diffusion MRI Microstructure Example Pipeline
#
# This notebook demonstrates the main capabilities of the `abcdmicro` package
# for extracting brain microstructure parameters from diffusion MRI data.
#
# ## Pipeline overview
#
# 1. Load DWI data
# 2. Denoise
# 3. Brain extraction
# 4. DTI estimation (FA, MD, eigenvalues)
# 5. NODDI estimation (NDI, ODI, FWF)
# 6. CSD / Fiber Orientation Distributions
# 7. TractSeg white matter tract segmentation
# 8. Saving results to disk

# %% [markdown]
# ## 1. Load DWI data
#
# A `Dwi` object bundles three resources: the 4D volume, b-values, and
# b-vectors. Resources are loaded lazily — nothing is read from disk until
# you call `.load()`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from abcdmicro.dwi import Dwi
from abcdmicro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource

# Point these to a BIDS-style DWI session directory containing
# <basename>.nii.gz, <basename>.bval, and <basename>.bvec files.
data_dir = Path("...")  # e.g. Path("/data/sub-XXX/ses-YYY/dwi")
basename = "..."  # e.g. "sub-XXX_ses-YYY_run-01_dwi"

dwi = Dwi(
    NiftiVolumeResource(data_dir / f"{basename}.nii.gz"),
    FslBvalResource(data_dir / f"{basename}.bval"),
    FslBvecResource(data_dir / f"{basename}.bvec"),
).load()

vol = dwi.volume.get_array()
bvals = dwi.bval.get()
print(f"Volume shape: {vol.shape}")
print(f"Unique b-values: {np.unique(np.round(bvals, -2))}")

# %% [markdown]
# Quick look at the mean b=0 image and a diffusion-weighted image side by side.
# `compute_mean_b0()` averages all b=0 volumes, which is also used internally
# by brain extraction.

# %%
mean_b0 = dwi.compute_mean_b0()
mean_b0_arr = mean_b0.get_array()
mid_slice = vol.shape[2] // 2

# %%
dwi_idx = np.argmax(bvals)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(mean_b0_arr[:, :, mid_slice].T, cmap="gray", origin="lower")
axes[0].set_title("Mean b = 0")
axes[1].imshow(vol[:, :, mid_slice, dwi_idx].T, cmap="gray", origin="lower")
axes[1].set_title(f"b = {bvals[dwi_idx]:.0f}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Denoising
#
# `Dwi.denoise()` applies DIPY's Patch2Self algorithm. It returns a new `Dwi`
# with the denoised volume (b-values and b-vectors are carried forward
# unchanged).

# %%
dwi_denoised = dwi.denoise()

# %%
orig = mean_b0_arr[:, :, mid_slice]
denoised = dwi_denoised.compute_mean_b0().get_array()[:, :, mid_slice]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(orig.T, cmap="gray", origin="lower")
axes[0].set_title("Original")
axes[1].imshow(denoised.T, cmap="gray", origin="lower")
axes[1].set_title("Denoised")
axes[2].imshow((orig - denoised).T, cmap="bwr", origin="lower")
axes[2].set_title("Residual (noise removed)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Brain extraction
#
# `extract_brain()` uses HD-BET to produce a binary brain mask from the mean
# b=0 image. The mask is returned as a `VolumeResource`.

# %%
mask = dwi_denoised.extract_brain()

# %%
# (This cell fixes a few notebook output issues caused by the HD-BET masking step above)

# %matplotlib inline

import sys
from IPython import get_ipython

kernel = get_ipython().kernel
sys.stdout = kernel._stdout

# %%
mask_arr = mask.get_array()
print(f"Mask shape: {mask_arr.shape}, voxels in brain: {mask_arr.sum()}")

fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(denoised.T, cmap="gray", origin="lower")
ax.contour(mask_arr[:, :, mid_slice].T, colors="red", linewidths=0.8)
ax.set_title("Brain mask overlay")
ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. DTI estimation
#
# `estimate_dti()` fits a diffusion tensor at each voxel using DIPY's
# TensorModel. The resulting `Dti` object gives access to:
#
# - **FA** (fractional anisotropy) — degree of diffusion directionality
# - **MD** (mean diffusivity) — average diffusion rate
# - **Eigenvalues / eigenvectors** — full tensor decomposition

# %%
dti = dwi_denoised.estimate_dti(mask=mask)
fa_vol, md_vol = dti.get_fa_md()

# %%
fa = fa_vol.get_array()
md = md_vol.get_array()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(fa[:, :, mid_slice].T, cmap="hot", origin="lower", vmin=0, vmax=1)
axes[0].set_title("Fractional Anisotropy (FA)")
plt.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(
    md[:, :, mid_slice].T, cmap="viridis", origin="lower", vmin=0, vmax=3e-3
)
axes[1].set_title("Mean Diffusivity (MD)")
plt.colorbar(im1, ax=axes[1], fraction=0.046)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Eigenvalue decomposition
#
# The eigenvalues ($\lambda_1 \ge \lambda_2 \ge \lambda_3$) reveal the shape
# of diffusion at each voxel.

# %%
evals_vol, evecs_vol = dti.get_eig()

# %%
evals = evals_vol.get_array()  # shape (x, y, z, 3)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, (ax, label) in enumerate(
    zip(axes, [r"$\lambda_1$ (axial)", r"$\lambda_2$", r"$\lambda_3$ (radial)"])
):
    im = ax.imshow(
        evals[:, :, mid_slice, i].T, cmap="inferno", origin="lower", vmin=0, vmax=3e-3
    )
    ax.set_title(label)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. NODDI estimation
#
# NODDI (Neurite Orientation Dispersion and Density Imaging) provides
# biophysically meaningful parameters:
#
# - **NDI** — neurite density index
# - **ODI** — orientation dispersion index
# - **FWF** — free water fraction

# %%
noddi = dwi_denoised.estimate_noddi(mask=mask)

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, arr, title, cmap in [
    (axes[0], noddi.ndi.get_array()[:, :, mid_slice], "NDI (neurite density)", "YlOrRd"),
    (axes[1], noddi.odi.get_array()[:, :, mid_slice], "ODI (orientation dispersion)", "YlGnBu"),
    (axes[2], noddi.fwf.get_array()[:, :, mid_slice], "FWF (free water)", "Blues"),
]:
    im = ax.imshow(arr.T, cmap=cmap, origin="lower", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. CSD and fiber orientation distributions
#
# Constrained Spherical Deconvolution estimates fiber orientation distributions
# (FODs) at each voxel. This is the basis for tractography and TractSeg.

# %%
from abcdmicro.csd import compute_csd_peaks, estimate_response_function

response = estimate_response_function(dwi_denoised, mask)
peak_dirs, peak_values = compute_csd_peaks(dwi_denoised, mask, response)

# %%
peak_dirs_arr = peak_dirs.get_array()  # (x, y, z, n_peaks, 3)
peak_vals_arr = peak_values.get_array()  # (x, y, z, n_peaks)
print(f"Peak directions shape: {peak_dirs_arr.shape}")
print(f"Peak values shape: {peak_vals_arr.shape}")

n_peaks_per_voxel = (peak_vals_arr > 0).sum(axis=-1)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(
    n_peaks_per_voxel[:, :, mid_slice].T, cmap="tab10", origin="lower", vmin=0, vmax=5
)
ax.set_title("Number of fiber peaks per voxel")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. TractSeg — white matter tract segmentation
#
# TractSeg segments 72 white matter bundles directly from CSD peaks. It can
# also produce tract endpoint regions and tract orientation maps (TOMs).

# %%
from abcdmicro.tractseg import extract_tractseg

tracts = extract_tractseg(dwi_denoised, mask, response, output_type="tract_segmentation")

# %%
from tractseg.data.dataset_specific_utils import get_bundle_names
all_names = get_bundle_names("All")[1:]  # skip BG

tracts_arr = tracts.get_array()  # (x, y, z, 72)
print(f"TractSeg output shape: {tracts_arr.shape}")

bundle_names = [
    "AF_left",
    "CST_right",
    "CC_1", # rostrum
    "CC_7", # splenium
]
bundle_indices = [all_names.index(n) for n in bundle_names]

fig, axes = plt.subplots(1, len(bundle_indices), figsize=(14, 4))
for ax, idx, name in zip(axes, bundle_indices, bundle_names):
    ax.imshow(denoised.T, cmap="gray", origin="lower")
    ax.contour(tracts_arr[:, :, mid_slice, idx].T, colors="lime", linewidths=0.8)
    ax.set_title(name)
    ax.axis("off")
plt.suptitle("Selected tract segmentations")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Saving results to disk
#
# All results can be saved to NIfTI files. `save()` returns a new object
# backed by on-disk resources (functional style — originals are unchanged).

# %%
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Save DTI tensor volume
dti_saved = dti.save(output_dir / "dti.nii.gz")

# Save individual FA and MD maps
NiftiVolumeResource.save(fa_vol, output_dir / "fa.nii.gz")
NiftiVolumeResource.save(md_vol, output_dir / "md.nii.gz")

# Save NODDI
noddi_saved = noddi.save(output_dir / "noddi.nii.gz")

# Save brain mask
NiftiVolumeResource.save(mask, output_dir / "brain_mask.nii.gz")

# Save the denoised DWI (volume + bval + bvec)
dwi_denoised.save(output_dir, basename="denoised_dwi")

print(f"Results saved to {output_dir.resolve()}")
