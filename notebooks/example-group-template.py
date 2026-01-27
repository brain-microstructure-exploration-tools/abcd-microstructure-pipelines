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
# # abcdmicro: Group Registration and Population Template Construction
#
# This notebook demonstrates how to build population-level templates from
# individual subject volumes using the `abcdmicro` registration tools.
#
# ## Pipeline overview
#
# 1. Load per-subject DWI data and estimate FA / MD via DTI
# 2. Build a single-metric population template
# 3. Build a multi-metric population template
# 4. Save templates to disk

# %% [markdown]
# ## 1. Load DWI data and compute per-subject FA / MD
#
# We construct `Dwi` objects from each subject's DWI files, fit a diffusion
# tensor, and extract FA and MD maps. These scalar maps are then used for
# template construction.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import tempfile

from abcdmicro.dwi import Dwi
from abcdmicro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource
from abcdmicro.masks import brain_extract_batch
from abcdmicro.resource import VolumeResource

# --- Fill in your DWI data paths here ---
data_root = Path("...") # Root data directory
subject_dirs = [
    # Subject subdirectories; these should be individual directories that contain subdirectories with the pattern sub-*/ses-*/dwi
    data_root/"...",
    data_root/"...",
    data_root/"...",
]

# Load all DWIs
dwis: list[Dwi] = []
for d in subject_dirs:
    dwi_dir = next(d.glob("sub-*/ses-*/dwi"))
    dwi_nifti = next(dwi_dir.glob("*_dwi.nii.gz"))
    basename = dwi_nifti.name.removesuffix(".nii.gz")

    dwis.append(Dwi(
        NiftiVolumeResource(dwi_dir / f"{basename}.nii.gz"),
        FslBvalResource(dwi_dir / f"{basename}.bval"),
        FslBvecResource(dwi_dir / f"{basename}.bvec"),
    ).load())

# Brain extraction in one batch (HD-BET initializes once)
with tempfile.TemporaryDirectory() as tmpdir:
    cases = [(dwi, Path(tmpdir) / f"mask_{i}.nii.gz") for i, dwi in enumerate(dwis)]
    # Load masks into memory before the temp directory is cleaned up
    masks = [m.load() for m in brain_extract_batch(cases)]

# Estimate DTI and extract FA/MD
fa_volumes: list[VolumeResource] = []
md_volumes: list[VolumeResource] = []

for dwi, mask in zip(dwis, masks):
    dti = dwi.estimate_dti(mask=mask)
    fa_vol, md_vol = dti.get_fa_md()
    fa_volumes.append(fa_vol)
    md_volumes.append(md_vol)

# %%
# (This cell fixes a few notebook output issues caused by the HD-BET masking step above)

# %matplotlib inline

import sys
from IPython import get_ipython

kernel = get_ipython().kernel
sys.stdout = kernel._stdout

# %% [markdown]
# Preview a slice of the individual FA and MD maps.

# %%
print(f"Computed FA/MD for {len(fa_volumes)} subjects")

n_show = min(len(fa_volumes), 4)
fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
if n_show == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    arr = fa_volumes[i].get_array()
    mid = arr.shape[2] // 2
    ax.imshow(arr[:, :, mid].T, cmap="hot", origin="lower", vmin=0, vmax=1)
    ax.set_title(f"Subject {i + 1}")
    ax.axis("off")
plt.suptitle("Individual FA maps")
plt.tight_layout()
plt.show()

n_show = min(len(md_volumes), 4)
fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
if n_show == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    arr = md_volumes[i].get_array()
    mid = arr.shape[2] // 2
    ax.imshow(arr[:, :, mid].T, cmap="viridis", origin="lower", vmin=0, vmax=3e-3)
    ax.set_title(f"Subject {i + 1}")
    ax.axis("off")
plt.suptitle("Individual MD maps")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Single-metric population template
#
# `build_template` constructs an unbiased group-wise mean template via
# iterative ANTs SyN registration. Each iteration:
#
# 1. Registers every subject to the current template
# 2. Averages the warped images
# 3. Corrects for mean shape bias using the inverse average transform
# 4. Sharpens the result
#
# The initial template is the simple voxel-wise average (no registration).

# %%
from abcdmicro.build_template import average_volumes, build_template

initial_avg = average_volumes(fa_volumes)

# %%
fa_template = build_template(fa_volumes, initial_template=initial_avg, iterations=3)

# %% [markdown]
# Compare the naive average with the registration-based template.

# %%
avg_arr = initial_avg.get_array()
tmpl_arr = fa_template.get_array()
mid = avg_arr.shape[2] // 2

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(avg_arr[:, :, mid].T, cmap="hot", origin="lower", vmin=0)
axes[0].set_title("Simple average (no registration)")
axes[1].imshow(tmpl_arr[:, :, mid].T, cmap="hot", origin="lower", vmin=0, vmax=1)
axes[1].set_title("Population template (3 iterations)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Multi-metric population template
#
# `build_multi_metric_template` uses multiple modalities simultaneously to
# drive the registration. The first modality is used for the affine step;
# all modalities contribute to the deformable (SyN) step.
#
# Each subject is passed as a dictionary mapping modality names to volumes.

# %%
from abcdmicro.build_template import build_multi_metric_template

subject_list = []
for fa, md in zip(fa_volumes, md_volumes):
    subject_list.append({"FA": fa, "MD": md})

# %%
multi_template = build_multi_metric_template(
    subject_list,
    weights={"FA": 1.0, "MD": 1.0},
    iterations=3,
)

# %% [markdown]
# Visualise the multi-metric template for each modality.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

fa_tmpl = multi_template["FA"].get_array()
md_tmpl = multi_template["MD"].get_array()
mid = fa_tmpl.shape[2] // 2

im0 = axes[0].imshow(fa_tmpl[:, :, mid].T, cmap="hot", origin="lower", vmin=0, vmax=1)
axes[0].set_title("FA template")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(md_tmpl[:, :, mid].T, cmap="viridis", origin="lower", vmin=0, vmax=3e-3)
axes[1].set_title("MD template")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

for ax in axes:
    ax.axis("off")
plt.suptitle("Multi-metric population templates")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Save templates to disk

# %%
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

NiftiVolumeResource.save(fa_template, output_dir / "fa_template.nii.gz")

for name, vol in multi_template.items():
    NiftiVolumeResource.save(vol, output_dir / f"{name.lower()}_template_multi.nii.gz")

print(f"Templates saved to {output_dir.resolve()}")
