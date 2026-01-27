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
# 1. Load per-subject scalar maps (e.g. FA, MD from DTI)
# 2. Build a single-metric population template
# 3. Build a multi-metric population template
# 4. Save templates to disk

# %% [markdown]
# ## 1. Load per-subject scalar maps
#
# Point `subject_dirs` to a list of directories, each containing the scalar
# maps you want to use for template construction. All volumes should share the
# same affine (i.e. they should already be in a common rigid/affine space).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from abcdmicro.io import NiftiVolumeResource
from abcdmicro.resource import VolumeResource

# --- Fill in your data paths here ---
subject_dirs = [
    Path("..."),  # e.g. Path("/data/derivatives/sub-001")
    Path("..."),  # e.g. Path("/data/derivatives/sub-002")
    Path("..."),  # e.g. Path("/data/derivatives/sub-003")
]

fa_volumes: list[VolumeResource] = []
md_volumes: list[VolumeResource] = []

for d in subject_dirs:
    fa_volumes.append(NiftiVolumeResource(d / "fa.nii.gz"))
    md_volumes.append(NiftiVolumeResource(d / "md.nii.gz"))

print(f"Loaded {len(fa_volumes)} subjects")

# %% [markdown]
# Quick look at the individual FA maps before building the template.

# %%
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
axes[0].imshow(avg_arr[:, :, mid].T, cmap="hot", origin="lower", vmin=0, vmax=1)
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
