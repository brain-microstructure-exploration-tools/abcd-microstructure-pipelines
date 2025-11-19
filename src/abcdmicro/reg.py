from __future__ import annotations

import tempfile
from pathlib import Path

import ants
import numpy as np

from abcdmicro.resource import InMemoryVolumeResource, VolumeResource
from abcdmicro.util import update_volume_metadata


def average_volumes(volume_list: list[VolumeResource]) -> VolumeResource:
    """
    Calculates the simple arithmetic average (mean) of a list of volumes.
    Args:
        volume_list: A list of VolumeResource objects to be averaged.
                     All volumes are treated with equal weight (1/N).
    Returns:
        A VolumeResource object containing the element-wise arithmetic mean of all input volumes.
    """

    ref_volume = volume_list[0]
    weights = np.repeat(
        1.0 / len(volume_list), len(volume_list)
    )  # Equal weighting for all volumes
    average_volume = volume_list[0].get_array() * 0
    for i in range(len(volume_list)):
        temp = volume_list[i].get_array() * weights[i]
        average_volume = average_volume + temp

    return InMemoryVolumeResource(
        average_volume,
        ref_volume.get_affine(),
        update_volume_metadata(ref_volume.get_metadata(), average_volume),
    )


def build_template(
    volume_list: list[VolumeResource],
    initial_template: VolumeResource | None = None,
    iterations: int = 3,
) -> VolumeResource:
    """
    Constructs an unbiased mean shape template from a list of input volumes using
    an iterative group-wise registration approach based on ANTs.

    The process follows the standard iterative unbiasing approach:
    1. Register all images to the current template (using SyN and Affine transforms).
    2. Average the warped images to create a new template estimate.
    3. Average the resulting transformations (warp and affine) to calculate the mean shift.
    4. Apply the inverse of the mean shift to the template to correct bias toward the true mean.
    5. Sharpen the template to enhance edge definition.

    NOTE: The current implementation assumes input images are roughly pre-aligned.

    Args:
        volume_list: A list of input 3D image volumes (VolumeResource objects).
        initial_template: An optional starting template volume. If None, the
                            initial template is the simple average of all input volumes.
        iterations: The number of iterations for the template refinement process.
    Returns:
        A VolumeResource object representing the final group-wise mean template.
    """

    # Convert from volume resource to ants image
    ants_image_list = []
    for vol in volume_list:
        vol_array = vol.get_array()
        if vol_array.ndim > 3:  # Check that the input volume is 3D
            error_message = f"Input volume dimensions must be 2D or 3D. Found {vol_array.ndim}D instead."
            raise ValueError(error_message)
        ants_image_list.append(ants.from_numpy(vol_array))

    if initial_template is None:
        current_template = ants.average_images(ants_image_list)
    else:
        ants_initial_template = ants.from_numpy(initial_template.get_array())
        current_template = ants_initial_template.clone()

    # Equal weighting of all images. ANTs has this as an input parameters.
    weights = np.repeat(1.0 / len(ants_image_list), len(ants_image_list))

    for _i in range(iterations):
        affine_list = []
        avg_warp = None
        avg_template = None

        for idx, moving_image in enumerate(ants_image_list):
            result = ants.registration(
                fixed=current_template,
                moving=moving_image,
                type_of_transform="SyN",  # Can also be 'Rigid'
            )

            # This assumes Syn result where result has L == 2 (warp, affine.mat)
            affine_list.append(result["fwdtransforms"][-1])  # Affine transform is last
            avg_warp = avg_warp + result["fwdtransforms"][0] * weights[idx]
            avg_template = avg_template + result["warpedmovout"] * weights[idx]

        # Average affine transforms and forward warp fields
        avg_affine_transform = ants.average_affine_transform(affine_list)

        # Save the transformations to file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Apply average transform to updated template
            aff_fn = str(Path(tmpdir) / "avgAffine.mat")
            ants.write_transform(avg_affine_transform, aff_fn)

            wavgA = ants.apply_transforms(
                fixed=avg_template,
                moving=avg_warp,
                imagetype=1,
                transformlist=aff_fn,
                whichtoinvert=[1],
            )
            wavg_fn = str(Path(tmpdir) / "avgWarp.nii.gz")
            ants.image_write(wavgA, wavg_fn)

            updated_template = ants.apply_transforms(
                fixed=avg_template,
                moving=avg_template,
                transformlist=[wavg_fn, aff_fn],
                whichtoinvert=[0, 1],
            )

        # Sharpen template
        blending_weight = 0.75
        if blending_weight is not None:
            updated_template = updated_template * blending_weight + ants.iMath(
                updated_template, "Sharpen"
            ) * (1.0 - blending_weight)

    return InMemoryVolumeResource(
        updated_template.numpy(),
        volume_list[0].get_affine(),
        update_volume_metadata(volume_list[0].get_metadata(), current_template.numpy()),
    )
