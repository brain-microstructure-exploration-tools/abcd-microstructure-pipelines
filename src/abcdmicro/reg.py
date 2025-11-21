from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import ants
import numpy as np

from abcdmicro.resource import InMemoryVolumeResource, VolumeResource
from abcdmicro.util import update_volume_metadata

# @dataclass
# class RegistrationResource(Resource):
#     @abstractmethod
#     def get_fixed():
#         pass

#     @abstractmethod
#     def get_moving():
#         pass

#     @abstractmethod
#     def get_trasform():
#         pass


# class InMemoryRegistrationResource(RegistrationResource):
#     pass


# def register_volumes(fixed: VolumeResource, moving: VolumeResource, transform_type: str = 'SyN'):
#     """
#     Performs a two-stage Affine and SyNOnly registration for high-accuracy
#     alignment of a moving volume to a fixed volume using ANTs.

#     Args:
#         fixed: The target VolumeResource (reference space).
#         moving: The source VolumeResource that is transformed.
#         transform_type: Placeholder (currently uses fixed Affine + SyNOnly pipeline).

#     Returns:
#         An InMemoryVolumeResource containing the 'moving' volume warped to the
#         'fixed' volume's space.
#     """

#     # TODO: Add check that the input is 3D - what does ANTs have?

#     fixed_image = ants.from_numpy(fixed.get_array())
#     moving_image = ants.from_numpy(moving.get_array())

#     result = ants.registration(
#         fixed=fixed_image,
#         moving=moving_image,
#         type_of_transform="Affine",
#     )

#     fwd_transforms = result['fwdtransforms']
#     # affine or rigid
#     if len(fwd_transforms) == 1:
#         affine = fwd_transforms[0]
#     else:
#         warp = fwd_transforms[0]
#         affine= fwd_transforms[-1]

#     # ---  Deformable Registration  ---
#     # Use Symmetric Normalization (SyN) for local, non-linear deformation.
#     # We use the affine result's transform as the initialization for SyN.
#     deformable_result = ants.registration(
#         fixed=fixed_image,
#         moving=moving_image,
#         type_of_transform="SyNOnly",
#         initial_transform=initial_transform,
#     )

#     moving_image_resliced = deformable_result["warpedmovout"]

#     # Read in transforms

#     return InMemoryVolumeResource(
#         moving_image_resliced,
#         moving.get_affine(),
#         update_volume_metadata(moving.get_metadata(), moving_image_resliced),
#     )


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
    for i, vol in enumerate(volume_list):
        temp = vol.get_array() * weights[i]
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
        ants_image_list.append(
            ants.from_numpy(vol_array)
        )  # This copies the data. https://github.com/ANTsX/ANTsPy/blob/4888946a8b59a55cc784585796423c65268369ab/ants/core/ants_image_io.py#L137

    if initial_template is None:
        current_template = ants.average_images(ants_image_list)
    else:
        ants_initial_template = ants.from_numpy(initial_template.get_array())
        current_template = ants_initial_template.clone()

    # Equal weighting of all images. ANTs has this as an input parameters.
    weights = np.repeat(1.0 / len(ants_image_list), len(ants_image_list))

    for _i in range(iterations):
        affine_list = []

        for idx, moving_image in enumerate(ants_image_list):
            result = ants.registration(
                fixed=current_template,
                moving=moving_image,
                type_of_transform="SyN",  # Can also be 'Rigid'
            )

            # This assumes Syn result where result has L == 2 (warp, affine.mat)
            affine_list.append(result["fwdtransforms"][-1])  # Affine transform is last

            if idx == 0:
                avg_warp = ants.image_read(result["fwdtransforms"][0]) * weights[idx]
                avg_template = result["warpedmovout"] * weights[idx]
            else:
                avg_warp = (
                    avg_warp
                    + ants.image_read(result["fwdtransforms"][0]) * weights[idx]
                )
                avg_template = avg_template + result["warpedmovout"] * weights[idx]

        # Average affine transforms and forward warp fields
        avg_affine_transform = ants.average_affine_transform(affine_list)

        # Save the transformations to file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Apply average transform to updated template
            aff_fn = str(Path(tmpdir) / "avgAffine.mat")
            ants.write_transform(avg_affine_transform, aff_fn)

            gradient_step = 0.2
            wscl = (-1.0) * gradient_step
            avg_warp = avg_warp * wscl
            avg_warp_resliced = ants.apply_transforms(
                fixed=avg_template,
                moving=avg_warp,
                imagetype=1,
                transformlist=aff_fn,
                whichtoinvert=[1],
            )
            wavg_fn = str(Path(tmpdir) / "avgWarp.nii.gz")
            ants.image_write(avg_warp_resliced, wavg_fn)

            current_template = ants.apply_transforms(
                fixed=avg_template,
                moving=avg_template,
                transformlist=[wavg_fn, aff_fn],
                whichtoinvert=[0, 1],
            )

        # Sharpen template
        blending_weight = 0.75
        if blending_weight is not None:
            sharpened_template = ants.iMath(current_template, "Sharpen")
            current_template = (
                current_template * blending_weight
                + sharpened_template * (1.0 - blending_weight)
            )

    # .numpy() also copies the data
    return InMemoryVolumeResource(
        current_template.numpy(),
        volume_list[0].get_affine(),
        update_volume_metadata(volume_list[0].get_metadata(), current_template.numpy()),
    )


def build_multi_metric_template(
    subject_list: list[dict[str, VolumeResource]],
    initial_template: dict[str, VolumeResource] | None = None,
    weights: list[np.floating] | None = None,
    iterations: int = 3,
) -> dict[str, InMemoryVolumeResource]:
    """
    Constructs an unbiased mean shape template from a list of input volumes using
    an iterative group-wise registration approach based on ANTs, utilizing
    multiple image metrics simultaneously. See 'build_template' for more details.

    TODO: Add primary metric argument?
    NOTE: The current implementation assumes input images are roughly pre-aligned.

    Args:
        subject_list: A list of input data, where each subject is a dictionary mapping metric names (string) to their corresponding
        3D image volumes (VolumeResource objects).
        initial_template: An optional starting template volume. If None, the
            initial template is the simple average of all input volumes.
        weights: The weight given to each volume type during the multivariate registration step. If None, equal weights are assumed.
            The length must match the number of volumes specified per subject.
        iterations: The number of iterations for the template refinement process.
    Returns:
        A VolumeResource object representing the final group-wise mean template (per metric)
    """

    # Check that all subjects have the same volumes types
    # The create a list of ANTs images for each type.
    metrics = list(subject_list[0].keys())

    # Equal weighting of all volume types
    if weights is None:
        weights = np.repeat(1.0 / len(metrics), len(metrics))
    assert len(weights) == len(metrics)
    weights = [x / sum(weights) for x in weights]

    ants_image_list = defaultdict(list)
    n_subj = len(subject_list)
    for subject_dict in subject_list:
        if set(subject_dict.keys()) != set(metrics):
            error_message = "All subjects must contain the exact same set of volumes."
            raise ValueError(error_message)

        for m, vol in subject_dict.items():
            vol_array = vol.get_array()
            if vol_array.ndim > 3:  # Check that the input volume is 3D
                error_message = f"Input volume dimensions must be 2D or 3D. Found {vol_array.ndim}D instead."
                raise ValueError(error_message)

            ants_image_list[m].append(ants.from_numpy(vol_array))

    # TODO: Keep or remove this?
    if initial_template is None:
        current_template = {}
        for m in metrics:
            current_template[m] = ants.average_images(ants_image_list[m])
    # else:
    #     #TODO: Update this
    #     ants_initial_template = ants.from_numpy(initial_template.get_array())
    #     current_template = ants_initial_template.clone()

    for _i in range(iterations):
        affine_list = []

        for idx in range(n_subj):
            # Assume first metric is the primary metric for affine.
            # TODO: user can specify this as argument

            # rigid/affine registration to capture global translation, rotation, and scaling
            main_metric = metrics[0]
            main_moving_image = ants_image_list[main_metric][idx]
            mat_result = ants.registration(
                fixed=current_template[main_metric],
                moving=main_moving_image,
                type_of_transform="Affine",  # Can also be 'Rigid'
            )

            affine_transform = mat_result["fwdtransforms"][0]

            multivariate_metrics = []
            for k in range(1, len(metrics)):
                moving_image = ants_image_list[metrics[k]][idx]
                multivariate_metrics.append(
                    [
                        "mattes",
                        current_template[metrics[k]],
                        moving_image,
                        weights[k],
                        1,
                    ]
                )

            # ---  Deformable Registration  ---
            # We use the affine result's transform as the initialization for SyN.
            deformable_result = ants.registration(
                fixed=current_template[main_metric],
                moving=main_moving_image,
                multivariate_extras=multivariate_metrics,
                type_of_transform="SyNOnly",
                initial_transform=affine_transform,
            )

            warp_field = deformable_result["fwdtransforms"][0]
            affine_list.append(affine_transform)
            if idx == 0:
                avg_warp = ants.image_read(warp_field) * 1 / n_subj
                current_template[main_metric] = deformable_result["warpedmovout"] * (
                    1 / n_subj
                )  # This is just the main metric

                # Update template for each metric
                for k in range(1, len(metrics)):
                    moving_image = ants_image_list[metrics[k]][idx]
                    current_template[metrics[k]] = (1 / n_subj) * ants.apply_transforms(
                        fixed=current_template[metrics[k]],
                        moving=moving_image,
                        imagetype=1,
                        transformlist=[
                            warp_field,
                            affine_transform,
                        ],
                        whichtoinvert=[0, 0],
                    )
            else:
                avg_warp = (
                    avg_warp
                    + ants.image_read(deformable_result["fwdtransforms"][0])
                    * 1
                    / n_subj
                )
                current_template[main_metric] = (
                    current_template[main_metric]
                    + deformable_result["warpedmovout"] * 1 / n_subj
                )  # This is just the main metric

                # Update template for each metric
                for k in range(1, len(metrics)):
                    moving_image = ants_image_list[metrics[k]][idx]
                    current_template[metrics[k]] = current_template[metrics[k]] + (
                        1 / n_subj
                    ) * ants.apply_transforms(
                        fixed=current_template[metrics[k]],
                        moving=moving_image,
                        imagetype=1,
                        transformlist=[
                            warp_field,
                            affine_transform,
                        ],
                        whichtoinvert=[0, 0],
                    )

        # Average affine transforms and forward warp fields
        avg_affine_transform = ants.average_affine_transform(affine_list)

        # Save the transformations to file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Apply average transform to updated template
            aff_fn = str(Path(tmpdir) / "avgAffine.mat")
            ants.write_transform(avg_affine_transform, aff_fn)

            gradient_step = 0.2
            wscl = (-1.0) * gradient_step
            avg_warp = avg_warp * wscl

            avg_warp_resliced = ants.apply_transforms(
                fixed=current_template[main_metric],
                moving=avg_warp,
                imagetype=1,
                transformlist=aff_fn,
                whichtoinvert=[1],
            )
            wavg_fn = str(Path(tmpdir) / "avgWarp.nii.gz")
            ants.image_write(avg_warp_resliced, wavg_fn)

            for m in metrics:
                current_template[m] = ants.apply_transforms(
                    fixed=current_template[m],
                    moving=current_template[m],
                    transformlist=[wavg_fn, aff_fn],
                    whichtoinvert=[0, 1],
                )

        # Sharpen template
        blending_weight = 0.75
        if blending_weight is not None:
            for m in metrics:
                sharpened_template = ants.iMath(current_template[m], "Sharpen")
                current_template[m] = current_template[
                    m
                ] * blending_weight + sharpened_template * (1.0 - blending_weight)

    reference_volume = subject_list[0][metrics[0]]

    # Return a dictionary of volume resource templates
    result = {}
    for m in metrics:
        result[m] = InMemoryVolumeResource(
            current_template[m].numpy(),
            reference_volume.get_affine(),
            update_volume_metadata(
                reference_volume.get_metadata(), current_template[m].numpy()
            ),
        )
    return result
