from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import (
    ConstrainedSphericalDeconvModel,
    mask_for_response_ssst,
    response_from_mask_ssst,
)
from dipy.reconst.shm import convert_sh_descoteaux_tournier

from abcdmicro.resource import (
    InMemoryResponseFunctionResource,
    ResponseFunctionResource,
    VolumeResource,
)
from abcdmicro.util import create_estimate_volume_resource

if TYPE_CHECKING:
    from abcdmicro.dwi import Dwi


def estimate_response_function(
    dwi: Dwi, mask: VolumeResource, flip_bvecs_x: bool = True
) -> InMemoryResponseFunctionResource:
    """Estimate the single-shell single-tissue response function from a DWI dataset using the SSST method.
    Args:
        dwi: The Diffusion Weighted Imaging (DWI) dataset.
        mask: A binary brain mask volume. This is used to extract an ROI at the center of the brain.
        flip_bvecs_x: Whether to flip the x-component of the b-vectors to match MRtrix3 conventio
    Returns: A resource containing the estimated single-tissue response function.
    """

    # Load data as numpy arrays
    bvals = dwi.bval.get()
    bvecs = dwi.bvec.get()
    volume_data = dwi.volume.get_array()
    mask_data = mask.get_array().astype(int)

    if flip_bvecs_x:
        bvecs[:, 0] = -bvecs[:, 0]

    # b-values above 1200 aren't great for DTI estimation. dipy uses DTI to model response functions.
    low_b_mask = bvals <= 1200
    gtab_low_b = gradient_table(bvals[low_b_mask], bvecs=bvecs[low_b_mask])
    data_low_b = volume_data[..., low_b_mask]

    # determine roi center based on brain mask (aiming for corpus callosum)
    i, j, k = np.where(mask_data)
    roi_center = np.round([i.mean(), j.mean(), k.mean()]).astype(int)  # COM of mask
    roi_radii = [(indices.max() - indices.min()) // 4 for indices in (i, j, k)]

    # get mask of voxels to use for response estimate
    mask_for_response = mask_for_response_ssst(
        gtab_low_b,
        data_low_b,
        roi_center=roi_center,
        roi_radii=roi_radii,
        fa_thr=0.8,
    )
    mask_for_response *= mask_data  # ensure we stay inside brain mask (almost certainly we already were but just in case)

    if mask_for_response.sum() < 100:
        logging.warning(
            "There are less than 100 voxels in the mask used to estimate the response function."
        )
        # If this happens maybe we need to decrease fa_thr or check what might be the issue

    # perform response function estimate using the selected voxels
    response, ratio = response_from_mask_ssst(
        gtab_low_b,
        data_low_b,
        mask_for_response,
    )

    if ratio > 0.3:
        logging.warning(
            "Ratio of response diffusion tensor eigenvalues is greater than 0.3. For a response function we expect more prolateness. Something may be wrong."
        )

    return InMemoryResponseFunctionResource(evals=response[0], avg_signal=response[1])


def compute_csd_fods(
    dwi: Dwi,
    mask: VolumeResource,
    response: ResponseFunctionResource | None = None,
    flip_bvecs_x: bool = True,
    mrtrix_format: bool = False,
    sh_order_max: int = 8,
) -> np.ndarray:
    """Computes Fiber Orientation Distributions (FODs) from a DWI dataset using Constrained Spherical Deconvolution (CSD).

    Args:
        dwi: The Diffusion Weighted Imaging (DWI) dataset.
        mask: A binary brain mask volume. FODs are computed only within this mask.
        response (Optional): The single-fiber response function. If `None`, the response function is estimated using an ROI in the center of the brain mask.
        flip_bvecs_x (Optional): Whether to flip the x-component of the b-vectors to match MRtrix3 convention.
        MRtrix3_format (Optional): If True, converts SH coefficients between legacy-descoteaux07 and tournier07.
        sh_order_max (Optional): Maximum spherical harmonic order to use in the CSD model. Default is 8.
    Returns: Array containing the spherical harmonic coefficients of the obtained FODs.
    """

    # Load data as numpy arrays
    bvals = dwi.bval.get()
    bvecs = dwi.bvec.get()
    volume_data = dwi.volume.get_array()
    mask_data = mask.get_array().astype(int)

    if flip_bvecs_x:
        bvecs[:, 0] = -bvecs[:, 0]

    if not response:
        logging.info("Estimating single-shell single-tissue response function...")
        response = estimate_response_function(
            dwi=dwi, mask=mask, flip_bvecs_x=flip_bvecs_x
        )

    gtab = gradient_table(bvals, bvecs=bvecs)
    csd_model = ConstrainedSphericalDeconvModel(
        gtab, response.get(), sh_order_max=sh_order_max
    )
    csd_fit = csd_model.fit(volume_data, mask=mask_data)

    coeffs = csd_fit.shm_coeff

    if mrtrix_format:
        coeffs = convert_sh_descoteaux_tournier(csd_fit.shm_coeff)

    return coeffs


def compute_csd_peaks(
    dwi: Dwi,
    mask: VolumeResource,
    response: ResponseFunctionResource | None = None,
    flip_bvecs_x: bool = True,
    n_peaks: int = 5,
    relative_peak_threshold: float = 0.5,
    min_separation_angle: float = 25,
) -> tuple[VolumeResource, VolumeResource]:
    """
    Compute Constrained Spherical Deconvolution peaks from a DWI resource. This involves
    estimating the response function, fitting the CSD model, and extracting the peaks.
    Args:
        dwi: The Diffusion Weighted Imaging (DWI) dataset.
        mask: A binary brain mask volume. CSD is computed only within this mask.
        response (Optional): The single-fiber response function. If `None`, the response function is estimated using an ROI in the center of the brain mask.
        flip_bvecs_x (Optional): Whether to flip the x-component of the b-vectors to match MRtrix3 convention.
        n_peaks (Optional): Number of peaks to extract per voxel. Default is 5.
        relative_peak_threshold (Optional): Only return peaks greater than relative_peak_threshold * m where m is the largest peak.
        min_separation_angle (Optional): The minimum distance between directions. If two peaks are too close only the larger of the two is
        returned. Must be in range [0,90]
    Returns: A tuple of VolumeResources containing the CSD peak directions stored as a 5-D array of shape [x,y,z,n_peaks,3],
    and the corresponding peak values stored as a 4D array of shape [x,y,z,n_peaks].
    """
    # Load data as numpy arrays
    bvals = dwi.bval.get()
    bvecs = dwi.bvec.get()
    volume_data = dwi.volume.get_array()
    mask_data = mask.get_array().astype(int)

    if flip_bvecs_x:
        bvecs[:, 0] = -bvecs[:, 0]  # flip x axis to match MRtrix3 convention

    if not response:
        logging.info("Estimating single-shell single-tissue response function...")
        response = estimate_response_function(
            dwi=dwi, mask=mask, flip_bvecs_x=flip_bvecs_x
        )

    gtab = gradient_table(bvals, bvecs=bvecs)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response.get(), sh_order_max=8)

    logging.info("Computing peaks...")
    csd_peaks = peaks_from_model(
        model=csd_model,
        data=volume_data,
        mask=mask_data,
        sphere=default_sphere,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,  # minimum angular distance required between any two peaks. Filtering step after peaks are found
        npeaks=n_peaks,
    )

    peak_dirs = create_estimate_volume_resource(
        array=csd_peaks.peak_dirs,
        reference_volume=dwi.volume,
        intent_name="CSD_PEAK_DIRS",
    )
    peak_values = create_estimate_volume_resource(
        array=csd_peaks.peak_values,
        reference_volume=dwi.volume,
        intent_name="CSD_PEAK_DIRS",
    )

    return (peak_dirs, peak_values)


def combine_csd_peaks_to_vector_volume(
    csd_peaks_dirs: VolumeResource, csd_peaks_values: VolumeResource
) -> VolumeResource:
    """
    Convert CSD peaks from separate direction and value VolumeResources (e.g. Dipy format) into a single volume
    containing a 4D array, where the last dimension is a n_peaks*3-length vector containing
    the x,y,z components of each peak (e.g. MRtrix3 format). The magnitude of each vector (peak value) is stored as its length (norm).
    Args:
        csd_peaks_dirs: VolumeResource containing peak directions (unit vectors).
        csd_peaks_values: VolumeResource containing peak values (amplitudes).
    Returns: VolumeResource with 4D array of shape [x,y,z,n_peaks*3].
    """
    dirs = csd_peaks_dirs.get_array()
    values = csd_peaks_values.get_array()

    values_expanded = np.expand_dims(values, axis=-1)
    csd_peaks = dirs * values_expanded
    spatial_dims = csd_peaks.shape[:-2]
    n_peak_dims = csd_peaks.shape[-2] * 3
    peaks_array = csd_peaks.reshape(*spatial_dims, n_peak_dims)

    return create_estimate_volume_resource(
        array=peaks_array,
        reference_volume=csd_peaks_dirs,
        intent_name="CSD_PEAK_VECTORS",
    )
