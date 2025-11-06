from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tractseg.python_api import run_tractseg

from abcdmicro.fods import combine_csd_peaks_to_vector_volume, compute_csd_peaks
from abcdmicro.resource import VolumeResource
from abcdmicro.util import create_estimate_volume_resource

if TYPE_CHECKING:
    from abcdmicro.dwi import Dwi


def extract_tractseg(
    dwi: Dwi, mask: VolumeResource, output_type: str = "tract_segmentation"
) -> VolumeResource:
    """Run TractSeg on a DWI dataset to segment white matter tracts.

    Args:
        dwi: The DWI dataset.
        mask: A brain mask VolumeResource.
        output_type: TractSeg can segment not only bundles, but also the end regions of bundles.
            Moreover it can create Tract Orientation Maps (TOM).
            'tract_segmentation' [DEFAULT]: Segmentation of bundles (72 bundles).
            'endings_segmentation': Segmentation of bundle end regions (72 bundles).
            'TOM': Tract Orientation Maps (20 bundles).

    Returns: A volume resource containing a 4D numpy array with the output of tractseg
        for tract_segmentation:     [x, y, z, nr_of_bundles]
        for endings_segmentation:   [x, y, z, 2*nr_of_bundles]
        for TOM:                    [x, y, z, 3*nr_of_bundles]
    """

    # Compute CSD peaks
    # MRtrix by default outputs 3 peaks not the dipy default of 5
    # Note: MRtrix3 performs peak finding internally via Newton optimization search.
    # Dipy using discrete sampling approach
    csd_peaks = compute_csd_peaks(dwi, mask, flip_bvecs_x=True, n_peaks=3)
    csd_peaks_vector = combine_csd_peaks_to_vector_volume(
        csd_peaks_dirs=csd_peaks[0], csd_peaks_values=csd_peaks[1]
    )
    # Run TractSeg
    # run_tractseg(data, output_type="tract_segmentation",
    #              single_orientation=False, dropout_sampling=False, threshold=0.5,
    #              bundle_specific_postprocessing=True, get_probs=False, peak_threshold=0.1,
    #              postprocess=False, peak_regression_part="All", input_type="peaks",
    #              blob_size_thr=50, nr_cpus=-1, verbose=False, manual_exp_name=None,
    #              inference_batch_size=1, tract_definition="TractQuerier+", bedpostX_input=False,
    #              tract_segmentations_path=None, TOM_dilation=1, unit_test=False):
    logging.info("Running tractseg...")
    segmentation = run_tractseg(
        data=csd_peaks_vector.get_array(), output_type=output_type
    )

    return create_estimate_volume_resource(
        array=segmentation,
        reference_volume=dwi.volume,
        intent_name="TRACTSEG_SEGMENTATION",
    )
