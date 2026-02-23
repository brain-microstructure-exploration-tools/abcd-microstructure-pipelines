from __future__ import annotations

from typing import TYPE_CHECKING

from dipy.denoise.patch2self import patch2self

from kwneuro.resource import InMemoryVolumeResource
from kwneuro.util import update_volume_metadata

if TYPE_CHECKING:
    from kwneuro.dwi import Dwi


def denoise_dwi(dwi: Dwi) -> InMemoryVolumeResource:
    """Run denoising on a DWI using Patch2Self from DIPY.

    :param input_dwi: Input DWI to be denoised.

    Returns the denoised DWI.
    """

    data_denoised = patch2self(
        dwi.volume.get_array(),
        dwi.bval.get(),
        shift_intensity=True,
        clip_negative_vals=False,
        b0_threshold=5,
        verbose=False,
    )

    return InMemoryVolumeResource(
        array=data_denoised,
        affine=dwi.volume.get_affine(),
        metadata=update_volume_metadata(
            dwi.volume.get_metadata(),
            data_denoised,
        ),
    )
