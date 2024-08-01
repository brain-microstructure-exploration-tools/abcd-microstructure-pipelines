from __future__ import annotations

from abcdmicro.resource import BvalResource, BvecResource, VolumeResource


class NiftiVolumeResrouce(VolumeResource):
    """A volume or volume stack that is saved to disk in the nifti file format."""


class FslBvalResource(BvalResource):
    """A b-value list that is saved to disk in the FSL text file format."""


class FslBvecResource(BvecResource):
    """A b-vector list that is saved to disk in the FSL text file format."""
