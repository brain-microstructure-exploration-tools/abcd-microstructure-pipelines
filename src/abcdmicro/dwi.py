from __future__ import annotations

from dataclasses import dataclass

from abcdmicro.event import AbcdEvent
from abcdmicro.resource import BvalResource, BvecResource, VolumeResource


@dataclass
class Dwi:
    """An ABCD diffusion weighted image."""

    event: AbcdEvent
    """The ABCD event associated with this DWI."""

    volume: VolumeResource
    """The DWI image volume."""

    bval: BvalResource
    """The DWI b-values"""

    bvec: BvecResource
    """The DWI b-vectors"""
