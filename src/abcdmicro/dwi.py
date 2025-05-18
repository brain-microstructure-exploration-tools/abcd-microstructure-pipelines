from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from abcdmicro.event import AbcdEvent
from abcdmicro.resource import (
    BvalResource,
    BvecResource,
    VolumeResource,
)


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

    def load(self) -> Dwi:
        """Load any on-disk resources into memory and return a Dwi with all loadable resources loaded."""
        raise NotImplementedError("TODO")

    def save(self, path: Path) -> Dwi:
        """Save all resources to disk and return a Dwi with all resources being on-disk. Provide a save directory in `path`."""
        raise NotImplementedError("TODO")
