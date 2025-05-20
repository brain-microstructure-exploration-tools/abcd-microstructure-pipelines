from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from abcdmicro.event import AbcdEvent
from abcdmicro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource
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
        return Dwi(
            event=self.event,
            volume=self.volume.load(),
            bval=self.bval.load(),
            bvec=self.bvec.load(),
        )

    def save(self, path: Path, basename: str) -> Dwi:
        """Save all resources to disk and return a Dwi with all resources being on-disk.

        Args:
            path: The desired save directory.
            basename: The desired file basenames, i.e. without an extension.

        Returns: A Dwi with its internal resources being on-disk.
        """
        if path.exists() and not path.is_dir():
            msg = "`path` should be the desired save directory"
            raise ValueError(msg)
        path.mkdir(exist_ok=True, parents=True)
        return Dwi(
            event=self.event,
            volume=NiftiVolumeResource.save(self.volume, path / f"{basename}.nii.gz"),
            bval=FslBvalResource.save(self.bval, path / f"{basename}.bval"),
            bvec=FslBvecResource.save(self.bvec, path / f"{basename}.bvec"),
        )
