from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from nibabel.nifti1 import Nifti1Header
from numpy.typing import NDArray

from abcdmicro.resource import (
    BvalResource,
    BvecResource,
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
    VolumeResource,
)


@dataclass
class NiftiVolumeResource(VolumeResource):
    """A volume or volume stack that is saved to disk in the nifti file format."""

    path: Path
    """Path to the underlying volume nifti file"""

    def load(self) -> InMemoryVolumeResource:
        """Load volume into memory"""
        data, affine, img = load_nifti(self.path, return_img=True)
        return InMemoryVolumeResource(
            array=data, affine=affine, metadata=dict(img.header)
        )

    def get_array(self) -> NDArray[np.number]:
        return self.load().get_array()

    def get_affine(self) -> NDArray[np.floating]:
        return self.load().get_affine()

    def get_metadata(self) -> dict[str, Any]:
        return self.load().get_metadata()

    @staticmethod
    def save(vol: VolumeResource, path: Path) -> NiftiVolumeResource:
        """Save volume data to a path, creating a NiftiVolumeResource."""
        header = Nifti1Header()
        for key, val in vol.get_metadata().items():
            header[key] = val
        save_nifti(
            fname=path,
            data=vol.get_array(),
            affine=vol.get_affine(),
            hdr=header,
        )
        return NiftiVolumeResource(path=path)


@dataclass
class FslBvalResource(BvalResource):
    """A b-value list that is saved to disk in the FSL text file format."""

    path: Path
    """Path to the underlying bval txt file"""

    def load(self) -> InMemoryBvalResource:
        """Load b-values into memory"""
        bvals_array, _ = read_bvals_bvecs(str(self.path), None)
        return InMemoryBvalResource(bvals_array)

    def get(self) -> NDArray[np.floating]:
        return self.load().get()

    @staticmethod
    def save(bvals: BvalResource, path: Path) -> FslBvalResource:
        """Save data to a path, creating a FslBvalResource."""
        np.savetxt(path, bvals.get(), fmt="%g")
        return FslBvalResource(path)


@dataclass
class FslBvecResource(BvecResource):
    """A b-vector list that is saved to disk in the FSL text file format."""

    path: Path
    """Path to the underlying bvec txt file"""

    def load(self) -> InMemoryBvecResource:
        """Load b-vectors into memory"""
        _, bvecs_array = read_bvals_bvecs(None, str(self.path))
        return InMemoryBvecResource(bvecs_array)

    def get(self) -> NDArray[np.floating]:
        return self.load().get()

    @staticmethod
    def save(bvecs: BvecResource, path: Path) -> FslBvecResource:
        """Save data to a path, creating a FslBvecResource."""
        np.savetxt(
            path,
            bvecs.get().T,  # transpose (N,3) to (3,N) for writing
            fmt="%.6f",
            delimiter=" ",
        )
        return FslBvecResource(path)
