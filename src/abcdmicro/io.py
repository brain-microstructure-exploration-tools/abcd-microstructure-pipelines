from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import itk
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
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
class NiftiVolumeResrouce(VolumeResource):
    """A volume or volume stack that is saved to disk in the nifti file format."""

    path: Path
    """Path to the underlying volume nifti file"""

    def load(self) -> InMemoryVolumeResource:
        return InMemoryVolumeResource(itk.imread(self.path))

    def get_array(self) -> NDArray[Any]:
        return self.load().get_array()

    def get_metadata(self) -> dict[Any, Any]:
        return self.load().get_metadata()


@dataclass
class FslBvalResource(BvalResource):
    """A b-value list that is saved to disk in the FSL text file format."""

    path: Path
    """Path to the underlying bval txt file"""

    def load(self) -> InMemoryBvalResource:
        bvals_array, _ = read_bvals_bvecs(self.path, None)
        return InMemoryBvalResource(bvals_array)

    def get(self) -> NDArray[np.floating]:
        return self.load().get()


@dataclass
class FslBvecResource(BvecResource):
    """A b-vector list that is saved to disk in the FSL text file format."""

    path: Path
    """Path to the underlying bvec txt file"""

    def load(self) -> InMemoryBvecResource:
        _, bvecs_array = read_bvals_bvecs(None, self.path)
        return InMemoryBvecResource(bvecs_array)

    def get(self) -> NDArray[np.floating]:
        return self.load().get()
