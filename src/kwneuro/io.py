from __future__ import annotations

import json
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import save_nifti
from nibabel.nifti1 import Nifti1Header
from numpy.typing import NDArray

from kwneuro.resource import (
    BvalResource,
    BvecResource,
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryResponseFunctionResource,
    InMemoryVolumeResource,
    ResponseFunctionResource,
    VolumeResource,
)
from kwneuro.util import PathLike, normalize_path

if TYPE_CHECKING:
    from dipy.reconst.csdeconv import AxSymShResponse


@dataclass
class NiftiVolumeResource(VolumeResource):
    """A volume or volume stack that is saved to disk in the nifti file format."""

    is_loaded: ClassVar[bool] = False

    path_in: InitVar[PathLike]
    """Path to the underlying volume nifti file"""

    path: Path = field(init=False)

    def __post_init__(self, path_in: PathLike) -> None:
        self.path = normalize_path(path_in)

    def load(self) -> InMemoryVolumeResource:
        """Load volume into memory"""
        img = nib.load(self.path, mmap=False)
        return InMemoryVolumeResource(
            array=img.get_fdata(), affine=img.affine, metadata=dict(img.header)
        )

    def get_array(self) -> NDArray[np.number]:
        return self.load().get_array()

    def get_affine(self) -> NDArray[np.floating]:
        return self.load().get_affine()

    def get_metadata(self) -> dict[str, Any]:
        return self.load().get_metadata()

    @staticmethod
    def save(vol: VolumeResource, path: PathLike) -> NiftiVolumeResource:
        """Save volume data to a path, creating a NiftiVolumeResource."""
        path = normalize_path(path)
        header = Nifti1Header()
        for key, val in vol.get_metadata().items():
            header[key] = val
        save_nifti(
            fname=path,
            data=vol.get_array(),
            affine=vol.get_affine(),
            hdr=header,
        )
        return NiftiVolumeResource(path)


@dataclass
class FslBvalResource(BvalResource):
    """A b-value list that is saved to disk in the FSL text file format."""

    is_loaded: ClassVar[bool] = False

    path_in: InitVar[PathLike]
    """Path to the underlying bval txt file"""

    path: Path = field(init=False)

    def __post_init__(self, path_in: PathLike) -> None:
        self.path = normalize_path(path_in)

    def load(self) -> InMemoryBvalResource:
        """Load b-values into memory"""
        bvals_array, _ = read_bvals_bvecs(str(self.path), None)
        return InMemoryBvalResource(bvals_array)

    def get(self) -> NDArray[np.floating]:
        return self.load().get()

    @staticmethod
    def save(bvals: BvalResource, path: PathLike) -> FslBvalResource:
        """Save data to a path, creating a FslBvalResource."""
        path = normalize_path(path)
        np.savetxt(path, bvals.get(), fmt="%g")
        return FslBvalResource(path)


@dataclass
class FslBvecResource(BvecResource):
    """A b-vector list that is saved to disk in the FSL text file format."""

    is_loaded: ClassVar[bool] = False

    path_in: InitVar[PathLike]
    """Path to the underlying bvec txt file"""

    path: Path = field(init=False)

    def __post_init__(self, path_in: PathLike) -> None:
        self.path = normalize_path(path_in)

    def load(self) -> InMemoryBvecResource:
        """Load b-vectors into memory"""
        _, bvecs_array = read_bvals_bvecs(None, str(self.path))
        return InMemoryBvecResource(bvecs_array)

    def get(self) -> NDArray[np.floating]:
        return self.load().get()

    @staticmethod
    def save(bvecs: BvecResource, path: PathLike) -> FslBvecResource:
        """Save data to a path, creating a FslBvecResource."""
        path = normalize_path(path)
        np.savetxt(
            path,
            bvecs.get().T,  # transpose (N,3) to (3,N) for writing
            fmt="%.6f",
            delimiter=" ",
        )
        return FslBvecResource(path)


@dataclass
class JsonResponseFunctionResource(ResponseFunctionResource):
    """A response function that is saved to disk in a json file."""

    is_loaded: ClassVar[bool] = False

    path_in: InitVar[PathLike]
    """Path to the underlying json file"""

    path: Path = field(init=False)

    def __post_init__(self, path_in: PathLike) -> None:
        self.path = normalize_path(path_in)

    def get(self) -> tuple[NDArray, np.floating]:
        """Get the underlying response function"""
        return self.load().get()

    def get_dipy_object(self) -> AxSymShResponse:
        """Get the underlying response function in a format compatible with Dipy"""
        return self.load().get_dipy_object()

    def load(self) -> InMemoryResponseFunctionResource:
        """Load volume into memory"""
        with self.path.open("r", encoding="utf-8") as file:
            response_list = json.load(file)
            return InMemoryResponseFunctionResource(
                sh_coeffs=np.array(response_list[0], dtype=float),
                avg_signal=np.float32(response_list[1]),
            )

    @staticmethod
    def save(
        response: ResponseFunctionResource, path: PathLike
    ) -> JsonResponseFunctionResource:
        """Save response function data to a path, creating a JsonResponseFunctionResource."""
        path = normalize_path(path)
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                [
                    [coeff.item() for coeff in response.get()[0]],
                    response.get()[
                        1
                    ].item(),  # json won't serialize numpy float type, it has to be native python float, hence ".item()"
                ],
                file,
            )
        return JsonResponseFunctionResource(path)
