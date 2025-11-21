from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from dipy.core.geometry import cart2sphere
from dipy.core.gradients import GradientTable
from dipy.reconst.csdeconv import AxSymShResponse, estimate_response
from dipy.reconst.shm import (
    lazy_index,
    real_sh_descoteaux_from_index,
    sph_harm_ind_list,
)
from numpy.typing import NDArray, Tuple


class Resource(ABC):
    """Base class for all Resources. A Resource is a piece of data that could live in memory or on disk."""

    is_loaded: ClassVar[bool] = True
    """Whether a resource corresponds to in-memory data, rather than for example on-disk data."""

    @abstractmethod
    def load(self) -> Resource:
        """Load a Resource. Specific functionality depends on the Resource subclass, but if the resource is considered to be
        'loaded' then this should be a no-op, returning the same Resource back."""


class VolumeResource(Resource):
    """Base class for resources representing a volume or volume stack.
    An n-D array where n >= 3 and where three of the dimensions are spatial
    and have associated header information describing a patient coordinate system."""

    @abstractmethod
    def get_array(self) -> NDArray[np.number]:
        """Get the underlying volume data array"""

    @abstractmethod
    def get_affine(self) -> NDArray[np.floating]:
        """Get the 4x4 affine matrix that maps index space to patient/scanner space"""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Get the volume image metadata"""

    def load(self) -> VolumeResource:
        return self


@dataclass
class InMemoryVolumeResource(VolumeResource):
    """A volume resource that is loaded into memory.
    An n-D array where n >= 3 and where three of the dimensions are spatial
    and have associated header information describing a patient coordinate system."""

    is_loaded: ClassVar[bool] = True

    array: NDArray[np.number]
    affine: NDArray[np.floating] = field(default_factory=lambda: np.eye(4))
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_array(self) -> NDArray[Any]:
        return self.array

    def get_affine(self) -> NDArray[np.floating]:
        return self.affine

    def get_metadata(self) -> dict[Any, Any]:
        return self.metadata


class BvalResource(Resource):
    """Base class for resources representing a list of b-values associated with a 4D DWI
    volume stack."""

    @abstractmethod
    def get(self) -> NDArray[np.floating]:
        """Get the underlying array of b-values"""

    def load(self) -> BvalResource:
        return self


@dataclass
class InMemoryBvalResource(BvalResource):
    """A b-value list that is loaded into memory."""

    is_loaded: ClassVar[bool] = True

    array: NDArray[np.floating]
    """The underlying array of b-values"""

    def get(self) -> NDArray[np.floating]:
        return self.array


class BvecResource(Resource):
    """Base class for resources representing a list of b-vectors associated with a 4D DWI
    volume stack."""

    @abstractmethod
    def get(self) -> NDArray[np.floating]:
        """Get the underlying array of b-vectors of shape (N,3)"""

    def load(self) -> BvecResource:
        return self


@dataclass
class InMemoryBvecResource(BvecResource):
    """A b-vector list that is loaded into memory."""

    is_loaded: ClassVar[bool] = True

    array: NDArray[np.floating]
    """The underlying array of b-vectors"""

    def __post_init__(self) -> None:
        # Check that b-vectors have the expected shape
        if self.array.ndim != 2 or self.array.shape[1] != 3:
            msg = f"Encountered wrong b-vector array shape {self.array.shape}. Expected shape (N,3)."
            raise ValueError(msg)

    def get(self) -> NDArray[np.floating]:
        return self.array


# class OldResponseFunctionResource(Resource):
#     """Base class for resources representing a response function associated with a DWI."""

#     @abstractmethod
#     def get(self) -> tuple[NDArray, np.floating]:
#         """Get the underlying response function"""

#     def load(self) -> ResponseFunctionResource:
#         return self


# @dataclass
# class OldInMemoryResponseFunctionResource(OldResponseFunctionResource):
#     """A response function that is loaded into memory."""

#     is_loaded: ClassVar[bool] = True

#     evals: NDArray[np.floating]
#     """Eigenvales of the response function"""

#     avg_signal: np.floating
#     """ The average non-diffusion weighted signal within the voxels used to calculate the response function"""

#     def __post_init__(self) -> None:
#         # Check that the eigen values have the expected shape
#         if self.evals.ndim != 1 or self.evals.shape[0] != 3:
#             msg = f"Encountered wrong eigen values array shape {self.evals.shape}. Expected shape (3,)."
#             raise ValueError(msg)

#     def get(self) -> tuple[NDArray, np.floating]:
#         return (self.evals, self.avg_signal)


class ResponseFunctionResource(Resource):
    """Base class for resources representing a response function associated with a DWI."""

    @abstractmethod
    def get(self) -> tuple[NDArray, np.floating]:
        """Get the underlying response function"""

    @abstractmethod
    def get_dipy_object(self) -> AxSymShResponse:
        """Get the underlying response function in a format compatible with Dipy"""

    def load(self) -> ResponseFunctionResource:
        return self


@dataclass
class InMemoryResponseFunctionResource(ResponseFunctionResource):
    """A response function that is loaded into memory."""

    is_loaded: ClassVar[bool] = True

    sh_coeffs: NDArray[np.floating]
    """Response function signal as coefficients to axially symmetric, even spherical harmonic."""

    avg_signal: np.floating
    """ The average non-diffusion weighted signal within the voxels used to calculate the response function"""

    def __post_init__(self) -> None:
        # Check that the eigen values have the expected shape
        if self.sh_coeffs.ndim != 1 or self.sh_coeffs.shape[0] != 3:
            msg = f"Encountered wrong eigen values array shape {self.evals.shape}. Expected shape (3,)."
            raise ValueError(msg)

    def get(self) -> tuple[NDArray, np.floating]:
        return (self.sh_coeffs, self.avg_signal)

    @staticmethod
    def estimate_from_prolate_tensor(
        response: Tuple[NDArray, np.floating],
        gtab: GradientTable,
        sh_order_max: int = 8,
    ):
        """Re-implmentation of the conversion performed in Dipy here:
        https://github.com/dipy/dipy/blob/f7b863f1485cd3fa6329c8e8f3388d8f58863f0d/dipy/reconst/csdeconv.py#L168.
        Convert from the tuple format to sh coeffs"""

        m_values, l_values = sph_harm_ind_list(sh_order_max)
        _where_dwi = lazy_index(~gtab.b0s_mask)

        x, y, z = gtab.gradients[_where_dwi].T
        _, theta, phi = cart2sphere(x, y, z)
        B_dwi = real_sh_descoteaux_from_index(
            m_values, l_values, theta[:, None], phi[:, None]
        )

        S_r = estimate_response(gtab, response[0], response[1])
        sh_coeffs = np.linalg.lstsq(B_dwi, S_r[_where_dwi], rcond=-1)[0]

        return InMemoryResponseFunctionResource(
            sh_coeffs=sh_coeffs, avg_signal=response[1]
        )

    def get_dipy_object(self) -> AxSymShResponse:
        return AxSymShResponse(S0=self.avg_signal, dwi_response=self.sh_coeffs)
