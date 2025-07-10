from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray


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
