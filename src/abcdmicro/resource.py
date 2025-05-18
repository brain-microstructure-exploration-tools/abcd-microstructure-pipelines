from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


class VolumeResource(ABC):
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


@dataclass
class InMemoryVolumeResource(VolumeResource):
    """A volume resource that is loaded into memory.
    An n-D array where n >= 3 and where three of the dimensions are spatial
    and have associated header information describing a patient coordinate system."""

    array: NDArray[np.number]
    affine: NDArray[np.floating] = field(default_factory=lambda: np.eye(4))
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_array(self) -> NDArray[Any]:
        return self.array

    def get_affine(self) -> NDArray[np.floating]:
        return self.affine

    def get_metadata(self) -> dict[Any, Any]:
        return self.metadata


class BvalResource(ABC):
    """Base class for resources representing a list of b-values associated with a 4D DWI
    volume stack."""

    @abstractmethod
    def get(self) -> NDArray[np.floating]:
        """Get the underlying array of b-values"""


@dataclass
class InMemoryBvalResource(BvalResource):
    """A b-value list that is loaded into memory."""

    array: NDArray[np.floating]
    """The underlying array of b-values"""

    def get(self) -> NDArray[np.floating]:
        return self.array


class BvecResource(ABC):
    """Base class for resources representing a list of b-vectors associated with a 4D DWI
    volume stack."""

    @abstractmethod
    def get(self) -> NDArray[np.floating]:
        """Get the underlying array of b-vectors of shape (N,3)"""


@dataclass
class InMemoryBvecResource(BvecResource):
    """A b-vector list that is loaded into memory."""

    array: NDArray[np.floating]
    """The underlying array of b-vectors"""

    def get(self) -> NDArray[np.floating]:
        return self.array
