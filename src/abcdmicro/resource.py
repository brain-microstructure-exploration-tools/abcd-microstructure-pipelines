from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class VolumeResource(ABC):
    """Base class for resources representing a volume or volume stack.
    An n-D array where n >= 3 and where three of the dimensions are spatial
    and have associated header information describing a patient coordinate system."""

    @abstractmethod
    def get_array(self) -> NDArray[np.floating]:
        """Get the underlying volume data array"""


class InMemoryVolumeResource(VolumeResource):
    """A volume resource that is loaded into memory.
    An n-D array where n >= 3 and where three of the dimensions are spatial
    and have associated header information describing a patient coordinate system."""


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
        """Get the underlying array of b-vectors"""


@dataclass
class InMemoryBvecResource(BvecResource):
    """A b-vector list that is loaded into memory."""

    array: NDArray[np.floating]
    """The underlying array of b-vectors"""

    def get(self) -> NDArray[np.floating]:
        return self.array
