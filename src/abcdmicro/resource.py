from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import itk
import numpy as np
from numpy.typing import NDArray


class VolumeResource(ABC):
    """Base class for resources representing a volume or volume stack.
    An n-D array where n >= 3 and where three of the dimensions are spatial
    and have associated header information describing a patient coordinate system."""

    @abstractmethod
    def get_array(self) -> NDArray[Any]:
        """Get the underlying volume data array"""

    @abstractmethod
    def get_metadata(self) -> dict[Any, Any]:
        """Get the volume image metadata"""


@dataclass
class InMemoryVolumeResource(VolumeResource):
    """A volume resource that is loaded into memory.
    An n-D array where n >= 3 and where three of the dimensions are spatial
    and have associated header information describing a patient coordinate system."""

    image: itk.Image
    """The underlying ITK image of the volume"""

    def get_array(self) -> NDArray[Any]:
        return itk.array_view_from_image(self.image)

    def get_metadata(self) -> dict[Any, Any]:
        return dict(self.image)


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
