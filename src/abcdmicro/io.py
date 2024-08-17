from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar, get_type_hints

import itk
from dipy.io.gradients import read_bvals_bvecs

from abcdmicro.resource import (
    BvalResource,
    BvecResource,
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
    VolumeResource,
)

T = TypeVar("T", bound="LoadableResource")


class LoadableResource(ABC):
    """Base class for on-disk resources that have a load method that converts them into in-memory resources"""

    @abstractmethod
    def load(self) -> Any:
        """Load this resource to get an in-memory version of it."""


def implement_via_loading(method_names: list[str]) -> Callable[[type[T]], type[T]]:
    """Decorator that implements the listed abstract methods of a LoadableResource class by calling the
    load() method and then using the loaded object's method of the same name."""

    def implement_via_loading_decorator(cls: type[T]) -> type[T]:
        for method_name in method_names:

            def method(self, method_name=method_name):  # type: ignore[no-untyped-def]
                return getattr(self.load(), method_name)()

            method.__name__ = method_name
            method.__doc__ = f"Automatically implemented method that returns `self.load().{method_name}()`."

            for parent_class in cls.__bases__:
                if hasattr(parent_class, method_name):
                    parent_method = getattr(parent_class, method_name)
                    return_type = get_type_hints(parent_method).get("return", Any)
                    method.__annotations__ = {"return": return_type}
                    break

            setattr(cls, method_name, method)

        # If the automatically implemented methods were abstract methods, then remove them from the set
        # to indicate that they have been implemented.
        if hasattr(cls, "__abstractmethods__"):
            cls.__abstractmethods__ = frozenset(
                name for name in cls.__abstractmethods__ if name not in method_names
            )

        return cls

    return implement_via_loading_decorator


@implement_via_loading(["get_array", "get_metadata"])
@dataclass
class NiftiVolumeResrouce(VolumeResource):  # type: ignore[type-var]
    """A volume or volume stack that is saved to disk in the nifti file format."""

    path: Path
    """Path to the underlying volume nifti file"""

    def load(self) -> InMemoryVolumeResource:
        return InMemoryVolumeResource(itk.imread(self.path))


@implement_via_loading(["get"])
@dataclass
class FslBvalResource(BvalResource, LoadableResource):
    """A b-value list that is saved to disk in the FSL text file format."""

    path: Path
    """Path to the underlying bval txt file"""

    def load(self) -> InMemoryBvalResource:
        bvals_array, _ = read_bvals_bvecs(self.path, None)
        return InMemoryBvalResource(bvals_array)


@implement_via_loading(["get"])
@dataclass
class FslBvecResource(BvecResource, LoadableResource):
    """A b-vector list that is saved to disk in the FSL text file format."""

    path: Path
    """Path to the underlying bvec txt file"""

    def load(self) -> InMemoryBvecResource:
        _, bvecs_array = read_bvals_bvecs(None, self.path)
        return InMemoryBvecResource(bvecs_array)
