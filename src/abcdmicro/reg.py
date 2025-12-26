from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import ants

from abcdmicro.resource import InMemoryVolumeResource, VolumeResource
from abcdmicro.util import PathLike, normalize_path


@dataclass
class TransformResource:
    """A registration result"""

    # Raw paths to ants output files
    _ants_fwd_paths: list[str]
    _ants_inv_paths: list[str]

    # Store the actual objects if they are already loaded
    _matrix: ants.ANTsTransform | None = None
    _warp: VolumeResource | None = None

    @staticmethod
    def initialize_from_ants(ants_result: dict[str, list[str]]) -> TransformResource:
        return TransformResource(
            _ants_fwd_paths=ants_result["fwdtransforms"],
            _ants_inv_paths=ants_result["invtransforms"],
        )

    @property
    def matrix(self) -> ants.ANTsTransform:
        """Returns the linear transformation matrix as an ANTsTransform if it exists."""
        if self._matrix is None:
            for path in self._ants_fwd_paths:
                if path.endswith(".mat"):
                    self._matrix = ants.core.read_transform(path)
        return self._matrix

    @property
    def warp_field(self) -> VolumeResource | None:
        """Returns the non-linear displacement field if it exists."""
        if self._warp is None:
            for path in self._ants_fwd_paths:
                if path.endswith(".nii.gz"):
                    ants_img = ants.image_read(path)
                    self._warp = InMemoryVolumeResource(array=ants_img.numpy())
        return self._warp

    def save(self, output_dir: PathLike) -> None:
        """Copies the underlying ANTs files to a permanent location."""
        output_dir = normalize_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for p in self._ants_fwd_paths:
            shutil.copy(p, output_dir / Path(p).name)

    def apply(
        self,
        fixed: VolumeResource,
        moving: VolumeResource,
        interpolation: str = "linear",
    ) -> InMemoryVolumeResource:
        """Wrapper around ants.apply_transforms using this result."""

        ants_fixed = ants.from_numpy(fixed.get_array())
        ants_moving = ants.from_numpy(moving.get_array())

        warped_ants = ants.apply_transforms(
            fixed=ants_fixed,
            moving=ants_moving,
            transformlist=self._ants_fwd_paths,
            interpolator=interpolation,
        )

        return InMemoryVolumeResource(
            array=warped_ants.numpy(),
            affine=fixed.get_affine(),
            metadata=fixed.get_metadata(),
        )


def register_volumes(
    fixed: VolumeResource,
    moving: VolumeResource,
    type_of_transform: str = "SyN",
    mask: VolumeResource | None = None,
    moving_mask: VolumeResource | None = None,
) -> tuple[InMemoryVolumeResource, TransformResource]:
    """
    Registers moving volume to fixed volume. By default, this performs a multi-stage registration (affine alignment followed
    by deformable registration, using  the mutual information metric). Optional spatial masks can be specified
    to weight the registration metric.
    """
    # Convert VolumeResource to ANTs Image
    ants_fixed = ants.from_numpy(fixed.get_array())
    ants_moving = ants.from_numpy(moving.get_array())

    ants_fixed_mask = None
    if mask is not None:
        ants_fixed_mask = ants.from_numpy(mask.get_array())

    ants_moving_mask = None
    if moving_mask is not None:
        ants_moving_mask = ants.from_numpy(moving_mask.get_array())

    # Perform registration
    reg = ants.registration(
        fixed=ants_fixed,
        moving=ants_moving,
        type_of_transform=type_of_transform,
        mask=ants_fixed_mask,
        moving_mask=ants_moving_mask,
    )

    transform = TransformResource.initialize_from_ants(reg)

    warpedmovout = InMemoryVolumeResource(
        array=reg["warpedmovout"].numpy(),
        affine=fixed.get_affine(),  # Warped image is in fixed space
        metadata=fixed.get_metadata(),
    )

    return warpedmovout, transform
