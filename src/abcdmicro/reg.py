from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import ants
import numpy as np
from numpy.typing import NDArray

from abcdmicro.resource import InMemoryVolumeResource, VolumeResource
from abcdmicro.util import PathLike, normalize_path


@dataclass()
class TransformResource:
    """A registration result"""

    # Raw paths to ants output files
    _ants_fwd_paths: list[str]
    _ants_inv_paths: list[str]

    _ref_affine: NDArray[np.floating]

    # Store the actual objects if they are already loaded
    _matrices: list[ants.ANTsTransform] | None = None
    _warps: list[InMemoryVolumeResource] | None = None

    @property
    def matrices(self) -> list[ants.ANTsTransform] | None:
        """Returns the affine transform if it exists."""
        if self._matrices is None:
            # Find the .mat file in paths and load it
            mat_paths = [p for p in self._ants_fwd_paths if p.endswith(".mat")]
            self._matrices = [ants.core.read_transform(p) for p in mat_paths]
        return self._matrices

    @property
    def warp_fields(self) -> list[InMemoryVolumeResource] | None:
        """Returns the non-linear displacement field if it exists."""

        if self._warps is None:
            # Get all warp files (typically .nii)
            warp_paths = [p for p in self._ants_fwd_paths if ".nii" in p]

            warp_volumes = []
            for p in warp_paths:
                ants_img = ants.image_read(p)
                warp_volumes.append(
                    InMemoryVolumeResource(
                        array=ants_img.numpy(), affine=self._ref_affine
                    )
                )
                self._warps = warp_volumes
        return self._warps

    @staticmethod
    def initialize_from_ants(
        ants_result: dict[str, list[str]], ref_volume: VolumeResource
    ) -> TransformResource:
        ants_fwdtransforms = ants_result["fwdtransforms"]

        return TransformResource(
            _ants_fwd_paths=ants_fwdtransforms,
            _ants_inv_paths=ants_result["invtransforms"],
            _ref_affine=ref_volume.get_affine(),
        )

    def apply(
        self,
        fixed: VolumeResource,
        moving: VolumeResource,
        invert: bool = False,
        interpolation: str = "linear",
    ) -> InMemoryVolumeResource:
        """Wrapper around ants.apply_transforms using this result."""

        ants_fixed = ants.from_numpy(fixed.get_array())
        ants_moving = ants.from_numpy(moving.get_array())

        transforms = self._ants_inv_paths if invert else self._ants_fwd_paths

        # Ants does not pre-invert the .mat files
        # so we need to explicitly specify that it needs to be inverted.
        # This is done in ANTs by default when the inv transform list is used but
        # we specify it here for clarity.
        if invert:
            whichtoinvert = [p.endswith(".mat") for p in transforms]
        else:
            whichtoinvert = [False] * len(transforms)

        warpedmovout = ants.apply_transforms(
            fixed=ants_fixed,
            moving=ants_moving,
            transformlist=transforms,
            whichtoinvert=whichtoinvert,
            interpolator=interpolation,
        )

        return InMemoryVolumeResource(
            array=warpedmovout.numpy(),
            affine=fixed.get_affine(),
            metadata=fixed.get_metadata(),
        )

    def save(self, output_dir: PathLike) -> None:
        """Copies the underlying ANTs files to a permanent location."""
        path = normalize_path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Files get saved with default temp names from ANTs
        all_paths = set(self._ants_fwd_paths + self._ants_inv_paths)
        for p in all_paths:
            shutil.copy(p, path / Path(p).name)


def register_volumes(
    fixed: VolumeResource,
    moving: VolumeResource,
    type_of_transform: str = "SyN",
    mask: VolumeResource | None = None,
    moving_mask: VolumeResource | None = None,
) -> tuple[InMemoryVolumeResource, TransformResource]:
    """
    Registers a moving volume to a fixed reference volume using ANTs.

    Args:
        fixed: The reference volume (assumed to share the same affine as moving).
        moving: The volume to be warped.
        type_of_transform: The transformation model (e.g., "SyN", "Rigid"). The full list of
            supported transforms can be found in the ANTs documentation.
        mask: Optional mask for the fixed image space.
        moving_mask: Optional mask for the moving image space.

    Returns:
        A tuple containing the registered volume and the transform object.

    """

    # Check input volume
    if fixed.get_array().ndim > 3 or moving.get_array().ndim > 3:
        error_message = "Input volume dimensions must be 2D or 3D."
        raise ValueError(error_message)

    # TODO: Nibabel to ants conversion when ants is released
    ants_fixed = ants.from_numpy(fixed.get_array())
    ants_moving = ants.from_numpy(moving.get_array())

    # Convert masks to ants images if provided
    ants_mask = ants.from_numpy(mask.get_array()) if mask is not None else None
    ants_moving_mask = (
        ants.from_numpy(moving_mask.get_array()) if moving_mask is not None else None
    )

    # Check that the mask dimensions match the fixed/moving images
    if ants_mask is not None and (ants_mask.shape != ants_fixed.shape):
        error_message = "Fixed mask dimensions do not match fixed image dimensions."
        raise ValueError(error_message)
    if ants_moving_mask is not None and (ants_moving_mask.shape != ants_moving.shape):
        error_message = "Moving mask dimensions do not match moving image dimensions."
        raise ValueError(error_message)

    # To catch any ANTs errors
    try:
        ants_result = ants.registration(
            fixed=ants_fixed,
            moving=ants_moving,
            mask=ants_mask,
            movingmask=ants_moving_mask,
            type_of_transform=type_of_transform,
        )
    except Exception as e:
        error_msg = f"ANTs registration failed: {e!s}"
        raise RuntimeError(error_msg) from e

    warpedmovout = InMemoryVolumeResource(
        array=ants_result["warpedmovout"].numpy(),
        affine=fixed.get_affine(),
        metadata=fixed.get_metadata(),
    )

    transform = TransformResource.initialize_from_ants(ants_result, ref_volume=fixed)

    return (warpedmovout, transform)
