from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from abcdmicro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource
from abcdmicro.resource import (
    BvalResource,
    BvecResource,
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
    VolumeResource,
)
from abcdmicro.util import deep_equal_allclose

if TYPE_CHECKING:
    from abcdmicro.event import AbcdEvent


@dataclass
class Dwi:
    """An ABCD diffusion weighted image."""

    event: AbcdEvent
    """The ABCD event associated with this DWI."""

    volume: VolumeResource
    """ The DWI image volume.
    It is assumed to be a 4D volume, with the first three dimensions being spatial and the final dimension indexing
    the diffusion weightings.
    """

    bval: BvalResource
    """The DWI b-values"""

    bvec: BvecResource
    """The DWI b-vectors"""

    def load(self) -> Dwi:
        """Load any on-disk resources into memory and return a Dwi with all loadable resources loaded."""
        return Dwi(
            event=self.event,
            volume=self.volume.load(),
            bval=self.bval.load(),
            bvec=self.bvec.load(),
        )

    def save(self, path: Path, basename: str) -> Dwi:
        """Save all resources to disk and return a Dwi with all resources being on-disk.

        Args:
            path: The desired save directory.
            basename: The desired file basenames, i.e. without an extension.

        Returns: A Dwi with its internal resources being on-disk.
        """
        if path.exists() and not path.is_dir():
            msg = "`path` should be the desired save directory"
            raise ValueError(msg)
        path.mkdir(exist_ok=True, parents=True)
        return Dwi(
            event=self.event,
            volume=NiftiVolumeResource.save(self.volume, path / f"{basename}.nii.gz"),
            bval=FslBvalResource.save(self.bval, path / f"{basename}.bval"),
            bvec=FslBvecResource.save(self.bvec, path / f"{basename}.bvec"),
        )

    @staticmethod
    def concatenate(dwis: list[Dwi]) -> Dwi:
        """Concatenate a list of `Dwi`s into a single (loaded) DWI.

        The event of the first `Dwi` in the list will be the event of the created Dwi,
        however a warning is logged if there is a mismatch of events among the `Dwi`s.

        Similarly, the affine and metadata of the first `Dwi` is used to concatenate volumes.
        """
        if len(dwis) == 0:
            msg = "Cannot concatenate an empty list of DWIs."
            raise ValueError(msg)

        # ensure all DWI resources are loaded into memory
        loaded_dwis = [d.load() for d in dwis]

        # use the first DWI as the reference for metadata
        ref_dwi = loaded_dwis[0]
        ref_event = ref_dwi.event
        ref_affine = ref_dwi.volume.get_affine()
        ref_metadata = ref_dwi.volume.get_metadata()

        # check for metadata consistency across all DWIs and log warnings
        for i, dwi in enumerate(loaded_dwis[1:], start=1):
            if dwi.event != ref_event:
                logging.warning(
                    "Event mismatch: Using event from DWI 0, but DWI %s has a different event.",
                    i,
                )
            if not np.allclose(dwi.volume.get_affine(), ref_affine):
                logging.warning(
                    "Affine mismatch: Using affine from DWI 0, but DWI %s has a different affine.",
                    i,
                )

            dwi_metadata = dwi.volume.get_metadata()
            if not deep_equal_allclose(dwi_metadata, ref_metadata):
                logging.warning(
                    "Metadata mismatch: Using metadata from DWI 0, but DWI %s has different metadata:",
                    i,
                )
                for key in ref_metadata:
                    if key not in dwi_metadata:
                        logging.warning(
                            "DWI 0 header has key '%s', but DWI %s header does not.",
                            key,
                            i,
                        )
                    elif not deep_equal_allclose(dwi_metadata[key], ref_metadata[key]):
                        logging.warning(
                            "At key '%s', values differ between DWI 0 and DWI %s.\n--> DWI 0: %s\n--> DWI %s: %s",
                            key,
                            i,
                            ref_metadata[key],
                            i,
                            dwi_metadata[key],
                        )

        # extract the numpy arrays from each resource.
        all_volumes_data = [d.volume.get_array() for d in loaded_dwis]
        all_bvals_data = [d.bval.get() for d in loaded_dwis]
        all_bvecs_data = [d.bvec.get() for d in loaded_dwis]

        # concatenate the data.
        concatenated_volume_data = np.concatenate(all_volumes_data, axis=-1)
        concatenated_bval_data = np.concatenate(all_bvals_data)
        concatenated_bvec_data = np.concatenate(all_bvecs_data, axis=0)

        # create new in-memory resources for the concatenated data.
        concatenated_volume = InMemoryVolumeResource(
            array=concatenated_volume_data,
            affine=ref_affine,
            metadata=ref_metadata,
        )
        concatenated_bval = InMemoryBvalResource(array=concatenated_bval_data)
        concatenated_bvec = InMemoryBvecResource(array=concatenated_bvec_data)

        # return a new Dwi object with the concatenated, in-memory data.
        return Dwi(
            event=ref_event,
            volume=concatenated_volume,
            bval=concatenated_bval,
            bvec=concatenated_bvec,
        )
