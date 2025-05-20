from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy.linalg import expm

from abcdmicro.dwi import Dwi
from abcdmicro.event import AbcdEvent
from abcdmicro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
)


@pytest.fixture
def abcd_event():
    return AbcdEvent(
        subject_id="NDAR_INV00U4FTRU",
        eventname="baseline_year_1_arm_1",
        image_download_path=Path("/this/is/a/path/for/images"),
        tabular_data_path=Path("/this/is/a/path/for/tables"),
        abcd_version="5.1",
    )


@pytest.fixture
def volume_array():
    rng = np.random.default_rng(1337)
    return rng.random(size=(3, 4, 5, 6), dtype=float)


@pytest.fixture
def random_affine() -> np.ndarray:
    rng = np.random.default_rng(18653)
    affine = np.eye(4)
    affine[:3, :3] = expm(
        (lambda A: (A - A.T) / 2)(rng.normal(size=(3, 3)))
    )  # generate a random orthogonal matrix
    affine[:3, 3] = rng.random(3)  # generate a random origin
    return affine


@pytest.fixture
def small_nifti_header():
    hdr = nib.Nifti1Header()
    hdr["descrip"] = b"an abcdmicro unit test header description"
    return hdr


@pytest.fixture
def dwi(abcd_event, volume_array, random_affine, small_nifti_header) -> Dwi:
    """An example in-memory Dwi"""
    return Dwi(
        event=abcd_event,
        volume=InMemoryVolumeResource(
            array=volume_array, affine=random_affine, metadata=dict(small_nifti_header)
        ),
        bval=InMemoryBvalResource(np.array([500.0, 1000.0, 200.0])),
        bvec=InMemoryBvecResource(
            np.array(
                [
                    [
                        1.0,
                        1.0,
                        1.0,
                    ],
                    [2.0, 0.0, -4.0],
                ]
            )
        ),
    )


def test_dwi_save(dwi: Dwi, tmp_path: Path):
    assert dwi.volume.is_loaded
    assert dwi.bval.is_loaded
    assert dwi.bvec.is_loaded
    dwi_saved = dwi.save(path=tmp_path, basename="test")
    assert not dwi_saved.volume.is_loaded
    assert not dwi_saved.bval.is_loaded
    assert not dwi_saved.bvec.is_loaded


def test_dwi_save_load(dwi: Dwi, tmp_path: Path):
    dwi_reloaded = dwi.save(path=tmp_path, basename="test").load()
    assert dwi_reloaded.volume.is_loaded
    assert dwi_reloaded.bval.is_loaded
    assert dwi_reloaded.bvec.is_loaded
    assert np.allclose(dwi_reloaded.bval.get(), dwi.bval.get())
    assert np.allclose(dwi_reloaded.bvec.get(), dwi.bvec.get())
    assert np.allclose(dwi_reloaded.volume.get_array(), dwi.volume.get_array())
