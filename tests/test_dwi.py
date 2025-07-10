from __future__ import annotations

import logging
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
from abcdmicro.util import deep_equal_allclose


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
    return rng.random(size=(6, 4, 5, 3), dtype=float)


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
        bval=InMemoryBvalResource(np.array([500.0, 1000.0])),
        bvec=InMemoryBvecResource(
            np.array(
                [
                    [
                        1.0 / np.sqrt(3),
                        1.0 / np.sqrt(3),
                        1.0 / np.sqrt(3),
                    ],
                    [2.0 / np.sqrt(20), 0.0, -4.0 / np.sqrt(20)],
                ]
            )
        ),
    )


@pytest.fixture
def dwi1(abcd_event, random_affine, small_nifti_header) -> Dwi:
    """An example in-memory Dwi with 6 volumes."""
    n_vols = 6
    rng = np.random.default_rng(4616)
    bvec_array = rng.random(size=(n_vols, 3), dtype=np.float32)
    bvec_array = bvec_array / np.sqrt((bvec_array**2).sum(axis=1, keepdims=True))
    return Dwi(
        event=abcd_event,
        volume=InMemoryVolumeResource(
            array=rng.random(size=(3, 4, 5, n_vols), dtype=np.float32),
            affine=random_affine,
            metadata=dict(small_nifti_header),
        ),
        bval=InMemoryBvalResource(array=rng.integers(0, 3000, n_vols).astype(float)),
        bvec=InMemoryBvecResource(array=bvec_array),
    )


@pytest.fixture
def dwi2(abcd_event, random_affine, small_nifti_header) -> Dwi:
    """A second example in-memory Dwi with 4 volumes and metadata that matches that of dwi1."""
    n_vols = 4
    rng = np.random.default_rng(7816)
    bvec_array = rng.random(size=(n_vols, 3), dtype=np.float32)
    bvec_array = bvec_array / np.sqrt((bvec_array**2).sum(axis=1, keepdims=True))
    return Dwi(
        event=abcd_event,  # event, affine, and metadata matches dwi1
        volume=InMemoryVolumeResource(
            array=rng.random(size=(3, 4, 5, n_vols), dtype=np.float32),
            affine=random_affine,
            metadata=dict(small_nifti_header),
        ),
        bval=InMemoryBvalResource(array=rng.integers(0, 3000, n_vols).astype(float)),
        bvec=InMemoryBvecResource(array=bvec_array),
    )


@pytest.fixture
def dwi3(abcd_event, random_affine, small_nifti_header) -> Dwi:
    """A n example in-memory Dwi with 6 volumes, 3 of which have b=0"""
    n_vols = 6
    rng = np.random.default_rng(7816)
    bvec_array = rng.random(size=(n_vols, 3), dtype=np.float32)
    bvec_array = bvec_array / np.sqrt((bvec_array**2).sum(axis=1, keepdims=True))
    return Dwi(
        event=abcd_event,
        volume=InMemoryVolumeResource(
            array=rng.random(size=(3, 4, 5, n_vols), dtype=np.float32),
            affine=random_affine,
            metadata=dict(small_nifti_header),
        ),
        bval=InMemoryBvalResource(array=np.array([0, 1000, 3000, 0, 0, 2000])),
        bvec=InMemoryBvecResource(array=bvec_array),
    )


def test_initialization_fails_with_bad_bvecs(volume_array):
    # Non unit b-vectors
    bvec = InMemoryBvecResource(
        np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    with pytest.raises(
        ValueError,
        match="must be unit vectors",
    ):
        Dwi(
            volume=InMemoryVolumeResource(array=volume_array),
            bval=InMemoryBvalResource(np.array([1000.0, 2000, 3000, 3000])),
            bvec=bvec,
        )

    # No exception should be raised if the offending b-vec has a b-val of 0:
    Dwi(
        volume=InMemoryVolumeResource(array=volume_array),
        bval=InMemoryBvalResource(np.array([1000.0, 0.0, 3000, 3000])),
        bvec=bvec,
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


def test_dwi_concatenate_empty_list():
    """Tests that concatenating an empty list raises a ValueError."""
    with pytest.raises(ValueError, match="Cannot concatenate an empty list of DWIs."):
        Dwi.concatenate([])


def test_dwi_concatenate_singleton_list(dwi1: Dwi):
    """
    Tests that concatenating a list with a single Dwi returns an
    equivalent Dwi.
    """
    dwi_cat = Dwi.concatenate([dwi1])

    assert dwi_cat.event == dwi1.event
    assert deep_equal_allclose(
        dwi_cat.volume.get_metadata(), dwi1.volume.get_metadata()
    )
    assert np.allclose(dwi_cat.volume.get_affine(), dwi1.volume.get_affine())
    assert np.allclose(dwi_cat.volume.get_array(), dwi1.volume.get_array())
    assert np.allclose(dwi_cat.bval.get(), dwi1.bval.get())
    assert np.allclose(dwi_cat.bvec.get(), dwi1.bvec.get())


def test_dwi_concatenate(dwi1: Dwi, dwi2: Dwi):
    """Tests the successful concatenation of two DWIs."""
    dwi_list = [dwi1, dwi2]

    # Perform concatenation
    dwi_cat = Dwi.concatenate(dwi_list)

    # 1. Check metadata from the first DWI is used
    assert dwi_cat.event == dwi1.event
    assert np.allclose(dwi_cat.volume.get_affine(), dwi1.volume.get_affine())
    assert deep_equal_allclose(
        dwi_cat.volume.get_metadata(), dwi1.volume.get_metadata()
    )

    # 2. Check volume data and shape
    expected_vol_data = np.concatenate(
        [dwi1.volume.get_array(), dwi2.volume.get_array()], axis=-1
    )
    assert dwi_cat.volume.get_array().shape == (3, 4, 5, 10)
    assert np.allclose(dwi_cat.volume.get_array(), expected_vol_data)

    # 3. Check bval data and shape
    expected_bval_data = np.concatenate([dwi1.bval.get(), dwi2.bval.get()])
    assert dwi_cat.bval.get().shape == (10,)
    assert np.allclose(dwi_cat.bval.get(), expected_bval_data)

    # 4. Check bvec data and shape
    expected_bvec_data = np.concatenate([dwi1.bvec.get(), dwi2.bvec.get()], axis=0)
    assert dwi_cat.bvec.get().shape == (10, 3)
    assert np.allclose(dwi_cat.bvec.get(), expected_bvec_data)


def test_dwi_concatenate_event_mismatch(dwi1: Dwi, dwi2: Dwi, caplog):
    """Tests that a warning is logged for mismatched events."""
    # create a new event for the second DWI
    dwi2.event = AbcdEvent("other_subject", "other_event", Path("/"), Path("/"), "5.1")

    with caplog.at_level(logging.WARNING):
        Dwi.concatenate([dwi1, dwi2])

    assert "Event mismatch" in caplog.text


def test_dwi_concatenate_affine_mismatch(dwi1: Dwi, dwi2: Dwi, caplog):
    """Tests that a warning is logged for mismatched affine matrices."""
    # create a slightly different affine for the second DWI's volume
    new_affine = dwi2.volume.get_affine().copy()
    new_affine[0, 3] += 1.0  # add a small translation
    dwi2.volume = InMemoryVolumeResource(
        array=dwi2.volume.get_array(),
        affine=new_affine,
        metadata=dwi2.volume.get_metadata(),
    )

    with caplog.at_level(logging.WARNING):
        dwi_cat = Dwi.concatenate([dwi1, dwi2])

    assert "Affine mismatch" in caplog.text
    # ensure the first DWI's affine was used in the final result
    assert np.allclose(dwi_cat.volume.get_affine(), dwi1.volume.get_affine())


def test_dwi_concatenate_metadata_mismatch(dwi1: Dwi, dwi2: Dwi, caplog):
    """Tests that a warning is logged for mismatched metadata headers."""
    # create a different header for the second DWI's volume
    new_metadata = dwi2.volume.get_metadata().copy()
    new_metadata["descrip"] = b"a different description"
    dwi2.volume = InMemoryVolumeResource(
        array=dwi2.volume.get_array(),
        affine=dwi2.volume.get_affine(),
        metadata=new_metadata,
    )

    with caplog.at_level(logging.WARNING):
        dwi_cat = Dwi.concatenate([dwi1, dwi2])

    assert "Metadata mismatch" in caplog.text
    # ensure the first DWI's metadata was used
    assert deep_equal_allclose(
        dwi_cat.volume.get_metadata(), dwi1.volume.get_metadata()
    )


def test_compute_b0_mean(dwi3: Dwi):
    """Test mean b0 computation"""
    b0_mean = dwi3.compute_mean_b0()
    dwi_array = dwi3.volume.get_array()
    assert b0_mean.get_array() == pytest.approx(
        (dwi_array[..., 0] + dwi_array[..., 3] + dwi_array[..., 4]) / 3
    )
