from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import ANY

import nibabel as nib
import numpy as np
import pytest
from scipy.linalg import expm

from abcdmicro.dwi import Dwi
from abcdmicro.event import AbcdEvent
from abcdmicro.io import NiftiVolumeResource
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
        image_download_path_in=Path("/this/is/a/path/for/images"),
        tabular_data_path_in=Path("/this/is/a/path/for/tables"),
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
            metadata=dict(small_nifti_header)
            | {
                "dim": np.array([4, 3, 4, 5, n_vols, 1, 1, 1], np.int16)
            },  # see nifti dim field
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
            metadata=dict(small_nifti_header)
            | {
                "dim": np.array([4, 3, 4, 5, n_vols, 1, 1, 1], np.int16)
            },  # see nifti dim field
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


@pytest.fixture
def dwi4(abcd_event, random_affine, small_nifti_header) -> Dwi:
    """An example in-memory Dwi with 10 volumes."""
    n_vols = 10
    rng = np.random.default_rng(4616)
    bvec_array = rng.random(size=(n_vols, 3), dtype=np.float32)
    bvec_array = bvec_array / np.sqrt((bvec_array**2).sum(axis=1, keepdims=True))
    return Dwi(
        event=abcd_event,
        volume=InMemoryVolumeResource(
            array=rng.random(size=(3, 4, 5, n_vols), dtype=np.float32),
            affine=random_affine,
            metadata=dict(small_nifti_header)
            | {
                "dim": np.array([4, 3, 4, 5, n_vols, 1, 1, 1], np.int16)
            },  # see nifti dim field
        ),
        bval=InMemoryBvalResource(array=rng.integers(0, 3000, n_vols).astype(float)),
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
    metadata, metadata1 = dwi_cat.volume.get_metadata(), dwi1.volume.get_metadata()
    assert all(
        deep_equal_allclose(
            metadata[k],
            metadata1[k],
        )
        for k in metadata
        if k
        != "dim"  # dim is allowed to not match as it must update for the concatenated array
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
    metadata, metadata1 = dwi_cat.volume.get_metadata(), dwi1.volume.get_metadata()
    assert all(
        deep_equal_allclose(
            metadata[k],
            metadata1[k],
        )
        for k in metadata
        if k
        != "dim"  # dim is allowed to not match as it must update for the concatenated array
    )


@pytest.mark.parametrize("test_with_array_of_nan", [True, False])
def test_dwi_concatenate_nan_metadata_equality(
    caplog, volume_array, random_affine, test_with_array_of_nan
):
    """If both DWIs have metadata that totally agrees and that includes NaN values,
    the NaNs should not trip up the equality check."""
    arr = volume_array[..., :3]  # pick any 3 volumes
    bval = InMemoryBvalResource(
        np.zeros(arr.shape[3])
    )  # all zero bvals so as not to worry about needing unit bvecs
    bvec = InMemoryBvecResource(np.zeros((arr.shape[3], 3)))
    nan_meta = (
        {"srow_x": np.array([np.nan, 7.2, 0, 0])}
        if test_with_array_of_nan
        else {"scl_slope": np.nan}
    )

    dwi1 = Dwi(
        volume=InMemoryVolumeResource(
            array=arr, affine=random_affine, metadata=nan_meta
        ),
        bval=bval,
        bvec=bvec,
    )
    dwi2 = Dwi(
        volume=InMemoryVolumeResource(
            array=arr, affine=random_affine, metadata=nan_meta
        ),
        bval=bval,
        bvec=bvec,
    )

    with caplog.at_level(logging.WARNING):
        Dwi.concatenate([dwi1, dwi2])
    assert "Metadata mismatch" not in caplog.text


def test_concatenate_updates_dim_in_metadata(dwi1: Dwi, dwi2: Dwi):
    """After concatenation, the header 'dim' field should reflect the new number of volumes."""
    n1 = dwi1.volume.get_array().shape[3]
    n2 = dwi2.volume.get_array().shape[3]

    concatenated = Dwi.concatenate([dwi1, dwi2])
    cat_meta = concatenated.volume.get_metadata()
    assert cat_meta["dim"][4] == n1 + n2


def test_compute_b0_mean(dwi3: Dwi):
    """Test mean b0 computation"""
    b0_mean = dwi3.compute_mean_b0()
    dwi_array = dwi3.volume.get_array()
    assert b0_mean.get_array() == pytest.approx(
        (dwi_array[..., 0] + dwi_array[..., 3] + dwi_array[..., 4]) / 3
    )


def test_extract_brain(dwi3: Dwi, random_affine: np.ndarray, mocker):
    """Test that calling brain_extract method calls and appropriately uses the brain masking utility in abcdmicro.masks"""
    with tempfile.TemporaryDirectory() as work_dir:
        # Mocking of brain_extract_single
        mask_path = Path(work_dir) / "blah.nii"
        rng = np.random.default_rng(18653)
        mask_in_memory = InMemoryVolumeResource(
            array=rng.random((2, 3, 4)), affine=random_affine
        )
        mask_on_disk = NiftiVolumeResource.save(mask_in_memory, mask_path)
        mock_brain_extract_single = mocker.patch(
            "abcdmicro.dwi.brain_extract_single",
            return_value=mask_on_disk,
        )

        # Test
        mask_actual = dwi3.extract_brain()
        mock_brain_extract_single.assert_called_once_with(dwi=dwi3, output_path=ANY)
        assert np.allclose(mask_actual.get_affine(), mask_in_memory.get_affine())
        assert np.allclose(mask_actual.get_array(), mask_in_memory.get_array())


def test_dwi_denoise(dwi4: Dwi):
    """Test that calling the denoise method calls and appropriately uses the denoising utility in abcdmicro.denoise
    Patch2self issues a warning if the input dwi has less than 10 volumes."""

    denoised = dwi4.denoise()

    # tests
    assert isinstance(denoised, Dwi)
    assert np.allclose(denoised.bval.get(), dwi4.bval.get())
    assert np.allclose(denoised.bvec.get(), dwi4.bvec.get())
    assert np.allclose(denoised.volume.get_affine(), dwi4.volume.get_affine())
    assert denoised.volume.get_array().shape == dwi4.volume.get_array().shape
    assert not np.allclose(
        denoised.volume.get_array(), dwi4.volume.get_array()
    )  # should be different arrays


def test_estimate_noddi(dwi3: Dwi):
    """Test that calling compute_noddi calls and appropriately uses the NODDI fitting utility in abcdmicro.noddi"""

    noddi = dwi3.estimate_noddi()
    assert noddi.volume.get_array().shape[3] == 3  # confirm that output has 3 maps
    assert (
        noddi.directions.get_array().shape[3] == 3
    )  # confirm direction vector has 3 components
    assert np.allclose(noddi.volume.get_affine(), dwi3.volume.get_affine())


def test_estimate_dti(dwi3: Dwi):
    """Test that calling estimate_dti calls and appropriately uses the DTI fitting utility in abcdmicro.dti"""
    dti = dwi3.estimate_dti()
    assert dwi3.volume.get_array().shape[3] == 6
    assert np.allclose(dti.volume.get_affine(), dwi3.volume.get_affine())
