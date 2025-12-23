from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from dipy.io.image import save_nifti
from scipy.linalg import expm

from abcdmicro.io import (
    FslBvalResource,
    FslBvecResource,
    JsonResponseFunctionResource,
    NiftiVolumeResource,
)
from abcdmicro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryResponseFunctionResource,
    InMemoryVolumeResource,
)


@pytest.fixture
def bval_array():
    return np.array([500.0, 1000.0, 200.0])


@pytest.fixture
def bvec_array():
    bvec = np.array(
        [
            [
                1.0,
                1.0,
                1.0,
            ],
            [2.0, 0.0, -4.0],
        ]
    )
    return bvec / np.linalg.norm(bvec, axis=1, keepdims=True)


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
def response_function():
    rng = np.random.default_rng(1337)
    sh_coeffs = rng.random(size=(10,))  # random array of coeeficients
    avg_signal = rng.random(size=(), dtype=np.float32)
    return [sh_coeffs, avg_signal]


@pytest.mark.filterwarnings("ignore:builtin type [sS]wig.* has no __module__ attribute")
def test_nifti_volume_resource(volume_array, random_affine, tmp_path):
    volume_file = Path(tmp_path) / "volume_file.nii"
    save_nifti(
        fname=volume_file,
        data=volume_array,
        affine=random_affine,
    )
    volume_resource = NiftiVolumeResource(volume_file)
    assert np.allclose(volume_resource.get_array(), volume_array)
    assert np.allclose(volume_resource.get_affine(), random_affine)


def test_nifti_volume_resource_save_load(
    volume_array, random_affine, small_nifti_header, tmp_path
):
    vol_mem = InMemoryVolumeResource(
        array=volume_array, affine=random_affine, metadata=dict(small_nifti_header)
    )
    path = tmp_path / "test.nii.gz"
    vol_disk = NiftiVolumeResource.save(vol_mem, path)
    vol_mem2 = vol_disk.load()
    assert np.allclose(vol_mem.get_array(), vol_mem2.get_array())
    assert np.allclose(vol_mem.get_affine(), vol_mem2.get_affine())
    assert vol_mem.get_metadata()["descrip"] == vol_mem2.get_metadata()["descrip"]


def test_fsl_bval_resource_save_load(bval_array, tmp_path):
    path = tmp_path / "test.bval"
    bval_mem = InMemoryBvalResource(bval_array)
    bval_mem2 = FslBvalResource.save(bval_mem, path).load()
    assert np.allclose(bval_mem.get(), bval_mem2.get())


def test_fsl_bvec_resource_save_load(bvec_array, tmp_path):
    path = tmp_path / "test.bvec"
    bvec_mem = InMemoryBvecResource(bvec_array)
    bvec_mem2 = FslBvecResource.save(bvec_mem, path).load()
    assert np.allclose(bvec_mem.get(), bvec_mem2.get())


def test_text_response_function_resource_save_load(response_function, tmp_path):
    response = InMemoryResponseFunctionResource(
        sh_coeffs=response_function[0], avg_signal=response_function[1]
    )

    path = tmp_path / "test_response.json"
    res_disk = JsonResponseFunctionResource.save(response, path)
    res_mem2 = res_disk.load()

    assert len(res_mem2.get()) == 2
    assert np.allclose(res_mem2.get()[0], response.sh_coeffs)
    assert res_mem2.get()[1] == response.avg_signal
