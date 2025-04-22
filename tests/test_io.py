from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from dipy.io.image import save_nifti
from scipy.linalg import expm

from abcdmicro.io import NiftiVolumeResrouce


@pytest.fixture
def bval_array():
    return np.array([500.0, 1000.0, 200.0])


@pytest.fixture
def bvec_array():
    return np.array(
        [
            [
                1.0,
                1.0,
                1.0,
            ],
            [2.0, 0.0, -4.0],
        ]
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


@pytest.mark.filterwarnings("ignore:builtin type [sS]wig.* has no __module__ attribute")
def test_nifti_volume_resource(volume_array, random_affine):
    with tempfile.TemporaryDirectory() as tmpdir:
        volume_file = Path(tmpdir) / "volume_file.nii"
        save_nifti(
            fname=volume_file,
            data=volume_array,
            affine=random_affine,
        )
        volume_resource = NiftiVolumeResrouce(path=volume_file)
        assert np.allclose(volume_resource.get_array(), volume_array)
        assert np.allclose(volume_resource.get_affine(), random_affine)
