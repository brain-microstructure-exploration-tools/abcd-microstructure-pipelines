from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from dipy.io.image import save_nifti

from abcdmicro.io import NiftiVolumeResrouce


@pytest.fixture()
def bval_array():
    return np.array([500.0, 1000.0, 200.0])


@pytest.fixture()
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


@pytest.fixture()
def volume_array():
    rng = np.random.default_rng(1337)
    return rng.random(size=(3, 4, 5, 6), dtype=float)


@pytest.mark.filterwarnings("ignore:builtin type [sS]wig.* has no __module__ attribute")
def test_nifti_volume_resource(volume_array):
    with tempfile.TemporaryDirectory() as tmpdir:
        volume_file = Path(tmpdir) / "volume_file.nii"
        save_nifti(
            fname=volume_file,
            data=volume_array,
            affine=np.array(
                [
                    [1.35, 0, 0, 0],
                    [0, 1.45, 0, 0],
                    [0, 0, 1.55, 0],
                    [0, 0, 0, 1],
                ]
            ),
        )
        volume_resource = NiftiVolumeResrouce(path=volume_file)
        assert np.allclose(volume_resource.get_array(), volume_array)
