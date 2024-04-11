from __future__ import annotations

import numpy as np
import pytest

from abcd_microstructure_pipelines.masks import compute_b0_mean


@pytest.fixture()
def dwi_data_small_random():
    rng = np.random.default_rng(1337)
    dwi_data = rng.random(size=(3, 4, 5, 6))
    bvals = [0, 1000, 500, 0, 0, 500]
    bvecs = rng.random(size=(6, 3))
    bvecs = bvecs / np.sqrt((bvecs**2).sum(axis=1, keepdims=True))
    return dwi_data, bvals, bvecs


def test_compute_b0_mean(dwi_data_small_random):
    dwi_data, bvals, bvecs = dwi_data_small_random
    b0_mean = compute_b0_mean(dwi_data, bvals, bvecs)
    assert b0_mean == pytest.approx(
        (dwi_data[..., 0] + dwi_data[..., 3] + dwi_data[..., 4]) / 3
    )
