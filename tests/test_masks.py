from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import scipy.linalg
from dipy.io.image import load_nifti, save_nifti

from abcdmicro.masks import compute_b0_mean, gen_b0_mean


@pytest.fixture()
def dwi_data_small_random():
    rng = np.random.default_rng(1337)
    dwi_data = rng.random(size=(3, 4, 5, 6))
    bvals = np.array([0, 1000, 500, 0, 0, 500], dtype=float)
    bvecs = rng.random(size=(6, 3))
    bvecs = bvecs / np.sqrt((bvecs**2).sum(axis=1, keepdims=True))
    return dwi_data, bvals, bvecs


@pytest.fixture()
def affine_random():
    rng = np.random.default_rng(7562)
    affine = np.eye(4, dtype=float)
    affine[:3, 3] = 100 * (
        rng.random(size=(3,), dtype=float) - 0.5
    )  # random translation
    random_3by3 = rng.random(size=(3, 3), dtype=float) - 0.5
    affine[:3, :3] = scipy.linalg.expm(
        random_3by3 - random_3by3.T
    )  # exponentiate a random skew-symmetric matrix to get some orthogonal matrix
    return affine


class TemporaryDwiFiles:
    """Context manager that creates a temporary directory and writes DWI data into it."""

    def __init__(
        self,
        dwi_data: npt.NDArray,
        affine: npt.NDArray,
        bvals: npt.NDArray,
        bvecs: npt.NDArray,
    ):
        self.dwi_data = dwi_data
        self.bvals = bvals
        self.bvecs = bvecs
        self.affine = affine

    def __enter__(self):
        self.temp_dir_context = tempfile.TemporaryDirectory()
        work_dir = Path(self.temp_dir_context.__enter__())
        dwi_file = (
            work_dir / "aaa.nii.gz"
        )  # TODO do version with nii and version nii.gz
        bval_file = work_dir / "aaa.bval"
        bvec_file = work_dir / "aaa.bvec"
        save_nifti(dwi_file, self.dwi_data, self.affine)
        with bval_file.open("w") as f:
            print(" ".join(map(str, self.bvals)), file=f, end="")
        with bvec_file.open("w") as f:
            for coord in self.bvecs.T:
                print(" ".join(map(str, coord)), file=f, end="\n")

        return {
            "dir": work_dir,
            "dwi": dwi_file,
            "bval": bval_file,
            "bvec": bvec_file,
        }

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.temp_dir_context.__exit__(exc_type, exc_val, exc_tb)


def test_compute_b0_mean(dwi_data_small_random):
    dwi_data, bvals, bvecs = dwi_data_small_random
    b0_mean = compute_b0_mean(dwi_data, bvals, bvecs)
    assert b0_mean == pytest.approx(
        (dwi_data[..., 0] + dwi_data[..., 3] + dwi_data[..., 4]) / 3
    )


def test_gen_b0_mean(dwi_data_small_random, affine_random):
    dwi_data, bvals, bvecs = dwi_data_small_random
    with TemporaryDwiFiles(dwi_data, affine_random, bvals, bvecs) as paths:
        output_path = paths["dir"] / "out.nii.gz"  # TODO test with nii and nii.gz
        gen_b0_mean(paths["dwi"], paths["bval"], paths["bvec"], output_path)
        output_b0_mean, affine = load_nifti(output_path)
        assert affine == pytest.approx(affine_random)  # test that affine is preserved
        expected_b0_mean = dwi_data[..., np.array(bvals) == 0].mean(axis=-1)
        assert output_b0_mean == pytest.approx(expected_b0_mean)
