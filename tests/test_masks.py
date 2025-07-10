from __future__ import annotations

import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pytest
import scipy.linalg
from dipy.io.image import save_nifti

from abcdmicro.masks import (
    Case,
    brain_extract_batch,
    extract_gen_b0_args,
    extract_hd_bet_args,
)


@pytest.fixture
def dwi_data_small_random():
    rng = np.random.default_rng(1337)
    dwi_data = rng.random(size=(3, 4, 5, 6))
    bvals = np.array([0, 1000, 500, 0, 0, 500], dtype=float)
    bvecs = rng.random(size=(6, 3))
    bvecs = bvecs / np.sqrt((bvecs**2).sum(axis=1, keepdims=True))
    return dwi_data, bvals, bvecs


@pytest.fixture
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


class DwiFiles(NamedTuple):
    dwi: Path
    bval: Path
    bvec: Path


def write_dwi_files(
    work_dir: Path,
    dwi_data: npt.NDArray,
    affine: npt.NDArray,
    bvals: npt.NDArray,
    bvecs: npt.NDArray,
    extension: str = "nii",
):
    dwi_file = work_dir / f"aaa.{extension}"
    bval_file = work_dir / "aaa.bval"
    bvec_file = work_dir / "aaa.bvec"
    save_nifti(dwi_file, dwi_data, affine)
    with bval_file.open("w") as f:
        print(" ".join(map(str, bvals)), file=f, end="")
    with bvec_file.open("w") as f:
        for coord in bvecs.T:
            print(" ".join(map(str, coord)), file=f, end="\n")

    return DwiFiles(dwi=dwi_file, bval=bval_file, bvec=bvec_file)


@pytest.mark.parametrize("extension", ["nii.gz"])  # TODO: add nii extension here
def test_extract_hd_bet_args(extension):
    cases = [
        Case(
            dwi=Path(f"1.{extension}"),
            bval=Path("2.bval"),
            bvec=Path("3.bvec"),
            b0_out=Path(f"4.{extension}"),
            mask_out=Path(f"5_mask.{extension}"),
        ),
        Case(
            dwi=Path(f"a.{extension}"),
            bval=Path("b.bval"),
            bvec=Path("c.bvec"),
            b0_out=Path(f"d.{extension}"),
            mask_out=Path(f"e_mask.{extension}"),
        ),
    ]
    inputs, outputs = extract_hd_bet_args(cases, overwrite=True)
    assert inputs == [f"4.{extension}", f"d.{extension}"]
    assert outputs == [f"5.{extension}", f"e.{extension}"]


def test_extract_hd_bet_args_no_overwrite():
    with tempfile.TemporaryDirectory() as temp_dir:
        existing_mask_file = Path(temp_dir) / "e_mask.nii.gz"
        existing_mask_file.touch()
        cases = [
            Case(
                dwi=Path("1.nii.gz"),
                bval=Path("2.bval"),
                bvec=Path("3.bvec"),
                b0_out=Path("4.nii.gz"),
                mask_out=Path("5_mask.nii.gz"),
            ),
            Case(
                dwi=Path("a.nii.gz"),
                bval=Path("b.bval"),
                bvec=Path("c.bvec"),
                b0_out=Path("d.nii.gz"),
                mask_out=existing_mask_file,
            ),
        ]
        inputs, outputs = extract_hd_bet_args(cases, overwrite=False)
        assert inputs == ["4.nii.gz"]
        assert outputs == ["5.nii.gz"]


def test_extract_hd_bet_args_warn_bad_mask_name(caplog):
    cases = [
        Case(
            dwi=Path("1.nii.gz"),
            bval=Path("2.bval"),
            bvec=Path("3.bvec"),
            b0_out=Path("4.nii.gz"),
            mask_out=Path("5_mask.nii.gz"),
        ),
        Case(
            dwi=Path("a.nii.gz"),
            bval=Path("b.bval"),
            bvec=Path("c.bvec"),
            b0_out=Path("d.nii.gz"),
            mask_out=Path("e_msak.nii.gz"),  # note misspelled "msak"
        ),
    ]
    extract_hd_bet_args(cases, overwrite=True)
    assert "Skipping" in caplog.text


def test_extract_gen_b0_args():
    with tempfile.TemporaryDirectory() as temp_dir:
        existing_b0_out = Path(temp_dir) / "d.nii.gz"
        existing_b0_out.touch()
        cases = [
            Case(
                dwi=Path("1.nii.gz"),
                bval=Path("2.bval"),
                bvec=Path("3.bvec"),
                b0_out=Path("4.nii.gz"),
                mask_out=Path("5_mask.nii.gz"),
            ),
            Case(
                dwi=Path("a.nii.gz"),
                bval=Path("b.bval"),
                bvec=Path("c.bvec"),
                b0_out=existing_b0_out,
                mask_out=Path("e_mask.nii.gz"),
            ),
        ]
        args = extract_gen_b0_args(cases, overwrite=False)
        assert (
            len(args) == 1
        )  # We should skip the second case because the output exists
        assert tuple(map(str, args[0])) == ("1.nii.gz", "2.bval", "3.bvec", "4.nii.gz")


@pytest.mark.parametrize(("extension", "parallel"), [("nii.gz", True), ("nii", False)])
def test_brain_extract_batch(
    mocker, extension, parallel, dwi_data_small_random, affine_random
):
    dwi_data, bvals, bvecs = dwi_data_small_random
    with tempfile.TemporaryDirectory() as work_dir:
        input_paths = write_dwi_files(
            Path(work_dir), dwi_data, affine_random, bvals, bvecs, extension
        )
        b0_out_path = Path(work_dir) / f"b0mean.{extension}"
        mask_out_path = Path(work_dir) / f"aaa_mask.{extension}"

        mock_run_hd_bet = mocker.patch("abcdmicro.masks._run_hd_bet")

        brain_extract_batch(
            cases=[
                Case(
                    dwi=input_paths.dwi,
                    bval=input_paths.bval,
                    bvec=input_paths.bvec,
                    b0_out=b0_out_path,
                    mask_out=mask_out_path,
                )
            ],
            overwrite=False,
            parallel=parallel,
        )

        assert b0_out_path.exists()  # b=0 mean should have been written out
        mock_run_hd_bet.assert_called_once()  # HD_BET should have been called
