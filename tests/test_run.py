from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from abcdmicro.run import gen_masks


@pytest.mark.parametrize(
    "extension", ["nii.gz"]
)  # More extensions can go here if we improve I/O
def test_gen_masks(mocker, extension):
    runner = CliRunner()
    mock_batch_generate = mocker.patch("abcdmicro.masks.batch_generate")
    with tempfile.TemporaryDirectory() as work_dir:
        input_dir = Path(work_dir) / "inputs"
        output_dir = Path(work_dir) / "outputs"
        case1_dir = input_dir / "a/b1/c"
        case2_dir = input_dir / "a/b2/"
        img1_name = "an_image"
        img2_name = "another_image"
        for case_dir, img_name in [(case1_dir, img1_name), (case2_dir, img2_name)]:
            case_dir.mkdir(parents=True)
            (case_dir / f"{img_name}_dwi.{extension}").touch()
            (case_dir / f"{img_name}.bval").touch()
            (case_dir / f"{img_name}.bvec").touch()

        runner.invoke(
            gen_masks, f"--inputs {input_dir} --outputs {output_dir} --overwrite"
        )
        mock_batch_generate.assert_called_once()
        cases, overwrite, parallel = mock_batch_generate.call_args.args
        assert overwrite
        assert not parallel
        assert len(cases) == 2  # ensure both cases were found for processing


def test_gen_masks_nonexistent_inputs():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as work_dir:
        input_dir = Path(work_dir) / "dir_that_does_not_exist"
        output_dir = Path(work_dir) / "outputs"
        output_dir.mkdir()

        # We will check that an exception is raised, but we cannot use "with pytests.raises..."
        # because the clock CliRunner swallows the exception. We have to check the result.
        click_result = runner.invoke(
            gen_masks, f"--inputs {input_dir} --outputs {output_dir} --overwrite"
        )
        assert isinstance(click_result.exception, FileNotFoundError)
