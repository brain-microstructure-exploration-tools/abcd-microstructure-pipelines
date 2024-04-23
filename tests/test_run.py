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

        # ipdb.set_trace()
        runner.invoke(
            gen_masks, f"--inputs {input_dir} --outputs {output_dir} --overwrite"
        )
        mock_batch_generate.assert_called_once()
        cases, overwrite, parallel = mock_batch_generate.call_args.args
        assert overwrite
        assert not parallel
        assert len(cases) == 2  # ensure both cases were found for processing
