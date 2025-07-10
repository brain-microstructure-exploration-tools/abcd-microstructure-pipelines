from __future__ import annotations

import itertools
import logging
import multiprocessing.pool
from pathlib import Path
from typing import NamedTuple

import torch

from abcdmicro.dwi import Dwi
from abcdmicro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource


class Case(NamedTuple):
    """All input and output files to generate a single mask."""

    dwi: Path
    """``_dwi.nii.gz`` input."""

    bval: Path
    """``.bval`` input."""

    bvec: Path
    """``.bvec`` input."""

    b0_out: Path
    """``_b0.nii.gz`` output."""

    mask_out: Path
    """``_mask.nii.gz`` output.

    HD-BET always outputs files ending in ``_mask.nii.gz``, so this file must have that suffix. Otherwise another
    file with that suffix would be created and ``mask_out`` will not exist.
    """


def gen_b0_mean(dwi: Path, bval: Path, bvec: Path, b0_out: Path) -> None:
    """
    Compute the mean of the b=0 images of a DWI file, and save the output.

    :param dwi: path to nifti file containing DWI input
    :param bval: path to b-values file in FSL format
    :param bvec: path to b-vectors file in FSL format
    :param b0_out: output path to save nifti file of the b=0 mean
    """

    b0_mean = Dwi(
        volume=NiftiVolumeResource(dwi),
        bval=FslBvalResource(bval),
        bvec=FslBvecResource(bvec),
    ).compute_mean_b0()
    b0_out.parent.mkdir(parents=True, exist_ok=True)
    logging.debug("generate %r", b0_out)
    NiftiVolumeResource.save(b0_mean, b0_out)


def extract_hd_bet_args(
    cases: list[Case], overwrite: bool
) -> tuple[list[str], list[str]]:
    """
    Extract arguments for ``HD_BET.run.run_hd_bet`` to process the cases. Do not include cases whose output already
    exists, unless ``overwrite`` is set.

    hd_bet expects arguments as a pair of lists, rather than a list of pairs. hd_bet also appends ``_mask`` to its
    output filenames, and this feature cannot be disabled, so check the outputs in ``tasks`` contain this suffix and
    choose the arguments to produce the correct output.

    Warn and skip tasks where this is not possible.

    :param cases: list of cases to process
    :param overwrite: include cases with already existing output.
    :return: (inputs, outputs) arguments suitable for ``HD_BET.run.run_hd_bet``
    """

    inputs = []
    outputs = []

    for case in cases:
        if not overwrite and case.mask_out.exists():
            continue

        # invert hd_bet behavior.
        output_arg = case.mask_out.with_name(
            case.mask_out.name.removesuffix("_mask.nii.gz") + ".nii.gz"
        )

        # match hd_bet behavior.
        output_real = output_arg.with_name(output_arg.name[:-7] + "_mask.nii.gz")

        if output_real != case.mask_out:
            logging.warning(
                "HD-BET will not output %r. Would output %r instead. Skipping.",
                case.mask_out.name,
                output_real.name,
            )
            continue

        inputs.append(str(case.b0_out))
        outputs.append(str(output_arg))

    return inputs, outputs


def extract_gen_b0_args(
    cases: list[Case], overwrite: bool
) -> list[tuple[Path, Path, Path, Path]]:
    """
    Extract arguments for ``gen_b0_mean`` to process each case. Do not include cases whose output already exists,
    unless ``overwrite`` is set.

    :param cases: list of cases to process
    :param overwrite: include cases with already existing output.
    :return: list of arguments for invocations to ``gen_b0_mean``, suitable for ``starmap``.
    """

    args = []
    for case in cases:
        if not overwrite and case.b0_out.exists():
            continue

        args.append((case.dwi, case.bval, case.bvec, case.b0_out))

    return args


def _run_hd_bet(
    hd_bet_input: list[str],
    hd_bet_output: list[str],
    device: str | None = None,
    use_tta: bool | None = None,
) -> None:  # TODO UPDATE THE DOCSTRING!
    """
    Run HD-BET inference on the given input files.
    The expensive HD-BET import and run call are isolated in this function.

    :param hd_bet_input: List of input filepaths. We have only observed nii.gz files to work.
    :param hd_bet_output: List of output filepaths. We have only observed nii.gz files to work.
    :param device: A string indicating the device on which to use pytorch. Defaults to 'cuda' if available, otherwise 'cpu'.
    :param use_tta: Whether to tell HD-BET to do test time augmentation. Takes a bit longer but is a bit more accurate.
        Default is to use it if the device is GPU and omit it if the device is CPU; this is HD-BET's recommendation.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_tta is None:
        use_tta = torch.device(device).type == "cuda"

    # try to catch certain issues before going into HD-BET
    if any(
        not Path(path).name.endswith(".nii.gz") for path in hd_bet_input + hd_bet_output
    ):
        logging.warning(
            "abcdmicro HD-BET runner has only been tested with *.nii.gz files. Masking might not work."
        )
    if any(Path(path).exists() for path in hd_bet_output):
        logging.warning(
            "HD-BET output already exists and will be overwritten for the following files: %s",
            [path for path in hd_bet_output if Path(path).exists()],
        )
    if any(not Path(path).exists() for path in hd_bet_input):
        msg = f"Some input files to HD-BET do not exist: {[path for path in hd_bet_input if not Path(path).exists()]}"
        raise FileNotFoundError(msg)

    logging.debug("Loading HD_BET")
    # don't import till now since it takes time to initialize.
    from HD_BET.checkpoint_download import (
        maybe_download_parameters,  # pylint: disable=import-outside-toplevel
    )
    from HD_BET.hd_bet_prediction import (
        get_hdbet_predictor,  # pylint: disable=import-outside-toplevel
    )

    maybe_download_parameters()
    predictor = get_hdbet_predictor(
        use_tta=use_tta,
        device=torch.device(device),
    )

    logging.debug("Generate %s masks", len(hd_bet_input))
    predictor.predict_from_files(
        list_of_lists_or_source_folder=[[i] for i in hd_bet_input],
        output_folder_or_list_of_truncated_output_files=hd_bet_output,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=8,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )


def brain_extract_batch(cases: list[Case], overwrite: bool, parallel: bool) -> None:
    """
    Generate ``b0_out`` and ``mask_out`` for each case. See ``extract_hd_bet_args`` for notes on HD_BET.

    :param cases: The cases to process.
    :param overwrite: Overwrite existing files only if this is set.
    :param parallel: Generate ``b0_out`` in parallel. HD_BET does not run in parallel.
    """

    b0_tasks = extract_gen_b0_args(cases, overwrite)

    hd_bet_input, hd_bet_output = extract_hd_bet_args(cases, overwrite)

    if parallel:
        logging.debug("generate %s b0_mean in parallel", len(b0_tasks))
        with multiprocessing.pool.Pool() as pool:
            for _ in pool.starmap(gen_b0_mean, b0_tasks):
                pass  # just consume the iterator. maybe wrap in tqdm?
    else:
        logging.debug("generate %s b0_mean sequentially", len(b0_tasks))
        for _ in itertools.starmap(gen_b0_mean, b0_tasks):
            pass  # just consume the iterator. maybe wrap in tqdm?

    _run_hd_bet(hd_bet_input, hd_bet_output)
