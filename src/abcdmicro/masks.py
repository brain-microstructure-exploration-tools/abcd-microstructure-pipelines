from __future__ import annotations

import itertools
import logging
import multiprocessing.pool
from pathlib import Path
from typing import NamedTuple

import dipy.core.gradients
import dipy.io
import dipy.io.image
import numpy.typing as npt


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


def compute_b0_mean(
    dwi_array: npt.NDArray, bvals: npt.NDArray, bvecs: npt.NDArray
) -> npt.NDArray:
    """
    Compute the mean of the b=0 images of a DWI.

    :param dwi_array: DWI image array of shape (H,W,D,N) where H,W,D are spatial dimensions and there are N DWI volumes.
    :param bvals: array of shape (N,) providing the b-values.
    :param bvecs: array of shape (N,3) providing the b-vectors. They must be unit vectors.
    :return: an array of shape (H,W,D) which is the mean of the b=0 images.
    """
    gtab = dipy.core.gradients.gradient_table(bvals, bvecs)
    return dwi_array[:, :, :, gtab.b0s_mask].mean(axis=3)


def gen_b0_mean(dwi: Path, bval: Path, bvec: Path, b0_out: Path) -> None:
    """
    Compute the mean of the b=0 images of a DWI file, and save the output.

    :param dwi: path to nifti file containing DWI input
    :param bval: path to b-values file in FSL format
    :param bvec: path to b-vectors file in FSL format
    :param b0_out: output path to save nifti file of the b=0 mean
    """

    data, affine, img = dipy.io.image.load_nifti(str(dwi), return_img=True)
    bvals, bvecs = dipy.io.read_bvals_bvecs(str(bval), str(bvec))

    b0_mean = compute_b0_mean(data, bvals, bvecs)

    b0_out.parent.mkdir(parents=True, exist_ok=True)

    logging.debug("generate %r", b0_out)
    dipy.io.image.save_nifti(str(b0_out), b0_mean, affine, img.header)


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


def batch_generate(cases: list[Case], overwrite: bool, parallel: bool) -> None:
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

    logging.debug("Loading HD_BET")
    # don't import till now since it takes time to initialize.
    import HD_BET.run  # pylint: disable=import-outside-toplevel

    logging.debug("Generate %s masks", len(hd_bet_input))
    HD_BET.run.run_hd_bet(hd_bet_input, hd_bet_output, overwrite=overwrite)
