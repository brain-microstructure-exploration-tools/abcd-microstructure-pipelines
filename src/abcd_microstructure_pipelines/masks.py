from __future__ import annotations

import itertools
import logging
import multiprocessing
import os
from collections.abc import Iterable
from pathlib import Path

import click
import dipy.core.gradients
import dipy.io
import dipy.io.image

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARN"))


def find_all_cases(root: Path) -> Iterable[Path]:
    """
    Recursively find all dwi cases in directory. Yield the case base names with file suffixes removed.
    """
    for dwi in root.rglob("*_dwi.nii.gz"):
        yield dwi.with_name(dwi.name.removesuffix(".nii.gz"))


def gen_b0_mean(dwi: Path, bval: Path, bvec: Path, b0_out: Path) -> None:
    """
    :param dwi: `_dwi.nii.gz` input
    :param bval: `.bval` input
    :param bvec: `.bvec` input
    :param b0_out: `.b0.nii.gz` output
    """
    data, affine, img = dipy.io.image.load_nifti(str(dwi), return_img=True)
    bvals, bvecs = dipy.io.read_bvals_bvecs(str(bval), str(bvec))

    gtab = dipy.core.gradients.gradient_table(bvals, bvecs)
    b0_mean = data[:, :, :, gtab.b0s_mask].mean(axis=3)

    b0_out.parent.mkdir(parents=True, exist_ok=True)

    logging.debug("generate %r", b0_out)
    dipy.io.image.save_nifti(str(b0_out), b0_mean, affine, img.header)


@click.command("gen_masks")
@click.option("--inputs", "-i", required=True, type=Path)
@click.option("--outputs", "-o", required=True, type=Path)
@click.option("--overwrite", is_flag=True)
@click.option("--parallel", "-j", is_flag=True)
def gen_masks(inputs: Path, outputs: Path, overwrite: bool, parallel: bool) -> None:
    """
    Recursively find and process dwi images and create hd_bet masks for each.
    :param inputs: Root directory to search for `_dwi.nii.gz` cases.
    :param outputs: Root directory for output. Produces `_dwi_mask.nii.gz` output for each case, preserving directory
        structure.
    :param overwrite: If true, recompute and overwrite existing output files.
    :param parallel: If true, compute intermediate `.b0.nii.gz` files in parallel. Note that HD_BET does _not_ compute
        in parallel.
    """
    b0_tasks: list[tuple[Path, Path, Path, Path]] = []  # args for gen_b0_mean

    hd_bet_input: list[str] = []
    hd_bet_output: list[str] = []

    for base in find_all_cases(inputs):
        base_out = outputs.joinpath(base.relative_to(inputs))

        dwi = base.with_suffix(".nii.gz")
        bval = base.with_suffix(".bval")
        bvec = base.with_suffix(".bvec")

        b0_out = base_out.with_suffix(".b0.nii.gz")

        if overwrite or not b0_out.exists():
            b0_tasks.append((dwi, bval, bvec, b0_out))

        # HD_BET will rename this to "_mask.nii.gz"
        mask_out = base_out.with_suffix(".nii.gz")
        mask_out_real = base_out.with_name(base_out.name + "_mask.nii.gz")

        if overwrite or not mask_out_real.exists():
            hd_bet_input.append(str(b0_out))
            hd_bet_output.append(str(mask_out))

    if parallel:
        logging.debug("Generate %s b0_mean in parallel", len(b0_tasks))
        with multiprocessing.Pool() as pool:
            for _ in pool.starmap(gen_b0_mean, b0_tasks):
                pass  # just consume the iterator. maybe wrap in tqdm?
    else:
        logging.debug("Generate %s b0_mean sequentially", len(b0_tasks))
        for _ in itertools.starmap(gen_b0_mean, b0_tasks):
            pass  # just consume the iterator. maybe wrap in tqdm?

    logging.debug("Loading HD_BET")
    # don't import till now since it takes time to initialize.
    import HD_BET.run  # pylint: disable=import-outside-toplevel

    logging.debug("Generate %s masks", len(hd_bet_input))
    HD_BET.run.run_hd_bet(hd_bet_input, hd_bet_output, overwrite=overwrite)
