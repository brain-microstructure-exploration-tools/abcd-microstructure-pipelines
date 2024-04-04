from __future__ import annotations

import logging
import os
from pathlib import Path

import click

from abcd_microstructure_pipelines import masks

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARN"))


@click.command("gen_masks")
@click.option(
    "--inputs",
    "-i",
    required=True,
    type=Path,
    help="Root directory to search for ``_dwi.nii.gz`` cases.",
)
@click.option(
    "--outputs",
    "-o",
    required=True,
    type=Path,
    help="Root directory for output. Produces ``_dwi_mask.nii.gz`` output for each case, preserving directory structure.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="If true, recompute and overwrite existing output files.",
)
@click.option(
    "--parallel",
    "-j",
    is_flag=True,
    help="If true, compute intermediate ``.b0.nii.gz`` files in parallel. Note that HD_BET does *not* compute in parallel.",
)
def gen_masks(inputs: Path, outputs: Path, overwrite: bool, parallel: bool) -> None:
    """
    Recursively find and process dwi images and create hd_bet masks for each.
    """

    masks.recursive_generate(inputs, outputs, overwrite, parallel)
