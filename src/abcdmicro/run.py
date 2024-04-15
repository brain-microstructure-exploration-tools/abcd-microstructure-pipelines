"""
.. click:: abcdmicro.run:gen_masks
    :prog: gen_masks
    :nested: short
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click

from abcdmicro import masks

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARN"))


@click.command("gen_masks")
@click.option(
    "--inputs",
    "-i",
    required=True,
    type=Path,
    help="Root directory to search for inputs.",
)
@click.option(
    "--outputs",
    "-o",
    required=True,
    type=Path,
    help="Root directory to write outputs.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Recompute and overwrite existing output files.",
)
@click.option(
    "--parallel",
    "-j",
    is_flag=True,
    help="Compute intermediate ``<ID>_b0.nii.gz`` files in parallel.\n"
    "Note that HD_BET does *not* compute in parallel.",
)
def gen_masks(inputs: Path, outputs: Path, overwrite: bool, parallel: bool) -> None:
    """
    Recursively find and process dwi images and create hd_bet masks for each.
    Preserves directory structure in output.

    Searches for input files: ``<ID>_dwi.nii.gz``, ``<ID>.bval``, ``<ID>.bvec``

    Produces output files: ``<ID>_b0.nii.gz``, ``<ID>_mask.nii.gz``
    \f
    See :func:`abcdmicro.masks.batch_generate` for details.
    """

    cases: list[masks.Case] = []

    for dwi in inputs.rglob("*_dwi.nii.gz"):
        base = dwi.with_name(dwi.name.removesuffix(".nii.gz"))
        base_out = outputs.joinpath(base.relative_to(inputs))

        cases.append(
            masks.Case(
                base.with_suffix(".nii.gz"),
                base.with_suffix(".bval"),
                base.with_suffix(".bvec"),
                base_out.with_name(base_out.name + "_b0.nii.gz"),
                base_out.with_name(base_out.name + "_mask.nii.gz"),
            )
        )

    masks.batch_generate(cases, overwrite, parallel)
