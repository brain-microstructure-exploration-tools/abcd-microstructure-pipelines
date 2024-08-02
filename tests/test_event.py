from __future__ import annotations

from pathlib import Path

from abcdmicro.event import AbcdEvent


def test_create_event():
    AbcdEvent(
        subject_id="NDAR_INV00U4FTRU",
        eventname="baseline_year_1_arm_1",
        image_download_path=Path("/this/is/a/path/for/images"),
        tabular_data_path=Path("/this/is/a/path/for/tables"),
        abcd_version="5.1",
    )
