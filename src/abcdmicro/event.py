from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AbcdEvent:
    """An ABCD event -- a particular subject and time point."""

    subject_id: str
    """The subject GUID defined in the NIMH Data Archive, for example 'NDAR_INV00U4FTRU'"""

    eventname: str
    """The ABCD Study event name, for example 'baseline_year_1_arm_1'"""

    image_download_path: Path
    """Path to the ABCD image download root directory. This would be the directory that
    contains `fmriresults01/abcd-mproc-release5/` with some compressed images in there"""

    tabular_data_path: Path
    """Path to the extracted ABCD tabular data directory. This would contain subdirectories
    like `core/mental-health/` with csv tables inside them."""
