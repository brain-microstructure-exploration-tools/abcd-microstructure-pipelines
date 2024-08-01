from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd


@dataclass
class AbcdEvent:
    """An ABCD event -- a particular subject and time point from a particular ABCD data release."""

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

    abcd_version: str
    """Version of the ABCD dataset release, for example '5.1'."""

    _tables: ClassVar[dict[str, dict[str, pd.DataFrame]]] = {}
    """A mapping (ABCD version string) -> (relative table path) -> (loaded table)"""

    def get_table(self, table_relative_path: str) -> pd.DataFrame:
        """Get a table, loading it from disk if it hasn't already been loaded.

        Args:
            table_relative_path: The relative path of the table from the table root directory.
                Example: 'core/mental-health/mh_p_pss.csv'

        Returns: The loaded table as a pandas DataFrame,
            with subject ID and eventname as a multi-index.
        """
        if self.abcd_version not in self._tables:
            self._tables[self.abcd_version] = {}
        path_to_table_mapping = self._tables[self.abcd_version]
        if table_relative_path not in path_to_table_mapping:
            table = pd.read_csv(
                self.tabular_data_path / table_relative_path,
                index_col=["src_subject_id", "eventname"],
            )
            path_to_table_mapping[table_relative_path] = table
        else:
            table = path_to_table_mapping[table_relative_path]
        return table
