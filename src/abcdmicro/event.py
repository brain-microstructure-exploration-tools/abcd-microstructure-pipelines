from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from abcdmicro.dwi import Dwi
from abcdmicro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource


@dataclass
class AbcdEvent:
    """An ABCD event -- a particular subject and time point from a particular ABCD data release."""

    subject_id: str
    """The subject GUID defined in the NIMH Data Archive, for example 'NDAR_INV00U4FTRU'"""

    eventname: str
    """The ABCD Study event name, for example 'baseline_year_1_arm_1'"""

    image_download_path: Path
    """Path to the ABCD image root directory. This would be the directory that
    contains paths like
        `sub-NDARINV6NU2WWNR/ses-2YearFollowUpYArm1/dwi/sub-NDARINV6NU2WWNR_ses-2YearFollowUpYArm1_run-01_dwi.nii`
    or
        `sub-NDARINV6NU2WWNR/ses-2YearFollowUpYArm1/dwi/sub-NDARINV6NU2WWNR_ses-2YearFollowUpYArm1_run-01_dwi.nii.gz`
    and
        `sub-NDARINV6NU2WWNR/ses-2YearFollowUpYArm1/dwi/sub-NDARINV6NU2WWNR_ses-2YearFollowUpYArm1_run-01_dwi.bval`
    and so on.
    """

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

    def get_row(self, table_relative_path: str) -> pd.Series:
        """Get the row from a table pertaining to this particular ABCD event.

        Args:
            table_relative_path: The relative path of the table from the table root directory.
                Example: 'core/mental-health/mh_p_pss.csv'
        Returns: A pandas Series containing the information in the requested table for this particular ABCD event.
        """
        return self.get_table(table_relative_path).loc[self.subject_id, self.eventname]

    def get_table_value(self, table_relative_path: str, variable_name: str) -> Any:
        """Get a value from a table pertaining to this particular ABCD event.

        Args:
            table_relative_path: The relative path of the table from the table root directory.
                Example: 'core/mental-health/mh_p_pss.csv'
            variable_name: The name of the column to query in the ABCD table.
                Example: 'pstr_not_cope_p'

        Returns: The queried value from the table.
        """
        return self.get_row(table_relative_path)[variable_name]

    def get_dwis(self) -> list[Dwi]:
        """Get a list of DWIs associated to this ABCD event. Usually there is just one, but for Philips scanners there can be two separate DWIs
        that need to be concatenated."""
        subject_string = f"sub-{self.subject_id.replace('_', '')}"
        eventname_camel_case = "".join(
            [s[0].upper() + s[1:] for s in self.eventname.split("_")]
        )
        session_string = f"ses-{eventname_camel_case}"
        session_dir = self.image_download_path / subject_string / session_string / "dwi"

        basenames = list(
            {
                filepath.name.split(".")[0]
                for filepath in session_dir.glob(f"{subject_string}_{session_string}*")
            }
        )
        # This contains unique basenames in the session directory such as "sub-NDARINV6NU2WWNR_ses-2YearFollowUpYArm1_run-01_dwi"

        dwi_list: list[Dwi] = []

        for basename in basenames:
            nifti_filepath: Path | None = None
            for nifti_extension in ["nii", "nii.gz"]:  # Could be compressed or not
                if (session_dir / f"{basename}.{nifti_extension}").exists():
                    nifti_filepath = session_dir / f"{basename}.{nifti_extension}"
                    break
            if nifti_filepath is None:
                msg = f"Did not find a nifti volume in {session_dir}"
                raise FileNotFoundError(msg)
            bval_filepath = session_dir / f"{basename}.bval"
            bvec_filepath = session_dir / f"{basename}.bvec"
            for path in [bval_filepath, bvec_filepath]:
                if not path.exists():
                    msg = f"Did not find {path}"
                    raise FileNotFoundError(msg)
            dwi_list.append(
                Dwi(
                    event=self,
                    volume=NiftiVolumeResource(nifti_filepath),
                    bval=FslBvalResource(bval_filepath),
                    bvec=FslBvecResource(bvec_filepath),
                )
            )
        if len(dwi_list) == 0:
            logging.warning(
                "Empty DWI list. Is the `image_download_path` set correctly? Have the images been extracted?"
            )
        return dwi_list
