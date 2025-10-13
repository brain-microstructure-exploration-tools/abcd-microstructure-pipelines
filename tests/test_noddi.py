from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

from abcdmicro.dwi import Dwi
from abcdmicro.noddi import Noddi
from abcdmicro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
)


@pytest.fixture
def affine_random():
    rng = np.random.default_rng(7562)
    affine = np.eye(4, dtype=float)
    affine[:3, 3] = 100 * (
        rng.random(size=(3,), dtype=float) - 0.5
    )  # random translation
    random_3by3 = rng.random(size=(3, 3), dtype=float) - 0.5
    affine[:3, :3] = scipy.linalg.expm(
        random_3by3 - random_3by3.T
    )  # exponentiate a random skew-symmetric matrix to get some orthogonal matrix
    return affine


@pytest.fixture
def dwi_data_small_random(affine_random) -> Dwi:
    rng = np.random.default_rng(1337)
    dwi_data = rng.random(size=(3, 4, 5, 6))
    bvals = np.array([0, 1000, 500, 0, 0, 500], dtype=float)
    bvecs = rng.random(size=(6, 3))
    bvecs = bvecs / np.sqrt((bvecs**2).sum(axis=1, keepdims=True))
    return Dwi(
        volume=InMemoryVolumeResource(array=dwi_data, affine=affine_random),
        bval=InMemoryBvalResource(bvals),
        bvec=InMemoryBvecResource(bvecs),
    )


@pytest.mark.parametrize("dpar_value", [None, 1.3e-3])
def test_amico_noddi(mocker, dwi_data_small_random, dpar_value, tmp_path: Path):
    """Test that Noddi.estimate_from_dwi correctly calls AMICO functions with the appropriate dpar parameter."""

    # Mocking of amico result
    mock_results = {
        "MAPs": np.zeros((2, 2, 2, 3)),
        "DIRs": np.ones((2, 2, 2, 3)),
    }

    # Check amico results
    mock_amico = mocker.patch("abcdmicro.noddi.amico", autospec=True)

    mock_eval_instance = mocker.MagicMock()
    mock_eval_instance.RESULTS = mock_results
    mock_eval_instance.model.maps_name = ["NDI", "ODI", "FWF"]
    mock_amico.Evaluation.return_value = mock_eval_instance

    # Call  the function
    if dpar_value is None:
        noddi = Noddi.estimate_from_dwi(dwi_data_small_random)
        expected_dpar = 1.7e-3
    else:
        noddi = Noddi.estimate_from_dwi(dwi_data_small_random, dpar=dpar_value)
        expected_dpar = dpar_value

    mock_eval_instance.generate_kernels.assert_called_once_with(regenerate=True)
    mock_eval_instance.fit.assert_called_once()

    # Verify that the dpar parameter was set correctly
    assert np.isclose(mock_eval_instance.model.dPar, expected_dpar)

    # verify output
    assert isinstance(noddi, Noddi)
    assert noddi.volume.get_array().shape == (2, 2, 2, 3)
    assert noddi.directions.get_array().shape == (2, 2, 2, 3)
    assert np.allclose(noddi.directions.get_array(), 1.0)

    # test save and reload
    path = tmp_path / "dummy_noddi.nii.gz"
    noddi_saved = noddi.save(path=path)
    assert not noddi_saved.volume.is_loaded
    assert not noddi_saved.directions.is_loaded

    noddi_reloaded = noddi_saved.load()
    assert noddi_reloaded.volume.is_loaded
    assert noddi_reloaded.directions.is_loaded

    assert np.allclose(noddi_reloaded.volume.get_array(), noddi.volume.get_array())
    assert np.allclose(
        noddi_reloaded.directions.get_array(), noddi.directions.get_array()
    )
