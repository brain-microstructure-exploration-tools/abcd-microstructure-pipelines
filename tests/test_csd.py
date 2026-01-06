from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.linalg
from dipy.direction.peaks import PeaksAndMetrics

from abcdmicro.csd import (
    combine_csd_peaks_to_vector_volume,
    combine_response_functions,
    compute_csd_fods,
    compute_csd_peaks,
)
from abcdmicro.dwi import Dwi
from abcdmicro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryResponseFunctionResource,
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


@pytest.fixture
def response_function() -> InMemoryResponseFunctionResource:
    rng = np.random.default_rng(13)
    sh_coeffs = rng.random(size=(10,))  # random array of coeeficients
    avg_signal = rng.random(size=(), dtype=np.float32)
    return InMemoryResponseFunctionResource(sh_coeffs=sh_coeffs, avg_signal=avg_signal)


def mock_mask_for_response(mocker, dwi_data_small_random: Dwi) -> Any:
    vol = dwi_data_small_random.volume
    mask_for_response = np.ones(vol.get_array().shape[:-1], dtype=int)
    return mocker.patch(
        "abcdmicro.csd.mask_for_response_ssst",
        return_value=mask_for_response,
    )


def mock_response_from_mask(mocker) -> Any:
    response_tensor = (np.array([0.00139919, 0.0003007, 0.0003007]), 416)
    return mocker.patch(
        "abcdmicro.csd.response_from_mask_ssst",
        return_value=(response_tensor, 0.2),
    )


@pytest.mark.parametrize("response_is_none", [True, False])
@pytest.mark.parametrize("flip_bvecs_x", [True, False])
def test_csd_peaks(
    mocker, dwi_data_small_random, response_is_none, response_function, flip_bvecs_x
):
    # Create mock csd peaks data
    rng = np.random.default_rng(17)
    vol = dwi_data_small_random.volume
    vol_shape = vol.get_array().shape[:-1]  #  x,y,z, dimensions only

    # Mask - all ones
    mask = InMemoryVolumeResource(array=np.ones(vol_shape, dtype=int))
    n_peaks = 3
    peak_dir_data = rng.random(size=(*vol_shape, n_peaks, 3))
    peak_value_data = rng.random(size=(*vol_shape, n_peaks))

    # Mock results and functions for computing csd peaks
    csd_peaks_mock = PeaksAndMetrics()
    csd_peaks_mock.peak_values = peak_value_data
    csd_peaks_mock.peak_dirs = peak_dir_data
    mocker_csd_peaks = mocker.patch(
        "abcdmicro.csd.peaks_from_model", return_value=csd_peaks_mock
    )

    # Patch estimate_response_function
    response = None if response_is_none else response_function

    mock_res_mask = mock_response_from_mask(mocker)
    mock_compute_mask = mock_mask_for_response(mocker, dwi_data_small_random)
    mock_estimate_response = mocker.patch(
        "abcdmicro.csd.InMemoryResponseFunctionResource.from_prolate_tensor",
        return_value=response_function,
    )

    # Mock CSD
    mock_csd_model = mocker.patch(
        "abcdmicro.csd.ConstrainedSphericalDeconvModel",
        return_value=mocker.Mock(),
    )

    mock_gradient_table = mocker.patch(
        "abcdmicro.csd.gradient_table",
        return_value=mocker.Mock(),
    )

    # Call function
    output = compute_csd_peaks(
        dwi_data_small_random, mask=mask, response=response, flip_bvecs_x=flip_bvecs_x
    )

    # ---Tests---
    mocker_csd_peaks.assert_called_once()
    mock_csd_model.assert_called_once()

    if response_is_none:
        # Test response function estimation
        mock_res_mask.assert_called_once()
        mock_compute_mask.assert_called_once()
        mock_estimate_response.assert_called_once()
        assert (
            mock_gradient_table.call_count == 2
        )  # called in estimate_response_function and compute_csd_peaks

    else:
        mock_gradient_table.assert_called_once()
        mock_res_mask.assert_not_called()
        mock_compute_mask.assert_not_called()
        mock_estimate_response.assert_not_called()

    # Test that bvecs are flipped correctly
    # even when called twice within the function
    _, kwargs = mock_gradient_table.call_args
    passed_bvecs = kwargs["bvecs"]

    expected_bvecs = dwi_data_small_random.bvec.get().copy()
    if flip_bvecs_x:
        expected_bvecs[:, 0] *= -1

    assert np.allclose(passed_bvecs, expected_bvecs)

    # Test output
    assert isinstance(output, tuple)

    dirs, values = output
    assert np.allclose(dirs.get_array(), peak_dir_data)
    assert np.allclose(values.get_array(), peak_value_data)


@pytest.mark.parametrize("response_is_none", [True, False])
@pytest.mark.parametrize("flip_bvecs_x", [True, False])
@pytest.mark.parametrize("mrtrix_format", [True, False])
def test_compute_csd_fods(
    mocker,
    dwi_data_small_random,
    response_is_none,
    response_function,
    flip_bvecs_x,
    mrtrix_format,
):
    # Create mask
    vol = dwi_data_small_random.volume
    vol_shape = vol.get_array().shape[:-1]  #  x,y,z, dimensions only
    mask = InMemoryVolumeResource(array=np.ones(vol_shape, dtype=int))

    # Patch estimate_response_function
    response = None if response_is_none else response_function
    mock_res_mask = mock_response_from_mask(mocker)
    mock_compute_mask = mock_mask_for_response(mocker, dwi_data_small_random)
    mock_estimate_response = mocker.patch(
        "abcdmicro.csd.InMemoryResponseFunctionResource.from_prolate_tensor",
        return_value=response_function,
    )

    # Mock CSD model and .fit()
    rng = np.random.default_rng(17)
    mock_csd_coeffs = rng.random(size=(45,))
    mock_csd_fit = mocker.Mock()
    mock_csd_fit.shm_coeff = mock_csd_coeffs

    mock_csd_model_instance = mocker.Mock()
    mock_csd_model_instance.fit.return_value = mock_csd_fit
    mock_csd_model = mocker.patch(
        "abcdmicro.csd.ConstrainedSphericalDeconvModel",
        return_value=mock_csd_model_instance,
    )

    mock_gradient_table = mocker.patch(
        "abcdmicro.csd.gradient_table",
        return_value=mocker.Mock(),
    )

    # Call function
    output = compute_csd_fods(
        dwi_data_small_random,
        mask=mask,
        response=response,
        flip_bvecs_x=flip_bvecs_x,
        mrtrix_format=mrtrix_format,
    )

    # ---Tests---
    mock_csd_model.assert_called_once()
    if response_is_none:
        # Test response function estimation
        mock_res_mask.assert_called_once()
        mock_compute_mask.assert_called_once()
        mock_estimate_response.assert_called_once()
        assert (
            mock_gradient_table.call_count == 2
        )  # called in estimate_response_function and compute_csd_fods

    else:
        mock_gradient_table.assert_called_once()
        mock_res_mask.assert_not_called()
        mock_compute_mask.assert_not_called()
        mock_estimate_response.assert_not_called()

    # Test that bvecs are flipped
    _, kwargs = mock_gradient_table.call_args
    passed_bvecs = kwargs["bvecs"]

    expected_bvecs = dwi_data_small_random.bvec.get().copy()
    if flip_bvecs_x:
        expected_bvecs[:, 0] *= -1

    assert np.allclose(passed_bvecs, expected_bvecs)

    # Test output
    assert isinstance(output, np.ndarray)

    # Test mrtrix_format conversion
    if mrtrix_format:
        assert not np.allclose(output, mock_csd_coeffs)
    else:
        assert np.allclose(output, mock_csd_coeffs)


def test_combine_csd_peaks_to_vector(dwi_data_small_random):
    # Create mock csd peaks data
    rng = np.random.default_rng(17)
    vol = dwi_data_small_random.volume
    vol_shape = vol.get_array().shape[:-1]  #  x,y,z, dimensions only

    # Create unit vector peak directions
    n_peaks = 3
    peak_dir_data = rng.random(size=(*vol_shape, n_peaks, 3))
    peak_dir_data /= np.linalg.norm(peak_dir_data, axis=-1, keepdims=True)
    peak_value_data = rng.random(size=(*vol_shape, n_peaks))

    csd_peak_dir = InMemoryVolumeResource(array=peak_dir_data)
    csd_peak_value = InMemoryVolumeResource(array=peak_value_data)

    combined_peaks = combine_csd_peaks_to_vector_volume(csd_peak_dir, csd_peak_value)
    combined_array = combined_peaks.get_array()
    assert combined_array.shape == (*vol_shape, n_peaks * 3)

    # Reshape back to (x,y,z,n_peaks, 3)
    combined_vectors = combined_array.reshape(*vol_shape, n_peaks, 3)
    combined_vector_norms = np.linalg.norm(combined_vectors, axis=-1)
    assert np.allclose(combined_vector_norms, peak_value_data)


def test_combine_response_functions_averaging():
    rng = np.random.default_rng(1337)

    # Example response functions
    l0_a = 200
    res_a = InMemoryResponseFunctionResource(
        sh_coeffs=np.concatenate(([l0_a], rng.random(44) * 100)), avg_signal=1000.0
    )
    l0_b = 200
    res_b = InMemoryResponseFunctionResource(
        sh_coeffs=np.concatenate(([l0_b], rng.random(44) * 100)), avg_signal=800.0
    )

    # Test combining
    result = combine_response_functions([res_a, res_b])
    expected_s0 = (1000.0 + 800.0) / 2.0
    # Since L=0 were already the same, the multipliers should be 1.0.
    # The coeffs should be a simple mean of the two responses.
    expected_coeffs = (res_a.sh_coeffs + res_b.sh_coeffs) / 2.0

    assert result.sh_coeffs.shape == (45,)
    assert result.avg_signal == expected_s0
    assert np.allclose(result.sh_coeffs, expected_coeffs)

    # Confirm that averaging identical responses returns the same response
    result = combine_response_functions([res_a, res_a])
    assert result.avg_signal == res_a.avg_signal
    assert np.allclose(result.sh_coeffs, res_a.sh_coeffs)


def test_combine_response_functions_mismatched_signals():
    rng = np.random.default_rng(1337)

    # Example response functions
    l0_a = 200
    res_a = InMemoryResponseFunctionResource(
        sh_coeffs=np.concatenate(([l0_a], rng.random(44) * 100)), avg_signal=1000.0
    )
    l0_b = 500
    res_b = InMemoryResponseFunctionResource(
        sh_coeffs=np.concatenate(([l0_b], rng.random(44) * 100)), avg_signal=800.0
    )

    # Test combining
    result = combine_response_functions([res_a, res_b])
    avg_l0 = (l0_a + l0_b) / 2.0
    expected_sh_coeffs = res_a.sh_coeffs * (avg_l0 / l0_a) + res_b.sh_coeffs * (
        avg_l0 / l0_b
    )
    expected_sh_coeffs /= 2.0

    assert np.allclose(result.sh_coeffs, expected_sh_coeffs)

    res_error = InMemoryResponseFunctionResource(
        sh_coeffs=np.concatenate(([l0_b], rng.random(42) * 100)), avg_signal=800.0
    )

    with pytest.raises(
        ValueError, match="must have the same number of SH coefficients"
    ):
        combine_response_functions([res_a, res_error])
