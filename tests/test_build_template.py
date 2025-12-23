from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import ants
import numpy as np
import pytest
from ants.core import ANTsImage

from abcdmicro.build_template import (
    _update_template,
    average_volumes,
    build_multi_metric_template,
    build_template,
)
from abcdmicro.dwi import Dwi
from abcdmicro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
    VolumeResource,
)


@pytest.fixture
def dwi1() -> Dwi:
    rng = np.random.default_rng(2656542)
    volume_array = rng.random(size=(3, 4, 5, 6), dtype=float)
    bvals = np.array([0, 1000, 500, 0, 0, 500], dtype=float)
    bvecs = rng.random(size=(6, 3))
    bvecs = bvecs / np.sqrt((bvecs**2).sum(axis=1, keepdims=True))
    return Dwi(
        volume=InMemoryVolumeResource(array=volume_array, affine=np.eye(4)),
        bval=InMemoryBvalResource(bvals),
        bvec=InMemoryBvecResource(bvecs),
    )


@pytest.fixture
def dwi2() -> Dwi:
    rng = np.random.default_rng(26540)
    volume_array = rng.random(size=(3, 5, 7, 6), dtype=float)
    bvals = np.array([0, 1000, 500, 0, 0, 500], dtype=float)
    bvecs = rng.random(size=(6, 3))
    bvecs = bvecs / np.sqrt((bvecs**2).sum(axis=1, keepdims=True))
    return Dwi(
        volume=InMemoryVolumeResource(array=volume_array, affine=np.eye(4)),
        bval=InMemoryBvalResource(bvals),
        bvec=InMemoryBvecResource(bvecs),
    )


def test_average_volumes(dwi1: Dwi, dwi2: Dwi):
    scalar_volume1 = dwi1.compute_mean_b0()
    scalar_volume2 = dwi2.compute_mean_b0()  # bigger size

    # Test that the output is in the space of the volume
    # with the bigger size
    volume_list = [scalar_volume1, scalar_volume2]
    average_volume = average_volumes(volume_list)
    assert average_volume.get_array().shape == scalar_volume2.get_array().shape

    # Test averaging
    average_volume = average_volumes([scalar_volume1, scalar_volume1])
    assert np.allclose(average_volume.get_array(), scalar_volume1.get_array())


def test_update_template():
    # Test data
    data = np.zeros((10, 10, 10))
    data[5, 5, 5] = 100.0
    template_img = ants.from_numpy(data)

    # Dummy average affine - shift 2 in X (negative value is shift to the right)
    aff = ants.new_ants_transform(transform_type="AffineTransform", dimension=3)
    aff_params = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -2, 0, 0])
    aff.set_parameters(aff_params)

    # Dummy warp - shift by 5 in X dimension
    warp_data = np.zeros((10, 10, 10, 1, 3))
    warp_data[..., 0] = -5.0
    avg_warp = ants.from_numpy(warp_data, has_components=True)

    updated_template: ANTsImage = _update_template(
        template=template_img, avg_warp=avg_warp, avg_affine_transform=aff
    )

    res_array = updated_template.numpy()
    new_location = np.unravel_index(np.argmax(res_array), res_array.shape)

    # If logic is correct:
    # new position = current_position (5) - Warp (-5*-0.2) - Affine (2) = 2
    assert new_location[0] == 2
    assert new_location[1] == 5
    assert new_location[2] == 5

    # Check that the intensity value is correct
    assert res_array[new_location] == 100


@pytest.mark.parametrize("provide_initial_template", [True, False])
def test_build_template(
    dwi1: Dwi, dwi2: Dwi, tmp_path: Path, provide_initial_template, mocker
):
    scalar_volume1 = dwi1.compute_mean_b0()
    scalar_volume2 = dwi2.compute_mean_b0()
    volume_list = [scalar_volume1, scalar_volume2]

    average_volume = average_volumes(volume_list)

    # Create dummy warp field & affine MAT file saved to disk
    warp_path = tmp_path / "dummy_warp.nii.gz"
    affine_path = tmp_path / "dummy_affine.mat"

    # Small 3D warp field (ANTS expects vector image, but scalar works for mocks)
    ants.from_numpy(np.zeros_like(average_volume.get_array())).to_file(str(warp_path))

    # Dummy identity affine transform
    identity_affine = ants.new_ants_transform(
        transform_type="AffineTransform", dimension=3
    )
    ants.write_transform(identity_affine, affine_path)

    # Mock ants.registration
    mock_ants_registration_return_value = {
        "fwdtransforms": [str(warp_path), str(affine_path)],
        "warpedmovout": ants.from_numpy(average_volume.get_array()),
    }
    mock_run_ants_registration = mocker.patch(
        "ants.registration",
        return_value=mock_ants_registration_return_value,
    )

    # Mock template sharpening
    mocker.patch(
        "ants.iMath",
        return_value=ants.from_numpy(average_volume.get_array()),
    )

    n_iter = 2
    if provide_initial_template:
        initial_template = scalar_volume1
        expected_initial_template = scalar_volume1
    else:
        initial_template = None
        expected_initial_template = average_volume

    template = build_template(
        volume_list=volume_list, iterations=n_iter, initial_template=initial_template
    )

    # ANTs registration call count - number of subjects * iterations
    assert mock_run_ants_registration.call_count == len(volume_list) * n_iter
    assert (
        template.get_array().shape == scalar_volume2.get_array().shape
    )  # The larger image.

    # Check that the template is being updated as expected
    registration_calls_list = mock_run_ants_registration.call_args_list
    assert len(registration_calls_list) == mock_run_ants_registration.call_count

    # Check that the template is updated as expected
    for i, (_, kwargs) in enumerate(registration_calls_list):
        # First time
        if i == 0:
            assert np.allclose(
                kwargs["fixed"].numpy(), expected_initial_template.get_array()
            )
            assert np.allclose(kwargs["moving"].numpy(), scalar_volume1.get_array())
        elif i == 1:
            assert np.allclose(
                kwargs["fixed"].numpy(), expected_initial_template.get_array()
            )
            assert np.allclose(kwargs["moving"].numpy(), scalar_volume2.get_array())
        elif i == 2:
            assert np.allclose(kwargs["fixed"].numpy(), average_volume.get_array())
            assert np.allclose(kwargs["moving"].numpy(), scalar_volume1.get_array())
        else:
            assert np.allclose(kwargs["fixed"].numpy(), average_volume.get_array())
            assert np.allclose(kwargs["moving"].numpy(), scalar_volume2.get_array())


@pytest.mark.parametrize("provide_initial_template", [True, False])
def test_build_multi_metric_template(
    dwi1: Dwi, dwi2: Dwi, tmp_path: Path, provide_initial_template, mocker
):
    b01 = dwi1.compute_mean_b0()
    b02 = dwi2.compute_mean_b0()

    # For testing, can provide same volume twice
    subject_list = [
        {"dummy_mod1": b01, "dummy_mod2": b01},
        {"dummy_mod1": b02, "dummy_mod2": b02},
    ]

    average_volume = average_volumes([b01, b02])
    if provide_initial_template:
        initial_template = {"dummy_mod1": average_volume, "dummy_mod2": average_volume}
    else:
        initial_template = None

    # Create dummy warp field & affine MAT file saved to disk
    warp_path = tmp_path / "dummy_warp.nii.gz"
    affine_path = tmp_path / "dummy_affine.mat"

    # Small 3D warp field (ANTS expects vector image, but scalar works for mocks)
    shape = average_volume.get_array().shape
    warp_data = np.zeros((*shape, 3))
    ants.from_numpy(warp_data, has_components=True).to_file(str(warp_path))
    assert warp_path.exists()

    # Dummy identity affine transform
    identity_affine = ants.ANTsTransform(transform_type="AffineTransform", dimension=3)
    identity_affine.set_parameters(np.eye(4).flatten())
    ants.write_transform(identity_affine, affine_path)

    mock_ants_registration_return_value = {
        "fwdtransforms": [str(warp_path), str(affine_path)],
        "warpedmovout": ants.from_numpy(average_volume.get_array()),
    }

    mock_run_ants_registration = mocker.patch(
        "ants.registration",
        return_value=mock_ants_registration_return_value,
    )

    n_iter = 2
    num_subjects = len(subject_list)
    template = build_multi_metric_template(
        subject_list=subject_list, initial_template=initial_template, iterations=n_iter
    )

    assert isinstance(template, dict)
    # ANTs registration call count - number of subjects * iterations * 2 (affine + SyN only)
    assert mock_run_ants_registration.call_count == num_subjects * n_iter * 2
    assert template["dummy_mod1"].get_array().shape == average_volume.get_array().shape

    with pytest.raises(ValueError, match="Inconsistent keys detected in 'weights'"):
        template = build_multi_metric_template(
            subject_list=subject_list,
            initial_template=initial_template,
            weights={"dummy_mod1": 1},
            iterations=n_iter,
        )


def test_invalid_inputs(dwi1: Dwi, dwi2: Dwi):
    scalar_volume = dwi1.compute_mean_b0()

    # Provide non scalar dwi volume as input
    subject_list: list[Mapping[str, VolumeResource]] = [
        {"dummy_mod1": scalar_volume, "dummy_mod2": scalar_volume},
        {"dummy_mod1": dwi2.volume, "dummy_mod2": dwi2.volume},  # Not a 3D volume
    ]

    volume_list = [scalar_volume, dwi2.volume]

    with pytest.raises(ValueError, match="Input volume dimensions must be 2D or 3D"):
        build_template(volume_list=volume_list, iterations=1)

    with pytest.raises(ValueError, match="Input volume dimensions must be 2D or 3D"):
        build_multi_metric_template(subject_list=subject_list, iterations=1)

    # Provide non scalar dwi volume as input
    subject_list_error: list[Mapping[str, VolumeResource]] = [
        {"dummy_mod1": scalar_volume, "dummy_mod2": scalar_volume},
        {"dummy_mod1": scalar_volume, "dummy_mod3": scalar_volume},  # Not a 3D volume
    ]

    with pytest.raises(
        ValueError, match="Inconsistent keys detected across subjects in subject_list"
    ):
        build_multi_metric_template(subject_list=subject_list_error, iterations=1)
