from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import AxSymShResponse, ConstrainedSphericalDeconvModel
from scipy.linalg import expm

from kwneuro.resource import (
    BvalResource,
    BvecResource,
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryResponseFunctionResource,
    InMemoryVolumeResource,
    Resource,
    ResponseFunctionResource,
    VolumeResource,
)


def test_resource_abstractness():
    with pytest.raises(TypeError):
        Resource()  # type: ignore[abstract]


def test_bval_abstractness():
    with pytest.raises(TypeError):
        BvalResource()  # type: ignore[abstract]


def test_bvec_abstractness():
    with pytest.raises(TypeError):
        BvecResource()  # type: ignore[abstract]


def test_volume_abstractness():
    with pytest.raises(TypeError):
        VolumeResource()  # type: ignore[abstract]


def test_response_abstractness():
    with pytest.raises(TypeError):
        ResponseFunctionResource()  # type: ignore[abstract]


@pytest.fixture
def bval_array():
    return np.array([500.0, 1000.0, 200.0])


@pytest.fixture
def bvec_array():
    return np.array(
        [
            [
                1.0 / np.sqrt(3),
                1.0 / np.sqrt(3),
                1.0 / np.sqrt(3),
            ],
            [2.0 / np.sqrt(20), 0.0, -4.0 / np.sqrt(20)],
        ]
    )


@pytest.fixture
def volume_array():
    rng = np.random.default_rng(1337)
    return rng.random(size=(3, 4, 5, 6), dtype=float)


@pytest.fixture
def random_affine() -> np.ndarray:
    rng = np.random.default_rng(18653)
    affine = np.eye(4)
    affine[:3, :3] = expm(
        (lambda A: (A - A.T) / 2)(rng.normal(size=(3, 3)))
    )  # generate a random orthogonal matrix
    affine[:3, 3] = rng.random(3)  # generate a random origin
    return affine


@pytest.fixture
def dipy_gradient_table():
    # Need datapoints > 45
    rng = np.random.default_rng(18653)
    bvals = np.linspace(0, 1000, 50)
    random_vectors = rng.random(size=(50, 3))
    norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
    bvecs = random_vectors / norms  # need unit vectors
    return gradient_table(bvals=bvals, bvecs=bvecs)


@pytest.fixture
def prolate_response_function():
    rng = np.random.default_rng(13)
    evals = rng.random(size=(3,))  # random array of coeeficients
    avg_signal = rng.random(size=(), dtype=np.float32) * 1000
    return (evals, avg_signal)


@pytest.fixture
def dipy_response_object() -> AxSymShResponse:
    rng = np.random.default_rng(13)
    sh_order_max = 8
    n_params = int(((sh_order_max + 1) * (sh_order_max + 2)) / 2)
    sh_coeffs = rng.random(size=(n_params,))  # random array of coeeficients
    avg_signal = rng.random(size=(), dtype=np.float32)
    return AxSymShResponse(S0=avg_signal, dwi_response=sh_coeffs)


def test_initialization_fails_with_bad_bvecs():
    # Bad dim
    with pytest.raises(
        ValueError,
        match="Encountered wrong b-vector array shape",
    ):
        InMemoryBvecResource(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))

    # Bad number of dims
    with pytest.raises(
        ValueError,
        match="Encountered wrong b-vector array shape",
    ):
        InMemoryBvecResource(
            np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        )


def test_bval_inmemory_get(bval_array):
    bval = InMemoryBvalResource(array=bval_array)
    assert (bval.get() == bval_array).all()


def test_bvec_inmemory_get(bvec_array):
    bvec = InMemoryBvecResource(array=bvec_array)
    assert (bvec.get() == bvec_array).all()


@pytest.mark.filterwarnings("ignore:builtin type [sS]wig.* has no __module__ attribute")
def test_volume_inmemory_get_array(volume_array, random_affine):
    vol = InMemoryVolumeResource(
        array=volume_array, affine=random_affine, metadata={"bleh": "some_info"}
    )
    assert (vol.get_array() == volume_array).all()


def test_volume_inmemory_get_metadata(volume_array, random_affine):
    vol = InMemoryVolumeResource(
        array=volume_array, affine=random_affine, metadata={"bleh": "some_info"}
    )
    assert vol.get_metadata()["bleh"] == "some_info"


def test_response_inmemory_get(prolate_response_function):
    res = InMemoryResponseFunctionResource(
        sh_coeffs=prolate_response_function[0], avg_signal=prolate_response_function[1]
    )
    assert isinstance(res.get(), tuple)
    assert res.get() == prolate_response_function


def test_dipy_conversion(
    prolate_response_function, dipy_gradient_table, dipy_response_object
):
    res = InMemoryResponseFunctionResource.from_prolate_tensor(
        response=prolate_response_function, gtab=dipy_gradient_table
    )
    res_dipy = res.get_dipy_object()

    # Test that both tuple and AxSymShResponse response functions initialize the same CSD model
    csd_model = ConstrainedSphericalDeconvModel(
        dipy_gradient_table, prolate_response_function
    )
    csd_model_dipy = ConstrainedSphericalDeconvModel(dipy_gradient_table, res_dipy)

    assert np.allclose(
        csd_model.B_reg, csd_model_dipy.B_reg
    )  # The regularization B matrix depends on the response function

    # Test conversion to and from AxSymShResponse object
    res = InMemoryResponseFunctionResource.from_dipy_object(dipy_response_object)
    dipy_object_out = res.get_dipy_object()

    assert np.allclose(dipy_response_object.dwi_response, dipy_object_out.dwi_response)
    assert dipy_response_object.S0 == dipy_object_out.S0

    with pytest.raises(TypeError, match="requires an AxSymShResponse instance"):
        res = InMemoryResponseFunctionResource.from_dipy_object(
            prolate_response_function
        )


def test_volume_resource_ants_conversion():
    # Create dummy voluem resource
    data = np.zeros((20, 25, 30), dtype=np.float32)
    data[5, 10, 15] = 100.0

    # Create a non-identity affine
    affine = np.diag([2, 2, 2, 1])
    affine[:3, 3] = [10, 20, 30]

    hdr = nib.Nifti1Header()
    hdr.set_xyzt_units(xyz="mm", t="sec")

    initial_volume = InMemoryVolumeResource(
        array=data, affine=affine, metadata=dict(hdr)
    )

    # Volume resource to ANTs conversion
    ants_img = initial_volume.to_ants_image()

    # Check ants image
    assert ants_img.shape == (20, 25, 30)
    assert ants_img.spacing == (2.0, 2.0, 2.0)

    ## ANTs to volume resource conversion
    final_volume = InMemoryVolumeResource.from_ants_image(ants_img)

    assert np.allclose(final_volume.get_array(), initial_volume.get_array())
    assert final_volume.get_array()[5, 10, 15] == 100.0

    # CHeck that the affine is preserved after orientation conversions (RAS <-> LPS)
    assert np.allclose(
        final_volume.get_affine(), initial_volume.get_affine(), atol=1e-5
    )

    # Check metadata
    final_hdr = nib.Nifti1Header()
    for key, val in final_volume.get_metadata().items():
        final_hdr[key] = val
    assert final_hdr.get_xyzt_units()[0] == "mm"
