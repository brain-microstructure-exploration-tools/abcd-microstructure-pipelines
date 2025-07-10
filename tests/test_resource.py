from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from abcdmicro.resource import (
    BvalResource,
    BvecResource,
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
    Resource,
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


def test_initialization_fails_with_bad_bvecs():
    # Non unit vectors
    with pytest.raises(
        ValueError,
        match="All b-vectors must be unit vectors.",
    ):
        InMemoryBvecResource(
            np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        )

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
