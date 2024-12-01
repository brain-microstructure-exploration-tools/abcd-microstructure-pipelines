from __future__ import annotations

import itk
import numpy as np
import pytest

from abcdmicro.resource import (
    BvalResource,
    BvecResource,
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
    VolumeResource,
)


def test_bval_abstractness():
    with pytest.raises(TypeError):
        BvalResource()  # type: ignore[abstract]


def test_bvec_abstractness():
    with pytest.raises(TypeError):
        BvecResource()  # type: ignore[abstract]


def test_volume_abstractness():
    with pytest.raises(TypeError):
        VolumeResource()  # type: ignore[abstract]


@pytest.fixture()
def bval_array():
    return np.array([500.0, 1000.0, 200.0])


@pytest.fixture()
def bvec_array():
    return np.array(
        [
            [
                1.0,
                1.0,
                1.0,
            ],
            [2.0, 0.0, -4.0],
        ]
    )


@pytest.fixture()
def volume_array():
    rng = np.random.default_rng(1337)
    return rng.random(size=(3, 4, 5, 6), dtype=float)


def test_bval_inmemory_get(bval_array):
    bval = InMemoryBvalResource(array=bval_array)
    assert (bval.get() == bval_array).all()


def test_bvec_inmemory_get(bvec_array):
    bvec = InMemoryBvecResource(array=bvec_array)
    assert (bvec.get() == bvec_array).all()


@pytest.mark.filterwarnings("ignore:builtin type [sS]wig.* has no __module__ attribute")
def test_volume_inmemory_get_array(volume_array):
    vol = InMemoryVolumeResource(image=itk.image_from_array(volume_array))
    assert (vol.get_array() == volume_array).all()


def test_volume_inmemory_get_metadata(volume_array):
    image = itk.image_from_array(volume_array)
    image["bleh"] = "some_info"
    vol = InMemoryVolumeResource(image=image)
    assert vol.get_metadata()["bleh"] == "some_info"
