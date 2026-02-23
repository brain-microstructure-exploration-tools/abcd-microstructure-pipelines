from __future__ import annotations

import importlib.metadata

import kwneuro as m


def test_version():
    assert importlib.metadata.version("kwneuro") == m.__version__
