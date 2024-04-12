from __future__ import annotations

import importlib.metadata

import abcdmicro as m


def test_version():
    assert importlib.metadata.version("abcd-microstructure-pipelines") == m.__version__
