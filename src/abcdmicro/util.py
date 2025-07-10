"""Common utility functions"""

from __future__ import annotations

from typing import Any

import numpy as np


def deep_equal_allclose(obj1: Any, obj2: Any) -> bool:
    """
    Recursively compares two objects, including nested lists, tuples,
    and dicts. Uses np.allclose for numpy arrays.
    """
    if type(obj1) is not type(obj2):
        return False

    if isinstance(obj1, np.ndarray):
        if obj1.shape != obj2.shape:
            return False
        if obj1.dtype != obj2.dtype:
            return False
        # if the arrays are numeric, compare them with a tolerance.
        if np.issubdtype(obj1.dtype, np.number) and np.issubdtype(
            obj2.dtype, np.number
        ):
            return bool(np.allclose(obj1, obj2))
        # Otherwise (e.g., for string arrays), require exact equality.
        return bool(np.array_equal(obj1, obj2))

    if isinstance(obj1, dict):
        if obj1.keys() != obj2.keys():
            return False
        return all(deep_equal_allclose(obj1[k], obj2[k]) for k in obj1)

    if isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        return all(
            deep_equal_allclose(item1, item2) for item1, item2 in zip(obj1, obj2)
        )

    # for all other types (int, str, etc.)
    return bool(obj1 == obj2)
