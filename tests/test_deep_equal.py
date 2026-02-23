from __future__ import annotations

import numpy as np

from kwneuro.util import deep_equal_allclose

# --- Tests for Primitive Types ---


def test_primitives_equal():
    """Tests that identical primitive types are equal."""
    assert deep_equal_allclose(5, 5)
    assert deep_equal_allclose("hello", "hello")
    assert deep_equal_allclose(True, True)
    assert deep_equal_allclose(None, None)


def test_primitives_unequal():
    """Tests that different primitive types are not equal."""
    assert not deep_equal_allclose(5, 6)
    assert not deep_equal_allclose("hello", "world")
    assert not deep_equal_allclose(True, False)


def test_primitives_type_mismatch():
    """Tests that different types are not equal."""
    assert not deep_equal_allclose(5, "5")
    assert not deep_equal_allclose(1, True)
    assert not deep_equal_allclose(0, False)
    assert not deep_equal_allclose(None, False)


# --- Tests for NumPy Arrays ---


def test_numeric_arrays_equal():
    """Tests numeric numpy arrays that should be equal."""
    # Exact equality
    assert deep_equal_allclose(np.array([1, 2]), np.array([1, 2]))
    # Floating point 'all close' equality
    assert deep_equal_allclose(np.array([1.0, 2.0]), np.array([1.00000001, 2.0]))


def test_numeric_arrays_unequal():
    """Tests numeric numpy arrays that should be unequal."""
    # Different values
    assert not deep_equal_allclose(np.array([1, 2]), np.array([1, 3]))
    # Different shapes
    assert not deep_equal_allclose(np.array([1, 2]), np.array([1, 2, 3]))
    # Not 'close enough'
    assert not deep_equal_allclose(np.array([1.0, 2.0]), np.array([1.1, 2.0]))


def test_string_arrays_equal():
    """Tests string numpy arrays for equality."""
    assert deep_equal_allclose(
        np.array(["a", "b"], dtype="|S1"), np.array(["a", "b"], dtype="|S1")
    )


def test_string_arrays_unequal():
    """Tests string numpy arrays that should be unequal."""
    assert not deep_equal_allclose(np.array(["a", "b"]), np.array(["a", "c"]))
    # Different dtypes should fail the initial type check
    assert not deep_equal_allclose(
        np.array(["a"], dtype="|S1"), np.array(["a"], dtype="|S2")
    )


def test_array_type_mismatch():
    """Tests comparison between different kinds of arrays."""
    assert not deep_equal_allclose(np.array([1, 2]), np.array(["1", "2"]))


# --- Tests for Nested Structures ---


def test_lists_and_tuples():
    """Tests lists and tuples, including nested ones."""
    # Equal
    assert deep_equal_allclose([1, [2, 3]], [1, [2, 3]])
    assert deep_equal_allclose((1, (2, 3)), (1, (2, 3)))
    # Unequal
    assert not deep_equal_allclose([1, [2, 3]], [1, [2, 4]])
    # Different lengths
    assert not deep_equal_allclose([1, 2], [1, 2, 3])
    # List vs. Tuple
    assert not deep_equal_allclose([1, 2, 3], (1, 2, 3))


def test_dictionaries():
    """Tests dictionaries, including nested ones."""
    # Equal
    assert deep_equal_allclose({"a": 1, "b": 2}, {"a": 1, "b": 2})
    # Unequal values
    assert not deep_equal_allclose({"a": 1, "b": 2}, {"a": 1, "b": 3})
    # Unequal keys
    assert not deep_equal_allclose({"a": 1, "b": 2}, {"a": 1, "c": 2})
    # Different number of keys
    assert not deep_equal_allclose({"a": 1, "b": 2}, {"a": 1})


def test_complex_nested_objects():
    """Tests deeply nested, complex objects."""
    obj1 = {
        "id": "test_01",
        "data": ([10, 20], np.array([1.00000001, 2.5, 3.0])),
        "params": {
            "desc": np.array(["coeff_a", "coeff_b"], dtype="|S7"),
            "values": np.array([4, 5, 6]),
        },
    }
    obj2 = {
        "id": "test_01",
        "data": ([10, 20], np.array([1.0, 2.5, 3.0])),
        "params": {
            "desc": np.array(["coeff_a", "coeff_b"], dtype="|S7"),
            "values": np.array([4, 5, 6]),
        },
    }
    # Should pass due to allclose on numeric arrays and equal on others
    assert deep_equal_allclose(obj1, obj2)

    # Now introduce a non-numeric difference
    obj3 = {
        "id": "test_01",
        "data": ([10, 20], np.array([1.0, 2.5, 3.0])),
        "params": {
            "desc": np.array(["coeff_X", "coeff_b"], dtype="|S7"),  # Difference here
            "values": np.array([4, 5, 6]),
        },
    }
    assert not deep_equal_allclose(obj1, obj3)
