"""Tests for the shared JSON-serialization rules.

Both write paths go through this module, so a value serialises the same way
whichever one handles it.
"""

import json
import math
import unittest
from pathlib import Path

import numpy as np

from graflag_runner.serialization import json_default, sanitize, to_jsonable


class NumpyCoercion(unittest.TestCase):
    def test_scalars(self):
        self.assertEqual(json_default(np.float32(1.5)), 1.5)
        self.assertEqual(json_default(np.int64(7)), 7)
        self.assertEqual(json_default(np.bool_(True)), True)

    def test_arrays(self):
        self.assertEqual(json_default(np.arange(3)), [0, 1, 2])
        self.assertEqual(json_default(np.array([[1.0, 2.0]])), [[1.0, 2.0]])

    def test_sets_and_paths(self):
        self.assertEqual(json_default({3, 1, 2}), [1, 2, 3])
        self.assertEqual(json_default(Path("/shared/x")), "/shared/x")

    def test_genuinely_unserializable_still_raises(self):
        with self.assertRaises(TypeError):
            json_default(object())

    def test_numpy_array_of_non_finite_is_cleaned(self):
        self.assertEqual(json_default(np.array([1.0, np.nan, np.inf])), [1.0, None, None])


class NonFiniteReplacement(unittest.TestCase):
    def test_scalars(self):
        self.assertIsNone(to_jsonable(float("nan")))
        self.assertIsNone(to_jsonable(float("inf")))
        self.assertIsNone(to_jsonable(float("-inf")))
        self.assertEqual(to_jsonable(0.5), 0.5)

    def test_nested_structures(self):
        value = {"a": [1.0, float("nan")], "b": {"c": (float("inf"), 2.0)}}
        self.assertEqual(
            to_jsonable(value), {"a": [1.0, None], "b": {"c": [None, 2.0]}}
        )

    def test_counts_what_it_replaced(self):
        cleaned, replaced = sanitize([float("nan"), 1.0, float("inf"), {"x": float("nan")}])
        self.assertEqual(replaced, 3)
        self.assertEqual(cleaned, [None, 1.0, None, {"x": None}])

    def test_clean_input_reports_zero(self):
        _, replaced = sanitize({"scores": [0.1, 0.2]})
        self.assertEqual(replaced, 0)

    def test_output_is_strict_json(self):
        cleaned, _ = sanitize({"scores": [float("nan"), float("inf"), 0.25]})

        def boom(c):
            raise ValueError(c)

        raw = json.dumps(cleaned)
        self.assertNotIn("NaN", raw)
        json.loads(raw, parse_constant=boom)

    def test_non_float_values_pass_through(self):
        value = {"name": "taddy", "n": 3, "ok": True, "none": None}
        self.assertEqual(to_jsonable(value), value)


if __name__ == "__main__":
    unittest.main()
