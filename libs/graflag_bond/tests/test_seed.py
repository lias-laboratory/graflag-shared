"""Tests for the seed of a bond run.

graflag_bond was the one integration that seeded nothing: two runs of the same
bond method with the same configuration trained from different initial
weights, so a repeated run could not tell a regression from noise.
"""

import os
import unittest
from unittest import mock

from graflag_bond import train


class RunSeed(unittest.TestCase):
    def test_defaults_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(train.run_seed(), train.DEFAULT_SEED)

    def test_reads_the_underscore_variable(self):
        with mock.patch.dict(os.environ, {"_SEED": "7"}, clear=True):
            self.assertEqual(train.run_seed(), 7)

    def test_a_non_integer_raises_instead_of_falling_back(self):
        with mock.patch.dict(os.environ, {"_SEED": "seven"}, clear=True):
            with self.assertRaises(ValueError):
                train.run_seed()

    def test_seed_is_not_handed_to_the_detector(self):
        """PyGOD constructors take no seed; passing one would raise."""

        class Detector:
            def __init__(self, hid_dim: int = 64):
                pass

        with mock.patch.dict(os.environ, {"_SEED": "7", "_HID_DIM": "8"}, clear=True):
            self.assertEqual(train.get_all_parameters(Detector), {"hid_dim": 8})


class SeededBeforeTheDetectorIsBuilt(unittest.TestCase):
    """The weights are drawn in the constructor, so the seed has to come first."""

    def test_order(self):
        calls = []

        class Detector:
            def __init__(self, **kwargs):
                calls.append("construct")

            def fit(self, data):
                calls.append("fit")

        writer = mock.Mock()
        with mock.patch.object(train, "seed_all",
                               side_effect=lambda s: calls.append(("seed", s))), \
             mock.patch.object(train.BondDetector, "from_method_name",
                               return_value="dominant"), \
             mock.patch.object(train.BondDetector, "get_detector_class",
                               return_value=Detector), \
             mock.patch.dict(os.environ, {}, clear=True):
            train.train_detector("bond_dominant", data=None, exp_dir=None,
                                 writer=writer, seed=11)

        self.assertEqual(calls, [("seed", 11), "construct", "fit"])


if __name__ == "__main__":
    unittest.main()
