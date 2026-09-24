"""Tests for the PyGOD detector registry.

Seventeen bond_* methods resolve their detector through this, so a name that
does not map is a method that cannot run.
"""

import inspect
import pathlib
import unittest

from graflag_bond.detectors import BondDetector


class RegistryContents(unittest.TestCase):
    def setUp(self):
        BondDetector._load_detectors()

    def test_abstract_bases_are_excluded(self):
        """Regression: filtering on module alone put Detector and
        DeepDetector in the registry, so list_detectors() advertised them and
        get_detector_class('detector') returned an uninstantiable class."""
        self.assertNotIn("detector", BondDetector._detectors)
        self.assertNotIn("deepdetector", BondDetector._detectors)

    def test_nothing_abstract_survives(self):
        for name, cls in BondDetector._detectors.items():
            with self.subTest(detector=name):
                self.assertFalse(inspect.isabstract(cls))

    def test_registry_is_not_empty(self):
        self.assertGreater(len(BondDetector._detectors), 5)

    def test_list_detectors_matches_the_registry(self):
        self.assertEqual(set(BondDetector.list_detectors()),
                         set(BondDetector._detectors))


class MethodNameResolution(unittest.TestCase):
    def test_bond_prefix_is_stripped(self):
        self.assertEqual(BondDetector.from_method_name("bond_dominant"), "dominant")

    def test_case_is_normalised(self):
        self.assertEqual(BondDetector.from_method_name("BOND_DOMINANT"), "dominant")
        self.assertEqual(BondDetector.from_method_name("DOMINANT"), "dominant")

    def test_bare_name_works(self):
        self.assertEqual(BondDetector.from_method_name("cola"), "cola")

    def test_unknown_name_lists_what_is_available(self):
        with self.assertRaises(ValueError) as ctx:
            BondDetector.from_method_name("bond_not_a_detector")
        self.assertIn("Supported:", str(ctx.exception))

    def test_abstract_base_is_not_resolvable(self):
        with self.assertRaises(ValueError):
            BondDetector.from_method_name("detector")


class ShippedMethodsResolve(unittest.TestCase):
    """Every bond_* directory must map to a detector this pygod provides."""

    def test_shipped_bond_methods(self):
        methods_dir = pathlib.Path(__file__).resolve().parents[3] / "methods"
        if not methods_dir.is_dir():
            self.skipTest("methods/ not alongside libs/")

        names = sorted(p.name for p in methods_dir.iterdir()
                       if p.is_dir() and p.name.startswith("bond_"))
        self.assertTrue(names, "no bond_* methods found")

        unresolved = []
        for name in names:
            try:
                BondDetector.from_method_name(name)
            except ValueError:
                unresolved.append(name)

        # CARD only exists in pygod master, which the bond_* images install;
        # an older pygod on the test machine is not a method defect.
        unresolved = [n for n in unresolved if n != "bond_card"]
        self.assertEqual(unresolved, [])


if __name__ == "__main__":
    unittest.main()
