"""Tests for the .env -> detector-kwargs translation.

Every bond_* method declares its whole parameter set in its `.env`, so any
mismatch between that file and the PyGOD signature is hit on a real run, not
in review. These are the two ways that mismatch used to end a run.
"""

import os
import unittest
from unittest import mock

from graflag_bond.utils import get_all_parameters


class Detector:
    """Stand-in for a PyGOD detector: a couple of typed parameters."""

    def __init__(self, hid_dim: int = 64, lr: float = 0.004, verbose=0):
        pass


class UnknownParametersAreDropped(unittest.TestCase):
    """Regression: the branch that skips a parameter the detector does not
    accept called `logger.debug(...)`, but utils.py imports no logger and no
    logging module -- so skipping raised NameError instead. It fires the
    moment a .env declares a key the installed pygod does not have, which
    a floating `pip install git+.../pygod.git` makes a question of when.
    """

    def test_a_parameter_the_detector_rejects_does_not_raise(self):
        with mock.patch.dict(os.environ, {"_NOT_A_REAL_PARAM": "1"}, clear=True):
            params = get_all_parameters(Detector)
        self.assertNotIn("not_a_real_param", params)

    def test_accepted_parameters_still_come_through(self):
        with mock.patch.dict(
            os.environ, {"_HID_DIM": "128", "_NOT_A_REAL_PARAM": "1"}, clear=True
        ):
            params = get_all_parameters(Detector)
        self.assertEqual(params, {"hid_dim": 128})

    def test_values_are_coerced_to_the_signature_type(self):
        with mock.patch.dict(os.environ, {"_HID_DIM": "128", "_LR": "0.01"}, clear=True):
            params = get_all_parameters(Detector)
        self.assertIsInstance(params["hid_dim"], int)
        self.assertIsInstance(params["lr"], float)

    def test_the_bare_underscore_bash_exports_is_ignored(self):
        """Bash exports `_` into every child process."""
        with mock.patch.dict(os.environ, {"_": "/usr/bin/python3"}, clear=True):
            self.assertEqual(get_all_parameters(Detector), {})

    def test_without_a_signature_everything_is_returned(self):
        with mock.patch.dict(os.environ, {"_ANYTHING": "7"}, clear=True):
            self.assertEqual(get_all_parameters()["anything"], 7)


class SupportedDatasetsCheck(unittest.TestCase):
    """Regression: load_graph_data() read SUPPORTED_DATA (the .env key is
    SUPPORTED_DATASETS) and compared for equality. `"".split(", ")` is `[""]`,
    which is truthy, so the "may not be officially tested" warning fired on
    every bond run -- and even with the right key, "bond_gen_100" is never
    equal to the pattern "bond_*" that all 17 declare.
    """

    def _warnings(self, environ, dataset):
        import pathlib
        import graflag_bond.train as train

        with mock.patch.dict(os.environ, environ, clear=True), \
                mock.patch.object(train, "load_data") as load, \
                mock.patch.object(train, "warning") as warn, \
                mock.patch.object(train, "info"):
            load.return_value = mock.Mock(num_nodes=1, num_edges=1, num_features=1)
            train.load_graph_data(pathlib.Path(f"/shared/datasets/{dataset}"))
        return [c.args[0] for c in warn.call_args_list]

    def test_a_matching_dataset_does_not_warn(self):
        self.assertEqual(
            self._warnings({"SUPPORTED_DATASETS": "bond_*"}, "bond_gen_100"), [])

    def test_no_declaration_does_not_warn(self):
        self.assertEqual(self._warnings({}, "bond_gen_100"), [])

    def test_a_genuinely_unsupported_dataset_still_warns(self):
        self.assertEqual(
            len(self._warnings({"SUPPORTED_DATASETS": "bond_*"}, "uci")), 1)

    def test_several_patterns_are_honoured(self):
        self.assertEqual(
            self._warnings({"SUPPORTED_DATASETS": "inj_*, bond_*"}, "bond_gen_100"), [])


if __name__ == "__main__":
    unittest.main()
