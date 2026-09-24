"""The skill's result check (gate 4), from graflag-shared's side.

The checks live in the GraFlag client (`graflag/verify.py`, with their own
tests there) and `.claude/skills/method-integration/scripts/verify_run.py` is
a shim over them. Two things are tested here, because only this repository
has what they need:

- `_summarise_pairs` is the part most likely to drift: the manager has no
  numpy, so it re-implements in plain Python what
  `graflag_evaluator.preprocessing` does with it. `AgreesWithTheEvaluator`
  runs both over the same inputs.
- the shim still reaches the checks under its old command line.
"""

import importlib.util
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".claude/skills/method-integration/scripts/verify_run.py"

# Beside this checkout in the workspace, and installed in CI.
sys.path.insert(0, str(ROOT.parent / "graflag"))
from graflag import verify as checks  # noqa: E402


class AgreesWithTheEvaluator(unittest.TestCase):
    """The plain-Python counting must match what numpy does in the evaluator.

    If it does not, the script reports an AUC computed over a sample the
    evaluator never used, which is the class of bug it exists to catch.
    """

    def setUp(self):
        sys.path.insert(0, str(ROOT / "libs"))
        try:
            from graflag_evaluator.preprocessing import prepare_pairs
        except ImportError as exc:                     # numpy/sklearn absent
            self.skipTest(f"graflag_evaluator not importable: {exc}")
        self.prepare_pairs = prepare_pairs

    def assert_agrees(self, scores, truth):
        _, _, report = self.prepare_pairs(scores, truth)
        mine = checks._summarise_pairs(scores, truth)
        self.assertEqual(mine["kept"], report.kept, "kept count differs")
        self.assertEqual(mine["dropped_unknown"], report.dropped_unknown)
        self.assertEqual(mine["dropped_inactive"], report.dropped_inactive)
        self.assertEqual(mine["dropped_non_finite"], report.dropped_non_finite)

    def test_a_plain_column(self):
        self.assert_agrees([0.1, 0.9, 0.4, 0.7], [0, 1, 0, 1])

    def test_the_two_sentinels_are_dropped_the_same_way(self):
        self.assert_agrees([0.1, -1.0, 0.4, -2.0, 0.7], [0, 1, 0, 1, 1])

    def test_non_finite_scores_are_dropped_the_same_way(self):
        self.assert_agrees([0.1, float("nan"), float("inf"), 0.7],
                           [0, 1, 0, 1])

    def test_nested_rows_flatten_the_same_way(self):
        self.assert_agrees([[0.1, 0.9], [0.4, 0.7]], [[0, 1], [0, 1]])


class TheShimKeepsTheOldCommandLine(unittest.TestCase):
    def test_it_is_the_clients_check(self):
        spec = importlib.util.spec_from_file_location("verify_run", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIs(module.main, checks.main)

    def test_it_runs(self):
        done = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("experiment", done.stdout)
        self.assertIn("--json", done.stdout)


if __name__ == "__main__":
    unittest.main()
