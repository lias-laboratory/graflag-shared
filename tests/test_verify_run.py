"""The skill's run verifier: does it fail the runs that should fail?

`.claude/skills/method-integration/scripts/verify_run.py` is the gate an agent
is told to put between "the run finished" and "the number is real". A gate that
passes everything is worse than none, so each check here is paired with input
that must trip it -- remove the check and the test fails.

`_summarise_pairs` is the part most likely to drift: the manager has no numpy,
so it re-implements in plain Python what `graflag_evaluator.preprocessing`
does with it. `AgreesWithTheEvaluator` runs both over the same inputs.
"""

import importlib.util
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".claude/skills/method-integration/scripts/verify_run.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify_run = _load(SCRIPT, "verify_run")


def probe(**overrides):
    """A probe summary of a run that passes everything, before overrides."""
    base = {
        "status": "completed",
        "results_present": True,
        "result_type": "EDGE_STREAM_ANOMALY_SCORES",
        "n_scores": 100, "n_truth": 100, "kept": 100,
        "dropped_unknown": 0, "dropped_inactive": 0, "dropped_non_finite": 0,
        "n_positive": 10, "distinct_scores": 50,
        "score_min": 0.0, "score_max": 1.0,
        "scored_split": "test", "declared_samples": 100,
        "reported_aucs": {"training_info.test_auc": 0.7806},
        "eval_metrics": {"auc_roc": 0.7806},
    }
    base.update(overrides)
    return base


def levels(findings, level):
    return [m for lvl, m in findings if lvl == level]


class AHealthyRunPasses(unittest.TestCase):
    def test_the_baseline_raises_nothing(self):
        findings = verify_run.check(probe())
        self.assertEqual(levels(findings, "ERROR"), [])
        self.assertEqual(levels(findings, "WARN"), [])


class TheSampleHasToBeUsable(unittest.TestCase):
    def test_a_run_that_did_not_complete_fails(self):
        findings = verify_run.check(probe(status="failed", status_error="OOM"))
        self.assertTrue(any("not 'completed'" in m for m in levels(findings, "ERROR")))

    def test_a_missing_results_file_fails(self):
        findings = verify_run.check({"status": "completed", "results_present": False})
        self.assertTrue(any("nothing was published" in m
                            for m in levels(findings, "ERROR")))

    def test_an_invalid_result_type_fails(self):
        findings = verify_run.check(probe(result_type="EDGE_SCORES"))
        self.assertTrue(any("not one of the nine" in m
                            for m in levels(findings, "ERROR")))

    def test_scores_and_labels_of_different_length_fail(self):
        """The slade defect: the publish sliced one array and not the other."""
        findings = verify_run.check(probe(n_scores=3618, n_truth=24186))
        self.assertTrue(any("not the same sample" in m
                            for m in levels(findings, "ERROR")))

    def test_one_class_ground_truth_fails(self):
        """Scoring the train split: the anomalies are all in the test half."""
        findings = verify_run.check(probe(n_positive=0))
        self.assertTrue(any("one class only" in m
                            for m in levels(findings, "ERROR")))

    def test_a_constant_score_column_fails(self):
        findings = verify_run.check(probe(distinct_scores=1, score_min=0.5))
        self.assertTrue(any("ranked nothing" in m
                            for m in levels(findings, "ERROR")))

    def test_filtering_away_most_of_the_sample_fails(self):
        findings = verify_run.check(
            probe(kept=10, dropped_non_finite=90, n_positive=3))
        self.assertTrue(any("dropped before scoring" in m
                            for m in levels(findings, "ERROR")))

    def test_filtering_away_a_little_only_warns(self):
        findings = verify_run.check(probe(kept=95, dropped_inactive=5, n_positive=10))
        self.assertEqual(levels(findings, "ERROR"), [])
        self.assertTrue(any("dropped before scoring" in m
                            for m in levels(findings, "WARN")))


class ThePublishedScoresAreTheMeasuredOnes(unittest.TestCase):
    """The cross-check: a method's own AUC against the evaluator's.

    It is the only check that looks at *which* scores were written, rather
    than at their shape, and it is cheap because both numbers already exist.
    """

    def test_a_disagreeing_auc_fails(self):
        findings = verify_run.check(probe(
            reported_aucs={"training_info.test_auc": 0.7806},
            eval_metrics={"auc_roc": 0.6829}))
        self.assertTrue(any("not the ones the method measured" in m
                            for m in levels(findings, "ERROR")))

    def test_any_one_reported_auc_matching_is_enough(self):
        """A method may report several; the published one need only be among them."""
        findings = verify_run.check(probe(
            reported_aucs={"best_test_auc": 0.7806, "test_auc": 0.6829},
            eval_metrics={"auc_roc": 0.6829}))
        self.assertEqual(levels(findings, "ERROR"), [])

    def test_rounding_to_four_decimals_still_matches(self):
        findings = verify_run.check(probe(
            reported_aucs={"test_auc": 0.78063285920},
            eval_metrics={"auc_roc": 0.7806}))
        self.assertEqual(levels(findings, "ERROR"), [])

    def test_a_declared_sample_count_that_does_not_match_fails(self):
        findings = verify_run.check(probe(declared_samples=7000))
        self.assertTrue(any("scored_samples=7000" in m
                            for m in levels(findings, "ERROR")))

    def test_no_evaluation_warns_rather_than_passing_quietly(self):
        findings = verify_run.check(probe(eval_metrics=None))
        self.assertEqual(levels(findings, "ERROR"), [])
        self.assertTrue(any("graflag evaluate" in m for m in levels(findings, "WARN")))

    def test_a_method_reporting_no_auc_warns(self):
        findings = verify_run.check(probe(reported_aucs={}))
        self.assertTrue(any("cannot be cross-checked" in m
                            for m in levels(findings, "WARN")))


class TheSplitHasToBeDeclared(unittest.TestCase):
    def test_a_missing_scored_split_warns(self):
        findings = verify_run.check(probe(scored_split=None))
        self.assertTrue(any("scored_split" in m for m in levels(findings, "WARN")))

    def test_scoring_something_other_than_test_warns(self):
        findings = verify_run.check(probe(scored_split="all"))
        self.assertTrue(any("fitted nothing" in m for m in levels(findings, "WARN")))


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
        mine = verify_run._summarise_pairs(scores, truth)
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


class TheRemoteProbeIsSelfContained(unittest.TestCase):
    """It is sent to a manager with no numpy and no graflag on sys.path."""

    def test_the_script_runs_on_a_bare_interpreter(self):
        import json
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            exp = pathlib.Path(tmp)
            (exp / "status.json").write_text('{"status": "completed"}')
            (exp / "results.json").write_text(json.dumps({
                "result_type": "EDGE_STREAM_ANOMALY_SCORES",
                "scores": [0.1, 0.9, -1.0],
                "ground_truth": [0, 1, 1],
                "metadata": {"summary": {"training_info": {"test_auc": 1.0},
                                         "dataset_info": {"scored_split": "test",
                                                          "scored_samples": 3}}},
            }))
            (exp / "eval").mkdir()
            (exp / "eval" / "evaluation.json").write_text(
                json.dumps({"metrics": {"auc_roc": 1.0}}))

            script = verify_run.remote_script(str(exp))
            # `sh -c` the way SSHManager's argument reaches the remote shell.
            done = subprocess.run(["sh", "-c", script], capture_output=True,
                                  text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stderr)
            out = json.loads(done.stdout.strip().splitlines()[-1])

        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["n_scores"], 3)
        self.assertEqual(out["kept"], 2)
        self.assertEqual(out["dropped_unknown"], 1)
        self.assertEqual(out["scored_split"], "test")
        self.assertEqual(out["declared_samples"], 3)
        self.assertEqual(out["reported_aucs"], {"training_info.test_auc": 1.0})
        self.assertEqual(out["eval_metrics"]["auc_roc"], 1.0)
        self.assertEqual(verify_run.check(out), verify_run.check(out))
        self.assertEqual(levels(verify_run.check(out), "ERROR"), [])


if __name__ == "__main__":
    unittest.main()
