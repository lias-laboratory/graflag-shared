"""Tests for metric correctness.

These pin down the numbers GraFlag reports. Where a value is asserted it was
computed by hand or from a definition, not copied from the implementation's
output -- a test that records whatever the code currently does cannot catch the
code being wrong.
"""

import unittest

import numpy as np

from graflag_evaluator.metrics import compute_classification_metrics as compute
from graflag_evaluator.preprocessing import prepare_pairs


class SentinelAndNonFiniteFiltering(unittest.TestCase):
    """RESULTS_STANDARD.md defines -1 as unknown and -2 as inactive."""

    def test_negative_scores_are_evaluated_not_discarded(self):
        """Regression: a heuristic keyed on max<=1 discarded every sample.

        Log-likelihood scores are all negative, so the old
        `(>=0)&(<=1) if max<=1` branch kept nothing and the function reported
        'only one class' -- a false diagnosis -- with null metrics.
        """
        scores = np.array([-0.5, -1.5, -3.0, -0.2, -2.5])
        gt = np.array([0, 1, 0, 1, 0])
        r = compute(scores, gt)

        self.assertEqual(r["num_samples"], 5)
        self.assertIsNotNone(r["auc_roc"])

    def test_sentinels_removed_regardless_of_score_range(self):
        """Regression: with any score > 1 the old mask was `> -2`, keeping -1."""
        scores = np.array([5.0, 3.0, -1.0, -2.0, 4.0, 0.5])
        gt = np.array([1, 0, 1, 0, 1, 0])
        r = compute(scores, gt)

        self.assertEqual(r["filtering"]["dropped_unknown"], 1)
        self.assertEqual(r["filtering"]["dropped_inactive"], 1)
        self.assertEqual(r["num_samples"], 4)

    def test_sentinels_removed_when_scores_are_probabilities(self):
        scores = np.array([0.1, 0.9, -1.0, 0.4, -2.0, 0.8])
        gt = np.array([0, 1, 1, 0, 0, 1])
        r = compute(scores, gt)
        self.assertEqual(r["num_samples"], 4)

    def test_one_nan_does_not_change_treatment_of_other_values(self):
        """Regression: np.max returning nan flipped the branch for the whole array."""
        scores = np.array([0.1, 0.9, np.nan, -1.0, 0.7, 0.2])
        gt = np.array([0, 1, 0, 1, 1, 0])
        r = compute(scores, gt)

        self.assertEqual(r["filtering"]["dropped_non_finite"], 1)
        self.assertEqual(r["filtering"]["dropped_unknown"], 1)
        self.assertEqual(r["num_samples"], 4)

    def test_infinity_is_dropped_rather_than_crashing_sklearn(self):
        scores = np.array([0.1, np.inf, 0.9, 0.3])
        gt = np.array([0, 1, 1, 0])
        r = compute(scores, gt)
        self.assertEqual(r["filtering"]["dropped_non_finite"], 1)
        self.assertIsNotNone(r["auc_roc"])

    def test_filtering_is_reported_not_silent(self):
        scores = np.array([0.1, -1.0, 0.9, -2.0])
        gt = np.array([0, 1, 1, 0])
        r = compute(scores, gt)
        self.assertEqual(r["filtering"]["total"], 4)
        self.assertEqual(r["filtering"]["kept"], 2)


class AtKMetrics(unittest.TestCase):
    def test_k_defaults_to_positive_count_and_is_reported(self):
        gt = np.array([1, 1, 0, 0, 0])
        r = compute(np.array([0.9, 0.8, 0.3, 0.2, 0.1]), gt)
        self.assertEqual(r["k"], 2)

    def test_perfect_ranking_scores_one(self):
        gt = np.array([1, 1, 0, 0, 0])
        r = compute(np.array([0.9, 0.8, 0.3, 0.2, 0.1]), gt)
        self.assertEqual(r["precision_at_k"], 1.0)
        self.assertEqual(r["recall_at_k"], 1.0)
        self.assertEqual(r["auc_roc"], 1.0)

    def test_explicit_k_makes_precision_and_recall_differ(self):
        """At the default k they coincide by construction; at any other k they must not."""
        gt = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

        r = compute(scores, gt, k=2)
        # top-2 are both positive: P = 2/2, R = 2/4
        self.assertEqual(r["k"], 2)
        self.assertEqual(r["precision_at_k"], 1.0)
        self.assertEqual(r["recall_at_k"], 0.5)
        self.assertAlmostEqual(r["f1_at_k"], 2 * 1.0 * 0.5 / 1.5, places=4)

    def test_k_larger_than_positives(self):
        gt = np.array([1, 0, 0, 0])
        r = compute(np.array([0.9, 0.8, 0.7, 0.6]), gt, k=2)
        # top-2 contains the single positive: P = 1/2, R = 1/1
        self.assertEqual(r["precision_at_k"], 0.5)
        self.assertEqual(r["recall_at_k"], 1.0)

    def test_k_is_clamped_to_sample_count(self):
        gt = np.array([1, 0, 1])
        r = compute(np.array([0.9, 0.1, 0.8]), gt, k=99)
        self.assertEqual(r["k"], 3)


class TieBreaking(unittest.TestCase):
    """Ties must not manufacture a signal out of array order."""

    def test_constant_scores_give_exactly_the_base_rate(self):
        """Regression: argsort's positional tie-break picked the last k rows,
        so a degenerate all-zero detector appeared to beat its base rate.

        `methods/anograph/train_graflag.py` really emits zeros when min == max.
        """
        gt = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])   # base rate 0.2
        r = compute(np.zeros(10), gt)
        self.assertEqual(r["precision_at_k"], 0.2)
        self.assertEqual(r["recall_at_k"], 0.2)

    def test_constant_scores_are_order_independent(self):
        gt = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        a = compute(np.zeros(10), gt)["precision_at_k"]
        b = compute(np.zeros(10), gt[::-1])["precision_at_k"]
        self.assertEqual(a, b)

    def test_partial_tie_at_the_boundary_is_averaged(self):
        # top score is unambiguous and positive; the next three tie for one slot,
        # and exactly one of those three is positive => E[TP] = 1 + 1*(1/3)
        gt = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.5, 0.5, 0.5])
        r = compute(scores, gt, k=2)
        self.assertAlmostEqual(r["precision_at_k"], (1 + 1 / 3) / 2, places=4)


class AucEstimators(unittest.TestCase):
    def test_auc_pr_uses_average_precision(self):
        """auc(recall, precision) is not a valid PR estimator; AP is."""
        from sklearn import metrics as M
        rng = np.random.default_rng(7)
        gt = (rng.random(300) < 0.1).astype(int)
        scores = rng.random(300) + gt * 0.3

        r = compute(scores, gt)
        self.assertAlmostEqual(
            r["auc_pr"], round(float(M.average_precision_score(gt, scores)), 4), places=4
        )

    def test_auc_roc_matches_sklearn(self):
        from sklearn import metrics as M
        rng = np.random.default_rng(3)
        gt = (rng.random(200) < 0.2).astype(int)
        scores = rng.random(200)
        r = compute(scores, gt)
        self.assertAlmostEqual(
            r["auc_roc"], round(float(M.roc_auc_score(gt, scores)), 4), places=4
        )


class BestF1Threshold(unittest.TestCase):
    def test_threshold_of_zero_is_not_reported_as_null(self):
        """Regression: `if best_f1_threshold` treated 0.0 as absent.

        Min-max normalised scores always have a minimum of exactly 0.0.
        """
        gt = np.array([1, 1, 0])
        scores = np.array([0.0, 0.0, 0.0])
        r = compute(scores, gt)
        self.assertIsNotNone(r["best_f1"])
        self.assertEqual(r["best_f1_threshold"], 0.0)


class DegenerateInputs(unittest.TestCase):
    """The early returns must keep one stable schema."""

    EXPECTED_KEYS = {
        "auc_roc", "auc_pr", "precision_at_k", "recall_at_k", "f1_at_k", "k",
        "best_f1", "best_f1_threshold", "num_anomalies", "num_samples",
        "anomaly_ratio", "filtering",
    }

    def test_single_class_returns_the_full_key_set(self):
        """Regression: it returned 2 keys where success returns 11, and omitted
        exactly the counts needed to diagnose the condition."""
        r = compute(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]))
        self.assertEqual(set(r), self.EXPECTED_KEYS)
        self.assertIsNone(r["auc_roc"])
        self.assertEqual(r["num_samples"], 3)
        self.assertEqual(r["num_anomalies"], 0)

    def test_all_samples_filtered_out_is_distinguishable_from_single_class(self):
        r = compute(np.array([-1.0, -2.0]), np.array([0, 1]))
        self.assertEqual(set(r), self.EXPECTED_KEYS)
        self.assertEqual(r["num_samples"], 0)
        self.assertEqual(r["filtering"]["kept"], 0)

    def test_success_path_has_the_same_keys(self):
        r = compute(np.array([0.9, 0.1]), np.array([1, 0]))
        self.assertEqual(set(r), self.EXPECTED_KEYS)


class Flattening(unittest.TestCase):
    """The nesting decision must be made per array, not inferred from scores."""

    def test_ragged_scores_and_ragged_ground_truth(self):
        scores = [[0.9, 0.1], [0.8]]
        gt = [[1, 0], [1]]
        r = compute(np.asarray(scores, dtype=object), np.asarray(gt, dtype=object))
        self.assertEqual(r["num_samples"], 3)

    def test_rectangular_scores_with_ragged_ground_truth(self):
        """Regression: the branch was chosen from scores and applied to gt, so
        a rectangular/ragged pair hit `.flatten()` on an object array and raised
        an opaque IndexError."""
        scores = np.array([[0.9, 0.1], [0.8, 0.2]])
        gt = [[1, 0], [1, 0]]
        r = compute(scores, gt)
        self.assertEqual(r["num_samples"], 4)

    def test_length_mismatch_raises_a_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            prepare_pairs(np.array([0.1, 0.2, 0.3]), np.array([1, 0]))
        self.assertIn("different lengths", str(ctx.exception))

    def test_empty_input_does_not_raise(self):
        r = compute(np.array([]), np.array([]))
        self.assertEqual(r["num_samples"], 0)


class PlotsAgreeWithMetrics(unittest.TestCase):
    def test_both_use_the_same_prepared_sample(self):
        """Regression: plots filtered `> -2` while metrics conditionally used
        [0,1], so the PNG legend and evaluation.json could disagree."""
        from graflag_evaluator import plots
        import inspect

        src = inspect.getsource(plots)
        self.assertNotIn("valid_mask = scores_flat > -2", src)
        self.assertIn("prepare_pairs", src)

    def test_pr_plot_uses_average_precision(self):
        from graflag_evaluator import plots
        import inspect

        src = inspect.getsource(plots.PlotGenerator.plot_pr_curve)
        self.assertIn("average_precision_score", src)
        self.assertNotIn("metrics.auc(recall, precision)", src)


if __name__ == "__main__":
    unittest.main()
