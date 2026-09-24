"""Tests for plugin loading and spot-file discovery."""

import shutil
import tempfile
import unittest
from pathlib import Path

from graflag_evaluator.evaluator import _is_spot_csv
from graflag_evaluator.metrics import MetricCalculator


class PluginRegistryIsIdempotent(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._saved = {k: list(v) for k, v in MetricCalculator._METRIC_REGISTRY.items()}
        self._saved_loaded = set(MetricCalculator._LOADED_PLUGINS)
        self.addCleanup(self._restore)

    def _restore(self):
        MetricCalculator._METRIC_REGISTRY.clear()
        MetricCalculator._METRIC_REGISTRY.update(self._saved)
        MetricCalculator._LOADED_PLUGINS.clear()
        MetricCalculator._LOADED_PLUGINS.update(self._saved_loaded)

    def _write_plugin(self, directory, name, metric):
        d = self.tmp / directory
        d.mkdir(exist_ok=True)
        (d / f"{name}.py").write_text(
            "from graflag_evaluator import MetricCalculator\n"
            f"def {metric}(scores, ground_truth, **kw):\n"
            f"    return {{'{metric}': 1.0}}\n"
            f"MetricCalculator.register_metric('T', {metric})\n"
        )
        return d

    def _count(self):
        return len(MetricCalculator._METRIC_REGISTRY.get("T", []))

    def test_loading_twice_registers_once(self):
        """Regression: load_plugins runs from Evaluator.__init__, so
        evaluating N experiments in one process registered N copies of every
        plugin metric and ran each of them N times."""
        d = self._write_plugin("plugins", "p", "my_metric")

        MetricCalculator.load_plugins(d)
        after_first = self._count()
        for _ in range(4):
            MetricCalculator.load_plugins(d)

        self.assertEqual(after_first, 1)
        self.assertEqual(self._count(), 1)

    def test_same_filename_in_two_directories_both_load(self):
        """Regression: both got module name graflag_plugin_<stem>."""
        a = self._write_plugin("plugins", "shared_name", "metric_a")
        b = self._write_plugin("custom_metrics", "shared_name", "metric_b")

        MetricCalculator.load_plugins(a, b)

        names = {f.__name__ for f in MetricCalculator._METRIC_REGISTRY.get("T", [])}
        self.assertEqual(names, {"metric_a", "metric_b"})

    def test_direct_double_registration_is_ignored(self):
        def dup(scores, ground_truth, **kw):
            return {"dup": 1.0}

        MetricCalculator.register_metric("T", dup)
        MetricCalculator.register_metric("T", dup)
        self.assertEqual(self._count(), 1)

    def test_a_broken_plugin_does_not_abort_the_others(self):
        d = self._write_plugin("plugins", "good", "good_metric")
        (d / "bad.py").write_text("raise RuntimeError('boom')\n")

        MetricCalculator.load_plugins(d)
        names = {f.__name__ for f in MetricCalculator._METRIC_REGISTRY.get("T", [])}
        self.assertIn("good_metric", names)


class SpotFileDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _csv(self, name, text):
        p = self.tmp / name
        p.write_text(text)
        return p

    def test_spot_csv_is_recognised(self):
        p = self._csv("training.csv", "timestamp,epoch,loss\n1,1,0.5\n")
        self.assertTrue(_is_spot_csv(p))

    def test_unrelated_csv_is_rejected(self):
        """Regression: every *.csv in $EXP was treated as a spot file, so a
        method's own metrics.csv had its string columns plotted as series."""
        p = self._csv("metrics.csv", "method,dataset,f1_score\nx,y,0.5\n")
        self.assertFalse(_is_spot_csv(p))

    def test_per_node_scores_csv_is_rejected(self):
        p = self._csv("anomaly_scores.csv", "node_id,score\n0,0.1\n")
        self.assertFalse(_is_spot_csv(p))

    def test_empty_file_is_rejected(self):
        self.assertFalse(_is_spot_csv(self._csv("empty.csv", "")))

    def test_missing_file_is_rejected(self):
        self.assertFalse(_is_spot_csv(self.tmp / "nope.csv"))


if __name__ == "__main__":
    unittest.main()
