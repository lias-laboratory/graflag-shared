"""Tests that a run cannot report success without a usable result.

An experiment that fails silently is worse than one that fails loudly: the
number never arrives, but the dashboard says it did.
"""

import json
import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from graflag_runner.results import ResultWriter
from graflag_runner.runner import MethodRunner


class ExpDirFixture(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        self.exp = tempfile.mkdtemp()
        self.data = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.exp, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.data, ignore_errors=True)
        self._old_exp = os.environ.get("EXP")
        os.environ["EXP"] = self.exp
        self.addCleanup(self._restore_exp)

    def _restore_exp(self):
        if self._old_exp is None:
            os.environ.pop("EXP", None)
        else:
            os.environ["EXP"] = self._old_exp

    @property
    def results_path(self):
        return Path(self.exp) / "results.json"

    def status(self):
        path = Path(self.exp) / "status.json"
        return json.loads(path.read_text())["status"] if path.exists() else None


class AtomicResultWrites(ExpDirFixture):
    """finalize() must leave either a complete file or no file."""

    def test_numpy_values_serialize(self):
        """Regression: save_scores stores kwargs verbatim, so a numpy value
        raised mid-write. methods/taddy really passes a raw ndarray."""
        w = ResultWriter()
        w.save_scores(
            result_type="NODE_ANOMALY_SCORES",
            scores=[1.0, 2.0],
            ground_truth=[0, 1],
            timestamps=np.arange(2),
            peak=np.float32(3.5),
        )
        w.finalize()

        data = json.loads(self.results_path.read_text())
        self.assertEqual(data["timestamps"], [0, 1])
        self.assertAlmostEqual(data["peak"], 3.5, places=4)

    def test_unserializable_value_leaves_no_results_file(self):
        """Regression: open(...,'w') truncated first, so a failed write left a
        partial results.json that still counted as a result -- making a failed
        run report completed and crashing the evaluator later."""
        w = ResultWriter()
        w.save_scores(result_type="NODE_ANOMALY_SCORES", scores=[1.0],
                      ground_truth=[0], bad=object())

        with self.assertRaises(TypeError):
            w.finalize()

        self.assertFalse(self.results_path.exists())

    def test_failed_write_leaves_no_temp_file(self):
        w = ResultWriter()
        w.save_scores(result_type="NODE_ANOMALY_SCORES", scores=[1.0],
                      ground_truth=[0], bad=object())
        with self.assertRaises(TypeError):
            w.finalize()

        leftovers = [p.name for p in Path(self.exp).iterdir()]
        self.assertEqual([p for p in leftovers if p.endswith(".tmp")], [])

    def test_an_existing_good_file_survives_a_failed_rewrite(self):
        good = ResultWriter()
        good.save_scores(result_type="NODE_ANOMALY_SCORES", scores=[0.5],
                         ground_truth=[1])
        good.finalize()
        before = self.results_path.read_text()

        bad = ResultWriter()
        bad.save_scores(result_type="NODE_ANOMALY_SCORES", scores=[0.5],
                        ground_truth=[1], bad=object())
        with self.assertRaises(TypeError):
            bad.finalize()

        self.assertEqual(self.results_path.read_text(), before)

    def test_streamable_outside_scores_is_still_streamed(self):
        """Regression: the dispatch only inspected `scores`, so a
        StreamableArray elsewhere reached json.dump and truncated the file."""
        from graflag_runner.streaming import StreamableArray

        w = ResultWriter()
        w.save_scores(
            result_type="NODE_ANOMALY_SCORES",
            scores=[0.1, 0.9],
            ground_truth=StreamableArray(iter([0, 1])),
        )
        w.finalize()
        self.assertTrue(self.results_path.exists())
        json.loads(self.results_path.read_text())  # must parse


class ExitCodeIsNotProofOfSuccess(ExpDirFixture):
    def _run(self, script):
        src = Path(self.data) / "method.py"
        src.write_text(script)
        runner = MethodRunner(
            data_dir=self.data, exp_dir=self.exp, method_name="t",
            command=f"python3 {src}", monitor_interval=0.05,
        )
        try:
            runner.run()
            return None
        except Exception as exc:
            return exc

    def test_method_exiting_zero_without_results_is_failed(self):
        """Regression: exit code 0 was recorded as completed regardless, so a
        method that swallowed its own exception produced an empty experiment
        the dashboard called successful."""
        err = self._run("import sys; sys.exit(0)")

        self.assertIsNotNone(err)
        self.assertEqual(self.status(), "failed")

    def test_method_writing_unparseable_results_is_failed(self):
        err = self._run(
            "import os\n"
            "open(os.path.join(os.environ['EXP'], 'results.json'), 'w')"
            ".write('{truncated')\n"
        )
        self.assertIsNotNone(err)
        self.assertEqual(self.status(), "failed")

    def test_results_without_scores_is_failed(self):
        err = self._run(
            "import os, json\n"
            "json.dump({'result_type': 'NODE_ANOMALY_SCORES'},"
            " open(os.path.join(os.environ['EXP'], 'results.json'), 'w'))\n"
        )
        self.assertIsNotNone(err)
        self.assertEqual(self.status(), "failed")

    def test_well_behaved_method_still_completes(self):
        err = self._run(
            "import os, json\n"
            "json.dump({'result_type': 'NODE_ANOMALY_SCORES',"
            " 'scores': [0.1, 0.9], 'ground_truth': [0, 1], 'metadata': {}},"
            " open(os.path.join(os.environ['EXP'], 'results.json'), 'w'))\n"
        )
        self.assertIsNone(err)
        self.assertEqual(self.status(), "completed")

    def test_nonzero_exit_is_still_failed(self):
        err = self._run("import sys; sys.exit(3)")
        self.assertIsNotNone(err)
        self.assertEqual(self.status(), "failed")


class RuntimeMetadataIsRecorded(ExpDirFixture):
    """RESULTS_STANDARD.md promises the runner sets resource metrics itself."""

    def _run(self, script):
        src = Path(self.data) / "method.py"
        src.write_text(script)
        runner = MethodRunner(
            data_dir=self.data, exp_dir=self.exp, method_name="t",
            command=f"python3 {src}", monitor_interval=0.05,
        )
        runner.run()
        return json.loads(self.results_path.read_text())["metadata"]

    WRITES_RESULTS = (
        "import os, json\n"
        "json.dump({'result_type': 'NODE_ANOMALY_SCORES', 'scores': [0.1, 0.9],"
        " 'ground_truth': [0, 1], 'metadata': %s},"
        " open(os.path.join(os.environ['EXP'], 'results.json'), 'w'))\n"
    )

    def test_exec_time_and_peak_memory_are_added(self):
        """Regression: the runner measured these and wrote them only to
        status.json, so results.json carried whatever the method recorded --
        usually nothing -- despite the documented promise."""
        meta = self._run(self.WRITES_RESULTS % "{'method_name': 'm'}")

        self.assertIn("exec_time_ms", meta)
        self.assertGreaterEqual(meta["exec_time_ms"], 0)
        self.assertIn("peak_memory_mb", meta)
        self.assertEqual(meta["method_name"], "m")

    def test_the_monitors_numbers_win_over_the_methods(self):
        """The monitor samples the whole process tree continuously; methods
        that track their own take a few point samples of their own process.
        On a real dynwalk run that was 907 MB vs 409 MB, so the method's
        figure must not be the one a benchmark table reads."""
        meta = self._run(
            self.WRITES_RESULTS % "{'exec_time_ms': 1.0, 'peak_memory_mb': 2.0}"
        )
        self.assertNotEqual(meta["exec_time_ms"], 1.0)
        self.assertNotEqual(meta["peak_memory_mb"], 2.0)

    def test_what_the_method_reported_is_preserved(self):
        meta = self._run(
            self.WRITES_RESULTS % "{'exec_time_ms': 1.0, 'peak_memory_mb': 2.0}"
        )
        self.assertEqual(meta["method_reported_exec_time_ms"], 1.0)
        self.assertEqual(meta["method_reported_peak_memory_mb"], 2.0)

    def test_no_shadow_key_when_the_method_reported_nothing(self):
        meta = self._run(self.WRITES_RESULTS % "{'method_name': 'm'}")
        self.assertNotIn("method_reported_peak_memory_mb", meta)

    def test_results_json_stays_valid(self):
        self._run(self.WRITES_RESULTS % "{}")
        json.loads(self.results_path.read_text())   # must still parse
        leftovers = [p.name for p in Path(self.exp).iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])


class ValidJsonOutput(ExpDirFixture):
    """NaN/Infinity are readable by Python but invalid per RFC 8259."""

    @staticmethod
    def _strict(raw):
        def boom(c):
            raise ValueError(f"non-RFC constant: {c}")
        return json.loads(raw, parse_constant=boom)

    def test_non_finite_scores_become_null(self):
        w = ResultWriter()
        w.save_scores(result_type="NODE_ANOMALY_SCORES",
                      scores=[0.1, float("nan"), float("inf"), 0.4],
                      ground_truth=[0, 1, 0, 1])
        w.finalize()

        raw = self.results_path.read_text()
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)
        self.assertEqual(self._strict(raw)["scores"], [0.1, None, None, 0.4])

    def test_streaming_path_is_also_valid(self):
        from graflag_runner.streaming import StreamableArray

        w = ResultWriter()
        w.save_scores(result_type="NODE_ANOMALY_SCORES",
                      scores=StreamableArray(iter([0.1, float("nan"), np.float32(0.7)])),
                      ground_truth=[0, 1, 0])
        w.finalize()
        self._strict(self.results_path.read_text())   # must not raise

    def test_finite_values_are_untouched(self):
        w = ResultWriter()
        w.save_scores(result_type="NODE_ANOMALY_SCORES", scores=[0.25, 0.75],
                      ground_truth=[0, 1])
        w.finalize()
        self.assertEqual(self._strict(self.results_path.read_text())["scores"],
                         [0.25, 0.75])


class SpotFilesAreNotClobbered(ExpDirFixture):
    """The schema lock is per-instance; the file on disk is the real state."""

    def test_second_writer_appends_instead_of_truncating(self):
        """Regression: a fresh ResultWriter opened the existing CSV with 'w'
        and destroyed every row a previous one had written."""
        a = ResultWriter()
        a.spot("training", epoch=1, loss=0.9)
        a.spot("training", epoch=2, loss=0.5)

        ResultWriter().spot("training", epoch=3, loss=0.3)

        rows = (Path(self.exp) / "training.csv").read_text().strip().splitlines()
        self.assertEqual(len(rows), 4)          # header + 3
        self.assertIn(",3,0.3", rows[-1])

    def test_mismatched_schema_refuses_to_overwrite(self):
        ResultWriter().spot("training", epoch=1, loss=0.9)
        with self.assertRaises(ValueError):
            ResultWriter().spot("training", epoch=2, accuracy=0.5)
        # the original file is still intact
        rows = (Path(self.exp) / "training.csv").read_text().strip().splitlines()
        self.assertEqual(rows[0], "timestamp,epoch,loss")

    def test_same_instance_schema_lock_still_applies(self):
        w = ResultWriter()
        w.spot("training", epoch=1, loss=0.9)
        with self.assertRaises(ValueError):
            w.spot("training", epoch=2, accuracy=0.1)


class MonitorIntervalValidation(unittest.TestCase):
    def test_bad_values_fall_back_or_clamp(self):
        from graflag_runner.runner import _parse_monitor_interval as parse

        self.assertEqual(parse(None), 1.0)
        self.assertEqual(parse(""), 1.0)
        self.assertEqual(parse("not-a-number"), 1.0)
        self.assertEqual(parse("nan"), 1.0)
        self.assertEqual(parse("inf"), 1.0)
        self.assertEqual(parse("0"), 0.05)      # would have been a busy loop
        self.assertEqual(parse("-5"), 0.05)     # would have killed the thread
        self.assertEqual(parse("30"), 2.0)      # would outlive the join timeout
        self.assertEqual(parse("0.5"), 0.5)     # valid values pass through


class StreamableArrayProtocol(unittest.TestCase):
    def test_iter_returns_a_real_iterator(self):
        """Regression: __iter__ returned the generator attribute itself, so a
        list raised 'iter() returned non-iterator' -- at write time, after the
        output file had been truncated."""
        from graflag_runner.streaming import StreamableArray

        self.assertEqual(list(StreamableArray([1, 2, 3])), [1, 2, 3])

    def test_second_pass_raises_instead_of_yielding_nothing(self):
        """A silently empty second pass wrote "scores": [] -- valid JSON that
        nothing downstream flags."""
        from graflag_runner.streaming import StreamableArray

        sa = StreamableArray(iter([1, 2, 3]))
        self.assertEqual(list(sa), [1, 2, 3])
        with self.assertRaises(RuntimeError):
            list(sa)


class LoggingIsNotHijacked(unittest.TestCase):
    def test_importing_the_package_keeps_existing_handlers(self):
        """Regression: logging_utils called basicConfig(force=True) at import,
        which removes and closes every root handler a method had installed."""
        import importlib, logging as pylogging

        root = pylogging.getLogger()
        sentinel = pylogging.NullHandler()
        root.addHandler(sentinel)
        try:
            import graflag_runner.logging_utils as lu
            importlib.reload(lu)
            self.assertIn(sentinel, pylogging.getLogger().handlers)
        finally:
            root.removeHandler(sentinel)


class FailuresAreAlwaysRecorded(ExpDirFixture):
    def test_a_crash_overwrites_a_stale_completed_status(self):
        """Regression: the handler only wrote when the existing status was
        "running", so if _save_status("running") had failed earlier (it only
        warns) a previous attempt's "completed" survived the crash."""
        (Path(self.exp) / "status.json").write_text(
            json.dumps({"status": "completed", "exit_code": 0})
        )

        src = Path(self.data) / "m.py"
        src.write_text("import sys; sys.exit(9)")
        runner = MethodRunner(data_dir=self.data, exp_dir=self.exp,
                              method_name="t", command=f"python3 {src}",
                              monitor_interval=0.05)
        with self.assertRaises(Exception):
            runner.run()

        self.assertEqual(self.status(), "failed")

    def test_startup_failure_leaves_a_diagnosis(self):
        """Regression: from_env() and the constructor run before run() writes
        anything, so an early failure left $EXP completely empty -- and the
        container log is gone once the Swarm task is reaped."""
        import subprocess, sys as _sys

        env = dict(os.environ, EXP=self.exp, DATA=self.data, METHOD_NAME="probe",
                   COMMAND="python3 -c pass",
                   SUPPORTED_DATASETS="a_different_dataset",
                   PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        proc = subprocess.run([_sys.executable, "-m", "graflag_runner"],
                              env=env, capture_output=True, text=True)

        self.assertEqual(proc.returncode, 1)
        status_file = Path(self.exp) / "status.json"
        self.assertTrue(status_file.is_file())
        payload = json.loads(status_file.read_text())
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["stage"], "startup")
        self.assertIn("error", payload)


class ResultWriterConstruction(unittest.TestCase):
    def test_output_dir_argument_is_accepted(self):
        """The docstring documented output_dir while the signature took none."""
        import tempfile as tf
        d = tf.mkdtemp()
        w = ResultWriter(output_dir=d)
        self.assertEqual(str(w.output_dir), d)

    def test_unset_exp_gives_a_clear_error(self):
        """Regression: Path(None) raised 'argument should be a str or an
        os.PathLike object, not NoneType'."""
        import os as _os
        from unittest import mock as _mock

        with _mock.patch.dict(_os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                ResultWriter()
        self.assertIn("EXP", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
