"""Tests that GraFlag.run() invokes graflag_data on the manager via SSH."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


GRAFLAG_SRC = Path(__file__).resolve().parents[4] / "graflag" / "graflag"


def _load_core():
    """Import graflag.core without needing graflag on sys.path.

    We synthesize a minimal 'graflag' package pointing at the upstream source
    tree and load only what core.py actually needs from it.
    """
    if "graflag.core" in sys.modules:
        return sys.modules["graflag.core"]

    if "graflag" not in sys.modules:
        pkg = types.ModuleType("graflag")
        pkg.__path__ = [str(GRAFLAG_SRC)]
        sys.modules["graflag"] = pkg

    for name in ("config", "ssh", "docker_ops", "utils", "models"):
        spec = importlib.util.spec_from_file_location(
            f"graflag.{name}", GRAFLAG_SRC / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"graflag.{name}"] = mod
        spec.loader.exec_module(mod)

    spec = importlib.util.spec_from_file_location(
        "graflag.core", GRAFLAG_SRC / "core.py"
    )
    core = importlib.util.module_from_spec(spec)
    sys.modules["graflag.core"] = core
    spec.loader.exec_module(core)
    return core


class EnsureDatasetTests(unittest.TestCase):
    def setUp(self):
        self.core = _load_core()
        # Build a GraFlag instance without running __init__ (avoids config file).
        self.gf = self.core.GraFlag.__new__(self.core.GraFlag)
        self.gf.config = mock.Mock()
        self.gf.config.remote_shared_dir = "/shared"
        self.gf.ssh = mock.Mock()

    def _ok(self):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    def _fail(self, stderr="boom"):
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)

    def test_skips_when_no_metadata(self):
        self.gf.ssh.path_exists.return_value = False
        self.gf._ensure_dataset("btc_alpha")
        self.gf.ssh.path_exists.assert_called_once_with(
            "/shared", "datasets/btc_alpha/metadata.json"
        )
        self.gf.ssh.execute.assert_not_called()

    def test_runs_fetch_when_metadata_present(self):
        self.gf.ssh.path_exists.return_value = True
        self.gf.ssh.execute.return_value = self._ok()
        self.gf._ensure_dataset("btc_alpha")

        self.gf.ssh.execute.assert_called_once()
        (cmd,), _ = self.gf.ssh.execute.call_args
        self.assertIn("PYTHONPATH=/shared/libs", cmd)
        self.assertIn("python3 -m graflag_data", cmd)
        self.assertIn("--root /shared/datasets", cmd)
        self.assertIn("fetch btc_alpha", cmd)

    def test_raises_on_fetch_failure(self):
        self.gf.ssh.path_exists.return_value = True
        self.gf.ssh.execute.return_value = self._fail("no such url")
        with self.assertRaises(self.core.GraFlagError) as ctx:
            self.gf._ensure_dataset("bad_ds")
        self.assertIn("bad_ds", str(ctx.exception))
        self.assertIn("no such url", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
