"""Tests for graflag_data.downloader using a local HTTP fixture server."""

from __future__ import annotations

import gzip
import io
import json
import shutil
import tarfile
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from graflag_data import (
    DatasetNotReadyError,
    fetch,
    fetch_all,
    is_ready,
    load_metadata,
    missing_files,
)


# --------------------------------------------------------------------------- #
# Fixture HTTP server
# --------------------------------------------------------------------------- #

FIXTURES: dict[str, bytes] = {}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = FIXTURES.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kw):  # silence
        return


def _make_fixtures() -> None:
    FIXTURES.clear()
    FIXTURES["/raw.csv"] = b"a,b,c\n1,2,3\n"

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(b"hello gunzip")
    FIXTURES["/raw.csv.gz"] = buf.getvalue()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("only.edges", b"1 2 100\n3 4 200\n")
    FIXTURES["/single.zip"] = buf.getvalue()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", b"ignore me")
        zf.writestr("inner.csv", b"u,i\n0,1\n")
    FIXTURES["/multi.zip"] = buf.getvalue()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        data = b"src dst ts\n1 2 0\n"
        info = tarfile.TarInfo("konect/out.net")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    FIXTURES["/konect.tar.bz2"] = buf.getvalue()


class _ServerThread:
    def __init__(self):
        _make_fixtures()
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_dataset(root: Path, name: str, meta: dict) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="graflag-data-test-"))
        self.srv = _ServerThread()
        self.base = self.srv.__enter__()

    def tearDown(self):
        self.srv.__exit__(None, None, None)
        shutil.rmtree(self.tmp)

    # ---- raw download -------------------------------------------------------
    def test_raw_download(self):
        d = _write_dataset(self.tmp, "raw_ds", {
            "name": "raw_ds",
            "files": [{"name": "data.csv", "url": f"{self.base}/raw.csv"}],
        })
        self.assertFalse(is_ready(d))
        self.assertEqual(fetch(d, verbose=False), ["data.csv"])
        self.assertEqual((d / "data.csv").read_bytes(), b"a,b,c\n1,2,3\n")
        self.assertTrue(is_ready(d))

    # ---- gunzip -------------------------------------------------------------
    def test_gunzip_extract(self):
        d = _write_dataset(self.tmp, "gz_ds", {
            "name": "gz_ds",
            "files": [{
                "name": "data.csv",
                "url": f"{self.base}/raw.csv.gz",
                "extract": "gunzip",
            }],
        })
        fetch(d, verbose=False)
        self.assertEqual((d / "data.csv").read_bytes(), b"hello gunzip")

    # ---- zip auto-pick the single regular file -----------------------------
    def test_zip_single_member(self):
        d = _write_dataset(self.tmp, "zip_single", {
            "name": "zip_single",
            "files": [{
                "name": "only.edges",
                "url": f"{self.base}/single.zip",
                "extract": "zip",
            }],
        })
        fetch(d, verbose=False)
        self.assertIn(b"1 2 100", (d / "only.edges").read_bytes())

    # ---- zip with members selector -----------------------------------------
    def test_zip_members_selector(self):
        d = _write_dataset(self.tmp, "zip_multi", {
            "name": "zip_multi",
            "files": [{
                "name": "edges.csv",
                "url": f"{self.base}/multi.zip",
                "extract": "zip",
                "members": ["inner.csv"],
            }],
        })
        fetch(d, verbose=False)
        self.assertEqual((d / "edges.csv").read_bytes(), b"u,i\n0,1\n")

    # ---- tar.bz2 with nested member ----------------------------------------
    def test_tar_bz2(self):
        d = _write_dataset(self.tmp, "tar_ds", {
            "name": "tar_ds",
            "files": [{
                "name": "net",
                "url": f"{self.base}/konect.tar.bz2",
                "extract": "tar.bz2",
                "members": ["konect/out.net"],
            }],
        })
        fetch(d, verbose=False)
        self.assertIn(b"src dst ts", (d / "net").read_bytes())

    # ---- already-present skip ----------------------------------------------
    def test_already_present_skip(self):
        d = _write_dataset(self.tmp, "skip_ds", {
            "name": "skip_ds",
            "files": [{"name": "data.csv", "url": f"{self.base}/raw.csv"}],
        })
        (d / "data.csv").write_bytes(b"sentinel")
        self.assertEqual(fetch(d, verbose=False), [])
        self.assertEqual((d / "data.csv").read_bytes(), b"sentinel")

    def test_force_redownload(self):
        d = _write_dataset(self.tmp, "force_ds", {
            "name": "force_ds",
            "files": [{"name": "data.csv", "url": f"{self.base}/raw.csv"}],
        })
        (d / "data.csv").write_bytes(b"sentinel")
        self.assertEqual(fetch(d, force=True, verbose=False), ["data.csv"])
        self.assertEqual((d / "data.csv").read_bytes(), b"a,b,c\n1,2,3\n")

    # ---- derived dataset ---------------------------------------------------
    def test_derived_noop(self):
        d = _write_dataset(self.tmp, "derived_ds", {
            "name": "derived_ds",
            "derived": True,
            "derived_from": "raw_ds",
            "files": [],
        })
        self.assertTrue(is_ready(d))  # derived is always "ready"
        self.assertEqual(fetch(d, verbose=False), [])

    # ---- missing URL raises ------------------------------------------------
    def test_missing_url_raises(self):
        d = _write_dataset(self.tmp, "broken", {
            "name": "broken",
            "files": [{"name": "nowhere.bin"}],
        })
        with self.assertRaises(DatasetNotReadyError):
            fetch(d, verbose=False)

    # ---- fetch_all aggregates ----------------------------------------------
    def test_fetch_all(self):
        _write_dataset(self.tmp, "a", {
            "name": "a",
            "files": [{"name": "data.csv", "url": f"{self.base}/raw.csv"}],
        })
        _write_dataset(self.tmp, "b", {
            "name": "b",
            "derived": True,
            "files": [],
        })
        report = fetch_all(self.tmp, verbose=False)
        self.assertEqual(set(report), {"a", "b"})
        self.assertEqual(report["a"], ["data.csv"])
        self.assertEqual(report["b"], [])

    def test_missing_files_helper(self):
        d = _write_dataset(self.tmp, "partial", {
            "name": "partial",
            "files": [
                {"name": "a.csv", "url": f"{self.base}/raw.csv"},
                {"name": "b.csv", "url": f"{self.base}/raw.csv"},
            ],
        })
        (d / "a.csv").write_bytes(b"x")
        miss = missing_files(d)
        self.assertEqual([f.name for f in miss], ["b.csv"])

    # ---- build step: derived-with-conversion-script -----------------------
    def test_build_step_runs_after_upstream_fetch(self):
        """A derived dataset with `build` fetches derived_from, then runs the
        build command to produce the listed outputs."""
        _write_dataset(self.tmp, "base_ds", {
            "name": "base_ds",
            "files": [{"name": "raw.csv", "url": f"{self.base}/raw.csv"}],
        })
        _write_dataset(self.tmp, "derived_ds", {
            "name": "derived_ds",
            "derived": True,
            "derived_from": "base_ds",
            "files": [],
            "build": {
                "command": [
                    "python3", "-c",
                    "import shutil, sys; shutil.copy(sys.argv[1], sys.argv[2])",
                    "../base_ds/raw.csv",
                    "out.csv",
                ],
                "produces": ["out.csv"],
            },
        })
        derived_dir = self.tmp / "derived_ds"
        self.assertFalse(is_ready(derived_dir))
        produced = fetch(derived_dir, verbose=False)
        # base_ds/raw.csv downloaded, then out.csv produced
        self.assertIn("base_ds/raw.csv", produced)
        self.assertIn("out.csv", produced)
        self.assertTrue((derived_dir / "out.csv").exists())
        self.assertTrue(is_ready(derived_dir))

    def test_build_step_skipped_if_outputs_present(self):
        _write_dataset(self.tmp, "base2", {
            "name": "base2",
            "files": [{"name": "raw.csv", "url": f"{self.base}/raw.csv"}],
        })
        d = _write_dataset(self.tmp, "derived2", {
            "name": "derived2",
            "derived": True,
            "derived_from": "base2",
            "files": [],
            "build": {
                "command": ["false"],  # would fail if actually run
                "produces": ["out.csv"],
            },
        })
        (d / "out.csv").write_bytes(b"pre-existing")
        (self.tmp / "base2" / "raw.csv").write_bytes(b"already here")
        produced = fetch(d, verbose=False)
        self.assertEqual(produced, [])
        self.assertEqual((d / "out.csv").read_bytes(), b"pre-existing")

    def test_build_step_failure_raises(self):
        _write_dataset(self.tmp, "base3", {
            "name": "base3",
            "files": [],
        })
        d = _write_dataset(self.tmp, "derived3", {
            "name": "derived3",
            "derived": True,
            "derived_from": "base3",
            "files": [],
            "build": {
                "command": ["python3", "-c", "import sys; sys.exit(2)"],
                "produces": ["out.csv"],
            },
        })
        with self.assertRaises(DatasetNotReadyError) as ctx:
            fetch(d, verbose=False)
        self.assertIn("build command failed", str(ctx.exception))

    def test_build_runs_after_direct_downloads(self):
        """Pattern used by email_snapshot: download a handful of files then
        run a build step that produces an additional file from them."""
        d = _write_dataset(self.tmp, "post_build", {
            "name": "post_build",
            "files": [
                {"name": "a.bin", "url": f"{self.base}/raw.csv"},
                {"name": "b.bin", "url": f"{self.base}/raw.csv"},
            ],
            "build": {
                "command": ["python3", "-c", "import shutil; shutil.copy('a.bin', 'split.bin')"],
                "produces": ["split.bin"],
            },
        })
        produced = fetch(d, verbose=False)
        self.assertEqual(set(produced), {"a.bin", "b.bin", "split.bin"})
        self.assertTrue((d / "split.bin").exists())
        self.assertEqual((d / "split.bin").read_bytes(), (d / "a.bin").read_bytes())

    def test_build_step_missing_produces_raises(self):
        d = _write_dataset(self.tmp, "derived4", {
            "name": "derived4",
            "derived": True,
            "files": [],
            "build": {
                "command": ["python3", "-c", "print('did nothing')"],
                "produces": ["never_created.csv"],
            },
        })
        with self.assertRaises(DatasetNotReadyError) as ctx:
            fetch(d, verbose=False)
        self.assertIn("did not produce expected files", str(ctx.exception))

    def test_load_metadata_reads_fields(self):
        d = _write_dataset(self.tmp, "meta_ds", {
            "name": "meta_ds",
            "description": "desc",
            "source": "https://example.com",
            "compatible_methods": ["x", "y"],
            "files": [],
        })
        m = load_metadata(d)
        self.assertEqual(m.name, "meta_ds")
        self.assertEqual(m.compatible_methods, ["x", "y"])
        self.assertEqual(m.source, "https://example.com")


class ConvertToStrgnnIntegrationTest(unittest.TestCase):
    """Exercise the real convert_to_strgnn.py script as used by the
    btc_alpha_snapshot / btc_otc_snapshot / uci_snapshot datasets."""

    def setUp(self):
        try:
            import numpy  # noqa: F401
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("numpy/scipy not installed")

        self.tmp = Path(tempfile.mkdtemp(prefix="graflag-convert-"))
        # Mirror the real layout: datasets/<name>/ and datasets/convert_to_strgnn.py
        real_script = (
            Path(__file__).resolve().parents[4]
            / "graflag-shared" / "datasets" / "convert_to_strgnn.py"
        )
        shutil.copy(real_script, self.tmp / "convert_to_strgnn.py")

        base = self.tmp / "btc_alpha"
        base.mkdir()
        # Minimal Bitcoin-Alpha-format CSV covering 2 snapshots of data.
        rows = []
        for i in range(40):
            rows.append(f"{i},{(i + 1) % 40},5,{1000 + i}\n")
            rows.append(f"{(i + 2) % 40},{i},3,{2000 + i}\n")
        (base / "soc-sign-bitcoinalpha.csv").write_text("".join(rows))
        (base / "metadata.json").write_text(json.dumps({
            "name": "btc_alpha",
            "files": [],
        }))

        # Snapshot dataset metadata matches production snapshot metadata.
        snap = self.tmp / "btc_alpha_snapshot"
        snap.mkdir()
        (snap / "metadata.json").write_text(json.dumps({
            "name": "btc_alpha_snapshot",
            "derived": True,
            "derived_from": "btc_alpha",
            "files": [],
            "build": {
                "command": [
                    "python3", "../convert_to_strgnn.py",
                    "../btc_alpha/soc-sign-bitcoinalpha.csv", ".",
                    "--format", "bitcoin",
                    "--snapshots", "4",
                    "--window", "1",
                ],
                "produces": ["acc_graph.npy", "sta_graph.npy", "split.npz"],
            },
        }))
        self.snap = snap

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_snapshot_produced_from_base_csv(self):
        produced = fetch(self.snap, verbose=False)
        self.assertIn("acc_graph.npy", produced)
        for name in ("acc_graph.npy", "sta_graph.npy", "split.npz"):
            self.assertTrue((self.snap / name).exists(), name)
        self.assertTrue(is_ready(self.snap))


if __name__ == "__main__":
    unittest.main(verbosity=2)
