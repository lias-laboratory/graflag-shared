"""Tests for Google Drive support in graflag_data.downloader.

We mock the ``gdown`` module so the tests never hit the network and don't
require gdown to be installed.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from graflag_data import DatasetNotReadyError, fetch, is_ready
from graflag_data import downloader as dl


def _install_fake_gdown(folder_files: dict, file_writes: dict | None = None):
    """Install a fake ``gdown`` module in sys.modules.

    ``folder_files`` maps basename -> bytes to populate when a folder is
    downloaded. ``file_writes`` maps URL -> bytes to write when
    ``gdown.download`` is called.
    """
    fake = types.ModuleType("gdown")

    def download_folder(url, output, quiet=True, use_cookies=False):
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        for name, data in folder_files.items():
            (out / name).write_bytes(data)

    def download(url, output, quiet=True, fuzzy=True):
        data = (file_writes or {}).get(url, b"")
        Path(output).write_bytes(data)

    fake.download_folder = download_folder
    fake.download = download
    sys.modules["gdown"] = fake


class GdriveFolderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gdrive-test-"))
        # Reset folder cache between tests
        dl._GDRIVE_FOLDER_CACHE.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp)
        sys.modules.pop("gdown", None)

    def _write(self, name, meta):
        d = self.tmp / name
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps(meta))
        return d

    def test_folder_download_and_member_selection(self):
        _install_fake_gdown({
            "btc_alpha_0.5_0.01.csv": b"rate-0.01",
            "btc_alpha_0.5_0.05.csv": b"rate-0.05",
            "btc_alpha_0.5_0.1.csv":  b"rate-0.10",
            "unrelated.txt": b"ignore",
        })
        folder_url = "https://drive.google.com/drive/folders/ABC"
        d = self._write("generaldyg_btc_alpha", {
            "name": "generaldyg_btc_alpha",
            "derived": True,
            "files": [
                {"name": "btc_alpha_0.5_0.01.csv", "url": folder_url,
                 "members": ["btc_alpha_0.5_0.01.csv"]},
                {"name": "btc_alpha_0.5_0.05.csv", "url": folder_url,
                 "members": ["btc_alpha_0.5_0.05.csv"]},
                {"name": "btc_alpha_0.5_0.1.csv",  "url": folder_url,
                 "members": ["btc_alpha_0.5_0.1.csv"]},
            ],
        })
        produced = fetch(d, verbose=False)
        self.assertEqual(set(produced), {
            "btc_alpha_0.5_0.01.csv",
            "btc_alpha_0.5_0.05.csv",
            "btc_alpha_0.5_0.1.csv",
        })
        self.assertEqual((d / "btc_alpha_0.5_0.01.csv").read_bytes(), b"rate-0.01")
        self.assertEqual((d / "btc_alpha_0.5_0.05.csv").read_bytes(), b"rate-0.05")
        self.assertEqual((d / "btc_alpha_0.5_0.1.csv").read_bytes(), b"rate-0.10")

    def test_folder_downloaded_once_and_cached(self):
        """gdown.download_folder must be called exactly once per URL, even
        when the metadata lists multiple files pointing at the same folder."""
        _install_fake_gdown({"a.csv": b"A", "b.csv": b"B"})
        import gdown
        spy = mock.Mock(wraps=gdown.download_folder)
        gdown.download_folder = spy
        folder_url = "https://drive.google.com/drive/folders/XYZ"
        d = self._write("cached", {
            "name": "cached",
            "derived": True,
            "files": [
                {"name": "a.csv", "url": folder_url, "members": ["a.csv"]},
                {"name": "b.csv", "url": folder_url, "members": ["b.csv"]},
            ],
        })
        fetch(d, verbose=False)
        self.assertEqual(spy.call_count, 1)

    def test_missing_member_raises(self):
        _install_fake_gdown({"present.csv": b"x"})
        d = self._write("bad_member", {
            "name": "bad_member",
            "derived": True,
            "files": [{
                "name": "absent.csv",
                "url": "https://drive.google.com/drive/folders/NOPE",
                "members": ["absent.csv"],
            }],
        })
        with self.assertRaises(DatasetNotReadyError) as ctx:
            fetch(d, verbose=False)
        self.assertIn("did not contain member", str(ctx.exception))

    def test_gdown_missing_raises(self):
        sys.modules.pop("gdown", None)
        # Force import to fail by shadowing with None
        sys.modules["gdown"] = None  # type: ignore[assignment]
        d = self._write("no_gdown", {
            "name": "no_gdown",
            "derived": True,
            "files": [{
                "name": "x.csv",
                "url": "https://drive.google.com/drive/folders/ID",
                "members": ["x.csv"],
            }],
        })
        try:
            with self.assertRaises(DatasetNotReadyError) as ctx:
                fetch(d, verbose=False)
            self.assertIn("gdown is required", str(ctx.exception))
        finally:
            sys.modules.pop("gdown", None)

    def test_is_ready_after_gdown_fetch(self):
        _install_fake_gdown({"only.csv": b"one"})
        d = self._write("ready_ds", {
            "name": "ready_ds",
            "derived": True,
            "files": [{
                "name": "only.csv",
                "url": "https://drive.google.com/drive/folders/A",
                "members": ["only.csv"],
            }],
        })
        self.assertFalse(is_ready(d))
        fetch(d, verbose=False)
        self.assertTrue(is_ready(d))


class GdriveFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gdrive-file-"))
        dl._GDRIVE_FOLDER_CACHE.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp)
        sys.modules.pop("gdown", None)

    def test_single_file_url(self):
        file_url = "https://drive.google.com/uc?id=FILE123"
        _install_fake_gdown({}, {file_url: b"file content"})
        d = self.tmp / "single"
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps({
            "name": "single",
            "derived": True,
            "files": [{"name": "data.bin", "url": file_url}],
        }))
        fetch(d, verbose=False)
        self.assertEqual((d / "data.bin").read_bytes(), b"file content")


if __name__ == "__main__":
    unittest.main(verbosity=2)
