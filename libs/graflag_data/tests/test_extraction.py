"""Tests for archive extraction and derived-dataset recursion."""

import io
import json
import shutil
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from graflag_data.downloader import (
    DatasetNotReadyError, _extract, fetch, is_ready, load_metadata,
)


def _meta(name, **extra):
    d = {"name": name}
    d.update(extra)
    return json.dumps(d)


class ArchiveMembers(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.base = self.tmp / "ds"
        self.base.mkdir()

    def _zip(self, entries):
        p = self.tmp / "a.zip"
        with zipfile.ZipFile(p, "w") as zf:
            for n, data in entries.items():
                zf.writestr(n, data)
        return p

    def _tar(self, entries):
        p = self.tmp / "a.tar.gz"
        with tarfile.open(p, "w:gz") as tf:
            for n, data in entries.items():
                info = tarfile.TarInfo(n)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        return p

    def _file(self, name, members=None, extract="zip"):
        meta_dir = self.tmp / "m"
        meta_dir.mkdir(exist_ok=True)
        payload = {"name": "ds", "files": [
            {"name": name, "url": "http://x/a", "extract": extract,
             "members": members}
        ]}
        (meta_dir / "metadata.json").write_text(json.dumps(payload))
        return load_metadata(meta_dir).files[0]

    def test_single_member_by_name(self):
        archive = self._zip({"data.csv": b"a,b\n1,2\n"})
        _extract(archive, self.base, self._file("data.csv"))
        self.assertEqual((self.base / "data.csv").read_bytes(), b"a,b\n1,2\n")

    def test_every_declared_member_is_extracted(self):
        """Regression: only members[0] was written, so a metadata author
        following the documented schema silently lost the rest -- and the
        dataset was then marked ready."""
        archive = self._zip({"a.csv": b"AAA", "b.csv": b"BBB", "c.csv": b"CCC"})
        _extract(archive, self.base, self._file("a.csv", members=["a.csv", "b.csv", "c.csv"]))

        self.assertEqual((self.base / "a.csv").read_bytes(), b"AAA")
        self.assertEqual((self.base / "b.csv").read_bytes(), b"BBB")
        self.assertEqual((self.base / "c.csv").read_bytes(), b"CCC")

    def test_tar_members_too(self):
        archive = self._tar({"x.npy": b"XX", "y.npy": b"YY"})
        _extract(archive, self.base,
                 self._file("x.npy", members=["x.npy", "y.npy"], extract="tar.gz"))
        self.assertEqual((self.base / "y.npy").read_bytes(), b"YY")

    def test_missing_member_is_reported(self):
        archive = self._zip({"a.csv": b"AAA"})
        with self.assertRaises(DatasetNotReadyError):
            _extract(archive, self.base, self._file("a.csv", members=["nope.csv"]))

    def test_member_with_a_traversal_name_stays_inside_the_dataset_dir(self):
        """metadata.json is not fully trusted: a member named ../escape.csv
        must not write outside the dataset directory."""
        archive = self._zip({"a.csv": b"AAA", "../escape.csv": b"NO"})
        names = zipfile.ZipFile(archive).namelist()
        traversal = next(n for n in names if "escape" in n)

        _extract(archive, self.base, self._file("a.csv", members=["a.csv", traversal]))

        self.assertFalse((self.tmp / "escape.csv").exists())
        self.assertTrue((self.base / "escape.csv").is_file())

    def test_nested_member_is_flattened_into_the_dataset_dir(self):
        archive = self._zip({"a.csv": b"AAA", "deep/nested/b.csv": b"BBB"})
        _extract(archive, self.base,
                 self._file("a.csv", members=["a.csv", "deep/nested/b.csv"]))
        self.assertEqual((self.base / "b.csv").read_bytes(), b"BBB")

    def test_large_member_is_streamed_not_buffered(self):
        """streamspot_all is ~2.2 GB uncompressed; reading a member with
        .read() materialised all of it before writing."""
        import graflag_data.downloader as dl
        import inspect

        src = inspect.getsource(dl._extract_archive_member)
        self.assertIn("copyfileobj", src)
        self.assertNotIn("read_fn(member)", src)


class DerivedRecursion(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _ds(self, name, **extra):
        d = self.root / name
        d.mkdir()
        (d / "metadata.json").write_text(_meta(name, **extra))
        return d

    def test_mutual_derived_from_is_rejected(self):
        """Regression: no cycle guard, so two datasets naming each other
        recursed until RecursionError."""
        build = {"command": ["true"], "produces": ["never.npy"]}
        self._ds("alpha", derived=True, derived_from="beta", build=build)
        self._ds("beta", derived=True, derived_from="alpha", build=build)

        with self.assertRaises(DatasetNotReadyError) as ctx:
            fetch(self.root / "alpha", verbose=False)
        self.assertIn("circular", str(ctx.exception))

    def test_a_normal_derived_chain_still_works(self):
        self._ds("upstream", derived=True)
        self._ds("downstream", derived=True, derived_from="upstream")

        fetch(self.root / "downstream", verbose=False)   # must not raise
        self.assertTrue(is_ready(self.root / "downstream"))


if __name__ == "__main__":
    unittest.main()
