"""Dataset downloader that reads per-dataset ``metadata.json`` files.

Metadata schema (``metadata.json`` at the root of each dataset folder)::

    {
      "name": "btc_alpha",
      "description": "Who-trusts-whom network of Bitcoin Alpha users.",
      "source": "https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html",
      "license": "CC BY-SA 4.0",
      "compatible_methods": ["taddy", "gady"],
      "format": "CSV: source,target,rating,timestamp",
      "files": [
        {
          "name": "soc-sign-bitcoinalpha.csv",
          "url": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
          "extract": "gunzip",
          "sha256": null
        }
      ]
    }

A file entry is considered present when ``dataset_dir / file.name`` exists.
If ``extract`` is set, the archive is downloaded to a temp file, then
unpacked into the dataset directory. Supported ``extract`` values:
``gunzip`` (single-file .gz), ``zip``, ``tar``, ``tar.gz`` / ``tgz``,
``tar.bz2``.

Derived datasets (preprocessed from another source) set
``"derived": true`` and typically list no files; running ``fetch`` on such
a dataset is a no-op and emits a warning pointing at the upstream source.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Union


METADATA_FILENAME = "metadata.json"


class DatasetNotReadyError(RuntimeError):
    """Raised when a dataset's files are missing and cannot be fetched."""


@dataclass
class DatasetFile:
    name: str
    url: Optional[str] = None
    extract: Optional[str] = None
    sha256: Optional[str] = None
    # When set, the file is produced by extracting a multi-file archive.
    # ``members`` names inside the archive to copy out (defaults: all).
    members: Optional[List[str]] = None


@dataclass
class BuildStep:
    """A post-fetch build command that produces files in the dataset dir.

    Used by derived datasets to run a conversion script (e.g.
    ``convert_to_strgnn.py``) once the base dataset has been fetched.
    """

    command: Union[str, List[str]]
    produces: List[str] = field(default_factory=list)
    cwd: Optional[str] = None  # relative to the dataset directory


@dataclass
class DatasetMetadata:
    name: str
    description: str = ""
    source: Optional[str] = None
    source_repo: Optional[str] = None
    license: Optional[str] = None
    compatible_methods: List[str] = field(default_factory=list)
    format: Optional[str] = None
    derived: bool = False
    derived_from: Optional[str] = None
    files: List[DatasetFile] = field(default_factory=list)
    build: Optional[BuildStep] = None
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetMetadata":
        files = [DatasetFile(**f) for f in data.get("files", [])]
        build_raw = data.get("build")
        build = BuildStep(**build_raw) if build_raw else None
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            source=data.get("source"),
            source_repo=data.get("source_repo"),
            license=data.get("license"),
            compatible_methods=list(data.get("compatible_methods", [])),
            format=data.get("format"),
            derived=bool(data.get("derived", False)),
            derived_from=data.get("derived_from"),
            files=files,
            build=build,
            notes=data.get("notes"),
        )


def load_metadata(dataset_dir: os.PathLike) -> DatasetMetadata:
    path = Path(dataset_dir) / METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {METADATA_FILENAME} in {dataset_dir}")
    with path.open("r", encoding="utf-8") as fh:
        return DatasetMetadata.from_dict(json.load(fh))


def missing_files(dataset_dir: os.PathLike, meta: Optional[DatasetMetadata] = None) -> List[DatasetFile]:
    meta = meta or load_metadata(dataset_dir)
    base = Path(dataset_dir)
    return [f for f in meta.files if not (base / f.name).exists()]


def is_ready(dataset_dir: os.PathLike) -> bool:
    try:
        meta = load_metadata(dataset_dir)
    except FileNotFoundError:
        return False
    base = Path(dataset_dir)
    if missing_files(dataset_dir, meta):
        return False
    if meta.build and meta.build.produces:
        return all((base / p).exists() for p in meta.build.produces)
    if not meta.files and meta.derived and not meta.build:
        # Derived dataset with neither download nor build step - treated as
        # pre-populated (upstream supplies the files manually).
        return True
    return True


def fetch(
    dataset_dir: os.PathLike,
    force: bool = False,
    verbose: bool = True,
) -> List[str]:
    """Download any missing files listed in ``metadata.json`` and, if the
    dataset declares a ``build`` step, run it once its inputs are available.

    Returns the list of items that were (re)produced — a mix of file names
    downloaded and build-step ``produces`` entries generated.
    Raises ``DatasetNotReadyError`` if a required file has no download URL,
    a build command fails, or a build step doesn't produce its declared files.
    """
    meta = load_metadata(dataset_dir)
    base = Path(dataset_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    produced: List[str] = []

    # 1. Ensure the upstream dataset is ready (for derived datasets with a
    #    build step, the build command usually reads from ../<derived_from>/).
    if meta.derived_from:
        upstream = base.parent / meta.derived_from
        if upstream.is_dir() and (upstream / METADATA_FILENAME).is_file():
            if force or not is_ready(upstream):
                produced.extend(
                    f"{meta.derived_from}/{n}"
                    for n in fetch(upstream, force=force, verbose=verbose)
                )

    # 2. Download declared files (direct-download datasets).
    for f in meta.files:
        target = base / f.name
        if target.exists() and not force:
            if verbose:
                print(f"[OK] {meta.name}: {f.name} already present")
            continue
        if not f.url:
            raise DatasetNotReadyError(
                f"{meta.name}: file {f.name!r} is missing and has no download URL "
                f"(source: {meta.source or meta.source_repo or 'unknown'})"
            )
        _download_file(f, base, verbose=verbose)
        produced.append(f.name)

    # 3. Run the build step if present and outputs are missing.
    if meta.build:
        outputs_present = bool(meta.build.produces) and all(
            (base / p).exists() for p in meta.build.produces
        )
        if force or not outputs_present:
            _run_build(meta, base, verbose=verbose)
            produced.extend(meta.build.produces)
    elif meta.derived and not meta.files:
        if verbose:
            upstream = meta.derived_from or meta.source_repo or meta.source or "upstream source"
            print(
                f"[INFO] {meta.name}: dataset is derived/preprocessed "
                f"(from {upstream}); nothing to download.",
                file=sys.stderr,
            )

    return produced


def _run_build(meta: "DatasetMetadata", base: Path, verbose: bool) -> None:
    build = meta.build
    cwd = base if not build.cwd else (base / build.cwd).resolve()
    cmd = build.command
    if isinstance(cmd, str):
        shell = True
        display = cmd
    else:
        shell = False
        display = " ".join(shlex.quote(c) for c in cmd)
    if verbose:
        print(f"[INFO] {meta.name}: running build -> {display}  (cwd={cwd})")
    result = subprocess.run(cmd, shell=shell, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise DatasetNotReadyError(
            f"{meta.name}: build command failed (exit {result.returncode})\n"
            f"  command: {display}\n"
            f"  stderr : {result.stderr.strip()[:2000]}"
        )
    missing = [p for p in build.produces if not (base / p).exists()]
    if missing:
        raise DatasetNotReadyError(
            f"{meta.name}: build ran but did not produce expected files: {missing}"
        )


def fetch_all(
    datasets_root: os.PathLike,
    names: Optional[Iterable[str]] = None,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Fetch multiple datasets. Returns a ``{name: [files]}`` report."""
    root = Path(datasets_root)
    targets: List[Path] = []
    if names:
        for n in names:
            p = root / n
            if not p.is_dir():
                raise FileNotFoundError(f"dataset folder not found: {p}")
            targets.append(p)
    else:
        for p in sorted(root.iterdir()):
            if p.is_dir() and (p / METADATA_FILENAME).is_file():
                targets.append(p)

    report: dict = {}
    for p in targets:
        try:
            report[p.name] = fetch(p, force=force, verbose=verbose)
        except DatasetNotReadyError as e:
            if verbose:
                print(f"[WARN] {e}", file=sys.stderr)
            report[p.name] = {"error": str(e)}
    return report


_GDRIVE_FOLDER_CACHE: dict = {}


def _is_gdrive_url(url: str) -> bool:
    return "drive.google.com" in url or url.startswith("gdrive://")


def _download_file(f: DatasetFile, base: Path, verbose: bool) -> None:
    if _is_gdrive_url(f.url):
        _download_gdrive(f, base, verbose=verbose)
        return

    if verbose:
        print(f"[INFO] downloading {f.name} <- {f.url}")

    with tempfile.NamedTemporaryFile(delete=False, dir=base, prefix=".dl-") as tmp:
        tmp_path = Path(tmp.name)
    try:
        _urlretrieve(f.url, tmp_path)
        if f.sha256:
            _verify_sha256(tmp_path, f.sha256)

        if not f.extract:
            shutil.move(str(tmp_path), str(base / f.name))
        else:
            _extract(tmp_path, base, f)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _urlretrieve(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "graflag-data/1.0"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got.lower() != expected.lower():
        raise DatasetNotReadyError(
            f"checksum mismatch for {path.name}: expected {expected}, got {got}"
        )


def _extract(archive: Path, base: Path, f: DatasetFile) -> None:
    kind = f.extract.lower()
    target = base / f.name

    if kind == "gunzip":
        with gzip.open(archive, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return

    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            _extract_archive_member(zf.namelist(), f, target, lambda m: zf.read(m))
        return

    if kind in ("tar", "tar.gz", "tgz", "tar.bz2"):
        mode = {"tar": "r", "tar.gz": "r:gz", "tgz": "r:gz", "tar.bz2": "r:bz2"}[kind]
        with tarfile.open(archive, mode) as tf:
            names = tf.getnames()
            def _read(m: str) -> bytes:
                fobj = tf.extractfile(m)
                if fobj is None:
                    raise DatasetNotReadyError(f"archive member {m!r} is not a file")
                return fobj.read()
            _extract_archive_member(names, f, target, _read)
        return

    raise DatasetNotReadyError(f"unknown extract kind: {f.extract!r}")


def _download_gdrive(f: DatasetFile, base: Path, verbose: bool) -> None:
    """Fetch a file from Google Drive.

    Supports two URL shapes:
      1. Folder: ``https://drive.google.com/drive/folders/<ID>``
         Downloads the whole folder once (cached), then copies
         ``f.members[0]`` (or ``f.name``) into ``base / f.name``.
      2. File:   ``https://drive.google.com/file/d/<ID>/view`` or
                 any URL pytesting ``?id=<ID>``
         Downloads that single file, then extracts if ``f.extract`` set.
    """
    try:
        import gdown  # type: ignore
    except ImportError as e:
        raise DatasetNotReadyError(
            "gdown is required for Google Drive URLs. "
            "Install with: pip install gdown"
        ) from e

    url = f.url
    target = base / f.name
    if "/folders/" in url or url.startswith("gdrive://folder/"):
        cache_dir = _GDRIVE_FOLDER_CACHE.get(url)
        if not cache_dir or not Path(cache_dir).is_dir():
            cache_dir = tempfile.mkdtemp(prefix="gdrive-folder-")
            if verbose:
                print(f"[INFO] gdown: downloading folder {url} -> {cache_dir}")
            gdown.download_folder(
                url=url if not url.startswith("gdrive://") else
                    f"https://drive.google.com/drive/folders/{url.split('/')[-1]}",
                output=cache_dir,
                quiet=not verbose,
                use_cookies=False,
            )
            _GDRIVE_FOLDER_CACHE[url] = cache_dir
        member = (f.members[0] if f.members else f.name)
        src = _find_in_tree(Path(cache_dir), member)
        if src is None:
            raise DatasetNotReadyError(
                f"gdrive folder {url} did not contain member {member!r}"
            )
        shutil.copy(src, target)
        return

    # Single-file URL
    if verbose:
        print(f"[INFO] gdown: downloading file {url}")
    with tempfile.NamedTemporaryFile(delete=False, dir=base, prefix=".dl-") as tmp:
        tmp_path = Path(tmp.name)
    try:
        gdown.download(url=url, output=str(tmp_path), quiet=not verbose, fuzzy=True)
        if f.sha256:
            _verify_sha256(tmp_path, f.sha256)
        if not f.extract:
            shutil.move(str(tmp_path), str(target))
        else:
            _extract(tmp_path, base, f)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _find_in_tree(root: Path, name: str) -> Optional[Path]:
    """Find a file whose path (relative to root) or basename equals ``name``."""
    direct = root / name
    if direct.is_file():
        return direct
    for p in root.rglob(Path(name).name):
        if p.is_file():
            return p
    return None


def _extract_archive_member(names, f, target, read_fn) -> None:
    """Copy one member out of an archive into ``target``.

    If ``f.members`` is set, use its first entry; otherwise pick the
    member whose basename matches ``f.name``; otherwise pick the only
    regular file.
    """
    if f.members:
        member = f.members[0]
    else:
        matches = [n for n in names if Path(n).name == f.name]
        if matches:
            member = matches[0]
        else:
            regulars = [n for n in names if not n.endswith("/")]
            if len(regulars) != 1:
                raise DatasetNotReadyError(
                    f"cannot infer which archive member to extract for {f.name!r}; "
                    f"set 'members' in metadata.json"
                )
            member = regulars[0]

    data = read_fn(member)
    with target.open("wb") as dst:
        dst.write(data)
