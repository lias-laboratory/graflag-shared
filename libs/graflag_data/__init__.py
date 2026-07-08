"""
GraFlag Data - Dataset metadata and downloader for GraFlag.

Each dataset folder contains a ``metadata.json`` file describing the dataset
and where its files can be downloaded from. This package provides the logic
to read those metadata files and fetch any missing files from their original
source, so datasets can be distributed as empty stubs and hydrated on demand.

Public API:
- ``load_metadata(path)``  -> dict
- ``is_ready(dataset_dir)`` -> bool
- ``fetch(dataset_dir, force=False)`` -> list of files downloaded
- ``fetch_all(datasets_root, names=None, force=False)``
"""

from .downloader import (
    BuildStep,
    DatasetMetadata,
    DatasetFile,
    DatasetNotReadyError,
    load_metadata,
    is_ready,
    fetch,
    fetch_all,
    missing_files,
)

__all__ = [
    "BuildStep",
    "DatasetMetadata",
    "DatasetFile",
    "DatasetNotReadyError",
    "load_metadata",
    "is_ready",
    "fetch",
    "fetch_all",
    "missing_files",
]
