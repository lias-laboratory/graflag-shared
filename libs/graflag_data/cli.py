"""CLI for graflag_data: fetch missing dataset files from their sources."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .downloader import (
    DatasetNotReadyError,
    fetch,
    fetch_all,
    is_ready,
    load_metadata,
    missing_files,
    METADATA_FILENAME,
)


def _default_root() -> Path:
    for env in ("GRAFLAG_DATASETS", "GRAFLAG_SHARED"):
        val = os.environ.get(env)
        if val:
            p = Path(val)
            if p.name == "datasets":
                return p
            cand = p / "datasets"
            if cand.is_dir():
                return cand
    cwd = Path.cwd()
    for cand in (cwd, cwd / "datasets", cwd / "graflag-shared" / "datasets"):
        if cand.is_dir() and any(c.is_dir() for c in cand.iterdir()):
            if (cand / "btc_alpha").exists() or cand.name == "datasets":
                return cand
    return cwd


def _cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.root)
    rows = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not (p / METADATA_FILENAME).is_file():
            continue
        meta = load_metadata(p)
        missing = missing_files(p, meta)
        status = "derived" if meta.derived else ("ready" if not missing else f"missing:{len(missing)}")
        rows.append((p.name, status, meta.source or meta.source_repo or ""))
    width = max((len(r[0]) for r in rows), default=8)
    for name, status, src in rows:
        print(f"{name:<{width}}  {status:<14}  {src}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    root = Path(args.root)
    names = args.datasets or None
    errors = 0
    if names:
        for n in names:
            try:
                downloaded = fetch(root / n, force=args.force)
                if not downloaded:
                    print(f"[OK] {n}: up to date")
            except (FileNotFoundError, DatasetNotReadyError) as e:
                print(f"[ERROR] {e}", file=sys.stderr)
                errors += 1
    else:
        report = fetch_all(root, force=args.force)
        if args.json:
            print(json.dumps(report, indent=2))
        for name, result in report.items():
            if isinstance(result, dict) and "error" in result:
                errors += 1
    return 1 if errors else 0


def _cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.root)
    ok = True
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not (p / METADATA_FILENAME).is_file():
            continue
        meta = load_metadata(p)
        if meta.derived:
            print(f"[--] {p.name}: derived")
            continue
        miss = missing_files(p, meta)
        if miss:
            ok = False
            print(f"[MISS] {p.name}: {', '.join(f.name for f in miss)}")
        else:
            print(f"[OK]   {p.name}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="graflag-data",
        description="Fetch GraFlag dataset files from their original sources.",
    )
    p.add_argument(
        "--root",
        default=str(_default_root()),
        help="Path to the datasets/ directory (default: autodetect or $GRAFLAG_DATASETS).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="List datasets and their readiness.")
    sp.set_defaults(func=_cmd_list)

    sp = sub.add_parser("status", help="Show missing files for each dataset.")
    sp.set_defaults(func=_cmd_status)

    sp = sub.add_parser("fetch", help="Download missing dataset files.")
    sp.add_argument("datasets", nargs="*", help="Datasets to fetch (default: all).")
    sp.add_argument("--force", action="store_true", help="Re-download even if present.")
    sp.add_argument("--json", action="store_true", help="Print a JSON report.")
    sp.set_defaults(func=_cmd_fetch)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
