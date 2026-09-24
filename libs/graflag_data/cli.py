"""CLI for graflag_data: fetch missing dataset files from their sources."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .downloader import (
    corrupt_files,
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
        if missing:
            status = f"missing:{len(missing)}"
        elif meta.derived:
            status = "derived"
        else:
            status = "ready"
        rows.append((p.name, status, meta.source or meta.source_repo or ""))
    width = max((len(r[0]) for r in rows), default=8)
    for name, status, src in rows:
        print(f"{name:<{width}}  {status:<14}  {src}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    root = Path(args.root)
    names = args.datasets or None
    errors = 0
    # --json is advertised on `fetch` but used to be honoured only in the
    # fetch-all branch, so `graflag-data fetch NAME --json | jq .` printed
    # nothing and exited 0.
    report = {}
    if names:
        for n in names:
            try:
                report[n] = fetch(root / n, force=args.force)
                if not report[n] and not args.json:
                    print(f"[OK] {n}: up to date")
            except (FileNotFoundError, DatasetNotReadyError) as e:
                report[n] = {"error": str(e)}
                if not args.json:
                    print(f"[ERROR] {e}", file=sys.stderr)
                errors += 1
    else:
        report = fetch_all(root, force=args.force)
        for name, result in report.items():
            if isinstance(result, dict) and "error" in result:
                errors += 1

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    return 1 if errors else 0


def _cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.root)
    ok = True
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not (p / METADATA_FILENAME).is_file():
            continue
        meta = load_metadata(p)
        miss = missing_files(p, meta)
        # `derived` does not imply "no files": email_snapshot and the
        # generaldyg_* datasets are derived *and* declare downloads. Skipping
        # them on the derived flag alone made `graflag-data status` exit 0 on a
        # completely unhydrated dataset, so CI passed while `graflag run` failed.
        if meta.derived and not miss:
            print(f"[--] {p.name}: derived")
            continue
        if miss:
            ok = False
            print(f"[MISS] {p.name}: {', '.join(f.name for f in miss)}")
        else:
            print(f"[OK]   {p.name}")
    return 0 if ok else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    """Re-hash what is on disk against what metadata.json pinned.

    ``fetch`` verifies a download and never looks again, so a dataset that was
    corrupted or replaced after it landed passed every check the library made.
    This is the command that asks. Exits non-zero when anything fails, so it
    works as a gate before a benchmark run.
    """
    root = Path(args.root)
    wanted = set(getattr(args, "datasets", None) or [])
    bad_total = 0
    checked = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / METADATA_FILENAME).is_file():
            continue
        if wanted and d.name not in wanted:
            continue
        meta = load_metadata(d)
        pinned = [f for f in meta.files if f.sha256]
        if not pinned:
            print(f"[--]   {d.name}: no sha256 pinned")
            continue
        bad = corrupt_files(d, meta)
        checked += len(pinned)
        if bad:
            bad_total += len(bad)
            for f in bad:
                print(f"[FAIL] {d.name}: {f.name} does not match its pinned sha256")
        else:
            print(f"[OK]   {d.name}: {len(pinned)} file(s) match")
    if bad_total:
        print(f"\n[ERROR] {bad_total} file(s) failed verification")
        return 1
    print(f"\n[OK] {checked} pinned file(s) verified")
    return 0


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

    sp = sub.add_parser(
        "verify",
        help="Re-hash resident files against the sha256 in metadata.json.")
    sp.add_argument("datasets", nargs="*", help="Datasets to verify (default: all).")
    sp.set_defaults(func=_cmd_verify)

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
