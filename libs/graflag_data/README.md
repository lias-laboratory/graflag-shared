# graflag_data

On-demand dataset downloader for GraFlag.

Every folder under `graflag-shared/datasets/` carries a `metadata.json` that
describes the dataset, lists the files it needs, and either:

- a `files[].url` pointing at the original source (direct-download datasets), or
- a `build` step that regenerates the dataset from its base dataset using a
  local conversion script (derived datasets, e.g. `*_snapshot` produced by
  `datasets/convert_to_strgnn.py`).

This package reads those metadata files and fetches/builds any missing
outputs, so datasets can ship as empty stubs and be hydrated at runtime.

## Install

```bash
pip install ./graflag-shared/libs/graflag_data
```

Installs the `graflag-data` CLI entry point.

## Usage

```bash
# List all datasets + readiness
graflag-data list

# Show missing outputs only (exits non-zero if anything is missing)
graflag-data status

# Fetch (download and/or build) for one dataset
graflag-data fetch btc_alpha
graflag-data fetch btc_alpha_snapshot   # downloads btc_alpha first, then runs convert_to_strgnn.py

# Fetch everything
graflag-data fetch

# Force re-download / re-build
graflag-data fetch uci_snapshot --force
```

Root resolution: `--root` > `$GRAFLAG_DATASETS` > `./datasets/` >
`./graflag-shared/datasets/` under the current working directory.

## Programmatic API

```python
from graflag_data import fetch, is_ready

dataset_dir = "graflag-shared/datasets/btc_alpha_snapshot"
if not is_ready(dataset_dir):
    fetch(dataset_dir)      # downloads btc_alpha then runs the build step
```

## How `fetch` works

For a given dataset directory, `fetch` runs these phases in order:

1. **Resolve the base dataset.** If `derived_from` is set, recurse into the
   sibling folder first (so `btc_alpha_snapshot` triggers a fetch of
   `btc_alpha`).
2. **Download declared files.** For each entry in `files`, download from
   `url`, verify `sha256` if set, and extract according to `extract`. Files
   already on disk are skipped unless `--force` is used.
3. **Run the build step.** If `build` is present and any of its `produces`
   outputs are missing, run the command in the dataset's cwd. The build step
   is skipped when all declared outputs already exist.

`is_ready(dir)` returns true when `build.produces` (if any) and every
declared file are all present on disk.

## metadata.json schema

```json
{
  "name": "btc_alpha_snapshot",
  "description": "Short human description.",
  "source": "Landing page URL of the upstream dataset.",
  "source_repo": "GitHub repo where the preprocessing lives (optional).",
  "license": "Free-text license or terms-of-use pointer.",
  "format": "Short description of the on-disk layout.",
  "compatible_methods": ["strgnn", "addgraph"],

  "derived": true,
  "derived_from": "btc_alpha",

  "files": [
    {
      "name": "soc-sign-bitcoinalpha.csv",
      "url": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
      "extract": "gunzip",
      "sha256": null,
      "members": null
    }
  ],

  "build": {
    "command": [
      "python3", "../convert_to_strgnn.py",
      "../btc_alpha/soc-sign-bitcoinalpha.csv", ".",
      "--format", "bitcoin"
    ],
    "produces": ["acc_graph.npy", "sta_graph.npy", "split.npz"],
    "cwd": null
  },

  "notes": "Free-text caveats."
}
```

### Fields

| Field | Purpose |
|---|---|
| `name` | Must match the folder name. Required. |
| `description` / `format` / `license` / `notes` | Free-text metadata. |
| `source` | Human-facing landing page URL. |
| `source_repo` | Repo where preprocessing or reference code lives. |
| `compatible_methods` | Omit on unprefixed datasets (`btc_alpha`, `uci`, ...); only include when the dataset folder name encodes a specific method. |
| `derived` | `true` when the dataset is preprocessed from another dataset. |
| `derived_from` | Sibling dataset folder whose files feed the build step. |
| `files[]` | Direct downloads. Each entry has `name`, `url`, optional `extract` (`gunzip` / `zip` / `tar` / `tar.gz` / `tgz` / `tar.bz2`), optional `sha256`, optional `members` (archive path selector). |
| `build.command` | Shell string OR argv list run in the dataset dir. |
| `build.produces` | Files the command is expected to create. Used for readiness checks and to skip the build when outputs already exist. |
| `build.cwd` | Optional working directory relative to the dataset dir (defaults to the dataset dir). |

### How derived datasets work in practice

`btc_alpha_snapshot`, `btc_otc_snapshot`, and `uci_snapshot` are produced by
`graflag-shared/datasets/convert_to_strgnn.py`. Their `metadata.json` points
`command` at `../convert_to_strgnn.py` with the base dataset's file as input:

```
datasets/
├── convert_to_strgnn.py
├── btc_alpha/
│   ├── metadata.json
│   └── soc-sign-bitcoinalpha.csv     ← fetched from SNAP
└── btc_alpha_snapshot/
    ├── metadata.json                  ← declares build step
    ├── acc_graph.npy                  ← produced
    ├── sta_graph.npy                  ← produced
    └── split.npz                      ← produced
```

Running `graflag-data fetch btc_alpha_snapshot` on an empty tree will:

1. Download `btc_alpha/soc-sign-bitcoinalpha.csv` from SNAP.
2. Run `python3 ../convert_to_strgnn.py ../btc_alpha/soc-sign-bitcoinalpha.csv . --format bitcoin`
   from inside `btc_alpha_snapshot/`.
3. Verify the three `produces` files appear.

For derived datasets without a known conversion script (e.g. `email_snapshot`,
`generaldyg_*`), set `derived: true` without a `build` block and use `notes`
to point at the upstream preprocessing pipeline — `fetch` becomes a no-op.

## Integration with the main graflag CLI

`graflag.core.GraFlag.run` calls `_ensure_dataset(dataset)` before deploying
the Docker service. That helper runs `graflag_data` on the swarm manager via
SSH so downloads/builds land directly on the NFS-mounted shared volume:

```
PYTHONPATH=/shared/libs python3 -m graflag_data --root /shared/datasets fetch <dataset>
```

Datasets without `metadata.json` are treated as pre-populated and skipped
silently, preserving backwards compatibility.

## Running the tests

```bash
cd graflag-shared/libs
PYTHONPATH=. python3 -m unittest discover -s graflag_data/tests -v
```

The suite spins up a local `http.server` for fixtures (no external network
calls), covers all `extract` kinds, exercises the build step end-to-end by
running the real `convert_to_strgnn.py` on a tiny synthetic CSV, and mocks
`SSHManager` to verify `GraFlag._ensure_dataset` issues the right remote
command.
