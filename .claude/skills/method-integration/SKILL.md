---
name: method-integration
description: Integrate a Graph Anomaly Detection method into GraFlag, or verify one that already exists. Use when adding a GAD method to graflag-shared/methods/, wrapping an upstream paper repository as a GraFlag method, writing or reviewing a train_graflag.py, pinning a method to an upstream commit, or checking whether a finished experiment's AUC is real.
---

# Integrating a GAD method into GraFlag

A method is integrated when GraFlag can build it, run it on a declared dataset,
and publish a number that means what it says. The last clause is the hard one.
Every scientific-validity defect this repository has recorded --- scores the
method never computed, an AUC over the training split, a checkpoint restore
that restored nothing --- shipped as a `completed` run with a plausible AUC.
So treat "it ran" as the start of the work, not the end.

## The four gates

Do not report a method as integrated until all four pass, in order. Each one
catches things the one before it cannot.

| Gate | Command | Catches |
|---|---|---|
| 1. Contract | `cd graflag-shared && python3 -m unittest discover -s tests` | The `.env`/Dockerfile schema, unpinned clones, `sed -i` on cloned source, COPY paths, GPU conventions, provenance |
| 2. Build and run | `graflag sync` then `graflag run -m NAME -d DATASET --build` | Anything that only exists on the share |
| 3. Result integrity | `python3 .claude/skills/method-integration/scripts/verify_run.py EXP` | Empty/one-class/constant scores, length mismatch, the published scores disagreeing with the method's own AUC |
| 4. Evaluation | `graflag evaluate -e EXP` | Metrics and plots; run it before gate 3 so gate 3 has an `auc_roc` to compare against |

Gate 3 is the one that does not exist anywhere else. Run
`graflag evaluate` first, then `verify_run.py`, and read every `[WARN]`
rather than skimming for `0 failed`.

## Before writing anything

Read the upstream repository and answer three questions. Getting these wrong
costs a rebuild, and a rebuild is ~9.6 GB and several minutes.

**Which pattern?**
- **A --- an integration script.** `COMMAND=python3 train_graflag.py`,
  `CMD ["python3", "-m", "graflag_runner"]`. Sixteen methods do this. Default to it.
- **B --- a library entry point.** `COMMAND=python3 -m graflag_bond.train`.
  Only for a PyGOD detector: add `IMAGE=bond_base` and write no Dockerfile.
  The seventeen `bond_*` methods share one image and stay distinct because
  `METHOD_NAME` is per service.
- `--pass-env-args` is a third option no method uses. It rewrites
  `_BATCH_SIZE=128` into `--batch_size 128`, lowercasing names and coercing
  nothing. Reach for it only when `COMMAND` is upstream's own entry point.

**`INTEGRATION=upstream` or `reimplementation`?** `upstream` means the image
fetches and runs the authors' code, and then `SOURCE_REF` must pin the
40-character commit. `reimplementation` means it does not, and `SOURCE_CODE`
is a citation rather than what ran --- say so in the README, or the method
reads as the authors' work when it is yours. `dynwalk` is how that happens.

**What is the test split, and does the method have one?** `RESULTS_STANDARD.md`
requires the published scores to come from the test split. Find, in upstream's
code, exactly which tensor holds the test predictions --- before writing
anything. If upstream has no validation split and selects on test (`generaldyg`,
`gady`), keep its protocol and say plainly in the README that the published
number is not a clean held-out measurement.

## The procedure

1. **Copy the template.** `cp -r methods/example methods/<name>`. It is
   annotated line by line and its comments are current.
2. **Write `.env`.** `METHOD_NAME` must equal the directory name. Required:
   `METHOD_NAME`, `DESCRIPTION`, `SOURCE_CODE`, `COMMAND`. Then `INTEGRATION`,
   `SOURCE_REF` if it clones, `SUPPORTED_DATASETS`, `IMAGE` if shared.
   Parameters are `_UPPERCASE`. Never declare `DATA`, `EXP`, `METHOD_NAME`,
   `COMMAND` or `MONITOR_INTERVAL` as parameters. A key outside `KNOWN_KEYS`
   in `tests/test_methods.py` fails gate 1 --- that is the schema working.
3. **Write the Dockerfile.** The build context is `SHARED_DIR/`, so every
   `COPY` is `methods/<name>/...`. Copy the `ARG GRAFLAG_LIBS` stanza from
   `methods/example/Dockerfile` verbatim. Pin the clone with `test -n` on both
   build args. Put fixes to cloned source in `patches/*.patch` applied with
   `git apply --verbose`, never `sed -i`.
4. **Write `train_graflag.py`** against `graflag_runner` --- see
   `reference/sdk.md`. Do not hand-roll parameter parsing, device selection,
   `sys.path` juggling or psutil sampling; all four are in the library, and
   the runner's resource numbers win over anything a method reports.
5. **Record the dataset** under `datasets/<name>/` if it is new, and hydrate
   it with `graflag-data fetch <name>`.
6. **Write `README.md`.** Required by gate 1. It must state what upstream does
   versus what this integration does, which split is scored, and any protocol
   the method inherits that a reader would otherwise mistake for a clean
   measurement.
7. **Run the gates.**

## Running on the cluster

```bash
cd graflag-shared/methods/<name> && graflag sync   # the build reads the share, not your checkout
graflag run -m <name> -d <dataset> --build --params N_EPOCHS=2   # smoke first
graflag logs -e <exp> -f
graflag evaluate -e <exp>
python3 .claude/skills/method-integration/scripts/verify_run.py <exp>
```

Four operational facts that cost real time when forgotten:

- **On a devcluster, one method at a time.** Its nodes share the host's RAM,
  which runs out before the GPU does: two concurrent runs get OOM-killed, and
  the method log just ends with `Killed` --- `dmesg -T | grep -i oom-kill` on
  the host is what reveals it.
- **`graflag sync` adds and overwrites; it never deletes.** A file removed
  from your checkout stays on the share, inside the build context, still
  importable. Delete it there too.
- **Images are reused unless `--build` is passed.** After editing a method,
  no `--build` means the previous image runs again and your edit is untested.
- **Disk.** A method image is several gigabytes (`bond_base` is ~9.6 GB), and
  on a devcluster every worker that pulls it keeps its own copy on the same
  disk. Reclaim with `graflag clear --apply --gc` before a build, not after it
  fails.

## Rules that are not negotiable

**Fail loudly.** Every defect in `VERIFICATION.md` is something that failed
open: `sed -i` exits 0 when it matches nothing, `git checkout --detach` with an
empty ref exits 0, a C++ reader answers a failed `fopen` with `exit(0)`. When
an input is missing or a patch does not apply, raise --- never skip, never
substitute a default, never `print` a warning and continue. `test_methods.py`
enforces this for method scripts and will fail a bare `print` used to signal a
problem. See `reference/traps.md`.

**Publish what the method computed.** If upstream writes a score file, publish
that file's contents. Do not compute scores yourself to fill a gap --- and if
you genuinely must, the README says so in its first paragraph and
`INTEGRATION` is `reimplementation`.

**Never report a run you did not verify.** If a method cannot run here ---
missing data, an environment it needs and this cluster lacks --- record it as
an environment limit with the traceback. A fabricated pass is worse than a
gap, because the gap gets fixed.

**A fail-open defect hides the next one behind it.** Twice now, fixing one
defect immediately exposed a second that had been unobservable: `anograph`'s
unreachable `_NUM_BUCKETS` was invisible while the binary exited instantly, and
`generaldyg`'s dead checkpoint restore was invisible while whole-stream scoring
made per-epoch AUCs incomparable. After fixing anything here, re-read the
method's own log against its published number.

## Where to look things up

- `reference/sdk.md` --- the `graflag_runner` API for integration scripts.
- `reference/traps.md` --- the fail-open catalogue, from real incidents.
- `methods/example/` --- the annotated template; `.env` and Dockerfile comments
  are authoritative and current.
- `docs/AGENT_METHOD_INTEGRATION.md` in the graflag repository --- the long-form reference
  (templates, a worked `gady` integration, dataset layout, common errors).
- `tests/test_methods.py` --- the contract as executable rules. When unsure
  whether something is allowed, read the test; it is more current than prose.
- Good scripts to copy: `methods/taddy/` (clean Pattern A),
  `methods/generaldyg/` (upstream argparse via `apply_params`),
  `methods/bond_cola/` (Pattern B, shared image).
