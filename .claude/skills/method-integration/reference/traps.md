# The fail-open catalogue

Every entry below is an incident, not a hypothetical. They share one shape:
**the failing step reported success.** That is why they survived review, and
it is why the rule in `SKILL.md` is to make new code raise rather than skip.

Two consequences worth internalising before reading the list:

- **A fail-open defect hides the next one behind it.** `anograph`'s binary
  exited instantly, so nobody could see its `_NUM_BUCKETS=1024` would never
  finish. `generaldyg` scored the whole stream, so nobody could see its
  checkpoint restore did nothing. Fix one and look again immediately.
- **The sample count is the fastest tell.** Two of the three scientific
  defects were visible as a number that was too large: 14,000 scores where
  the test split has 7,000; 1,097,070 where the method scores 2,751 windows.
  Check the count before the AUC.

## Shell and build

| Trap | What happens | Instead |
|---|---|---|
| `sed -i 's/x/y/' cloned.py` | Exits 0 when it matches nothing. After a `SOURCE_REF` bump the patch is silently gone. | `patches/*.patch` + `git apply --verbose` |
| `git checkout --detach ${SOURCE_REF}` with `SOURCE_REF` unset | Exits 0, leaves the clone on today's default branch. The pin is gone and the build says nothing. | `test -n "${SOURCE_REF}" && git -C src checkout --detach ${SOURCE_REF}` |
| `if/else` on `GRAFLAG_LIBS` | A typo reads as the other arm and builds an image whose libraries are not the ones asked for. | `case` with a `*)` arm that `exit 1`s |
| `COPY methods/x/f.py` where `f.py` was deleted from the checkout | Fails only after the whole context uploads, on the manager. | `test_copy_sources_exist` in `tests/test_methods.py` catches it locally |
| Editing a method and re-running without `--build` | The previous registry image runs. The edit is untested and the run looks fine. | Always `--build` after an edit |
| Editing a method and running `--build` without `graflag sync` | The build context is the share, not your checkout. `--build` rebuilds the same code that already failed. | `cd methods/<name> && graflag sync` first |
| Deleting a file from a method directory and syncing | `graflag sync` carries no `--delete`. The file stays on the share, inside the build context, still importable. | Remove it on the share as well |
| `graflag copy -s ./methods --dest methods -r` | Copies the directory *into* the destination: `/shared/methods/methods`. | Pass the contents, or use `sync` |
| No `.dockerignore` on the share | Build still succeeds, at 3.65 GB of context instead of 804 kB. | `graflag copy -s ./.dockerignore --dest .` once |
| `registry garbage-collect` against a live registry | Deletes blobs the registry still answers `HEAD` for from cache. The next push reports "Layer already exists", uploads nothing, and leaves an image the manager can run and a worker cannot pull. | `graflag clear --apply --gc`, which scales the registry to 0 and back in a `finally` |

## Python

| Trap | What happens | Instead |
|---|---|---|
| `model.state_dict().copy()` as a checkpoint | `state_dict()` returns the *live* tensors; `.copy()` duplicates the dict only. The optimizer updates them in place, so the snapshot tracks training and `load_state_dict` restores the weights onto themselves. | `{k: v.detach().clone() for k, v in model.state_dict().items()}` |
| `f"cuda:{gpu}"` from `_GPU` | `-1` means CPU, so a CPU run asks torch for `cuda:-1`. | `device()` |
| `os.environ.get("DATA", ".")` | Reads the working directory and reports an empty dataset as a real result. | `paths()`, which raises |
| `--pass-env-args` into upstream argparse | Abbreviation matching: upstream declares `--gpus` and no `--gpu`, so `_GPU=0` quietly sets `gpus=0`. | `apply_params(ns, ignore={"gpu"})` |
| Accepting a parameter and not using it | `gady` took `_LR_G`, `_ALPHA`, `_BETAA` and discarded them; a sweep over them produced identical runs. | Wire it, or delete it from `.env` and say why in the README |
| `print("[WARN] no data, skipping")` | A skipped input becomes a completed run over nothing. | Raise |
| `open("results.json", "w")` then `json.dump` | Truncates first, so one unserialisable value leaves a truncated file that still counts as a result. | `ResultWriter.finalize()` |
| `exit(0)` on a failed `fopen` (C/C++ upstream) | The binary "succeeds" having read nothing. `anograph` shipped this way for months. | Check the exit path before trusting an upstream binary's silence |

## Scientific validity

These are the ones that pass every mechanical check.

**Scoring the whole stream.** `generaldyg` published 14,000 scores where the
test split holds 7,000, most of them over edges the model had fitted. `slade`
published 24,186 where the test mask selects 3,618. Both looked like ordinary
completed runs with plausible AUCs. `RESULTS_STANDARD.md` requires the test
split; record `scored_split` and `scored_samples` so a reader can tell.

**Publishing scores the method did not compute.** `anograph` filled a
perceived gap with a local density heuristic and published that as AnoGraph's
output — 0.9652 over 1,097,070 per-edge scores, against the binary's actual
0.9480 over 2,751 windows. The premise ("the binary only emits a scalar AUC")
was simply wrong. Verify against upstream's own entry point: running
`demo.sh` on the same dataset now matches GraFlag element for element.

**Selecting on the test split.** `generaldyg` and `gady` pick their published
checkpoint by the test score, because upstream has no validation split. Keep
the protocol — changing it changes what the method does — but say in the
README that the number is not a clean held-out measurement, so nobody
compares it against a method that does hold one out.

**Re-randomised evaluation features.** `generaldyg`'s removed
`dataset_all.py` drew a third independent `np.random.uniform` feature matrix,
so the scored inputs matched neither training nor test features. Upstream
draws random features by design; drawing them a *third* time was ours.

## The cluster

- **Run one method at a time.** RAM is the binding resource, not the GPU.
  An OOM-killed method log just ends with `Killed`; `dmesg -T | grep -i
  oom-kill` on the manager is what names it.
- **A method image is ~9.6 GB** on a share that sits above 90%. One `--build`
  can fill the disk outright, and when it does the host's `/tmp` goes with it,
  so tool output comes back empty. Reclaim first with `graflag clear --apply
  --gc`.
- **`graflag stop -e <exp> --rm` deletes the experiment directory**, not just
  the service. Without `--rm` it leaves the directory alone.
- **`graflag cleanup` keeps a service** unless the run is diagnosable from
  disk, because `graflag logs` falls back to `method_output.txt`. Removing the
  service for a run that died before writing it destroys the only record.
