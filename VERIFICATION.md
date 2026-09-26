# Method verification

The 27 method directories that existed when this file was written (the 17
PyGOD detectors and 10 others, the annotated `example/` among them) were built
and run on a cluster, and the result recorded here. 26 of 27 produced a
`results.json` that `graflag evaluate` could score; the rest are explained
below rather than quietly omitted. Methods integrated since (`ad_gcl`,
`ada_gad`, `diffgad`, `f_fade`, `huge_gad`, `midas`, `rare`) record their
runs, or why none has completed, in the `## Verification` section of their
own README.

This file exists because "the method is integrated" and "the method runs" had
never been the same claim. Several methods here had never been executed once.

## How each run was made

    graflag run -m <method> -d <dataset> --build [--params EPOCHS=2]
    graflag evaluate -e <experiment>

on a five-node Docker Swarm sharing one GPU, so the runs are serial. Epoch
counts are reduced where a `.env` declared a long default -- the question
being asked is whether the integration works end to end, not what the method
scores at convergence. Where a run used reduced epochs the parameters are in
the table; where the cell is empty the method ran at its declared defaults.

Datasets are the smallest member of each method's declared family, hydrated
on the manager by `graflag_data` before the run.

## The 17 PyGOD methods

All 17 share one image (`IMAGE=bond_base`), built once. They stay
distinct because `METHOD_NAME` is set per service and `graflag_bond.train`
selects the detector from it -- which this matrix is the first evidence for,
since before it no bond method had ever run.

The sharing is visible in what the runs recorded, not only in the code. The
first run logs the build once:

    [INFO] Rebuilding shared image bond_base (used by bond_dominant)

and no later run logs a build at all. Every `service_config.json` in the
family names the same image with a different method:

    exp__bond_dominant__...  image 192.168.100.10:5000/bond_base:latest  METHOD_NAME bond_dominant
    exp__bond_cola__...      image 192.168.100.10:5000/bond_base:latest  METHOD_NAME bond_cola
    exp__bond_adone__...     image 192.168.100.10:5000/bond_base:latest  METHOD_NAME bond_adone

Their scores are 0.7913, 0.1795 and 0.8218. One image running one detector
under three names would have produced one number three times.

Every `_FOO` in the 17 `.env` files was also checked against the
constructor it is meant to reach: 16 of the 17 detectors resolve under
the locally installed PyGOD and **none of them declares a key its constructor
does not accept** (`bond_card` is newer than that release and was not
checkable this way). So the warning `params()` now emits on a dropped
parameter is not noise the family would produce anyway -- it fires on a
`--params` request that reached nothing, or on upstream drift, which is the
only way either becomes visible.

| Method | Dataset | Status | AUC-ROC | AUC-PR | Note |
|---|---|---|---|---|---|
| `bond_adone` | `bond_gen_100` | completed | 0.8218 | 0.6962 |  |
| `bond_anomalous` | `bond_gen_100` | completed | 0.3503 | 0.1416 |  |
| `bond_anomalydae` | `bond_gen_100` | completed | 0.7283 | 0.5341 |  |
| `bond_card` | `bond_gen_100` | completed | 0.6430 | 0.3466 |  |
| `bond_cola` | `bond_gen_100` | completed | 0.1795 | 0.1129 |  |
| `bond_conad` | `bond_gen_100` | completed | 0.7913 | 0.7133 |  |
| `bond_dmgd` | `bond_gen_100` | completed | 0.2087 | 0.1182 | CPU: upstream GPU defect, declared in `.env` |
| `bond_dominant` | `bond_gen_100` | completed | 0.7913 | 0.7134 |  |
| `bond_done` | `bond_gen_100` | completed | 0.6944 | 0.3937 |  |
| `bond_gaan` | `bond_gen_100` | failed | -- | -- | upstream, both devices -- see [its README](methods/bond_gaan/README.md) |
| `bond_gadnr` | `bond_gen_100` | completed | 0.7981 | 0.7489 | CPU: upstream GPU defect, declared in `.env` |
| `bond_gae` | `bond_gen_100` | completed | 0.6440 | 0.3183 |  |
| `bond_guide` | `bond_gen_100` | completed | 0.8401 | 0.7313 |  |
| `bond_ocgnn` | `bond_gen_100` | completed | 0.7581 | 0.4445 |  |
| `bond_one` | `bond_gen_100` | completed | 0.3699 | 0.1440 |  |
| `bond_radar` | `bond_gen_100` | completed | 0.5413 | 0.2090 |  |
| `bond_scan` | `bond_gen_100` | completed | 0.7534 | 0.4768 | `EPOCH=2` reached nothing -- SCAN has no `epoch`; ran its declared defaults |

## The 10 other methods

| Method | Dataset | Status | AUC-ROC | AUC-PR | Note |
|---|---|---|---|---|---|
| `addgraph` | `email_snapshot` | completed | 0.8143 | 0.0304 | reimplementation, not the cited repository; `NUM_EPOCHS=2` |
| `anograph` | `anograph_iscx` | completed | 0.9480 | 0.5230 | upstream's binary, reproduced exactly (see below) |
| `dynwalk` | `email_snapshot` | completed | 0.8378 | 0.0030 | reimplementation, not the cited repository; `EPOCHS=2` |
| `example` | `email_snapshot` | completed | 0.4812 | 0.0008 | `EPOCHS=2`; placeholder random scores over the held-out half (18,837 edges, 10 anomalous), `graflag verify` 0 failed, 0 warned |
| `gady` | `gady_email_dnc` | completed | 0.9770 | 0.8388 | adversarial loop restored; only `_LR` is inert now; `N_EPOCH=2` |
| `generaldyg` | `generaldyg_btc_alpha` | completed | 0.7806 | 0.0797 | test split, best checkpoint; eval features are re-randomised; `N_EPOCHS=2` |
| `slade` | `slade_bitcoinalpha` | completed | 0.7588 | 0.1563 | test split (3,618 of 24,186 edges); `N_EPOCH=2` |
| `streamspot` | `streamspot_all` | completed | 0.8942 | 0.4574 | labels are the paper's scenario layout; the stream is verified to match it before scoring |
| `strgnn` | `email_snapshot` | completed | 0.6456 | 0.0192 | `NUM_EPOCHS=2` |
| `taddy` | `uci` | completed | 0.8521 | 0.4271 | `MAX_EPOCH=2` |

## The four scientific-validity defects, and how each was closed

An earlier pass through this matrix found four places where a method produced
a number that was not the number it appeared to be, and recorded them as out
of scope -- they were defects in what a method *computes*, not in how GraFlag
runs it. They are fixed now. Each was closed inside the integration layer;
none of them changed a line of upstream's algorithm.

Six are listed below, not four. Closing a defect that fails open exposes
whatever was standing behind it: `anograph`'s binary was exiting before it
scored anything, which is why nobody could see that its declared
`_NUM_BUCKETS` would not finish, and `generaldyg` was scoring the whole
stream, which is why nobody could see that its checkpoint restore did
nothing. Both pairs are kept together here, because the second one is not a
separate discovery so much as what the first was hiding.

- **`anograph` published scores it had computed itself.** The AnoGraph binary
  was being handed its input at `../data/<name>.csv` while its readers open
  `../data/<name>/Data.csv`, and every one of those readers answers a failed
  `fopen` with `exit(0)` -- so the binary "succeeded" instantly, having read
  nothing, and the integration filled the gap with a local density heuristic
  and published that. The premise written down at the time, that the binary
  emits only a scalar AUC, was wrong: it writes a full `score label` file, the
  same one upstream's own `metrics.py` reads.

  Fixed by writing the data where upstream looks for it. The published scores
  are now the binary's, and that is checked rather than asserted: running
  upstream's own `demo.sh` path on the same dataset gives 2751 scores that
  match GraFlag's element for element, `max |difference| = 0`, AUC 0.9480 both
  sides. `anograph_iscx` is byte-identical to upstream's ISCX, which is what
  makes the comparison meaningful.

  The heuristic's own number was 0.9652, over 1,097,070 per-edge scores. It is
  higher than the method's, and it was never AnoGraph's: 2751 is how many
  windows AnoGraph scores, and the count is the quickest way to tell the two
  apart in an old `evaluation.json`.

- **`anograph` also could not have finished.** `_NUM_BUCKETS=1024`, against
  the `32` upstream passes on all four of its datasets. The greedy peel in
  `getAnographDensity` is cubic in the bucket count -- about 10^9 operations
  at 32, roughly 3.5 * 10^13 at 1024 -- and a run at 1024 sat thirteen minutes
  on a worker without scoring one of its 2751 windows. At 32 the method takes
  4.2 s. This defect was only reachable once the first one was fixed: while
  the binary was exiting instantly on missing input, the scoring path never
  ran, so nothing could reveal that the declared parameters would not finish.

- **`generaldyg` published scores for the whole edge stream.** A local
  `dataset_all.py` -- upstream has no such class; it was ours -- re-ran the
  model over every edge, so most of what `results.json` carried was
  **training** data, and it drew a third independent `np.random.uniform`
  feature matrix so the scored inputs matched neither the training nor the
  test features. Scoring is now `eval_epoch(loader_test, ...)`: the same
  loader and the same code path the reported AUC already came from.
  `metadata.summary` records `scored_split: "test"`.

  The old AUC of 0.6702 in this file's history came from the whole-stream
  scores and is **not comparable** with what the method publishes now.

- **`generaldyg` then turned out never to have restored its checkpoint.** The
  snapshot was `model.state_dict().copy()`. `state_dict()` returns the live
  parameter tensors and `copy()` duplicates the dict around them, so the
  optimizer kept updating the very tensors the "best epoch" was supposedly
  held in, and `load_state_dict` restored the current weights onto
  themselves. The first run after the split fix showed it in one line: best
  AUC at epoch 1 (0.7806), epoch 2 at 0.6829, and the published score 0.6829
  -- the last epoch's -- under a log line saying the best checkpoint had been
  loaded.

  Like `anograph`'s second defect, this one was only reachable once the first
  was closed: while the final pass scored the whole stream, its AUC was not
  comparable with any per-epoch number, so a restore that did nothing looked
  exactly like one that worked. The snapshot now clones each tensor, and the
  re-run publishes 0.7806 -- epoch 1's -- with the method and the evaluator
  agreeing to four decimals. The per-epoch columns were renamed with it:
  `training.csv` carries `test_loss` and `test_auc`, and the summary key is
  `best_test_auc`, because upstream's `val` naming described a split
  GeneralDyG does not have.

- **`slade` published scores for the whole edge stream**, the same defect. The
  full-stream forward pass is kept, because SLADE is a streaming method whose
  memory state depends on having seen the earlier edges; only the *publish* is
  sliced, by `test_mask` -- 3,618 edges of 24,186.

  0.6345 and 0.6318, the two numbers this file used to carry for `slade`, are
  whole-stream AUCs and do not belong in a column with the 0.7588 it publishes
  now. Unlike `generaldyg`, there is no checkpoint defect hiding behind this
  one: `slade` reports its best epoch and publishes its last, so nothing was
  being selected on the test split to begin with.

- **`streamspot` took its ground truth from a constant.**
  `ATTACK_GRAPH_IDS = set(range(300, 400))` was transcribed from the paper
  rather than derived from the data, so a stream that did not match the
  paper's layout would have been scored against labels belonging to a
  different dataset. The ids are now derived -- `ATTACK_SCENARIOS` from the
  scenario table, `ATTACK_GRAPH_IDS` from `gid // 100` -- and
  `verify_paper_layout()` checks the stream against that layout and fails the
  run when it disagrees, instead of scoring on.

  The AUC is unchanged at 0.8942, which is the expected result: on the paper's
  own dataset the check passes and the labels it guards are the same ones. The
  run log now carries the evidence -- `89,770,902 edges over 600 graphs,
  matching the paper's layout` -- so the assumption is recorded as tested
  rather than made silently.

## Results that are still not what they look like

These remain, and are documented at the top of each method's own README.
Changing what a method computes to make its output look better is not
integration work, so none of them was silently patched.

- **`generaldyg`'s features are random, and that is upstream's design.**
  `train.py:106-107` builds the train and test datasets back to back, each
  drawing its own `np.random.uniform` matrix. Removing `dataset_all.py` took
  away the *third* draw that was ours; the two that remain are GeneralDyG's.
- **`dynwalk`** and **`addgraph`** are reimplementations. Their `.env` files
  say `INTEGRATION=reimplementation`; both previously cited a repository whose
  code never ran.
- **`bond_gaan`** does not run at all, on either device. It is the only
  detector in the family that passes the dense adjacency to
  `binary_cross_entropy` as a *target*, and the published BOND graphs contain
  duplicate edges, so that target contains 2s.

## What running them actually found

Each of these was found by running the method, and by nothing else. None of
the six test suites could have caught them: they are defects in how a method
calls upstream's code -- or in how much memory that takes -- and upstream's
code is not installed where the tests run.

- **`gady` had never executed a batch.** `Generator(...)` was constructed and
  never passed to `TGN(...)`, so the first batch died on `None.eval()`. Behind
  that sat five more, each unreachable until the one before it was fixed: a
  loss called with three arguments where it takes two and those two reversed,
  a call that left the model's positional encodings `None` for the next batch,
  an `eval_edge_prediction` call matching no signature upstream has, a module
  never moved to the device, and a published score that was inverted. The loop
  is now upstream's; see [its README](methods/gady/README.md).
- **`gady` allocated two Adam states for the same 440 million parameters.**
  The generator is a submodule of the discriminator, and upstream hands all of
  `discriminator.parameters()` to the discriminator's optimizer
  (`train.py:153`), so both optimizers held the same 440,074,746 of them.
  Torch 1.9's `zero_grad()` zeroes gradients rather than clearing them, so from
  the second batch on those parameters carry zero-valued grads rather than
  `None` -- and Adam allocates state for any parameter that has a gradient. The
  second `exp_avg`/`exp_avg_sq` pair is 3.3 GiB, and it exhausted an 11.6 GiB
  card inside `d_optimizer.step()` with 9.64 GiB allocated and 102 MiB
  requested. It had never moved a weight: under a zero gradient, with
  `weight_decay` at 0 and `amsgrad` off, every update it computed was exactly
  0. Holding the generator out of `d_optimizer` changes no number and returns
  the 3.3 GiB -- a full epoch now peaks at 11,475 MB on a 12,282 MB card.
- **`gady`'s edge features are one row short of the indices used to read
  them.** Upstream numbers edges from 1 and then sizes the feature matrix to
  one row per edge, so the last edge's index is exactly one past the end of the
  array two call sites index with it. Training never reaches it -- the train
  split stops at 70% and its neighbour finder is built from that split alone --
  while evaluation swaps in a finder built from the whole stream and the read
  goes off the end. It surfaced as `CUDA error: device-side assert triggered`
  against `memory.py:40`, a correct line on a correctly sized array: CUDA
  kernels report asynchronously, so the operation that synchronises and raises
  is the one *after* the out-of-bounds gather, and nothing in the traceback
  named the edge features. On `gady_email_dnc` it took a full epoch and most of
  an evaluation pass to arrive. The matrix is now sized from
  `max(edge_idxs) + 1`; its entries are zeros, so the added row changes no
  number the method computes -- it only makes the last edge indexable. The
  re-run cleared the pass that had failed and finished both epochs, and epoch
  1's losses came back identical to the failed run's -- discriminator 1.0330,
  generator 18.2169 -- which is what "changes no number" means in practice.
- **`bond_gaan` cannot run on the published BOND graphs**, for a reason in the
  data rather than in the integration -- four duplicate edges in `gen_100.pt`
  make the dense adjacency contain a 2.
- **DMGD and GAD-NR cannot run on a GPU** at the PyGOD commit the image pins.
  Both now declare `_GPU=-1` in their `.env` with the upstream traceback
  recorded beside it.
- **`generaldyg` held the same edge stream in memory three times.** The first
  run was killed with exit code 137 after training had finished, one forward
  pass from the end; `resources.csv` recorded 6.3 GB in the last sample. Its
  `main()` builds a train dataset, a test dataset and then a third covering
  the whole stream, and released none of them -- each one a dense float64
  padding of a 305 MB pickle. The training dataset and its loader are now
  freed before the final pass; the test loader stays, because it is the one
  being scored. Nothing reads the freed objects after that point, so no
  number changes -- and the re-run completed on the same host, with the same
  3 GB still on the `tmpfs`, peaking at 5,340 MB. The third dataset is gone
  outright now, taken away with the whole-stream scoring it existed to serve.

One result that looks like a regression and is not:

- **`dynwalk`** scores 0.8378 on `email_snapshot` here against 0.8526
  previously. Those are different datasets: the earlier number is
  `btc_alpha_snapshot`, which two separate runs reproduced to four decimals.

## What the matrix confirms about the integration

- **Every clone is pinned.** Each `.env` declares a 40-character `SOURCE_REF`
  and each Dockerfile clones with
  `git -C src checkout --detach ${SOURCE_REF}`, which the build log shows as
  the step's text. Before this, all eight clones floated on a default branch.

  Checking that left the pin itself unrecorded, though: `docker build` never
  echoes a `--build-arg` value, so a log holding only docker's output contains
  that step with `${SOURCE_REF}` unexpanded and names no commit anywhere.
  GraFlag now writes `[INFO] Upstream pinned at <40-hex>` and `[INFO] GraFlag
  libraries from <source>` into `build.log` as well as to the terminal.
  `gady`'s is the first build log on this cluster to open with them, and so far
  the only one -- 1 of 38. Every other run in this table predates the change
  and is pinned by its `.env` rather than by its own record.
- **Every build in this table shipped the whole share to the daemon.**
  `graflag-shared/.dockerignore` is what keeps `datasets/` and `experiments/`
  out of the build context, and it had never been on the cluster: `sync` copies
  a method directory, `sync --lib` copies a library, and nothing copies the
  root. The build succeeds either way -- docker prints the context size and
  moves on -- so every run here paid it without saying so. Docker's own
  accounting, on the same share either way: `Sending build context to Docker
  daemon  3.651GB` without the file against `804.4kB` with it.
  `build_method_image()` now probes the context root and writes
  a `[WARN]` naming the remedy into `build.log`, which is where a later reader
  of the experiment would look.
- **The libraries came from the share, not from PyPI.** 15 of the 38 builds
  ran the install step rather than hitting the layer cache, and all 15 log
  `Processing /tmp/graflag_libs/graflag_runner` -- the `GRAFLAG_LIBS=local`
  arm. None downloaded the published wheel. That is what makes
  `graflag sync --lib` mean anything, and it is why this matrix tests the
  working tree rather than the last release.
- **No rewrite can silently do nothing.** `sed -i` exits 0 when it matches
  nothing, which is how a patch disabled by upstream drift produces an image
  that builds and is wrong. anograph's and streamspot's are now `git apply` of
  a `patches/*.patch` against the pinned commit, so drift fails the build.
  generaldyg's CRLF strip cannot be a patch -- it touches 11 files and changes
  no content -- so it is bracketed instead: one `grep` asserting there is
  something to strip, another asserting nothing survived. taddy's was deleted,
  because at its pin no file in the clone had a CR and the step had only ever
  been a pass over the tree that reported success.
- **One image serves seventeen methods**, which is what made running the whole
  family possible on the available disk at all.
- **A method that dies before writing output keeps its service.** Four runs
  here failed with no `method_output.txt` at all; `cleanup_services()` kept
  their services, and `docker service ps` was the only surviving record of
  why. That rule is the reason those failures were diagnosable instead of
  being an empty directory with a `failed` status.

## Reproducing this

The runs are driven serially, recording each row as it completes and pruning
each method's image from every node afterwards -- five nodes on one disk fill
up fast (440 G, 96% full with one method's image resident). Nothing from this
matrix is therefore still in the registry, so re-running any row of it needs
`--build`. Four warnings, all learned here:

- Do not run `registry garbage-collect` on a live registry between runs. It
  deletes blob files while the registry still answers `HEAD` for them from
  cache, so the next `docker push` reports `Layer already exists`, skips the
  upload, and produces an image that the manager can run and a worker cannot
  pull (`No such image`).
- Run `graflag` from outside a directory holding a `.env`. A `.env` defining
  `MANAGER_IP` overrides `~/.config/graflag/config.env`.
- Send `.dockerignore` to the share once, with `graflag copy -s
  ./.dockerignore --dest .`. It is the only thing keeping `datasets/` and
  `experiments/` out of a build context rooted at `SHARED_DIR`, and no
  `sync` copies it. Every build behind this matrix ran without it, which
  costs time rather than correctness: docker's own accounting, same share
  either way, is `Sending build context to Docker daemon  3.651GB` without
  the file against `804.4kB` with it.
- `graflag sync` adds and overwrites; it does not delete. Its rsync carries no
  `--delete`, so a file removed from a method directory stays on the share
  after the sync that was supposed to reconcile them. Removing `dataset_all.py`
  from `generaldyg` left it on the manager, inside the build context, still
  importable -- a deleted file that a stale `COPY` or import could go on
  finding. Delete it on the share too. `test_copy_sources_exist` in
  `tests/test_methods.py` catches the matching half of this, a `COPY` left
  behind pointing at a file that is gone from the checkout.

- Sync the checkout to the share before a row runs. Builds happen on the
  manager against `SHARED_DIR`, so a fix that exists only in git is not a fix
  under test -- one gady re-run rebuilt the image and rebuilt the same
  unpatched script, and read as the fix not working. The driver now rsyncs
  `methods/` and `libs/` before the first row. `graflag copy -s ./methods
  --dest methods -r` is not the way to do it by hand: it copies the directory
  *into* the destination and leaves a `/shared/methods/methods` behind.
