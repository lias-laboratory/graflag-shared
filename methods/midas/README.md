# MIDAS

Microcluster-Based Detector of Anomalies in Edge Streams (AAAI 2020).

Upstream: https://github.com/Stream-AD/MIDAS

## What runs

Upstream's algorithm at the commit `SOURCE_REF` pins. MIDAS is **header-only**:
`src/NormalCore.hpp`, `src/RelationalCore.hpp` and `src/FilteringCore.hpp` are
the method, and this integration does not touch them. The one patched file,
`example/Demo.cpp`, is a driver in upstream too.

The method is **unsupervised** and fits nothing. It is a streaming detector:
each edge is scored against the count-min sketch state built by the edges
before it, in constant time and memory. There is no training phase, no epoch
and no train/test split, which is why `scored_split` is `all_edges`.

`_CORE` selects the published variant:

| `_CORE` | Variant | |
|---|---|---|
| `normal` | MIDAS | the AAAI 2020 method |
| `relational` | MIDAS-R | adds temporal and spatial relations |
| `filtering` | MIDAS-F | filters likely-anomalous edges out of the sketch before they corrupt it; the paper's best variant, and the default here |

Upstream selects between them by editing two commented lines in `Demo.cpp`.
Here it is a parameter, because all three are published variants and a
benchmark wants all three.

## The dataset needs no conversion at all

MIDAS reads a header-less CSV of `source,destination,timestamp`. That is
already GraFlag's `Data.csv`. Better than compatible:
**`datasets/anograph_darpa/Data.csv` is byte-identical to
`data/DARPA/darpa_processed.csv` in the MIDAS repository**, and
`Label.csv` is byte-identical to `darpa_ground_truth.csv` — 4,554,344 records
either way. GraFlag's copy arrived via AnoGraph, from the same authors.

So this method reads the mounted dataset directly, with no staging, no
rendering and no format shim.

## What the patch changes, and what it does not

`patches/graflag-driver.patch` parameterises `example/Demo.cpp`. Upstream's
version hardcodes the DARPA paths under `SOLUTION_DIR`, picks a core by
uncommenting a line, and has its **score-writing block commented out** — so it
can only benchmark the dataset the repository ships, and it prints an AUC
without emitting the scores that AUC was computed from. GraFlag publishes the
scores a method computed, so the driver has to hand them over.

Two behaviours change deliberately:

- **The seed is an argument, not `time(nullptr)`.** Upstream prints the clock
  seed "in case of reproduction"; recording it after the fact is not the same
  as choosing it, and two runs of one GraFlag experiment should be one
  experiment. `_SEED` carries it in.
- **A short or malformed data file is now an error.** `fopen()` succeeding says
  nothing about shape, and upstream discards `fscanf`'s return value by
  compiler flag (`-Wno-unused-result`), so a truncated file left the tail of
  the arrays uninitialised and scored whatever was in memory, silently.

Applied with `git apply --verbose`, which exits non-zero on drift. A `sed -i`
would exit 0 on no match and produce a binary that scores the repository's own
DARPA copy instead of the mounted dataset — the same number every run, for the
wrong reason.

## DARPA is 60% anomalous

`anograph_darpa` carries 2,737,209 anomalous edges out of 4,554,344 — **60.10%**.
That is the dataset, not a defect, but an AUC over a graph whose majority class
is the anomaly is not comparable with one over `bond_*` at 3–5%, and
`precision_at_k` will read high for structural reasons. `anograph_iscx`
(1,097,070 edges) is the saner default for a first run.

## No GPU

MIDAS is CPU-only by design — the paper's claim is constant time and memory per
edge. The image carries no CUDA base and no torch, which is why it is small.
`_GPU` is not declared; there is nothing for it to select.

## Verification

| Dataset | Gate 1 | Gate 2 | Gate 3 (`auc_roc`) | Gate 4 |
|---|---|---|---|---|
| `anograph_darpa` | 78 tests | completed, 2.9 s, 417 MB | **0.9844** | 0 failed, 1 warned, 3 passed |
| `anograph_iscx` | 78 tests | completed, 2.9 s, 138 MB | 0.3714 | 0 failed, 2 warned, 2 passed |

On DARPA the driver's own C++ `AUROC` prints **0.984358** and the evaluator
computes **0.9844** independently from the published vector. That agreement is
the check that matters: it says the scores in `results.json` are the ones the
method measured itself on, not a recomputation. 4,554,344 edges scored in
**527 ms** — the paper's constant-time-per-edge claim, on this hardware.

### Against upstream's own program: the same 4,554,344 scores

On a three-node GPU cluster (2026-11-17) upstream's `example/Demo.cpp` was built
at the pinned commit with two edits only -- seed 42 instead of `time(nullptr)`,
and upstream's own commented-out `Score.txt` block enabled
(`validation/reference/midas_upstream.sh`) -- and run on upstream's copy of
DARPA. That copy and the files GraFlag mounts have the same SHA-256, both
`darpa_processed.csv`/`Data.csv` and the labels. Upstream printed ROC-AUC
0.9844.

The scores then agree exactly, not approximately. GraFlag's driver writes each
float with `%.9g` (every line parses back to a float32 that prints to the same
line), upstream writes `%f`; each GraFlag score narrowed to float32 and printed
with `%f` is the same text as upstream's line for all 4,554,344 edges
(`validation/reference/midas_exact.py`), and `results.json` holds exactly the
values in the driver's score file. A naive numeric comparison of the two text
files reports 3,682,846 differing values, up to 496 apart. That is the two
formats and nothing else: `%f` rounds small scores to six decimals
(0.005952 for 0.00595238106), and `%.9g` keeps nine significant digits of
large ones (8.76332712e+11 for 876332711936, still the same float32). The AUC
from both vectors is 0.9843577267715715.

### ISCX scores below random, and that is the dataset, not a defect

`anograph_iscx` gives **0.3714**. Worth stating plainly rather than burying:
MIDAS does not publish on ISCX, and the two datasets differ in exactly the way
the method is sensitive to. MIDAS detects *microclusters* — bursts of activity
sharing a source, destination and tick — and ISCX has 69,745 distinct
timestamps in its first 400,000 records against DARPA's 11,624. Ticks that fine
spread a burst across many of them, so the signal MIDAS looks for is diluted.

The score distribution says the same thing: on ISCX the anomalous edges have a
much higher *mean* than the normal ones (10,384,862 against 343,094) but a much
lower *median* (309 against 4,489). A few edges score enormously, and the
majority of anomalies rank below the majority of normals -- which is what an
AUC below 0.5 is.

Run it on ISCX if you want that data point; do not read it as MIDAS's
performance. `_NUM_COLUMN` and `_THRESHOLD` were left at upstream's Demo values
for both runs, and no sweep was done.
