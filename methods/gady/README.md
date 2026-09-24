# GADY

Unsupervised Anomaly Detection on Dynamic Graphs (WSDM 2024) —
<https://github.com/mufeng-74/GADY>

## Upstream's code runs here — but upstream's `train.py` does not

The Dockerfile clones GADY into `/app/src` at the commit `SOURCE_REF` pins, and
`train_graflag.py` imports from it: `model.tgn.TGN`, `modules.GAN.Generator`,
`utils.utils` (`EarlyStopMonitor`, `RandEdgeSampler`, `get_neighbor_finder`,
`get_data_settings`, `GenFGANLoss`, `DiscFGANLoss`) and
`utils.data_processing.get_data`. `data_loader.py` additionally runs
`src/prepare_data.py` and `src/preproc_new.py` as subprocesses. The model, the
generator, the losses, the samplers and the whole data pipeline are upstream's.

The training loop is reproduced here rather than imported, because upstream's
`train.py` cannot be run at this commit:

```python
# train.py:100-102 -- and no --alpha/--betaa/--gamma is ever declared
alpha = args.alpha
betaa = args.betaa
gamma = args.gamma
```

`argparse` produces no `alpha` attribute, so the module raises
`AttributeError` before it reaches any training code. The loop in
`run_gady_training` follows `train.py:244-272` step for step; the differences
that remain are listed under "Where this loop and upstream's differ".

## The loop this replaced had never executed a batch

Every one of these was in the integration, and each one is downstream of the
last, so only the first ever showed up in a log. Recorded because the shape
repeats: a method can look integrated, be tested by nothing, and be wrong in
six places at once.

| # | What it did | What happened |
|---|---|---|
| 1 | built `Generator(...)` and never passed it to `TGN(...)` | `discriminator.Generator` stayed `None`; the first batch died on `.eval()` — `AttributeError: 'NoneType' object has no attribute 'eval'` |
| 2 | `criterion_disc(pos_prob, neg_prob2, args.alpha)` | `DiscFGANLoss.forward(d_out_fake, d_out_real)` takes two arguments, and the two it was given were the wrong way round |
| 3 | called `compute_neg_edge_probabilities` without `next_V`/`next_R` | both functions end with `self.V, self.R = next_V, next_R` (`tgn.py:366,385`), so the model's positional encodings became `None` and the next batch raised on `discriminator.V + …` |
| 4 | `eval_edge_prediction(model=…, test_data=…, train_data=…, args=…, test_rand_sampler=…, partition_size=…, device=…)` | upstream's signature is `(model, negative_edge_sampler, data, n_neighbors, batch_size=200, vs=None, rs=None)` returning `(ap, auc, vs, rs)` — a `TypeError`, and an AP reported as an AUC if it had not been |
| 5 | never called `discriminator.to(dev)` | `TGN.__init__` moves the feature matrices itself but not its parameters, so a GPU run would have hit "Expected all tensors to be on the same device" |
| 6 | published `1 - P(edge)` as the score | `DiscFGANLoss` drives a real edge toward 0 and a generated one toward 1, so the raw output *is* the anomaly score — the published scores were inverted |

Three more were silent rather than fatal: the neighbour finder was never
switched to `full_ngh_finder` for evaluation, so the model scored test edges
without seeing test-time neighbours; `reset_VR()` was never called, so each
epoch began on the previous one's encodings; and the positional-feature
savepoint was re-read once per batch instead of once per partition.

## Where this loop and upstream's differ

- **Evaluation is `evaluate_test_split`, not `eval_edge_prediction`.** Same
  batching, same `model.V + vs[k]` accumulation, same per-batch mean of AP and
  AUC — it additionally keeps the per-edge probabilities, which `results.json`
  needs. Computing them in a second pass would score a different V/R state than
  the AUC printed beside them.
- **A batch with one class present is skipped for the metric, not fatal.**
  Upstream lets `roc_auc_score` raise.
- **`discriminator.memory.detach_memory()` after every batch.** Not upstream's;
  without it the second backward pass raises "trying to backward through the
  graph a second time".
- **`d_optimizer` is built without the generator's parameters.** Upstream passes
  it all of `discriminator.parameters()` (`train.py:153`), and the generator is a
  submodule of the discriminator, so the same 440,074,746 parameters sit in both
  optimizers. Torch 1.9's `zero_grad()` zeroes gradients instead of clearing
  them, so from the second batch on those parameters have zero-valued grads
  rather than `None`, and Adam -- which allocates state for any parameter with a
  gradient -- builds a second 3.3 GiB `exp_avg`/`exp_avg_sq` pair for them. That
  is what exhausted an 11.6 GiB card here, inside `d_optimizer.step()`, with
  9.64 GiB allocated and 102 MiB requested. The duplicate state never moved a
  weight: under a zero gradient `exp_avg` stays zero, `weight_decay` is 0 and
  `amsgrad` is off, so every update it computed was exactly 0. Holding the
  generator out of `d_optimizer` changes no number and frees the 3.3 GiB.
- **`_ALPHA`, `_BETAA` and `_GAMMA` are wired** to `GenFGANLoss(alpha_, beta_)`
  and `DiscFGANLoss(gamma_)`. Upstream names the same three values and uses
  none of them, in the lines that stop its `train.py` from running at all, so
  there is no upstream behaviour to be faithful to. The values in `.env` are
  GADY's published ones; `_BETAA=10` against the library default of 15 is the
  one place they disagree.
- **`--mode 1` (ablation) follows `train.py:274-289`**: random negatives from
  `RandEdgeSampler` and a BCE with real edges as class 0. It previously used
  `-log(pos_prob)`, which is not that ablation.

## The edge features are one row short

Upstream numbers edges from 1 — `prepare_data.py` appends
`idx_list = [int(x)+1 for x in range(np.size(all_data, 0))]` as the index
column — and then sizes the feature matrix to one row per edge,
`edge_features = np.zeros((data_full.shape[0], 172))`. The last edge's index is
exactly one past the end of the array two different call sites index with it:
`tgn.py:421` for the message it builds, and `embedding_module.py:131,205` for
every neighbour the finder returns.

Training never reaches it. The train split stops at 70%, and `train_ngh_finder`
is built from that split alone, so no index it produces exceeds the array.
Evaluation swaps in `full_ngh_finder`, which is built from the whole stream, and
the read goes off the end.

It surfaced as `RuntimeError: CUDA error: device-side assert triggered` pointing
at `memory.py:40`, `self.memory[node_idxs, :]` — a line that is correct, on an
array that is correctly sized. CUDA kernels report asynchronously: the
out-of-bounds gather launches, returns, and the next indexing operation is the
one that synchronises and raises. Nothing in the traceback named the edge
features. On `gady_email_dnc` it took a full epoch and most of an evaluation
pass to arrive, 38,544 edges against 38,544 rows.

`train_graflag.py` now sizes the matrix from the indices that will be used,
`max(full_data.edge_idxs) + 1`, and logs when it pads. The features are all
zeros, so the added row changes no number this method produces; it only makes
the last edge indexable. The check is written as a comparison rather than a
fixed `+ 1` so a dataset whose indices start at 0 gets no padding and no
message.

The re-run logged `[INFO] Edge features sized 38544 for edge indices up to
38544; padding to 38545 rows`, cleared the evaluation pass that had raised, and
finished both epochs — `auc_roc` 0.9770 from `graflag evaluate`, the first
number this method has produced here. Epoch 1's losses came back identical to
the run that died: discriminator 1.0330, generator 18.2169. That equality is
the evidence for "changes no number" — the padded row is never read during
training, and the split that does read it is the one that used to crash.

## Parameters that do nothing

`INERT_PARAMS` in `train_graflag.py` is the list, and setting any of them off
its default prints a `[WARN]` naming it:

| Parameter | Why it is inert |
|---|---|
| `_LR` | GADY trains two optimizers, on `_LR_D` and `_LR_G`. Upstream declares `--lr` (`train.py:33`) and never reads it either. |

It stays in the `.env` and in `results.json` deliberately: dropping a key would
make a `--params LR=...` sweep look like it worked while every run came back
identical.

`_LR_G`, `_LR_D` and `_BETAA` were on this list until the adversarial loop was
restored. They are live now.

## Booleans are spelled out

`_USE_MEMORY=true`, not `_USE_MEMORY=`. The empty spelling used to work through
`--pass-env-args`, which turns an empty value into a bare `--use_memory` flag
that argparse read as `True`. `graflag_runner.params()` reads an empty string
as `False`, so the old spelling would have turned memory off without saying so.
Every flag in this `.env`, including the commented-out ones, now takes an
explicit `true`.

`_USE_MEMORY=false` is untested. Every `memory` call in the loop is now guarded
by it, but nothing has run that path.

## Parameters

`_GPU=-1` selects CPU. Note that upstream's `preproc_new.py` is handed the same
`-1` as a subprocess argument and may not honour it; the script warns when it
passes one.

`_SEED` seeds `random`, numpy and torch before training. It does **not** reach
the anomaly injection, which happens earlier in the `src/prepare_data.py`
subprocess under upstream's own seeding.

## Datasets

`DATASET_CONFIG` in `data_loader.py` maps three datasets; the `gady_` prefix is
stripped from the mounted folder name.

| Dataset | Raw file | Format |
|---|---|---|
| `gady_uci` | `uci` | whitespace `src dst label ts` |
| `gady_btc_alpha` | `soc-sign-bitcoinalpha.csv` | `src,dst,rating,ts` |
| `gady_email_dnc` | `email-dnc.edges` | `src,dst,ts` |

Like `taddy`, **this method injects its own anomalies** — `prepare_data.py`
generates the test split at `_ANOMALY_PER` — so it does not use the
`*_snapshot` datasets that carry pre-built labels.

Upstream's `get_data_settings` (`utils/utils.py:5`) knows exactly four names —
`uci`, `btc_otc`, `btc_alpha`, `email_dnc` — and falls off the end of its
`if/elif` chain for anything else, raising `UnboundLocalError`. A fourth
dataset needs an entry there, not just one in `DATASET_CONFIG`.

## Scores

`EDGE_STREAM_ANOMALY_SCORES`. The score is the discriminator's output from
`compute_edge_probabilities`, unmodified: `DiscFGANLoss` trains a real edge
toward 0 and a generated one toward 1, so it already rises with anomalousness,
the same direction as the injected-anomaly labels and as upstream's own
`roc_auc_score(true_label, pos_prob)`. **No inversion is applied and none is
needed.** Scores cover the test split only.

The published scores come from the checkpoint the early stopper selects on —
the epoch with the best test AP, across every run — so they and
`metadata.summary.training_info.best_auc` describe the same model. `best_auc`
is that epoch's AUC, not the maximum over epochs: a maximum taken from a
different epoch than the scores would make `results.json` and the summary
disagree.

The AUC `graflag evaluate` reports is still not identical to `best_auc`.
Upstream's metric is the **mean of per-batch AUCs**; the evaluator computes one
AUC over all the scores at once. They answer different questions and both are
correct. `best_auc` is in the summary for comparison with the paper; the
evaluator's number is the one to compare across methods.

The last evaluation batch is dropped when `pos_features` holds one entry fewer
than there are test batches, which is how upstream's own loop is written
(`range(num_test_batch - 1)`). The scores then cover every test edge except
that final partial batch.

## Logs

`setup_logging` writes to `log/` inside the container, which is not on the NFS
share — that file does not survive the run. `method_output.txt` in the
experiment directory is the record that does.
