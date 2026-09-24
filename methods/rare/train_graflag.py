"""
GraFlag integration for RARE.
RARE: Rarity-based Anomaly Detection in Graphs via Order-Embedding Subgraph
Mining (WISE 2026).

RARE scores the NODES of a static graph by how rare the subgraph patterns
anchored at them are. A matcher pretrained on synthetic graphs (DSAN, an
order-embedding GNN) estimates how often a pattern occurs in the graph; nodes
whose neighbourhoods are outliers in that embedding space seed a beam search
that grows the rarest patterns, and a node's score comes from the rarest
pattern verified at it. Nothing is trained on the target graph, and every
node is scored.

This script runs upstream's pipeline under upstream's configuration for the
dataset, and publishes the per-node vector upstream's own evaluation scores.
"""

import dataclasses
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from graflag_runner import ResultWriter, device, info, params, paths, upstream

# The RARE clone. upstream() anchors on this file's directory and raises if
# the checkout is missing, instead of failing later with an ImportError.
SRC = upstream("src")

import rare.evaluation.metrics as rare_metrics                    # noqa: E402
import rare.pipeline as rare_pipeline                             # noqa: E402
import rare.utils.device as rare_device                           # noqa: E402
from rare.callbacks.logging_cb import LoggingCallback             # noqa: E402
from rare.config.loader import load_config, merge_cli_overrides   # noqa: E402
from rare.utils.serialization import save_json                    # noqa: E402


@dataclass
class Config:
    """The parameters this method accepts, and their defaults.

    The first four are declared in the .env. The rest each override one field
    of the upstream configuration and stay None unless passed with --params:
    their values differ per dataset, so a default in the .env would override
    all three upstream configurations at once.
    """

    config: str = "auto"
    task: str = "struct-anomaly"
    seed: int = 42
    gpu: int = 0                # GPU index; -1 means CPU
    max_freq: Optional[float] = None
    outlier_max_freq: Optional[float] = None
    max_steps: Optional[int] = None
    n_beams: Optional[int] = None
    n_neighborhoods: Optional[int] = None


# Config field -> the upstream key it sets. Applied through upstream's own
# merge_cli_overrides(), which is what `python -m rare --config ...
# search.max_freq=20` does, so upstream's schema decides the types.
UPSTREAM_KEYS = {
    "task": "dataset.task",
    "seed": "seed",
    "max_freq": "search.max_freq",
    "outlier_max_freq": "search.outlier_max_freq",
    "max_steps": "search.max_steps",
    "n_beams": "search.n_beams",
    "n_neighborhoods": "sampling.n_neighborhoods",
}

# GraFlag dataset -> (upstream's name for it, the configuration upstream's
# README lists as its benchmark run). The .pt on the share is the archive
# upstream's loader downloads (github.com/pygod-team/data), stored under
# GraFlag's name.
DATASETS = {
    "bond_inj_cora": ("inj_cora", "cora_order_glass_fast"),
    "bond_inj_amazon": ("inj_amazon", "amazon_order_glass_canon"),
    "bond_inj_flickr": ("inj_flickr", "flickr_order_glass_v2"),
}


def stage_dataset(run, name):
    """A directory in which upstream's loader finds the dataset as <name>.pt.

    load_pygod_dataset() reads <cache_dir>/<name>.pt and downloads the file
    when it is absent. Linking the share's copy in under that name means a
    missing dataset stops the run here instead of becoming a download.
    """
    source = run.data / f"{run.dataset}.pt"
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} not found. Hydrate it with "
            f"`graflag-data fetch {run.dataset}`.")
    staged = Path(tempfile.mkdtemp(prefix="rare_data_"))
    (staged / f"{name}.pt").symlink_to(source)
    return staged


def build_config(cfg, run):
    """Upstream's configuration for this run, pointed at the share."""
    if run.dataset not in DATASETS:
        raise ValueError(
            f"No upstream configuration for dataset {run.dataset!r}. "
            f"Supported: {', '.join(DATASETS)}")
    name, benchmark = DATASETS[run.dataset]

    config_name = benchmark if cfg.config == "auto" else cfg.config
    config_file = SRC / "configs" / f"{config_name}.yaml"
    if not config_file.is_file():
        available = sorted(p.stem for p in (SRC / "configs").glob("*.yaml"))
        raise FileNotFoundError(
            f"Upstream has no configuration {config_file.name}. "
            f"Available: {', '.join(available)}")
    config = load_config(config_file)

    overrides = {UPSTREAM_KEYS[key]: str(value)
                 for key, value in asdict(cfg).items()
                 if key in UPSTREAM_KEYS and value is not None}
    config = merge_cli_overrides(config, overrides)

    if config.training.enabled:
        raise ValueError(
            f"{config_file.name} trains the matcher. This integration runs "
            f"detection with the pretrained one; train with upstream's "
            f"`python -m rare --train` instead.")

    # The mounted dataset decides the data. A configuration naming another one
    # would otherwise send upstream's loader off to download that one.
    if config.dataset.name != name:
        info(f"[INFO] {config_file.name} is written for "
             f"{config.dataset.name}; running it on {name}")
    config.dataset.name = name
    config.dataset.cache_dir = str(stage_dataset(run, name))

    # Upstream resolves the checkpoint against its working directory, which
    # here is not the clone.
    config.model.model_path = str(SRC / config.model.model_path)
    return config_name, config


class ScoreCapture:
    """The per-node vectors behind RARE's own AUROC.

    get_stat_results() turns the verified patterns into one score per node --
    1 - beam.score at each verified anchor and 0 at every other node, or an
    IsolationForest score under search.structural_scoring -- passes that
    vector and the labels to roc_auc_score, and returns only the metrics.
    Recording the arguments of that call publishes the vector upstream
    evaluated. Rebuilding it from the beams here would be a second copy of
    upstream's scoring rule, free to drift from the one behind its AUROC.
    """

    def __init__(self):
        self.last = None
        self._vectors = None
        self._roc_auc_score = rare_metrics.roc_auc_score
        self._get_stat_results = rare_pipeline.get_stat_results
        rare_metrics.roc_auc_score = self._record_vectors
        rare_pipeline.get_stat_results = self._record_call

    def _record_vectors(self, y_true, y_score, *args, **kwargs):
        self._vectors = (list(y_true), list(y_score))
        return self._roc_auc_score(y_true, y_score, *args, **kwargs)

    def _record_call(self, *args, **kwargs):
        self._vectors = None
        stats = self._get_stat_results(*args, **kwargs)
        if self._vectors is None:
            raise RuntimeError(
                "RARE's get_stat_results() no longer hands its scores to "
                "roc_auc_score, so they cannot be recorded; this integration "
                "needs updating for the pinned SOURCE_REF")
        self.last = (stats, *self._vectors)
        return stats

    def final(self, results):
        """(labels, scores) of the call that produced results['stat_results']."""
        if self.last is None or self.last[0] is not results["stat_results"]:
            raise RuntimeError(
                "RARE's reported metrics were not produced by the last "
                "recorded get_stat_results() call")
        _, y_true, y_score = self.last
        return y_true, y_score


def restore_strictly():
    """Refuse a matcher checkpoint that does not fit the configured model.

    Upstream restores it with load_state_dict(strict=False), which neither
    restores nor reports a key the model and the checkpoint do not share: a
    mismatched checkpoint leaves the matcher on its random initialisation and
    the run still completes. At the pinned commit all 45 keys match for the
    three benchmark configurations; this keeps it that way.
    """
    load = rare_pipeline.RAREPipeline._load_or_train_model

    def checked(self):
        model = load(self)
        path = self.config.model.model_path
        saved = set(torch.load(path, map_location="cpu", weights_only=True))
        wanted = set(model.state_dict())
        if saved != wanted:
            raise RuntimeError(
                f"{path} does not fit the configured matcher: missing "
                f"{sorted(wanted - saved)[:5]}, unexpected "
                f"{sorted(saved - wanted)[:5]}")
        return model

    rare_pipeline.RAREPipeline._load_or_train_model = checked


def refuse_a_constant_ranking(results, y_score, config):
    """Stop when RARE verified nothing, instead of publishing all zeros.

    A node scores only if a pattern anchored at it was verified, i.e. stayed
    below T_F. With none verified every node scores 0: the run still exits
    0, and the evaluator reports AUC 0.5 -- chance by construction, which
    reads as a measurement of a method that in fact ranked nothing. On Cora
    the benchmark configuration ends this way for some seeds, on the GPU as
    on the CPU; see README.md.
    """
    if len(set(y_score)) > 1:
        return
    raise RuntimeError(
        f"RARE verified {results['verified_count']} pattern(s) from "
        f"{results['starting_nodes_count']} starting nodes, so every node "
        f"scores {y_score[0]} and there is nothing to rank. No candidate "
        f"stayed below T_F (search.max_freq={config.search.max_freq}) on "
        f"{config.device.device}. rare_results.json holds upstream's record.")


def publish(results, y_true, y_score, run, cfg, config_name, config):
    """Write results.json."""
    stats = results["stat_results"]

    writer = ResultWriter()
    writer.save_scores(
        result_type="NODE_ANOMALY_SCORES",
        scores=[float(s) for s in y_score],
        ground_truth=[int(t) for t in y_true],
        node_ids=list(range(len(y_score))),
    )
    writer.add_metadata(
        exp_name=run.experiment,
        method_name="RARE",
        dataset=run.dataset,
        method_parameters={
            **{k: v for k, v in asdict(cfg).items() if v is not None},
            "upstream_config": f"configs/{config_name}.yaml",
            "resolved": dataclasses.asdict(config),
        },
        summary={
            "description": "RARE, upstream pipeline with the pretrained "
                           "DSAN matcher",
            "task": "node_anomaly_detection",
            "dataset_info": {
                "name": run.dataset,
                "upstream_name": config.dataset.name,
                # Transductive and unsupervised: nothing is fitted on this
                # graph, so every node is scored. See README.md.
                "scored_split": "all",
                "scored_samples": len(y_score),
                "labels": config.dataset.task,
                "num_anomalies": int(sum(y_true)),
            },
            "detection_info": {
                "rare_auc_roc": stats["auroc"],
                "rare_ap": stats["ap"],
                "rare_precision": stats["precision"],
                "rare_recall": stats["recall"],
                "rare_f1": stats["f1"],
                "verified_patterns": results["verified_count"],
                "starting_nodes": results["starting_nodes_count"],
                "search_time": results["total_time"],
            },
        },
    )
    info(f"[OK] Results written to {writer.finalize()}")


def main():
    run = paths()
    cfg = Config(**params(Config))
    config_name, config = build_config(cfg, run)

    dev = device(cfg.gpu)
    config.device.device = str(dev)
    # Upstream's model-based detector and batching helpers call get_device(),
    # which answers cuda whenever a GPU is visible, not the configured device.
    # Seeding its cache makes _GPU=-1 mean CPU for them too.
    rare_device._device_cache = dev

    # Upstream caches embeddings and starting nodes under /tmp/rare_savings,
    # keyed by dataset name and parameters but not by the graph itself. A new
    # container starts with it empty; a fresh directory keeps that true
    # wherever this script runs, so no run reuses another's detection.
    rare_pipeline.CACHE_DIR = Path(tempfile.mkdtemp(prefix="rare_cache_"))

    info(f"[INFO] RARE on {run.dataset} with configs/{config_name}.yaml: "
         f"{asdict(cfg)}")

    capture = ScoreCapture()
    restore_strictly()

    pipeline = rare_pipeline.RAREPipeline(config)
    pipeline.add_callback(LoggingCallback())
    # Upstream's own switch for a caller that wants the results rather than
    # the report (its grid study sets it too): it skips anomalies.json, the
    # diagnostic plots and the pattern renderings, which would otherwise be
    # timed as part of detection.
    pipeline._search_only = True
    results = pipeline.run()

    y_true, y_score = capture.final(results)
    info(f"[INFO] RARE's own AUROC {results['stat_results']['auroc']:.4f}, "
         f"AP {results['stat_results']['ap']:.4f} over {len(y_score)} nodes")

    # Upstream's record of the detection: every verified pattern with its
    # anchor, nodes, frequency and score, plus its metrics. Written before
    # anything can refuse the run, so a refused run is diagnosable from disk.
    save_json(results, run.exp / "rare_results.json")

    refuse_a_constant_ranking(results, y_score, config)
    publish(results, y_true, y_score, run, cfg, config_name, config)


if __name__ == "__main__":
    main()
