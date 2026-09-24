"""
==============================================================================
GraFlag Method Integration Template
==============================================================================

Copy this directory to start a new method:

    cp -r methods/example methods/your_method

Then rewrite `YourModel` and point `.env` at your dataset. Everything else --
reading parameters, locating the dataset, choosing a device, seeding, timing,
measuring memory -- is provided by `graflag_runner`, so this file stays about
your method and nothing else.

What GraFlag gives you (see graflag_runner/method.py):

    params(...)      the `_FOO` variables from .env, typed and named for you
    paths()          the DATA and EXP directories for this run
    load_dataset()   (edges, labels) from any of the four dataset layouts
    device()         the torch device, honouring _GPU=-1 for CPU
    seed_all(seed)   seeds random, numpy and torch together
    upstream(...)    puts a cloned upstream repository on sys.path
    ResultWriter     writes results.json in the standard schema

What you must NOT do here (the runner already does it, and its numbers win):

    time.time() around main(), psutil memory sampling, torch.cuda memory
    stats. `graflag_runner` measures all three from outside the method and
    merges them into metadata; anything you record yourself is preserved
    under `method_reported_*` but is not what gets reported.

==============================================================================
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from graflag_runner import ResultWriter, info, load_dataset, params, paths, seed_all


# ==============================================================================
# Step 1: Your method
# ==============================================================================
# The constructor's signature IS the parameter contract. `params(YourModel)`
# reads `_LEARNING_RATE=0.001` from .env, lowercases it, coerces it with the
# annotation below, and drops anything this constructor does not accept -- so
# adding a parameter is one line here plus one line in .env, and a stale .env
# key can no longer reach the model as a surprise keyword argument.

class YourModel:
    """Replace this with your detector.

    The contract is narrow on purpose: learn from the graph, then score every
    edge, higher meaning more anomalous.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # If your model uses torch, take the device from the SDK rather than
        # building one from _GPU yourself -- it is the single place that knows
        # -1 means CPU, which is what `graflag run --no-gpu` sets. Three
        # methods built f"cuda:{gpu}" by hand and died on "Invalid device
        # string: 'cuda:-1'":
        #
        #     from graflag_runner import device
        #     self.device = device()

    def train(self, edges, writer=None):
        """Fit the model.

        Args:
            edges: DataFrame with columns [src, dst, timestamp]
            writer: ResultWriter, for per-epoch metrics
        """
        info(f"[INFO] Training for {self.epochs} epochs on {len(edges)} edges")

        for epoch in range(1, self.epochs + 1):
            # ----- your training step here -----
            loss = 1.0 / epoch  # placeholder

            # spot() appends a row to training.csv, which `graflag evaluate`
            # turns into training_curves.png. The schema is locked after the
            # first call per key, so pass the same fields every epoch.
            if writer is not None:
                writer.spot("training", epoch=epoch, loss=loss)

            if epoch % 10 == 0:
                info(f"[INFO] Epoch {epoch}/{self.epochs}  loss={loss:.6f}")

    def predict(self, edges):
        """Return one anomaly score per edge, higher = more anomalous."""
        # ----- your scoring here -----
        return np.random.rand(len(edges))  # placeholder


# ==============================================================================
# Step 2: Wire it to GraFlag
# ==============================================================================

def main():
    run = paths()                       # DATA and EXP, or a clear error
    config = params()                   # every _FOO, for the record
    seed_all(int(config.get("seed", 42)))

    info(f"[INFO] Dataset {run.dataset} from {run.data}")
    edges, labels = load_dataset()

    num_nodes = len(set(edges["src"]) | set(edges["dst"]))
    num_anomalies = int(labels.sum())
    info(f"[INFO] {len(edges)} edges, {num_nodes} nodes, {num_anomalies} anomalies")

    # Scores must come from the test split: evaluation needs both classes
    # present, and a run that scores only normal edges produces a results.json
    # the evaluator cannot score.
    model = YourModel(**params(YourModel))
    writer = ResultWriter()
    model.train(edges, writer=writer)

    scores = np.asarray(model.predict(edges), dtype=float)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    auc = None
    if len(labels) == len(scores) and 0 < labels.sum() < len(labels):
        auc = float(roc_auc_score(labels, scores))
        info(f"[INFO] AUC {auc:.4f}")
    else:
        info("[WARN] Not both classes present; skipping AUC")

    writer.save_scores(
        # One of the types in graflag_runner/results.py: {NODE,EDGE,GRAPH}_-,
        # TEMPORAL_- and -_STREAM_ANOMALY_SCORES.
        result_type="EDGE_STREAM_ANOMALY_SCORES",
        scores=scores.tolist(),
        edges=edges[["src", "dst"]].values.tolist(),
        timestamps=edges["timestamp"].tolist(),
        ground_truth=labels.tolist(),
    )

    writer.add_metadata(
        method_name="example",          # must match METHOD_NAME in .env
        dataset=run.dataset,
        exp_name=run.experiment,
        method_parameters=config,
        summary={
            "task": "edge_anomaly_detection",
            "dataset_info": {
                "num_edges": len(edges),
                "num_nodes": num_nodes,
                "num_anomalies": num_anomalies,
            },
            "results": {"auc": auc},
        },
    )

    info(f"[OK] Results written to {writer.finalize()}")


if __name__ == "__main__":
    main()
