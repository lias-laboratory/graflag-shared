"""What results.json says about the input is read before the detector runs.

PyGOD's DMGD replaces data.x in place: on inj_cora, 1,433 feature columns
become 2,708 during fit(). graflag_bond described the graph after fit(), so
the published bond_dmgd run on a three-node GPU cluster (2026-11-17) said
`num_features: 2708` for Cora. The description, labels included, is now
taken once before training and is the only thing save_results() sees.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    import torch
    from torch_geometric.data import Data
except ImportError:                                    # pragma: no cover
    torch = None

from graflag_bond import train


def cora_like():
    return Data(x=torch.ones(4, 3), y=torch.tensor([0, 2, 0, 1]),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]), num_nodes=4)


def fit_like_dmgd(method_name, data, exp_dir, writer, seed=0):
    """A detector that does to its input what PyGOD's DMGD does."""
    data.x = torch.ones(data.num_nodes, data.num_nodes)
    data.y = torch.zeros(data.num_nodes, dtype=torch.long)
    model = mock.Mock()
    model.decision_score_ = np.array([0.1, 0.9, 0.2, 0.8])
    return model


@unittest.skipIf(torch is None, "torch_geometric is not installed")
class DescribeGraph(unittest.TestCase):
    def test_sizes_and_binarized_labels(self):
        self.assertEqual(train.describe_graph(cora_like()), {
            "num_nodes": 4, "num_edges": 3, "num_features": 3,
            "ground_truth": [0, 1, 0, 1]})

    def test_plain_ints_for_json(self):
        record = train.describe_graph(cora_like())
        for key in ("num_nodes", "num_edges", "num_features"):
            self.assertIs(type(record[key]), int)


@unittest.skipIf(torch is None, "torch_geometric is not installed")
class TheInputIsDescribedBeforeFit(unittest.TestCase):
    def run_main(self):
        captured = {}

        def save(model, graph, exp_dir, writer, method_name, dataset_name, **kw):
            captured["graph"] = graph

        exp = Path(tempfile.mkdtemp())
        env = {"METHOD_NAME": "bond_dmgd", "DATA": "/shared/datasets/bond_inj_cora",
               "EXP": str(exp)}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(train, "load_graph_data", return_value=cora_like()), \
                mock.patch.object(train, "train_detector", side_effect=fit_like_dmgd), \
                mock.patch.object(train, "save_results", side_effect=save), \
                mock.patch.object(train, "ResultWriter"), \
                mock.patch.object(train, "info"):
            train.main()
        return captured["graph"]

    def test_features_are_counted_before_the_detector_changes_them(self):
        self.assertEqual(self.run_main()["num_features"], 3)

    def test_labels_are_read_before_the_detector_changes_them(self):
        self.assertEqual(self.run_main()["ground_truth"], [0, 1, 0, 1])

    def test_save_results_publishes_the_record(self):
        writer = mock.Mock()
        model = fit_like_dmgd("bond_dmgd", cora_like(), None, None)
        record = {"num_nodes": 4, "num_edges": 3, "num_features": 3,
                  "ground_truth": [0, 1, 0, 1]}
        with mock.patch.object(train, "info"):
            train.save_results(model, record, Path("/tmp"), writer, "bond_dmgd",
                               "bond_inj_cora", seed=42)
        info = writer.add_metadata.call_args.kwargs["summary"]["dataset_info"]
        self.assertEqual((info["num_nodes"], info["num_edges"], info["num_features"]),
                         (4, 3, 3))
        self.assertEqual(writer.save_scores.call_args.kwargs["ground_truth"],
                         [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
