"""GAAN, and only GAAN, gets the graph with repeated edges removed.

PyGOD's inj_cora repeats 6 of its 11,060 (u, v) pairs. GAAN's loss is a
binary cross-entropy against the dense adjacency, which to_dense_adj builds by
summing repeats, so the 2s fail PyGOD 1.1.0's own target check and bond_gaan
never ran. The other sixteen detectors ran, and verified, on the graph as it
is in the file; they must keep doing so.
"""

import unittest

try:
    import torch
    from torch_geometric.data import Data
except ImportError:                                    # pragma: no cover
    torch = None

from graflag_bond import train


@unittest.skipIf(torch is None, "torch_geometric is not installed")
class Deduplicate(unittest.TestCase):
    def graph(self, pairs, **extra):
        ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        return Data(edge_index=ei, num_nodes=4, **extra)

    def test_repeats_are_removed_first_occurrence_and_order_kept(self):
        g = self.graph([[0, 1], [2, 3], [0, 1], [1, 0], [2, 3], [3, 0]])
        self.assertEqual(train.deduplicate_edges(g), 2)
        self.assertEqual(g.edge_index.t().tolist(), [[0, 1], [2, 3], [1, 0], [3, 0]])

    def test_both_directions_of_an_undirected_edge_are_kept(self):
        g = self.graph([[0, 1], [1, 0]])
        self.assertEqual(train.deduplicate_edges(g), 0)
        self.assertEqual(g.edge_index.size(1), 2)

    def test_the_dense_adjacency_becomes_binary(self):
        from torch_geometric.utils import to_dense_adj
        g = self.graph([[0, 1], [0, 1], [1, 2]])
        self.assertEqual(to_dense_adj(g.edge_index, max_num_nodes=4).max().item(), 2.0)
        train.deduplicate_edges(g)
        self.assertEqual(to_dense_adj(g.edge_index, max_num_nodes=4).max().item(), 1.0)

    def test_edge_attributes_are_refused(self):
        g = self.graph([[0, 1], [0, 1]], edge_attr=torch.ones(2, 1))
        with self.assertRaises(ValueError):
            train.deduplicate_edges(g)


@unittest.skipIf(torch is None, "torch_geometric is not installed")
class OnlyGaan(unittest.TestCase):
    def graph(self):
        ei = torch.tensor([[0, 1], [0, 1], [1, 2]], dtype=torch.long).t().contiguous()
        return Data(edge_index=ei, num_nodes=3)

    def test_gaan_is_deduplicated_and_says_so(self):
        g = self.graph()
        changes = train.prepare_graph("gaan", g)
        self.assertEqual(changes, {"num_edges_in_file": 3, "duplicate_edges_removed": 1})
        self.assertEqual(g.edge_index.size(1), 2)

    def test_every_other_detector_gets_the_file_as_it_is(self):
        from graflag_bond.detectors import BondDetector
        for name in BondDetector.list_detectors() if hasattr(BondDetector, "list_detectors") \
                else ("dominant", "anomalydae", "cola", "done", "guide"):
            if name.lower() == "gaan":
                continue
            with self.subTest(detector=name):
                g = self.graph()
                self.assertEqual(train.prepare_graph(name, g), {})
                self.assertEqual(g.edge_index.size(1), 3)

    def test_the_set_is_exactly_gaan(self):
        self.assertEqual(train.DEDUPLICATE_EDGES, {"gaan"})


if __name__ == "__main__":
    unittest.main()
