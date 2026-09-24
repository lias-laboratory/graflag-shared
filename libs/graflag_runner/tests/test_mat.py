"""The .mat <-> PyG bridge, which lives container-side.

Both directions are here rather than in graflag_data because both need torch
and torch_geometric. The manager that fetches datasets has neither, and
converting at fetch time failed there with ModuleNotFoundError: No module
named 'torch'. Methods convert inside their own image, where the ML stack
already exists -- and the .mat readers (HUGE-GAD, UNPrompt, AD-GCL) convert
nothing at all, because the stored form is what they already read.

Two of the .mat files these methods ship are datasets GraFlag already stores as
bond_* graphs, which makes this the rare converter with free ground truth:
Reddit.mat must come out as bond_reddit and Disney.mat as bond_disney.
"""

import glob
import tempfile
import unittest
from pathlib import Path


def _importable(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


HAVE = _importable("torch_geometric") and _importable("scipy")
DATASETS = Path(__file__).resolve().parents[3] / "datasets"


def _find_mat(filename):
    hits = glob.glob(f"/tmp/**/upstream/**/{filename}", recursive=True)
    return Path(hits[0]) if hits else None


def _edges(edge_index):
    src, dst = edge_index.tolist()
    return set(zip(src, dst))


def _undirected(edge_set):
    return edge_set | {(b, a) for a, b in edge_set}


@unittest.skipUnless(HAVE, "torch_geometric/scipy unavailable")
class GroundTruth(unittest.TestCase):
    def _compare(self, mat_name, bond_name):
        import torch
        from graflag_runner import read_mat

        mat = _find_mat(mat_name)
        if mat is None:
            self.skipTest(f"{mat_name} absent (upstream clone not present)")
        bond_file = DATASETS / bond_name / f"{bond_name}.pt"
        if not bond_file.is_file():
            self.skipTest(f"{bond_name} not hydrated")

        got = read_mat(mat)
        want = torch.load(bond_file, weights_only=False)

        self.assertEqual(got.num_nodes, want.num_nodes)
        self.assertTrue(torch.allclose(got.x, want.x, atol=1e-5), "features differ")
        self.assertTrue(torch.equal((got.y > 0).long(), (want.y > 0).long()),
                        "binary labels differ")
        # Equivalence as undirected graphs, not byte equality: the shipped
        # bond_* files disagree with each other about whether both directions
        # of an edge are stored. bond_reddit keeps all 168,016 entries;
        # bond_disney keeps 335 of its 670.
        self.assertEqual(_undirected(_edges(got.edge_index)),
                         _undirected(_edges(want.edge_index)))

    def test_reddit_mat_is_bond_reddit(self):
        self._compare("Reddit.mat", "bond_reddit")

    def test_disney_mat_is_bond_disney(self):
        self._compare("Disney.mat", "bond_disney")


@unittest.skipUnless(HAVE, "torch_geometric/scipy unavailable")
class RoundTrip(unittest.TestCase):
    def _toy(self):
        import torch
        from torch_geometric.data import Data
        return Data(x=torch.randn(5, 4),
                    edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
                    y=torch.tensor([0, 1, 0, 1, 0]))

    def test_write_then_read_preserves_the_graph(self):
        import torch
        from graflag_runner import read_mat, write_mat
        original = self._toy()
        out = Path(tempfile.mkdtemp()) / "toy.mat"
        write_mat(original, out)
        back = read_mat(out)
        self.assertEqual(back.num_nodes, original.num_nodes)
        self.assertEqual(back.x.shape, original.x.shape)
        self.assertEqual(_undirected(_edges(back.edge_index)),
                         _undirected(_edges(original.edge_index)))
        self.assertTrue(torch.equal(back.y, original.y))

    def test_written_adjacency_is_unweighted(self):
        """coo_matrix sums duplicates, so an already-symmetric edge_index plus
        symmetrize=True would put 2s in Network. The readers treat it as
        unweighted."""
        import scipy.io as sio
        import torch
        from torch_geometric.data import Data
        from graflag_runner import write_mat
        both = Data(x=torch.randn(3, 2),
                    edge_index=torch.tensor([[0, 1], [1, 0]]),
                    y=torch.tensor([0, 1, 0]))
        out = Path(tempfile.mkdtemp()) / "both.mat"
        write_mat(both, out)
        network = sio.loadmat(str(out))["Network"]
        self.assertEqual(set(network.tocoo().data.tolist()), {1.0})

    def test_wrong_keys_raise_naming_what_was_found(self):
        import scipy.io as sio
        from graflag_runner import read_mat
        out = Path(tempfile.mkdtemp()) / "wrong.mat"
        sio.savemat(str(out), {"adjacency": [[0, 1], [1, 0]]})
        with self.assertRaises(KeyError) as cm:
            read_mat(out)
        self.assertIn("adjacency", str(cm.exception))

    def test_missing_file_raises(self):
        from graflag_runner import read_mat
        with self.assertRaises(FileNotFoundError):
            read_mat("/nonexistent/nope.mat")
