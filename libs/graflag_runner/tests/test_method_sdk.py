"""Tests for the helpers integration scripts share (graflag_runner.method).

Each of these locks in a convention that ten scripts previously each had their
own version of, and where the versions disagreed a real run failed.
"""

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from graflag_runner import logging_utils, method


class Model:
    def __init__(self, hid_dim: int = 64, lr: float = 0.004, use_bias=True, tag=None):
        pass


class Params(unittest.TestCase):
    def test_prefix_is_stripped_and_lowercased(self):
        with mock.patch.dict(os.environ, {"_HID_DIM": "128"}, clear=True):
            self.assertEqual(method.params(), {"hid_dim": 128})

    def test_names_the_signature_rejects_are_dropped(self):
        with mock.patch.dict(os.environ, {"_HID_DIM": "128", "_NOPE": "1"}, clear=True):
            self.assertEqual(method.params(Model), {"hid_dim": 128})

    def test_a_dropped_name_is_warned_about_not_whispered(self):
        """Regression: this was a debug() line, so nothing reached the log.

        `graflag run -m bond_scan --params EPOCH=2` ran SCAN's declared
        defaults instead -- PyGOD's SCAN takes no `epoch` -- and said so
        nowhere, which left the run recorded as a reduced-epoch one it was
        not. apply_params() warned for the same event all along.
        """
        with mock.patch.dict(os.environ, {"_NOPE": "1"}, clear=True):
            # The runner names its logger after METHOD_NAME at import time,
            # so the object is the reliable handle, not a literal name.
            with self.assertLogs(logging_utils._logger, level="WARNING") as caught:
                method.params(Model)
        self.assertTrue(any("_NOPE" in line and "Model" in line
                            for line in caught.output), caught.output)

    def test_values_are_coerced_to_the_declared_type(self):
        with mock.patch.dict(os.environ, {"_HID_DIM": "128", "_LR": "1"}, clear=True):
            got = method.params(Model)
        self.assertIsInstance(got["hid_dim"], int)
        self.assertIsInstance(got["lr"], float)

    def test_booleans_and_none_are_recognised(self):
        with mock.patch.dict(os.environ, {"_USE_BIAS": "False", "_TAG": "none"},
                             clear=True):
            self.assertEqual(method.params(Model), {"use_bias": False, "tag": None})

    def test_a_declared_but_untyped_parameter_is_not_dropped(self):
        """Regression: the type map was built only from annotations and from
        non-None defaults, so a parameter declared as `backbone=None` was
        treated as one the callee does not accept and silently discarded --
        a .env setting it had no effect and said nothing."""
        with mock.patch.dict(os.environ, {"_TAG": "x"}, clear=True):
            self.assertEqual(method.params(Model), {"tag": "x"})

    def test_the_bare_underscore_bash_exports_is_ignored(self):
        with mock.patch.dict(os.environ, {"_": "/usr/bin/python3"}, clear=True):
            self.assertEqual(method.params(Model), {})

    def test_a_custom_converter_is_used(self):
        with mock.patch.dict(os.environ, {"_TAG": "x"}, clear=True):
            got = method.params(Model, convert=lambda n, v, t: f"seen:{v}")
        self.assertEqual(got, {"tag": "seen:x"})


class Device(unittest.TestCase):
    """`_GPU` is an index and -1 means CPU -- PyGOD's convention, which
    `graflag run --no-gpu` sets. Scripts that built f"cuda:{gpu}" straight
    from it asked torch for "cuda:-1" and died with Invalid device string."""

    def _device(self, environ, cuda_available):
        import torch
        with mock.patch.dict(os.environ, environ, clear=True), \
                mock.patch.object(torch.cuda, "is_available", return_value=cuda_available):
            return str(method.device())

    def test_minus_one_means_cpu_even_with_a_gpu_present(self):
        self.assertEqual(self._device({"_GPU": "-1"}, True), "cpu")

    def test_an_index_is_honoured(self):
        self.assertEqual(self._device({"_GPU": "1"}, True), "cuda:1")

    def test_no_gpu_present_falls_back_to_cpu(self):
        self.assertEqual(self._device({"_GPU": "0"}, False), "cpu")

    def test_an_unset_gpu_still_uses_one_when_present(self):
        self.assertEqual(self._device({}, True), "cuda:0")

    def test_an_unreadable_value_does_not_crash(self):
        self.assertEqual(self._device({"_GPU": "auto"}, False), "cpu")


class Paths(unittest.TestCase):
    def test_data_and_exp_are_exposed_with_their_names(self):
        env = {"DATA": "/shared/datasets/uci_snapshot/",
               "EXP": "/shared/experiments/exp__m__d__t/"}
        with mock.patch.dict(os.environ, env, clear=True):
            p = method.paths()
        self.assertEqual(p.dataset, "uci_snapshot")
        self.assertEqual(p.experiment, "exp__m__d__t")
        self.assertEqual(p.data, Path("/shared/datasets/uci_snapshot"))

    def test_a_missing_variable_is_an_error_not_a_default(self):
        """Regression: one script used os.environ.get('DATA', '.'), so running
        it without DATA read the working directory and reported the empty
        result as a successful run."""
        with mock.patch.dict(os.environ, {"EXP": "/tmp/e"}, clear=True):
            with self.assertRaises(RuntimeError) as raised:
                method.paths()
        self.assertIn("DATA", str(raised.exception))


class Upstream(unittest.TestCase):
    def setUp(self):
        self._path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self._path))
        self.root = Path(tempfile.mkdtemp())

    def test_the_clone_is_added_to_sys_path(self):
        (self.root / "src").mkdir()
        returned = method.upstream("src", root=self.root)
        self.assertEqual(returned, (self.root / "src").resolve())
        self.assertIn(str((self.root / "src").resolve()), sys.path)

    def test_several_directories_are_all_added(self):
        for sub in ("src/detection", "src/pytorch_DGCNN"):
            (self.root / sub).mkdir(parents=True)
        method.upstream("src/detection", "src/pytorch_DGCNN", root=self.root)
        for sub in ("src/detection", "src/pytorch_DGCNN"):
            self.assertIn(str((self.root / sub).resolve()), sys.path)

    def test_a_missing_clone_says_so_here(self):
        """Without this the failure surfaced as an ImportError several lines
        later, naming a module rather than the directory that is absent."""
        with self.assertRaises(FileNotFoundError) as raised:
            method.upstream("src", root=self.root)
        self.assertIn("src", str(raised.exception))


class SnapshotLoading(unittest.TestCase):
    """The snapshot loader was copy-pasted into five scripts. The labels it
    produces are the ground truth every one of those methods is scored on."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def _write(self, adjacency, test_neg, test_neg_id=None):
        np.save(self.dir / "acc_x.npy", np.asarray(adjacency))
        payload = {"test_neg": np.asarray(test_neg)}
        if test_neg_id is not None:
            payload["test_neg_id"] = np.asarray(test_neg_id)
        np.savez(self.dir / "split.npz", **payload)

    def test_edges_are_read_once_per_undirected_pair(self):
        self._write([[[0, 1], [1, 0]]], test_neg=np.empty((0, 2), dtype=int),
                    test_neg_id=np.empty(0, dtype=int))
        edges, labels = method.load_dataset(self.dir)
        self.assertEqual(list(edges.itertuples(index=False, name=None)), [(0, 1, 0)])
        self.assertEqual(list(labels), [0.0])

    def test_an_anomalous_edge_present_in_the_graph_is_labelled(self):
        self._write([[[0, 1], [1, 0]]], test_neg=[[0, 1]], test_neg_id=[0])
        edges, labels = method.load_dataset(self.dir)
        self.assertEqual(len(edges), 1)
        self.assertEqual(list(labels), [1.0])

    def test_an_anomalous_edge_absent_from_the_graph_is_appended(self):
        self._write([[[0, 1], [1, 0]]], test_neg=[[0, 2]], test_neg_id=[0])
        edges, labels = method.load_dataset(self.dir)
        self.assertEqual(len(edges), 2)
        self.assertEqual(list(labels), [0.0, 1.0])
        self.assertEqual(tuple(edges.iloc[1]), (0, 2, 0))

    def test_a_transposed_test_neg_is_accepted(self):
        """The split files store it (2, N) about as often as (N, 2)."""
        self._write([[[0, 1], [1, 0]]], test_neg=[[0], [2]], test_neg_id=[0])
        edges, labels = method.load_dataset(self.dir)
        self.assertEqual(tuple(edges.iloc[1]), (0, 2, 0))

    def test_the_static_graph_file_is_not_mistaken_for_the_snapshots(self):
        self._write([[[0, 1], [1, 0]]], test_neg=np.empty((0, 2), dtype=int))
        np.save(self.dir / "sta_acc_x.npy", np.zeros((1, 9, 9)))
        edges, _ = method.load_dataset(self.dir)
        self.assertEqual(len(edges), 1)

    def test_a_two_edge_split_is_not_transposed(self):
        """Regression: test_neg is stored as either (2, N) or (N, 2), and the
        loader transposed whenever shape[0] == 2. For a split holding exactly
        two edges both readings fit, so [[1,2],[0,2]] was read as [[1,0],[2,2]]
        -- src and dst swapped, and the ground truth with them."""
        self._write([[[0, 1, 1], [1, 0, 0], [1, 0, 0]]],
                    test_neg=[[1, 2], [0, 2]], test_neg_id=[0, 0])
        edges, labels = method.load_dataset(self.dir)
        anomalous = {tuple(r) for r, l in zip(edges.to_numpy(), labels) if l}
        self.assertEqual(anomalous, {(1, 2, 0), (0, 2, 0)})

    def test_labels_line_up_with_edges(self):
        adjacency = [[[0, 1, 1], [1, 0, 0], [1, 0, 0]]]
        self._write(adjacency, test_neg=[[1, 2], [0, 2]], test_neg_id=[0, 0])
        edges, labels = method.load_dataset(self.dir)
        self.assertEqual(len(edges), len(labels))
        anomalous = {tuple(r) for r, l in zip(edges.to_numpy(), labels) if l}
        self.assertEqual(anomalous, {(1, 2, 0), (0, 2, 0)})


class SplitTestEdges(unittest.TestCase):
    """The anomaly is test_neg, and an upstream link predictor says otherwise.

    strgnn published `graph_list[-1].label` straight from upstream, where
    dyn_links2subgraphs labels test_pos 1 and test_neg 0 -- real edge vs
    sampled non-edge. On these datasets test_neg *is* the injected anomaly, so
    that label is 1 for normal. Paired with a score that was P(class 1), both
    sides were inverted at once and auc_roc still looked right.
    """

    def split(self, **over):
        base = {
            "test_pos": np.array([[1, 2], [3, 4], [5, 6]]),
            "test_neg": np.array([[7, 8]]),
            "test_pos_id": np.array([0, 1, 1]),
            "test_neg_id": np.array([2]),
        }
        base.update(over)
        return base

    def test_test_neg_is_the_positive_class(self):
        _, _, truth = method.split_test_edges(self.split())
        self.assertEqual(truth, [0, 0, 0, 1])

    def test_edges_and_timestamps_follow_the_same_order(self):
        """pos then neg -- the order dyn_links2subgraphs concatenates them in,
        which is what lets a method zip its scores against this."""
        edges, stamps, truth = method.split_test_edges(self.split())
        self.assertEqual(edges, [[1, 2], [3, 4], [5, 6], [7, 8]])
        self.assertEqual(stamps, [0, 1, 1, 2])
        self.assertEqual(len(edges), len(stamps))
        self.assertEqual(len(edges), len(truth))

    def test_a_2xn_split_is_read_as_src_dst(self):
        split = self.split(test_pos=np.array([[1, 3, 5], [2, 4, 6]]))
        edges, _, _ = method.split_test_edges(split)
        self.assertEqual(edges[:3], [[1, 2], [3, 4], [5, 6]])

    def test_exactly_two_edges_are_not_transposed(self):
        """(2, 2) fits both readings; treating it as (2, N) swapped src and dst."""
        split = self.split(test_pos=np.array([[1, 2], [3, 4]]),
                           test_pos_id=np.array([0, 1]))
        edges, _, _ = method.split_test_edges(split)
        self.assertEqual(edges[:2], [[1, 2], [3, 4]])

    def test_a_split_without_ids_falls_back_to_the_snapshot_given(self):
        split = self.split()
        del split["test_pos_id"], split["test_neg_id"]
        _, stamps, _ = method.split_test_edges(split, default_snapshot=9)
        self.assertEqual(stamps, [9, 9, 9, 9])

    def test_a_split_without_ids_and_no_default_says_so(self):
        """Silently placing every test edge at snapshot 0 would score them
        against a graph they are not in."""
        split = self.split()
        del split["test_pos_id"]
        with self.assertRaises(KeyError) as raised:
            method.split_test_edges(split)
        self.assertIn("test_pos_id", str(raised.exception))


class SnapshotFiles(unittest.TestCase):
    """addgraph reads these files itself, so the discovery has to work the same
    for it as for the loader -- it is the one part it still shares."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_the_real_layout_resolves_to_acc_and_split(self):
        """What datasets/*_snapshot actually holds: acc_graph.npy (the snapshot
        series), sta_graph.npy (the static graph) and split.npz."""
        for name in ("acc_graph.npy", "sta_graph.npy"):
            np.save(self.dir / name, np.zeros((1, 2, 2)))
        np.savez(self.dir / "split.npz", test_neg=np.empty((0, 2)))
        graph, split = method.snapshot_files(self.dir)
        self.assertEqual((graph.name, split.name), ("acc_graph.npy", "split.npz"))

    def test_the_static_graph_is_never_returned_as_the_snapshots(self):
        """The guard only bites on the '*.npy' fallback, where 'sta_graph.npy'
        sorts before 'uci.npy' and would otherwise be picked -- scoring the
        method on a single static graph instead of the snapshot series."""
        for name in ("sta_graph.npy", "uci.npy"):
            np.save(self.dir / name, np.zeros((1, 2, 2)))
        np.savez(self.dir / "split.npz", test_neg=np.empty((0, 2)))
        self.assertEqual(method.snapshot_files(self.dir)[0].name, "uci.npy")

    def test_split_npz_wins_over_another_npz(self):
        np.save(self.dir / "acc_x.npy", np.zeros((1, 2, 2)))
        np.savez(self.dir / "aaa_other.npz", x=np.zeros(1))
        np.savez(self.dir / "split.npz", test_neg=np.empty((0, 2)))
        self.assertEqual(method.snapshot_files(self.dir)[1].name, "split.npz")

    def test_a_directory_with_neither_reports_neither(self):
        self.assertEqual(method.snapshot_files(self.dir), (None, None))

    def test_a_graph_named_neither_acc_nor_graph_is_still_found(self):
        """addgraph's loader fell through to a bare '*.npy'; keep that."""
        np.save(self.dir / "uci.npy", np.zeros((1, 2, 2)))
        self.assertEqual(method.snapshot_files(self.dir)[0].name, "uci.npy")


class CsvLoading(unittest.TestCase):
    def test_data_and_label_csv_are_preferred(self):
        d = Path(tempfile.mkdtemp())
        (d / "Data.csv").write_text("0,1,0\n1,2,1\n")
        (d / "Label.csv").write_text("0\n1\n")
        edges, labels = method.load_dataset(d)
        self.assertEqual(len(edges), 2)
        self.assertEqual(list(labels), [0, 1])

    def test_an_edge_list_is_read_with_comments_skipped(self):
        d = Path(tempfile.mkdtemp())
        (d / "edges.txt").write_text("% comment\n0 1 5\n1 2 6\n")
        edges, labels = method.load_dataset(d)
        self.assertEqual(list(edges["timestamp"]), [5, 6])
        self.assertEqual(list(labels), [0.0, 0.0])


class Seeding(unittest.TestCase):
    def test_the_same_seed_gives_the_same_draws(self):
        method.seed_all(7)
        first = np.random.rand(3).tolist()
        method.seed_all(7)
        self.assertEqual(np.random.rand(3).tolist(), first)


class InjectedParams(unittest.TestCase):
    """GRAFLAG_PARAMS is what separates a method parameter from a variable the
    shell happened to export. Without it, running a method by hand -- which is
    what the integration guide tells people to do -- recorded conda's
    _CE_CONDA and zsh's _P9K_TTY in results.json as method parameters.
    """

    NOISE = {"_CE_CONDA": "", "_CE_M": "", "_P9K_TTY": "/dev/pts/8"}
    REAL = {"_EPOCHS": "20", "_LR": "0.01"}

    def test_the_manifest_excludes_the_shells_own_variables(self):
        env = {**self.NOISE, **self.REAL,
               "GRAFLAG_PARAMS": "_EPOCHS,_LR"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(method.injected_params(), self.REAL)

    def test_params_drops_them_too(self):
        """Regression: `params()` returned ce_conda/p9k_tty as parameters."""
        env = {**self.NOISE, **self.REAL,
               "GRAFLAG_PARAMS": "_EPOCHS,_LR"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(method.params(), {"epochs": 20, "lr": 0.01})

    def test_without_a_manifest_every_underscore_var_is_a_parameter(self):
        """An image built before GRAFLAG_PARAMS existed still has to work."""
        with mock.patch.dict(os.environ, dict(self.REAL), clear=True):
            self.assertEqual(method.injected_params(), self.REAL)

    def test_an_empty_manifest_means_no_parameters(self):
        """Distinct from absent: a method with no `_FOO` in its .env declares
        an empty manifest, and must not then fall back to scanning."""
        env = {**self.NOISE, "GRAFLAG_PARAMS": ""}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(method.injected_params(), {})

    def test_the_shells_bare_underscore_is_never_a_parameter(self):
        with mock.patch.dict(os.environ, {"_": "/usr/bin/python3"}, clear=True):
            self.assertEqual(method.injected_params(), {})


class ApplyParams(unittest.TestCase):
    """generaldyg has no parameters of its own: it adopts upstream's argparse
    namespace, so the names and types come from whatever is already on it.
    """

    def upstream(self):
        return argparse.Namespace(hidden_dim=258, learning_rate=0.0001,
                                  n_epochs=200, ckpt_file="./src/", gpus=1)

    def test_values_are_coerced_to_the_type_already_there(self):
        env = {"_HIDDEN_DIM": "64", "_LEARNING_RATE": "0.01", "_N_EPOCHS": "2",
               "GRAFLAG_PARAMS": "_HIDDEN_DIM,_LEARNING_RATE,_N_EPOCHS"}
        config = self.upstream()
        with mock.patch.dict(os.environ, env, clear=True):
            applied = method.apply_params(config)

        self.assertEqual((config.hidden_dim, config.learning_rate, config.n_epochs),
                         (64, 0.01, 2))
        for name, value in (("hidden_dim", int), ("learning_rate", float),
                            ("n_epochs", int)):
            self.assertIsInstance(getattr(config, name), value)
        self.assertEqual(sorted(applied), ["hidden_dim", "learning_rate", "n_epochs"])

    def test_upstreams_other_defaults_are_untouched(self):
        env = {"_HIDDEN_DIM": "64", "GRAFLAG_PARAMS": "_HIDDEN_DIM"}
        config = self.upstream()
        with mock.patch.dict(os.environ, env, clear=True):
            method.apply_params(config)
        self.assertEqual(config.n_epochs, 200)
        self.assertEqual(config.ckpt_file, "./src/")

    def test_a_name_upstream_does_not_have_is_reported_not_applied(self):
        """Regression: with --pass-env-args, _GPU=0 arrived as `--gpu 0` and
        argparse's abbreviation matching set gpus=0 instead."""
        env = {"_GPU": "0", "GRAFLAG_PARAMS": "_GPU"}
        config = self.upstream()
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(method, "warning") as warned:
                applied = method.apply_params(config)

        self.assertEqual(config.gpus, 1, "_GPU must not reach --gpus")
        self.assertFalse(hasattr(config, "gpu"))
        self.assertEqual(applied, {})
        self.assertIn("_GPU", warned.call_args[0][0])

    def test_ignore_silences_a_parameter_read_elsewhere(self):
        """`_GPU` is read by device(), not by the method's config object."""
        env = {"_GPU": "-1", "GRAFLAG_PARAMS": "_GPU"}
        config = self.upstream()
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(method, "warning") as warned:
                method.apply_params(config, ignore={"gpu"})
        warned.assert_not_called()

    def test_the_shells_own_variables_are_not_applied(self):
        env = {"_CE_CONDA": "", "_HIDDEN_DIM": "64",
               "GRAFLAG_PARAMS": "_HIDDEN_DIM"}
        config = self.upstream()
        with mock.patch.dict(os.environ, env, clear=True):
            applied = method.apply_params(config)
        self.assertEqual(applied, {"hidden_dim": 64})


class PassEnvArgs(unittest.TestCase):
    """The --pass-env-args path has the same exposure: an empty _CE_CONDA
    becomes a bare `--ce_conda`, and argparse exits 2 on it.
    """

    def _args(self, env):
        from graflag_runner.runner import MethodRunner
        runner = MethodRunner.__new__(MethodRunner)
        runner.command = "python3 train.py"
        with mock.patch.dict(os.environ, env, clear=True):
            return runner._build_command_with_env_args()

    def test_shell_variables_do_not_become_flags(self):
        cmd = self._args({"_CE_CONDA": "", "_P9K_TTY": "/dev/pts/8",
                          "_EPOCHS": "20", "GRAFLAG_PARAMS": "_EPOCHS"})
        self.assertEqual(cmd, "python3 train.py --epochs 20")

    def test_declared_empty_values_are_still_bare_flags(self):
        """An empty value means a bare flag, not `--use_memory ''`.

        methods/gady used to spell its flag this way; it now writes `true`,
        but the convention is documented and still applies to any method
        passing --pass-env-args.
        """
        cmd = self._args({"_USE_MEMORY": "", "GRAFLAG_PARAMS": "_USE_MEMORY"})
        self.assertEqual(cmd, "python3 train.py --use_memory")


if __name__ == "__main__":
    unittest.main()


def _importable(name):
    """find_spec is not enough: dgl installs a Python package whose C++ half
    can be missing for the installed torch, so the import raises although the
    spec resolves."""
    try:
        __import__(name)
        return True
    except Exception:
        return False


@unittest.skipUnless(_importable("torch_geometric"), "torch_geometric unavailable")
class LoadAttributedGraph(unittest.TestCase):
    """The static counterpart to load_dataset.

    Every assertion here is a way a static integration previously failed open:
    scoring a graph it never found, or publishing test-split scores for a
    dataset that defines no test split.
    """

    def _graph(self, with_masks=True):
        import torch
        from torch_geometric.data import Data
        d = Data(x=torch.randn(6, 3),
                 edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
                 y=torch.tensor([0, 1, 0, 0, 1, 0]))
        if with_masks:
            d.train_mask = torch.tensor([1, 1, 0, 0, 0, 0]).bool()
            d.val_mask = torch.tensor([0, 0, 1, 0, 0, 0]).bool()
            d.test_mask = torch.tensor([0, 0, 0, 1, 1, 1]).bool()
        return d

    def _dir(self, name, data):
        import torch
        tmp = Path(tempfile.mkdtemp()) / name
        tmp.mkdir()
        torch.save(data, tmp / f"{name}.pt")
        return tmp

    def test_loads_the_file_named_after_the_directory(self):
        d = self._dir("bond_toy", self._graph())
        out = method.load_attributed_graph(d)
        self.assertEqual(out.num_nodes, 6)
        self.assertEqual(int(out.y.sum()), 2)

    def test_missing_directory_raises_rather_than_returning_empty(self):
        with self.assertRaises(FileNotFoundError):
            method.load_attributed_graph(Path(tempfile.mkdtemp()) / "nope")

    def test_directory_without_a_pt_raises(self):
        """Fail loudly: a method that scores nothing still writes a plausible
        results.json, so returning an empty graph here is undetectable."""
        empty = Path(tempfile.mkdtemp())
        (empty / "readme.txt").write_text("no graph here")
        with self.assertRaises(FileNotFoundError) as cm:
            method.load_attributed_graph(empty)
        self.assertIn("readme.txt", str(cm.exception))

    def test_require_masks_rejects_a_graph_with_no_test_split(self):
        d = self._dir("bond_nomask", self._graph(with_masks=False))
        method.load_attributed_graph(d)                      # fine unmasked
        with self.assertRaises(ValueError):
            method.load_attributed_graph(d, require_masks=True)

    def test_rejects_a_pt_that_is_not_a_pyg_data(self):
        import torch
        tmp = Path(tempfile.mkdtemp()) / "bogus"
        tmp.mkdir()
        torch.save({"not": "a graph"}, tmp / "bogus.pt")
        with self.assertRaises(ValueError):
            method.load_attributed_graph(tmp)


@unittest.skipUnless(_importable("torch_geometric"), "torch_geometric unavailable")
class SingleClassSplits(unittest.TestCase):
    """bond_weibo ships with every anomaly in train_mask and none in val or
    test. Training converges, scores get written, and the AUC is undefined
    rather than wrong -- so only an explicit check catches it."""

    def _dir(self, y, train, val, test):
        import torch
        from torch_geometric.data import Data
        n = len(y)
        d = Data(x=torch.randn(n, 3),
                 edge_index=torch.tensor([[0, 1], [1, 2]]),
                 y=torch.tensor(y))
        d.train_mask, d.val_mask, d.test_mask = (
            torch.tensor(m).bool() for m in (train, val, test))
        tmp = Path(tempfile.mkdtemp()) / "ds"
        tmp.mkdir()
        torch.save(d, tmp / "ds.pt")
        return tmp

    def test_test_split_without_anomalies_is_rejected(self):
        d = self._dir([1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])
        method.load_attributed_graph(d)              # unsupervised use is fine
        with self.assertRaises(ValueError) as cm:
            method.load_attributed_graph(d, require_masks=True)
        self.assertIn("single class", str(cm.exception))

    def test_both_classes_present_passes(self):
        d = self._dir([1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1])
        self.assertIsNotNone(method.load_attributed_graph(d, require_masks=True))
