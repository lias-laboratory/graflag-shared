"""Helpers for integration scripts, so each one stops reinventing them.

Ten `train_graflag.py` scripts were each solving the same five problems, and
each divergence cost a run:

* **Device selection** had six variants. Three honoured ``_GPU=-1`` (the CPU
  sentinel a ``--no-gpu`` run sets), three did not -- one hardcoded ``cuda:0``,
  one ignored ``_GPU`` entirely, and one produced the invalid ``"cuda:-1"``.
* **Parameters** were read three ways: an argparse parser per script, a
  hand-written env-name mapping, and the upstream repo's own parser.
* **`sys.path` into the clone** was built four ways, two of them hardcoding
  ``/app/...``, which ties the script to one Dockerfile's WORKDIR.
* **Dataset loading** was one function copy-pasted into five scripts, so a fix
  to one left four wrong.
* **Resource tracking** was ~30 lines of psutil per script whose numbers the
  runner then overrode anyway (see ``runner._merge_runtime_metadata``).

Nothing here is speculative: every helper replaces code that already existed in
several places. torch, numpy and pandas are imported lazily, so importing this
module stays as cheap as the rest of ``graflag_runner`` and does not add a
dependency to images that do not use them.
"""

import inspect
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .logging_utils import debug, info, warning

__all__ = [
    "ExperimentPaths",
    "injected_params",
    "apply_params",
    "params",
    "device",
    "paths",
    "upstream",
    "load_dataset",
    "load_snapshots",
    "snapshot_files",
    "split_test_edges",
    "seed_all",
]


# ==========================================================================
# Parameters
# ==========================================================================

def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes", "on")


def convert_value(env_name: str, raw: str, expected_type: type = None) -> Any:
    """Coerce one ``_FOO`` environment value to a Python value.

    Every value arrives as a string because it came from a `.env`. The
    parameter's declared type wins when there is one; otherwise the shape of
    the string decides, which is what the argparse parsers were doing by hand.
    """
    if raw.strip().lower() == "none":
        return None
    if raw.strip().lower() in ("true", "false"):
        return _str_to_bool(raw)

    if expected_type is not None:
        try:
            if expected_type is bool:
                return _str_to_bool(raw)
            if expected_type in (int, float, str):
                return expected_type(raw)
        except (TypeError, ValueError):
            pass

    try:
        return int(raw) if "." not in raw else float(raw)
    except (AttributeError, ValueError):
        return raw


def _declared_types(signature) -> Dict[str, type]:
    """Map parameter name -> type for a class, callable or Signature."""
    if signature is None:
        return {}
    if isinstance(signature, inspect.Signature):
        sig = signature
    else:
        target = signature.__init__ if inspect.isclass(signature) else signature
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            return {}

    types = {}
    for name, param in sig.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        if param.annotation is not inspect.Parameter.empty:
            types[name] = param.annotation
        elif param.default is not inspect.Parameter.empty and param.default is not None:
            types[name] = type(param.default)
        else:
            # Declared but untyped: accepted, coerced by value shape.
            types[name] = None
    return types


def injected_params() -> Dict[str, str]:
    """The `_FOO` environment variables that are this method's parameters.

    GraFlag names them in ``GRAFLAG_PARAMS`` (written by
    ``docker_ops._build_service_env``), which is the only authoritative
    answer: a `_FOO` variable in the environment is not necessarily a
    parameter. Inside a container it nearly always is, but on a workstation
    -- where the integration guide tells people to run a method by hand --
    zsh exports ``_P9K_TTY`` and conda exports ``_CE_CONDA`` and ``_CE_M``.
    Those were being recorded as method parameters in `results.json`, and
    with ``--pass-env-args`` the empty ``_CE_CONDA`` was appended to the
    command as a bare ``--ce_conda``, which argparse rejects outright.

    Falls back to scanning every `_FOO` when ``GRAFLAG_PARAMS`` is absent, so
    an image built before this existed keeps working.

    Returns:
        ``{env_var_name: raw_string}``, e.g. ``{"_EPOCHS": "100"}``.
    """
    manifest = os.environ.get("GRAFLAG_PARAMS")
    declared = None
    if manifest is not None:
        declared = {n.strip() for n in manifest.split(",") if n.strip()}

    found = {}
    for name, raw in os.environ.items():
        if not name.startswith("_") or len(name) == 1:
            # Bare "_" is the shell's last-argument variable, exported into
            # every child process.
            continue
        if declared is not None and name not in declared:
            debug("Ignoring %s: not in GRAFLAG_PARAMS", name)
            continue
        found[name] = raw
    return found


def params(signature=None, convert=convert_value) -> Dict[str, Any]:
    """The method's parameters, read from the ``_FOO`` environment variables.

    GraFlag injects every `.env` parameter with an underscore prefix, and
    ``--params LR=0.01`` overrides one in place. This turns them back into
    keyword arguments: ``_HID_DIM=64`` becomes ``{"hid_dim": 64}``.

    Args:
        signature: a class, callable or ``inspect.Signature`` whose parameters
            the result must fit. Names it does not accept are dropped, and
            values are coerced to its declared types. With ``None`` every
            ``_FOO`` is returned, typed by the shape of its value.
        convert: hook for turning one string into a value, for methods whose
            parameters are not plain scalars -- ``graflag_bond`` passes one
            that resolves ``torch.nn.functional.relu`` to the function.

    Returns:
        Keyword arguments, ready to splat into the model's constructor.
    """
    declared = _declared_types(signature)
    out = {}

    for env_name, raw in injected_params().items():
        name = env_name[1:].lower()
        if declared and name not in declared:
            # A warning, not a debug line, and the same one apply_params()
            # emits for the same event. `--params EPOCH=2` against a detector
            # whose constructor has no `epoch` is a request that reached
            # nothing, and a run that silently used its declared defaults
            # instead reads afterwards as a reduced-epoch run it never was.
            warning(f"[WARN] Ignoring {env_name}: "
                    f"{getattr(signature, '__name__', signature)} "
                    f"has no parameter '{name}'")
            continue
        out[name] = convert(env_name, raw, declared.get(name))

    return out


def apply_params(target, ignore=(), convert=convert_value):
    """Override an existing configuration object from the ``_FOO`` variables.

    For a method that adopts upstream's own configuration rather than
    declaring its own: ``generaldyg`` does ``from option import args`` and
    gets an argparse namespace carrying upstream's defaults, including keys
    its `.env` never mentions. :func:`params` cannot help there, because a
    namespace has no signature to read names and types from.

    Each `_FOO` is matched to an existing attribute ``foo`` and coerced to the
    type of the value already there, so upstream's own defaults define both
    what is accepted and how it is read.

    This is the alternative to ``--pass-env-args`` for such a method, and the
    safer one. Passing parameters through argv means argparse's abbreviation
    matching gets a say: with upstream declaring ``--gpus`` and no ``--gpu``,
    ``_GPU=0`` arrives as ``--gpu 0`` and argparse quietly sets ``gpus=0``.

    Args:
        target: the object to write onto, e.g. an ``argparse.Namespace``.
        ignore: parameter names that are deliberately not upstream's --
        ``"gpu"`` is read by :func:`device` rather than by the method.
        convert: hook for turning one string into a value, as in
            :func:`params`.

    Returns:
        ``{name: value}`` for what was applied, ready to record in
        ``results.json``.
    """
    ignore = set(ignore)
    applied = {}

    for env_name, raw in injected_params().items():
        name = env_name[1:].lower()
        if not hasattr(target, name):
            if name not in ignore:
                warning(f"[WARN] Ignoring {env_name}: "
                        f"{type(target).__name__} has no '{name}'")
            continue
        current = getattr(target, name)
        value = convert(env_name, raw, type(current) if current is not None else None)
        setattr(target, name, value)
        applied[name] = value

    return applied


# ==========================================================================
# Device
# ==========================================================================

def device(index=None):
    """The torch device this run should use.

    ``_GPU`` is an index, and **-1 means CPU** -- that is PyGOD's convention,
    which GraFlag adopted, and it is what ``graflag run --no-gpu`` sets. Three
    scripts built ``f"cuda:{gpu}"`` straight from it, so a CPU run asked torch
    for ``cuda:-1`` and died with "Invalid device string". Three others ignored
    ``_GPU`` and took a GPU whenever one was visible, including on runs
    scheduled without one.

    Falls back to CPU when no GPU is present, so a method is runnable on a
    CPU-only node without editing its `.env`.
    """
    import torch

    raw = os.environ.get("_GPU") if index is None else index
    try:
        wanted = 0 if raw is None or str(raw).strip() == "" else int(raw)
    except (TypeError, ValueError):
        warning(f"[WARN] Ignoring unreadable _GPU={raw!r}; using GPU 0 if present")
        wanted = 0

    if wanted < 0:
        info("[INFO] Using CPU (_GPU=-1)")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        info("[INFO] Using CPU (no GPU visible)")
        return torch.device("cpu")

    info(f"[INFO] Using GPU cuda:{wanted}")
    return torch.device(f"cuda:{wanted}")


# ==========================================================================
# Paths
# ==========================================================================

@dataclass(frozen=True)
class ExperimentPaths:
    """Where this run reads its dataset and writes its output."""

    data: Path
    exp: Path

    @property
    def dataset(self) -> str:
        """The dataset directory's name, e.g. ``uci_snapshot``."""
        return self.data.name

    @property
    def experiment(self) -> str:
        """The experiment directory's name, ``exp__method__dataset__stamp``."""
        return self.exp.name


def paths() -> ExperimentPaths:
    """The DATA and EXP directories GraFlag set for this run.

    Both are always set by ``docker_ops._build_service_env``, so a missing one
    means the script is being run outside GraFlag. Saying that is better than
    the ``os.environ.get("DATA", ".")`` fallback one script used, which read
    the working directory and reported an empty dataset as a real result.
    """
    missing = [k for k in ("DATA", "EXP") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not set. GraFlag sets these for every "
            f"run; set them by hand to run this script outside a container."
        )
    return ExperimentPaths(Path(os.environ["DATA"]), Path(os.environ["EXP"]))


# ==========================================================================
# Upstream source
# ==========================================================================

def upstream(*subdirs: str, root=None) -> Path:
    """Put the cloned upstream repository on ``sys.path``.

    Anchors on the calling script's directory rather than a hardcoded
    ``/app/...``, so the method does not depend on one Dockerfile's WORKDIR,
    and fails here with the path it looked for instead of as an ImportError
    several lines later.

    Args:
        *subdirs: directories under the script, relative to it. Defaults to
            ``"src"``, which is where every method's Dockerfile clones.
        root: the directory to anchor on. Defaults to the caller's.

    Returns:
        The first path added, i.e. the clone root for the usual one-argument
        call.
    """
    if root is None:
        frame = inspect.currentframe().f_back
        caller = frame.f_globals.get("__file__") if frame else None
        root = Path(caller).resolve().parent if caller else Path.cwd()
    root = Path(root)

    resolved = []
    for sub in (subdirs or ("src",)):
        path = (root / sub).resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"Upstream source not found at {path}. The Dockerfile should "
                f"`git clone` it there; check SOURCE_CODE and the clone step."
            )
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        resolved.append(path)

    return resolved[0]


# ==========================================================================
# Datasets
# ==========================================================================

def snapshot_files(path=None):
    """Find the snapshot dataset's ``(graph_file, split_file)``, or ``(None, None)``.

    Separate from :func:`load_dataset` because a method may want the raw
    adjacency matrices rather than the flattened edge stream: addgraph trains
    per snapshot and needs the four-way split with its snapshot ids, so it
    reads these files itself. Finding them is the part every such script had
    its own copy of, down to the same two patterns below.
    """
    data_dir = Path(path) if path is not None else paths().data

    graph_file = None
    for pattern in ("acc_*.npy", "graph.npy", "*.npy"):
        # 'sta_*' files are the static graph, not the snapshot series.
        matches = [m for m in sorted(data_dir.glob(pattern)) if "sta_" not in m.name]
        if matches:
            graph_file = matches[0]
            break

    split_file = None
    for pattern in ("split.npz", "*.npz"):
        matches = sorted(data_dir.glob(pattern))
        if matches:
            split_file = matches[0]
            break

    return graph_file, split_file


def _as_edge_pairs(edges):
    """Normalise a split's edge array to ``(N, 2)``.

    Stored either way round. Transposing on ``shape[0] == 2`` alone silently
    swapped src and dst for a split holding exactly two edges, which is the one
    case where both readings fit; treat that as ``(N, 2)``, the layout the rest
    of this module assumes.
    """
    import numpy as np

    edges = np.asarray(edges)
    if edges.ndim == 2 and edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.T
    return edges


def split_test_edges(split, default_snapshot=None):
    """The test half of a snapshot split as ``(edges, timestamps, ground_truth)``.

    ``split`` is a loaded ``split.npz``, or any mapping with the same keys.
    Edges come back as ``[[src, dst], ...]`` in ``test_pos`` then ``test_neg``
    order -- the order upstream subgraph extractors are handed them -- with
    ``ground_truth`` 0 for every ``test_pos`` and **1 for every ``test_neg``**.

    That labelling is the part worth stating once instead of per method.
    ``test_neg`` holds the sampled non-edges ``datasets/convert_to_strgnn.py``
    injects as the anomalies, so here it is the *positive* class; an upstream
    link predictor calls the real edges its positive class and labels those 1.
    Publishing an upstream label unchanged therefore inverts the contract, and
    inverts it on both sides at once, which leaves ``auc_roc`` looking right
    while ``precision_at_k`` and the anomaly counts describe the normal class.
    ``scores`` must rise with anomalousness and ``ground_truth`` must be 1 for
    an anomaly.

    ``timestamps`` are the split's own ``test_*_id`` snapshot ids. A split
    without them needs ``default_snapshot`` -- usually the last snapshot index.
    """
    import numpy as np

    pos = _as_edge_pairs(split["test_pos"])
    neg = _as_edge_pairs(split["test_neg"])

    def ids(key, count):
        if key in split:
            return np.asarray(split[key]).astype(int)
        if default_snapshot is None:
            raise KeyError(
                f"split has no {key!r} and no default_snapshot was given, so "
                f"the test edges cannot be placed in time"
            )
        return np.full(count, int(default_snapshot))

    edges = [[int(s), int(d)] for s, d in np.concatenate([pos, neg])]
    timestamps = np.concatenate([ids("test_pos_id", len(pos)),
                                 ids("test_neg_id", len(neg))]).tolist()
    ground_truth = [0] * len(pos) + [1] * len(neg)

    return edges, timestamps, ground_truth


def load_dataset(path=None):
    """Load a GraFlag dataset as ``(edges, labels)``.

    One copy of the loader five scripts each carried their own copy of. The
    formats are the ones already present under ``datasets/``:

    1. ``Data.csv`` + ``Label.csv``  -- edge stream with per-edge labels
    2. ``acc_*.npy`` + ``*.npz``     -- discrete snapshots with a train/test split
    3. ``edges.txt`` / ``edges.csv`` -- plain edge list
    4. a single file named after the dataset directory

    Returns:
        ``(edges, labels)`` where ``edges`` is a DataFrame with columns
        ``src``, ``dst``, ``timestamp`` and ``labels`` is a float array,
        1 for anomalous.
    """
    import numpy as np
    import pandas as pd

    data_dir = Path(path) if path is not None else paths().data

    # --- 1. Data.csv + Label.csv -----------------------------------------
    data_file, label_file = data_dir / "Data.csv", data_dir / "Label.csv"
    if data_file.exists() and label_file.exists():
        info(f"[INFO] Loading Data.csv/Label.csv from {data_dir}")
        edges = pd.read_csv(data_file, header=None,
                            names=["src", "dst", "timestamp"])
        labels = pd.read_csv(label_file, header=None,
                             names=["label"])["label"].to_numpy()
        return edges, labels

    # --- 2. snapshots ----------------------------------------------------
    graph_file, split_file = snapshot_files(data_dir)
    if graph_file and split_file:
        info(f"[INFO] Loading snapshots from {data_dir}")
        return load_snapshots(graph_file, split_file)

    # --- 3/4. edge lists -------------------------------------------------
    for name in ("edges.txt", "edges.csv", "edge_list.txt", data_dir.name):
        candidate = data_dir / name
        if candidate.is_file():
            info(f"[INFO] Loading edge list from {candidate}")
            return load_edge_list(candidate)

    for candidate in sorted(data_dir.iterdir()):
        if (candidate.is_file() and candidate.suffix in ("", ".txt", ".csv", ".tsv")
                and candidate.name != "README.md"):
            info(f"[INFO] Loading edge list from {candidate}")
            return load_edge_list(candidate)

    raise ValueError(f"No recognised dataset format in {data_dir}")


def load_snapshots(graph_file, split_file):
    """Load the snapshot format: adjacency matrices plus a train/test split.

    ``test_neg`` holds the anomalous (negative) edges. Those already in the
    graph are labelled; those that are not are appended, because the split's
    negatives are sampled rather than drawn from the observed edges.
    """
    import numpy as np
    import pandas as pd

    net = np.load(graph_file, allow_pickle=True)
    split = np.load(split_file, allow_pickle=True)

    num_snapshots = len(net) if net.dtype == object else net.shape[0]

    rows = []
    for t in range(num_snapshots):
        adj = net[t]
        if hasattr(adj, "toarray"):
            adj = adj.toarray()
        src, dst = np.where(adj > 0)
        upper = src < dst  # one row per undirected edge
        rows.append(pd.DataFrame({
            "src": src[upper].astype(int),
            "dst": dst[upper].astype(int),
            "timestamp": t,
        }))

    edges = (pd.concat(rows, ignore_index=True) if rows
             else pd.DataFrame(columns=["src", "dst", "timestamp"], dtype=int))

    test_neg = _as_edge_pairs(split["test_neg"])
    timestamps = (split["test_neg_id"] if "test_neg_id" in split
                  else np.full(len(test_neg), num_snapshots - 1))

    anomalies = pd.DataFrame({
        "src": np.asarray(test_neg)[:, 0].astype(int),
        "dst": np.asarray(test_neg)[:, 1].astype(int),
        "timestamp": np.asarray(timestamps).astype(int),
    })

    # One merge, not one full-DataFrame boolean mask per anomalous edge --
    # that inner loop is why snapshot loading was slow on the larger sets.
    key = ["src", "dst", "timestamp"]
    marked = edges.merge(anomalies.drop_duplicates().assign(_anom=1.0),
                         on=key, how="left")
    labels = marked["_anom"].fillna(0.0).to_numpy(dtype=float)

    seen = anomalies.merge(edges.drop_duplicates().assign(_seen=1), on=key, how="left")
    absent = anomalies[seen["_seen"].isna().to_numpy()]
    if len(absent):
        edges = pd.concat([edges, absent], ignore_index=True)
        labels = np.concatenate([labels, np.ones(len(absent))])

    return edges, labels


def load_edge_list(file_path):
    """Load a whitespace-separated edge list; a third column is a timestamp."""
    import numpy as np
    import pandas as pd

    rows = []
    with open(file_path) as handle:
        for line in handle:
            if line.startswith(("%", "#")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                timestamp = int(float(parts[2])) if len(parts) > 2 else len(rows)
                rows.append([int(parts[0]), int(parts[1]), timestamp])

    edges = pd.DataFrame(rows, columns=["src", "dst", "timestamp"])
    return edges, np.zeros(len(edges))


# ==========================================================================
# Reproducibility
# ==========================================================================

def seed_all(seed: int = 42) -> int:
    """Seed every generator a method might draw from.

    No integration script seeded anything, so re-running one with identical
    parameters gave a different AUC and there was no way to tell a real
    regression from noise. torch is seeded when installed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    info(f"[INFO] Seeded with {seed}")
    return seed


def load_attributed_graph(path=None, require_masks=False):
    """Load a static attributed graph as a PyTorch Geometric ``Data`` object.

    The counterpart to :func:`load_dataset`, which understands the dynamic
    formats (edge streams, snapshots, edge lists) and nothing else. The
    ``bond_*`` datasets are a single ``<name>.pt`` holding a PyG ``Data`` with
    ``x``, ``edge_index`` and ``y``.

    **This is a convenience, not a contract.** Most upstream methods load data
    their own way -- DiffGAD calls ``pygod.utils.load_data`` from inside its
    own ``__call__``, ADA-GAD reads its vendored copy of PyGOD -- and an
    integration that forced them through this function would have to patch
    upstream code to do it. The supported shape is the opposite: stage the
    dataset where the method already looks, let it load its own input, and use
    this to read ``y`` for the ``ground_truth`` the result contract requires,
    or to check the dataset before handing it over.

    Args:
        path: dataset directory. Defaults to ``paths().data``.
        require_masks: raise when the graph carries no ``test_mask``. Pass it
            in a supervised method, where scoring the wrong nodes is not
            detectable from the output.

    Returns:
        The ``Data`` object, unmodified.

    Raises:
        FileNotFoundError: no ``.pt`` in the directory. Never returns an empty
            graph or a default -- a method that silently scores nothing still
            writes a plausible ``results.json``.
        ValueError: the file loads but is not a PyG ``Data``, or
            ``require_masks`` is set and there is no ``test_mask``.
    """
    import torch

    data_dir = Path(path) if path is not None else paths().data
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {data_dir}")

    # Prefer the file named after the directory; fall back to a lone .pt.
    candidates = [data_dir / f"{data_dir.name}.pt"]
    candidates += sorted(p for p in data_dir.glob("*.pt")
                         if p not in candidates)
    graph_file = next((p for p in candidates if p.is_file()), None)
    if graph_file is None:
        raise FileNotFoundError(
            f"no .pt graph in {data_dir}; looked for {data_dir.name}.pt then "
            f"any *.pt. Contents: {sorted(p.name for p in data_dir.iterdir())}")

    info(f"[INFO] Loading attributed graph from {graph_file}")
    data = torch.load(graph_file, weights_only=False)

    if not hasattr(data, "edge_index") or not hasattr(data, "y"):
        raise ValueError(
            f"{graph_file} is a {type(data).__name__} without edge_index/y; "
            "expected a torch_geometric.data.Data")

    if require_masks:
        if getattr(data, "test_mask", None) is None:
            raise ValueError(
                f"{graph_file} carries no test_mask, and this method publishes "
                "test-split scores. Splitting here would invent a split the "
                "dataset does not define.")
        # A split holding one class is the quiet version of no split at all:
        # training still converges, scores still get written, and the AUC is
        # undefined rather than wrong -- so nothing downstream reports a
        # problem. bond_weibo is shipped this way: all 347 of its anomalies
        # are in train_mask and both val and test hold none.
        for name in ("train_mask", "val_mask", "test_mask"):
            mask = getattr(data, name, None)
            if mask is None:
                continue
            selected = data.y[mask]
            positives = int(selected.sum())
            if positives == 0 or positives == int(mask.sum()):
                raise ValueError(
                    f"{graph_file}: {name} holds {int(mask.sum())} nodes of a "
                    f"single class ({positives} anomalous). A supervised split "
                    "needs both classes -- training on it, or scoring it, "
                    "produces a number that cannot be an AUC.")

    n_anom = int(data.y.sum()) if data.y is not None else 0
    info(f"[INFO] {data.num_nodes} nodes, {data.num_edges} edges, "
         f"{n_anom} anomalies ({n_anom / max(data.num_nodes, 1):.2%})")
    return data


#: The three arrays a GAD ``.mat`` carries. HUGE-GAD, UNPrompt and AD-GCL all
#: read this shape, which is why one pair of converters serves all of them.
MAT_KEYS = ("Network", "Attributes", "Label")


def _dense(matrix):
    """Return a dense float32 ndarray from a dense or scipy-sparse matrix."""
    import numpy as np
    if hasattr(matrix, "todense"):
        matrix = matrix.todense()
    return np.asarray(matrix, dtype=np.float32)


def write_mat(data, mat_path, symmetrize: bool = True) -> Path:
    """Render a PyG ``Data`` as the ``.mat`` the upstream loaders read.

    The container-side half of GraFlag's static-graph conversion. HUGE-GAD,
    UNPrompt and AD-GCL each load ``Network``/``Attributes``/``Label`` from a
    MATLAB file, and none of them can be redirected at a ``.pt`` without
    patching their data path -- which would change what runs. So the
    integration renders what the method already reads, and every canonical
    dataset becomes available to it without a second copy existing anywhere.

    Offered, not imposed: a method that reads something else converts to that
    instead, and one that reads ``.pt`` (ADA-GAD, DiffGAD) just copies the file.

    Args:
        data: the graph.
        mat_path: destination.
        symmetrize: add the reverse of every edge. On by default because the
            ``.mat`` readers in this literature treat ``Network`` as an
            undirected adjacency, and ``bond_disney`` stores only one direction
            -- handing that over unsymmetrised would give the method half the
            neighbourhood with nothing raising.

    Returns:
        The path written.
    """
    import numpy as np
    import scipy.io as sio
    import scipy.sparse as sp

    path = Path(mat_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    src, dst = data.edge_index.cpu().numpy()
    if symmetrize:
        src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
    n = int(data.num_nodes)
    adjacency = sp.coo_matrix(
        (np.ones(len(src), dtype=np.float64), (src, dst)), shape=(n, n)).tocsr()
    # Duplicates sum in coo_matrix, so an edge present in both directions
    # already would become a 2. The readers treat Network as unweighted.
    adjacency.data[:] = 1.0

    y = data.y.cpu().numpy().ravel()
    sio.savemat(str(path), {
        "Network": adjacency,
        "Attributes": sp.csr_matrix(data.x.cpu().numpy().astype(np.float64)),
        "Label": (y > 0).astype(np.float64).reshape(1, -1),
    })
    return path


def read_mat(mat_path, binarize_labels: bool = True):
    """Load a ``.mat`` GAD dataset as a PyG ``Data``, inside the container.

    The inverse of :func:`write_mat`, and the other half of GraFlag's static
    static-graph bridge. Both live here rather than in ``graflag_data``
    because both need torch and torch_geometric, and the manager that does the
    fetching has neither -- converting at fetch time failed there with
    ``ModuleNotFoundError: No module named 'torch'`` and would have meant
    installing a GPU stack on a thin orchestration node.

    Args:
        mat_path: the ``.mat`` file.
        binarize_labels: store ``y`` as 0/1. The BOND graphs encode the outlier
            *type* in the label bits (1 contextual, 2 structural, 3 both) while
            the ``.mat`` files are already binary; keeping both on the same
            convention is what lets one evaluator read either. The raw column
            is preserved as ``y_raw`` regardless.

    Raises:
        FileNotFoundError: no such file.
        KeyError: the file is not this format -- named so the message says
            which keys were found, because a ``.mat`` with different key names
            loads perfectly well and would otherwise fail much later as a
            shape error.
    """
    import numpy as np
    import scipy.io as sio
    import torch
    from torch_geometric.data import Data

    path = Path(mat_path)
    if not path.is_file():
        raise FileNotFoundError(f"no such .mat file: {path}")

    raw = sio.loadmat(str(path))
    present = [k for k in raw if not k.startswith("__")]
    missing = [k for k in MAT_KEYS if k not in raw]
    if missing:
        raise KeyError(
            f"{path.name} is missing {missing}; it holds {present}. Expected a "
            "GAD .mat with Network/Attributes/Label.")

    adjacency = raw["Network"].tocoo()
    edge_index = torch.tensor(
        np.vstack([adjacency.row, adjacency.col]), dtype=torch.long)

    x = torch.from_numpy(_dense(raw["Attributes"]))
    y_raw = torch.tensor(np.asarray(raw["Label"]).ravel(), dtype=torch.long)
    y = (y_raw > 0).long() if binarize_labels else y_raw

    data = Data(x=x, edge_index=edge_index, y=y)
    data.y_raw = y_raw
    # Kept because they are the only record of *why* a node is an outlier, and
    # two of the shipped files carry them. Dropping them would make a
    # contextual-vs-structural breakdown impossible to reconstruct later.
    for optional in ("str_anomaly_label", "attr_anomaly_label", "Class"):
        if optional in raw:
            setattr(data, optional.lower(),
                    torch.tensor(np.asarray(raw[optional]).ravel()))
    return data
