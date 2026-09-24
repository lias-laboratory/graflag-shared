"""
GADY GraFlag Integration - Train wrapper for GADY method.

GADY: Unsupervised Anomaly Detection on Dynamic Graphs (WSDM 2024)
https://github.com/mufeng-74/GADY

This script integrates GADY with the GraFlag benchmarking framework by:
1. Setting up data directories from GraFlag mounts
2. Running GADY's data preparation and preprocessing
3. Running GADY training and evaluation
4. Outputting results in GraFlag format using ResultWriter

The training loop follows upstream's generator/discriminator alternation
(train.py:244-272). It is reproduced here rather than imported because
upstream's own train.py does not run: it reads args.alpha, args.betaa and
args.gamma at lines 100-102 without declaring the flags, so the module raises
AttributeError before it trains. What this file imports is upstream's model,
generator, losses, samplers and data pipeline -- which is what SOURCE_REF pins.
README.md lists every place this loop and upstream's differ.
"""

import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from graflag_runner import (
    ResultWriter, device, info, params, paths, seed_all, upstream, warning,
)

# The GADY clone. upstream() anchors on this file's directory and raises if the
# checkout is missing, instead of failing later with an unexplained ImportError.
# data_loader.py runs src/prepare_data.py and src/preproc_new.py as subprocesses
# against the same directory.
upstream("src")

from data_loader import (                                       # noqa: E402
    ensure_data_ready,
    get_dataset_name_from_path,
)


@dataclass
class Config:
    """Every parameter GADY accepts, with the default it takes when unset.

    This replaces three competing paths that used to decide one value: an
    argparse parser fed by ``--pass-env-args``, a 30-key ``env_mappings`` dict,
    and a loop in ``main()`` copying the second over the first. The dict read
    ``ANOMALY_PER`` where GraFlag injects ``_ANOMALY_PER``, so all 30 lookups
    returned None and the entire path was dead -- two of its entries were
    additionally misspelled (``BATCH_SIZE`` for ``_BS``, ``lr_G`` for the
    ``lr_g`` argparse produced), which nothing could reveal while it never ran.

    ``params(Config)`` is now the single route in, and a field's annotation is
    what types the value.

    ``data`` is a field because upstream's helpers read it by attribute off
    the same object -- ``get_data`` and ``get_data_settings`` both take the
    dataset name that way.
    """

    # Filled from the DATA mount in main(), not from a `_DATA` parameter.
    data: str = 'uci'

    # Data
    anomaly_per: float = 0.1
    train_per: float = 0.7
    bs: int = 200

    # Model architecture
    n_layer: int = 2
    n_degree: int = 10
    memory_dim: int = 172
    message_dim: int = 100
    node_dim: int = 100
    time_dim: int = 1

    # Memory
    use_memory: bool = False
    memory_update_at_end: bool = False
    message_function: str = 'identity'
    memory_updater: str = 'gru'
    aggregator: str = 'last'

    # Positional features
    r_dim: int = 4
    beta: float = 0.00001

    # Training
    n_epoch: int = 50
    n_runs: int = 1
    lr: float = 0.0001
    patience: int = 5
    seed: int = 142
    gpu: int = 0

    # GAN loss
    alpha: float = 0.1
    betaa: float = 10.0
    gamma: float = 0.1
    lr_g: float = 0.000001
    lr_d: float = 0.000001

    # Mode and sampling
    mode: int = 0
    uniform: bool = False
    randomize_features: bool = False
    use_destination_embedding_in_message: bool = False
    use_source_embedding_in_message: bool = False
    different_new_nodes: bool = False
    prefix: str = ''


# Parameters this integration accepts but never reads. Keeping them typed and
# recorded is deliberate -- they are GADY's own knobs, and silently dropping
# them would make a `--params LR=...` sweep look like it worked while every run
# came back identical. Warning on each one that is actually moved off its
# default is what makes that visible. Delete an entry when it gets wired up --
# _LR_G, _LR_D and _BETAA left this dict when the adversarial loop was restored.
INERT_PARAMS = {
    'lr': "GADY trains two optimizers, on _LR_D and _LR_G; upstream declares "
          "--lr (train.py:33) and never reads it either",
}


def warn_about_inert_params(config: Config) -> None:
    """Say out loud which parameters were changed but will not be used."""
    defaults = {f.name: f.default for f in fields(Config)}
    for name, reason in INERT_PARAMS.items():
        if getattr(config, name) != defaults[name]:
            warning(f"[WARN] _{name.upper()} was set but has no effect: {reason}")


def setup_logging(data_name: str, anomaly_per: float):
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    Path("log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f'log/{data_name}_{anomaly_per}_{time.time()}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_positional_features(path: str, dev=None):
    """Load one savepoint of GADY's positional encodings.

    ``preproc_new.py`` writes V and R as a list of sparse tensors, one per
    batch, under ``pos_features/``. Training reads one file per partition,
    evaluation one file for the whole test split.

    A missing file used to be caught per batch and skipped with ``continue``,
    which turned a preprocessing step that had not run into an epoch that
    quietly trained on nothing. It is an error.
    """
    try:
        V, R = torch.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"positional features missing: {path}. They are written by "
            f"upstream's preproc_new.py, which data_loader.py runs before "
            f"training -- a run cannot proceed without them.") from None
    if dev is not None:
        V = [v.to(dev) for v in V]
        R = [r.to(dev) for r in R]
    return V, R


def evaluate_test_split(model, test_data, test_V, test_R, n_neighbors, batch_size):
    """Score the test split, returning the metrics *and* the scores.

    This is upstream's ``eval_edge_prediction`` (evaluation/evaluation.py:8)
    with the per-edge probabilities kept instead of discarded, and it is local
    for two reasons.

    Upstream's signature is ``(model, negative_edge_sampler, data, n_neighbors,
    batch_size=200, vs=None, rs=None)`` and it returns ``(ap, auc, vs, rs)``.
    The call this integration made passed ``test_data=``, ``train_data=``,
    ``args=``, ``test_rand_sampler=``, ``partition_size=`` and ``device=`` and
    unpacked two values in the order ``(auc, ap)`` -- so it raised TypeError on
    the first epoch that reached it, and would have reported AP as AUC if it
    had not.

    And results.json needs the scores themselves. Deriving them in a second
    pass would score a different V/R state than the AUC reported beside them;
    one pass is what keeps the two consistent.

    The score is the discriminator's output, unmodified. DiscFGANLoss drives a
    real edge toward 0 and a generated one toward 1, so higher means more
    anomalous -- which is the direction ``test_data.labels`` uses and the one
    upstream evaluates, ``roc_auc_score(true_label, pos_prob)``. The previous
    code published ``1 - prob``, inverting every score it produced.
    """
    scores, labels, edges, timestamps = [], [], [], []
    batch_aps, batch_aucs = [], []

    num_test = len(test_data.sources)
    num_test_batch = math.ceil(num_test / batch_size)
    # test_V holds one entry per evaluation batch, and upstream stops one batch
    # short of the end -- which is where it runs out.
    usable = min(num_test_batch, len(test_V))

    with torch.no_grad():
        model.eval()
        for k in range(usable):
            s_idx = k * batch_size
            e_idx = min(num_test, s_idx + batch_size)
            src_np = test_data.sources[s_idx:e_idx]
            dst_np = test_data.destinations[s_idx:e_idx]
            ts_np = test_data.timestamps[s_idx:e_idx]
            true_label = np.asarray(test_data.labels[s_idx:e_idx]).reshape(-1)

            pos_prob = model.compute_edge_probabilities(
                torch.tensor(src_np), torch.tensor(dst_np), torch.tensor(ts_np),
                test_data.edge_idxs[s_idx:e_idx], n_neighbors,
                update_memory=True,
                next_V=model.V + test_V[k].to_dense().to(model.device),
                next_R=model.R + test_R[k].to_dense().to(model.device),
            )
            pred = pos_prob.squeeze().cpu().numpy().reshape(-1)

            scores.extend(pred.tolist())
            labels.extend(true_label.tolist())
            edges.extend([[int(s), int(d)] for s, d in zip(src_np, dst_np)])
            timestamps.extend(np.asarray(ts_np).reshape(-1).tolist())

            # A batch holding one class has no AUC. Upstream raises on it;
            # averaging over the batches that do hold both is the metric it
            # reports when it does not.
            if len(np.unique(true_label)) > 1:
                batch_aps.append(average_precision_score(true_label, pred))
                batch_aucs.append(roc_auc_score(true_label, pred))

    if not scores:
        return 0.0, 0.0, None, None, None, None

    ap = float(np.mean(batch_aps)) if batch_aps else 0.0
    auc = float(np.mean(batch_aucs)) if batch_aucs else 0.0
    return ap, auc, scores, labels, edges, timestamps


def run_gady_training(args, logger, writer):
    """
    Run GADY training and evaluation.

    Args:
        args: Command line arguments
        logger: Logger instance
        writer: ResultWriter instance for tracking metrics

    Returns:
        dict: Results including AUC-ROC scores, predictions, and edge-level scores
    """
    # Import GADY modules
    from modules.GAN import Generator
    from model.tgn import TGN
    from utils.utils import (EarlyStopMonitor, RandEdgeSampler,
                             get_neighbor_finder, get_data_settings,
                             GenFGANLoss, DiscFGANLoss)
    from utils.data_processing import get_data, compute_time_statistics

    # One call seeds `random`, numpy and torch; the two lines here left
    # `random` unseeded, so upstream's negative sampling was never repeatable.
    seed_all(args.seed)
    
    logger.info(args)
    
    # Load data
    node_features, edge_features, full_data, train_data, test_data = get_data(
        args.data,
        different_new_nodes_between_val_and_test=args.different_new_nodes,
        randomize_features=args.randomize_features,
        anomaly_per=args.anomaly_per
    )

    # Upstream numbers edges from 1 -- `prepare_data.py` appends
    # `idx_list = [int(x)+1 for x in range(np.size(all_data, 0))]` as the edge
    # index column -- but sizes the edge feature matrix to one row per edge,
    # `edge_features = np.zeros((data_full.shape[0], 172))`. The last edge's
    # index is therefore exactly one past the end of the array that
    # `tgn.py:421` indexes with it.
    #
    # The train split stops at 70% and never reaches that index, so training
    # runs clean; the first evaluation pass reaches it in its final batch. The
    # out-of-bounds read is a CUDA kernel, and CUDA kernels report
    # asynchronously, so it surfaced as `device-side assert triggered` raised
    # against the *next* indexing operation -- the memory lookup two lines
    # later -- which is why the traceback named `memory.py:40` and nothing in
    # it pointed at the edge features. On 38,544 edges it took one full epoch
    # and 62 evaluation batches to arrive.
    #
    # Sizing the matrix from the indices that will actually be used fixes it
    # for any split. The features are all zeros, so the added row changes
    # nothing this method computes -- it only makes the last edge indexable.
    needed = int(np.max(full_data.edge_idxs)) + 1
    if needed > edge_features.shape[0]:
        info(f"[INFO] Edge features sized {edge_features.shape[0]} for edge "
             f"indices up to {needed - 1}; padding to {needed} rows")
        edge_features = np.vstack([
            edge_features,
            np.zeros((needed - edge_features.shape[0], edge_features.shape[1]),
                     dtype=edge_features.dtype),
        ])

    # Initialize neighbor finders
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)
    # Only the ablation (--mode 1) draws random negatives; mode 0 gets them
    # from the generator. The sampler was built twice, once unconditionally and
    # once again inside `if args.mode == 1`, with the same arguments.
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)

    # device() is the shared resolver: a negative index means CPU -- which is
    # what `graflag run --no-gpu` sets -- and so does CUDA being unavailable.
    dev = device(args.gpu)
    logger.info(f'Using device: {dev}')
    
    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
    
    # Store best results across runs
    # -1 rather than 0: an AP of exactly 0 is a legitimate (if dire)
    # result, and starting at 0 would discard the scores that produced it
    # and then report "no epoch produced test scores".
    best_auc = -1.0
    best_ap = -1.0
    best_scores = None
    best_labels = None
    best_edges = None
    best_timestamps = None
    all_results = []

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / args.bs)
    partition_size, _ = get_data_settings(args.data)

    # Positional features for the test split. One file for the whole split,
    # indexed per evaluation batch, so it is read once rather than per epoch.
    # They are left on the CPU: the evaluation pass moves one batch at a time,
    # which is what upstream does and what keeps a long test split off the GPU.
    test_V, test_R = load_positional_features(
        f'pos_features/{args.data}_VR_test_bs_{args.bs}'
        f'_rdim_{args.r_dim}{args.anomaly_per}')

    # GADY's generator is its negative sampler, and it lives *inside* the
    # discriminator: TGN.__init__ takes it as `Generator` and keeps it as a
    # submodule (model/tgn.py:28,52), which is how compute_neg_edge_probabilities
    # reaches it (:231). Building it and not passing it -- which is what this
    # integration did -- left `discriminator.Generator` as None, so the very
    # first training batch died on `.eval()`. The method had never run.
    train_generator = Generator(n_neighbors=args.n_degree, batch_size=args.bs, device=dev)

    for run_idx in range(args.n_runs):
        logger.info(f'\n=== Run {run_idx + 1}/{args.n_runs} ===')

        # Initialize model
        discriminator = TGN(
            neighbor_finder=train_ngh_finder,
            node_features=node_features,
            edge_features=edge_features,
            device=dev,
            n_layers=args.n_layer,
            use_memory=args.use_memory,
            message_dimension=args.message_dim,
            memory_dimension=args.memory_dim,
            memory_update_at_start=not args.memory_update_at_end,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=args.n_degree,
            mean_time_shift_src=mean_time_shift_src,
            std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst,
            std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            beta=args.beta,
            r_dim=args.r_dim,
            Generator=train_generator,
        )
        # TGN.__init__ moves the feature matrices and V/R to `device` itself but
        # not its own parameters, so the module still has to be moved. Without
        # this the first matmul on a GPU run raises "Expected all tensors to be
        # on the same device" -- which nothing could reveal while the run died
        # three lines earlier.
        discriminator = discriminator.to(dev)

        # Losses, constructed as upstream constructs them (utils/utils.py:146,169).
        # _ALPHA, _BETAA and _GAMMA are wired to their keyword arguments here.
        # Upstream names the same three values at train.py:100-102 and then uses
        # none of them -- and does not declare the flags they read, so
        # `python train.py` raises AttributeError on line 100 before it trains.
        # Passing them is therefore the only reading under which the keys mean
        # anything; the values in .env are GADY's published ones.
        criterion_gen = GenFGANLoss(alpha_=args.alpha, beta_=args.betaa).to(dev)
        criterion_disc = DiscFGANLoss(gamma_=args.gamma).to(dev)
        criterion_bce = torch.nn.BCELoss().to(dev)

        # Two optimizers, not one. GADY is adversarial: the discriminator learns
        # to separate real edges from generated ones, the generator learns to
        # fool it. A single optimizer over discriminator.parameters() -- which
        # now contains the generator, since it is a submodule -- would drive
        # both toward the same objective, which is not training GADY. Upstream
        # keeps them apart (train.py:153-154), and that is what makes _LR_D and
        # _LR_G mean something. The discriminator step detaches the generator's
        # samples (nograd=True below), so d_optimizer leaves it alone.
        #
        # It leaves it alone in the arithmetic, but upstream still hands it the
        # generator's parameters, and torch 1.9's Adam allocates state for any
        # parameter whose .grad is not None. zero_grad() there zeroes gradients
        # rather than clearing them, so from the second batch on the generator's
        # grads are zero tensors, not None -- and d_optimizer.step() materialises
        # a full exp_avg/exp_avg_sq pair for all 440,074,746 of them. That is a
        # second 3.3 GiB copy of the state g_optimizer already holds, and it is
        # what put this run over an 11.6 GiB card: 9.64 GiB was already allocated
        # when the step asked for 102 MiB more.
        #
        # The duplicate never moved a weight. exp_avg stays identically zero
        # under a zero gradient, weight_decay is 0 and amsgrad is off, so every
        # update Adam computed for those parameters was 0/(0 + eps) = 0. Holding
        # them out of d_optimizer therefore changes no number this method
        # produces; it only stops the generator being optimised twice over.
        gen_param_ids = {id(p) for p in discriminator.Generator.parameters()}
        disc_params = [p for p in discriminator.parameters()
                       if id(p) not in gen_param_ids]
        d_optimizer = torch.optim.Adam(disc_params, lr=args.lr_d)
        g_optimizer = torch.optim.Adam(discriminator.Generator.parameters(), lr=args.lr_g)

        early_stopper = EarlyStopMonitor(max_round=args.patience)

        run_auc, run_ap, run_scores = -1.0, -1.0, None

        for epoch in range(args.n_epoch):
            logger.info(f'Epoch {epoch + 1}/{args.n_epoch}')

            # V and R accumulate over an epoch; starting the next one on the
            # previous one's encodings is not what upstream trains (train.py:183).
            discriminator.reset_VR()
            if args.use_memory:
                discriminator.memory.__init_memory__()
            discriminator.set_neighbor_finder(train_ngh_finder)

            d_losses, g_losses = [], []
            next_V, next_R = [], []

            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.bs
                end_idx = min(num_instance, start_idx + args.bs)

                src_np = train_data.sources[start_idx:end_idx]
                dst_np = train_data.destinations[start_idx:end_idx]
                sources_batch = torch.tensor(src_np)
                destinations_batch = torch.tensor(dst_np)
                timestamps_batch = torch.tensor(train_data.timestamps[start_idx:end_idx])
                edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
                size = end_idx - start_idx

                # One file per partition, not one per batch: the previous loop
                # re-read the same savepoint `partition_size` times per partition.
                if batch_idx % partition_size == 0:
                    prt = batch_idx // partition_size
                    next_V, next_R = load_positional_features(
                        f'pos_features/{args.data}_nextVR_part_{prt}'
                        f'_bs_{args.bs}_rdim_{args.r_dim}{args.anomaly_per}', dev)

                idx = batch_idx % partition_size
                if idx >= len(next_V):
                    # The last savepoint covers fewer batches than a full
                    # partition. Say so and end the epoch, rather than index
                    # past it or skip the batch in silence.
                    warning(f"[WARN] positional features exhausted at batch "
                            f"{batch_idx}/{num_batch}; ending epoch {epoch + 1} here")
                    break

                discriminator.train()
                d_optimizer.zero_grad()

                if args.mode == 0:
                    # Discriminator step. nograd=True detaches the generator's
                    # samples, which is what keeps d_optimizer -- whose parameter
                    # list contains the generator -- from training it here.
                    discriminator.Generator.eval()
                    pos_prob = discriminator.compute_edge_probabilities(
                        sources_batch, destinations_batch, timestamps_batch,
                        edge_idxs_batch, args.n_degree,
                        update_memory=True,
                        next_V=discriminator.V + next_V[idx].to_dense(),
                        next_R=discriminator.R + next_R[idx].to_dense(),
                    )
                    neg_prob2, _ = discriminator.compute_neg_edge_probabilities(
                        sources_batch, destinations_batch, timestamps_batch,
                        edge_idxs_batch, args.n_degree,
                        update_memory=False,
                        next_V=discriminator.V + next_V[idx].to_dense(),
                        next_R=discriminator.R + next_R[idx].to_dense(),
                        nograd=True,
                    )
                    # DiscFGANLoss.forward is (d_out_fake, d_out_real) and takes
                    # nothing else. The call here used to pass three arguments,
                    # and the two it did pass were the wrong way round -- so it
                    # raised TypeError the moment the line was reached.
                    d_loss = criterion_disc(neg_prob2.squeeze(), pos_prob.squeeze())
                    d_loss.backward()
                    d_optimizer.step()

                    # Generator step. criterion_gen was constructed and never
                    # called before this, so the generator was never trained.
                    discriminator.eval()
                    discriminator.Generator.train()
                    neg_prob, neg_samples = discriminator.compute_neg_edge_probabilities(
                        sources_batch, destinations_batch, timestamps_batch,
                        edge_idxs_batch, args.n_degree,
                        update_memory=False,
                        next_V=discriminator.V + next_V[idx].to_dense(),
                        next_R=discriminator.R + next_R[idx].to_dense(),
                    )
                    g_loss = criterion_gen(neg_prob.squeeze(), neg_samples)
                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()
                    g_losses.append(g_loss.item())
                else:
                    # Ablation (--mode 1): random negatives and a plain BCE,
                    # with real edges as class 0, as upstream runs it
                    # (train.py:274-289).
                    edges = np.hstack((src_np.reshape(-1, 1), dst_np.reshape(-1, 1)))
                    _, negatives_batch = train_rand_sampler.sample(edges)
                    with torch.no_grad():
                        pos_label = torch.zeros(size, dtype=torch.float, device=dev)
                        neg_label = torch.ones(size, dtype=torch.float, device=dev)
                    pos_prob = discriminator.compute_edge_probabilities(
                        sources_batch, destinations_batch, timestamps_batch,
                        edge_idxs_batch, args.n_degree,
                        update_memory=True,
                        next_V=discriminator.V + next_V[idx].to_dense(),
                        next_R=discriminator.R + next_R[idx].to_dense(),
                    )
                    neg_prob = discriminator.compute_edge_probabilities(
                        sources_batch, torch.tensor(negatives_batch), timestamps_batch,
                        edge_idxs_batch, args.n_degree,
                        update_memory=False,
                        next_V=discriminator.V + next_V[idx].to_dense(),
                        next_R=discriminator.R + next_R[idx].to_dense(),
                    )
                    d_loss = (criterion_bce(pos_prob.squeeze(), pos_label)
                              + criterion_bce(neg_prob.squeeze(), neg_label))
                    d_loss.backward()
                    d_optimizer.step()

                d_losses.append(d_loss.item())

                # Not upstream's, kept deliberately: without it the memory keeps
                # the previous batch's graph alive and the second backward pass
                # raises "trying to backward through the graph a second time".
                if args.use_memory:
                    discriminator.memory.detach_memory()

            avg_d_loss = float(np.mean(d_losses)) if d_losses else float('nan')
            logger.info(f'Epoch {epoch + 1} mean discriminator loss: {avg_d_loss:.4f}')
            if g_losses:
                logger.info(f'Epoch {epoch + 1} mean generator loss: {np.mean(g_losses):.4f}')

            # Evaluation. The full neighbour finder is what upstream evaluates
            # against (train.py:304); leaving the training finder in place hides
            # every test-time neighbour from the model.
            discriminator.set_neighbor_finder(full_ngh_finder)
            memory_backup = discriminator.memory.backup_memory() if args.use_memory else None

            test_ap, test_auc, scores, labels, edges_out, ts_out = evaluate_test_split(
                discriminator, test_data, test_V, test_R, args.n_degree, args.bs)

            if args.use_memory:
                discriminator.memory.restore_memory(memory_backup)
            discriminator.set_neighbor_finder(train_ngh_finder)

            logger.info(f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')

            metrics = {'epoch': epoch + 1, 'd_loss': avg_d_loss,
                       'test_auc': test_auc, 'test_ap': test_ap}
            if args.mode == 0:
                metrics['g_loss'] = float(np.mean(g_losses)) if g_losses else float('nan')
            writer.spot("training", **metrics)

            # The published scores come from the checkpoint the early stopper
            # selects on, so `graflag evaluate` and the summary below describe
            # the same epoch. Reporting a maximum taken from a different epoch
            # than the scores would make results.json and the summary disagree.
            if scores is not None and test_ap > run_ap:
                run_ap, run_auc = test_ap, test_auc
                run_scores = (scores, labels, edges_out, ts_out)

            if early_stopper.early_stop_check(test_ap):
                logger.info(f'Early stopping at epoch {epoch + 1}')
                break

        all_results.append({'run': run_idx, 'auc': run_auc, 'ap': run_ap})
        logger.info(f'Run {run_idx + 1} - AUC: {run_auc:.4f}, AP: {run_ap:.4f}')

        if run_scores is not None and run_ap >= best_ap:
            best_ap, best_auc = run_ap, run_auc
            best_scores, best_labels, best_edges, best_timestamps = run_scores

    if best_scores is None:
        raise RuntimeError(
            "no epoch produced test scores -- the evaluation split yielded no "
            "batch with both classes present, so there is nothing to publish")

    final_results = {
        'method': 'GADY',
        'dataset': args.data,
        'anomaly_rate': args.anomaly_per,
        'best_auc': best_auc,
        'best_ap': best_ap,
        'all_runs': all_results,
        'mean_auc': float(np.mean([r['auc'] for r in all_results])),
        'std_auc': float(np.std([r['auc'] for r in all_results])),
        'mean_ap': float(np.mean([r['ap'] for r in all_results])),
        'std_ap': float(np.std([r['ap'] for r in all_results])),
        'scores': best_scores,
        'labels': best_labels,
        'edges': best_edges,
        'timestamps': best_timestamps,
    }

    return final_results


def main():
    """Prepare the data GADY expects, train it, and publish the scores."""
    info('=' * 60)
    info('GADY - Unsupervised Anomaly Detection on Dynamic Graphs')
    info('GraFlag Integration')
    info('=' * 60)

    run = paths()
    config = Config(**params(Config))
    # The dataset name comes from the mount, not from a parameter: GraFlag
    # sets DATA, and the folder `gady_uci` is what GADY knows as `uci`.
    config.data = get_dataset_name_from_path(run.data)
    warn_about_inert_params(config)

    info(f'Dataset: {config.data} (from {run.data})')
    info(f'Anomaly rate: {config.anomaly_per} | GPU: {config.gpu} | '
         f'Epochs: {config.n_epoch} | Runs: {config.n_runs}')
    if config.gpu < 0:
        warning('[WARN] _GPU=-1 puts training on CPU, but upstream\'s '
                'preproc_new.py is handed the same -1 and may not honour it')

    logger = setup_logging(config.data, config.anomaly_per)
    writer = ResultWriter()

    try:
        # One call for the three steps main() used to inline: symlink the raw
        # file, run prepare_data.py, run preproc_new.py.
        ensure_data_ready(
            run.data, config.data,
            anomaly_per=config.anomaly_per,
            train_per=config.train_per,
            batch_size=config.bs,
            r_dim=config.r_dim,
            gpu=config.gpu,
        )

        results = run_gady_training(config, logger, writer)

        # The score is the discriminator's output unmodified: DiscFGANLoss
        # trains a real edge toward 0 and a generated one toward 1, so it
        # already rises with anomalousness -- the same direction as the
        # injected-anomaly labels and as upstream's own AUC. This used to
        # publish `1 - P(edge)`, which inverted every score. Do not invert
        # either side.
        writer.save_scores(
            result_type="EDGE_STREAM_ANOMALY_SCORES",
            scores=results['scores'],
            edges=results['edges'],
            timestamps=results['timestamps'],
            ground_truth=results['labels'],
        )

        # asdict() instead of the hand-written dict this replaced: that one
        # listed 19 of the parameters, so the rest of what a run used went
        # unrecorded and could not drift back into agreement on its own.
        writer.add_metadata(
            exp_name=run.experiment,
            method_name="gady",
            dataset=config.data,
            method_parameters={k: v for k, v in asdict(config).items()
                               if k != 'data'},
            threshold=None,
            summary={
                "description": "GADY: Unsupervised Anomaly Detection on Dynamic Graphs (WSDM 2024)",
                "task": "edge_stream_anomaly_detection",
                "dataset_info": {
                    "name": config.data,
                    "anomaly_rate": config.anomaly_per,
                    "total_test_edges": len(results['scores']),
                    "n_anomalies": sum(results['labels']),
                },
                "training_info": {
                    "n_runs": config.n_runs,
                    "best_auc": float(results['best_auc']),
                    "best_ap": float(results['best_ap']),
                    "mean_auc": float(results['mean_auc']),
                    "std_auc": float(results['std_auc']),
                    "mean_ap": float(results['mean_ap']),
                    "std_ap": float(results['std_ap']),
                },
            },
        )

        # No add_resource_metrics and no psutil sampling: graflag_runner
        # measures exec time, peak memory and peak GPU from outside the
        # method, and _merge_runtime_metadata makes its numbers the ones that
        # land in results.json.
        results_file = writer.finalize()

        info('[OK] GADY completed')
        info(f'   Best AUC: {results["best_auc"]:.4f}')
        info(f'   Best AP: {results["best_ap"]:.4f}')
        info(f'   Results saved to: {results_file}')

    except Exception as exc:
        logger.error(f'Error during GADY training: {exc}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
