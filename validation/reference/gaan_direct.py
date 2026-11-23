"""PyGOD's GAAN called directly on the deduplicated inj_cora, outside GraFlag.

    python3 gaan_direct.py SEED [GPU]

Same constructor arguments as methods/bond_gaan/.env, the same repeated-edge
removal graflag_bond applies, nothing else. Prints the AUC and score spread.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pygod.detector import GAAN
from pygod.utils import load_data
from sklearn.metrics import roc_auc_score
import random

seed = int(sys.argv[1]); gpu = int(sys.argv[2]) if len(sys.argv) > 2 else -1
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
data = load_data("bond_inj_cora", cache_dir="/shared/datasets/bond_inj_cora")
ei = data.edge_index.numpy()
key = ei[0].astype(np.int64) * data.num_nodes + ei[1]
_, first = np.unique(key, return_index=True)
data.edge_index = data.edge_index[:, np.sort(first)]
model = GAAN(hid_dim=64, num_layers=4, dropout=0., weight_decay=0., act=F.relu,
             backbone=None, contamination=0.1, lr=0.004, epoch=100, gpu=gpu,
             batch_size=0, num_neigh=-1, weight=0.5, verbose=0)
model.fit(data)
s = model.decision_score_.numpy() if hasattr(model.decision_score_, "numpy") else np.asarray(model.decision_score_)
y = (data.y != 0).numpy()
print(f"seed {seed} gpu {gpu} edges {data.edge_index.size(1)} auc {roc_auc_score(y, s):.4f} "
      f"unique {len(np.unique(s))} std {s.std():.3e}")
