import argparse, os, sys, numpy as np
from graflag_runner import ResultWriter

ap = argparse.ArgumentParser()
ap.add_argument("--max_epoch", type=int, default=5)
ap.add_argument("--hidden_dims", default="64")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mode", default="ok")
ap.add_argument("--use_memory", action="store_true")
a = ap.parse_args()
print(f"[method] use_memory={a.use_memory}")
print(f"[method] max_epoch={a.max_epoch} hidden_dims={a.hidden_dims!r} seed={a.seed}")
if a.mode == "slow":
    import time
    for i in range(40):
        print(f"[method] slow tick {i}", flush=True); time.sleep(1)
if a.mode == "crash":
    raise RuntimeError("deliberate failure for the e2e probe")
if a.mode == "silent":
    print("[method] exiting 0 without writing results"); sys.exit(0)
print(f"[method] DATA={os.environ['DATA']} EXP={os.environ['EXP']}")
sys.stdout.buffer.write(b"[method] non-utf8 marker: \xff\xfe\n"); sys.stdout.flush()

d = np.load(os.path.join(os.environ["DATA"], "data.npz"))
gt = d["labels"]
rng = np.random.default_rng(a.seed)
w = ResultWriter()
for e in range(1, a.max_epoch + 1):
    w.spot("training", epoch=e, loss=float(1.0 / e))
scores = rng.random(len(gt)) + gt * 0.5
w.save_scores(result_type="NODE_ANOMALY_SCORES", scores=scores.tolist(),
              ground_truth=gt.tolist(), node_ids=np.arange(len(gt)))   # numpy on purpose
w.add_metadata(method_name="e2e_probe", dataset="e2e_data")
w.finalize()
print("[method] done")
