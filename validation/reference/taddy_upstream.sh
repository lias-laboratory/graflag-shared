#!/bin/bash
# Reference for the temporal check: upstream TADDY's own scripts, no GraFlag.
#
# Runs inside the `taddy` image, whose /app/src is the upstream clone at the
# commit methods/taddy/.env pins, unmodified. It copies that clone to /work,
# then runs upstream's two entry points exactly as its README gives them --
# 0_prepare_data.py (anomaly injection, seed fixed upstream at 1) and
# 1_train.py (seed 1, its default) -- on upstream's own copy of the UCI data.
# Upstream trains on the CPU only; it prints a TOTAL AUC over the test
# snapshots every print_feq epochs, and the last one is its result.
#
#   docker run --rm -v /root/validation:/work IMAGE bash /work/taddy_upstream.sh uci 0.1
set -euo pipefail
DATASET=${1:-uci}
ANOMALY_PER=${2:-0.1}
OUT=/work/out/taddy_upstream_${DATASET}_${ANOMALY_PER}
rm -rf "$OUT" && mkdir -p "$OUT"
cp -a /app/src "$OUT/src"
cd "$OUT/src"
git rev-parse HEAD > "$OUT/commit.txt"
git status --porcelain > "$OUT/dirty.txt"
sha256sum data/raw/* > "$OUT/raw_sha256.txt"
# Upstream writes to these and does not ship them (git keeps no empty dirs).
mkdir -p data/interim data/percent data/eigen
start=$(date +%s.%N)
python3 0_prepare_data.py --dataset "$DATASET" --anomaly_per "$ANOMALY_PER" --train_per 0.5 \
    > "$OUT/prepare.log" 2>&1
python3 1_train.py --dataset "$DATASET" --anomaly_per "$ANOMALY_PER" --train_per 0.5 --seed 1 \
    > "$OUT/train.log" 2>&1
end=$(date +%s.%N)
python3 - "$OUT" "$start" "$end" <<'PY'
import json, re, sys
out, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
log = open(f"{out}/train.log").read()
totals = [float(x) for x in re.findall(r"TOTAL AUC:([0-9.]+)", log)]
json.dump({"total_auc_by_print": totals, "final_total_auc": totals[-1] if totals else None,
           "wall_s": round(end - start, 1)}, open(f"{out}/summary.json", "w"), indent=1)
print(open(f"{out}/summary.json").read())
PY
