#!/bin/bash
# Reference for the streaming check: upstream MIDAS's own driver, no GraFlag.
#
# Clones upstream at the commit methods/midas/.env pins, into a container of
# the `midas` image (for its compiler), and compiles upstream's
# example/Demo.cpp against upstream's own copy of DARPA, with exactly two
# edits, each checked to apply once:
#   1. the seed is 42 instead of time(nullptr), so a run can be compared;
#   2. upstream's own score-writing block, shipped commented out, is enabled.
# The core (FilteringCore(2, 1024, 1e3f)) is upstream's default, which is
# also GraFlag's default for this method.
#
#   docker run --rm -e https_proxy=... -v /root/validation:/work IMAGE bash /work/midas_upstream.sh REF
set -euo pipefail
REF=${1:?pinned commit}
OUT=/work/out/midas_upstream
rm -rf "$OUT" && mkdir -p "$OUT"
git clone -q https://github.com/Stream-AD/MIDAS "$OUT/MIDAS"
cd "$OUT/MIDAS"
git checkout -q --detach "$REF"
git rev-parse HEAD > "$OUT/commit.txt"
python3 - <<'PY'
from pathlib import Path
p = Path("example/Demo.cpp")
s = p.read_text()
edits = [
    ("const unsigned seed = time(nullptr);", "const unsigned seed = 42;"),
    ('\t// const auto pathScore = SOLUTION_DIR"temp/Score.txt";\n'
     '\t// const auto fileScore = fopen(pathScore, "w");\n'
     '\t// for (int i = 0; i < n; i++)\n'
     '\t// \tfprintf(fileScore, "%f\\n", score[i]);\n'
     '\t// fclose(fileScore);\n',
     '\tconst auto pathScore = SOLUTION_DIR"temp/Score.txt";\n'
     '\tconst auto fileScore = fopen(pathScore, "w");\n'
     '\tfor (int i = 0; i < n; i++)\n'
     '\t\tfprintf(fileScore, "%f\\n", score[i]);\n'
     '\tfclose(fileScore);\n'),
]
for old, new in edits:
    assert s.count(old) == 1, f"edit does not apply exactly once: {old[:40]!r}"
    s = s.replace(old, new)
p.write_text(s)
PY
git diff > "$OUT/reference.diff"
mkdir -p temp
g++ -O3 -std=c++11 -Isrc -Iutil -DSOLUTION_DIR="\"$OUT/MIDAS/\"" example/Demo.cpp -o "$OUT/demo"
sha256sum data/DARPA/darpa_processed.csv data/DARPA/darpa_ground_truth.csv > "$OUT/data_sha256.txt"
"$OUT/demo" | tee "$OUT/demo.log"
cp temp/Score.txt "$OUT/Score.txt"
