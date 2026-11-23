"""Are GraFlag's MIDAS scores the very float32 values upstream's Demo.cpp computed?

GraFlag's driver writes each float score with "%.9g", which round-trips any
float32 exactly; upstream's Score.txt block writes it with "%f". So each
GraFlag score is parsed, narrowed to float32, printed with "%f" as upstream
prints it (C promotes the float to double exactly, and Python's % formatting
of that double is correctly rounded, as glibc's printf is), and compared with
upstream's line as text. Every line equal means every score is the same
float32 value.
"""
import json
import sys
import numpy as np

exp, ref = sys.argv[1], sys.argv[2]
with open(f"{exp}/midas_scores.txt") as fh:
    graflag_lines = [line.rstrip("\n") for line in fh]
graflag = np.array([float(line) for line in graflag_lines], dtype=np.float64)
with open(ref) as fh:
    upstream = [line.rstrip("\n") for line in fh]
as32 = graflag.astype(np.float32)
# The float32 recovered from each line prints back to that line: "%.9g"
# lost nothing, so as32 is the value the driver held.
roundtrip_exact = all("%.9g" % float(v) == line for v, line in zip(as32, graflag_lines))
mismatch = [i for i, (v, u) in enumerate(zip(as32, upstream)) if "%f" % float(v) != u]
published = np.asarray(json.load(open(f"{exp}/results.json"))["scores"], dtype=np.float64).ravel()
print(json.dumps({
    "n_graflag": int(graflag.size), "n_upstream": len(upstream),
    "graflag_file_roundtrips_through_float32": roundtrip_exact,
    "lines_differing_as_printed_by_upstream": len(mismatch),
    "first_mismatches": [[int(i), "%f" % float(as32[i]), upstream[i]] for i in mismatch[:3]],
    "results_json_equals_score_file": bool(np.array_equal(published, graflag)),
}))
