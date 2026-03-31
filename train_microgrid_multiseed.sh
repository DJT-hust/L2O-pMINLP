#!/usr/bin/env bash
set -euo pipefail

# Batch runner for multi-seed microgrid experiments.
# Usage examples:
#   train_microgrid_multiseed.sh
#   SEEDS="42 43 44" MC_PRESET=mc_strong train_microgrid_multiseed.sh
#   SEEDS="42 43 44 45 46" train_microgrid_multiseed.sh --policy_mc_noise 0.009

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="${SEEDS:-45 46 47 48 49 50 51 52 53 54}"
MC_PRESET="${MC_PRESET:-mc_strong}"

mkdir -p runs/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="runs/logs/microgrid_multiseed_${STAMP}.csv"

EXTRA_ARGS=("$@")

echo "seed,policy_obj_mean,policy_obj_median,solver_obj_mean,solver_obj_median,gap_mean,gap_median,speedup_mean,speedup_median,infeasible,mc_improvements,log_file" > "$SUMMARY_CSV"

echo "[INFO] Multi-seed batch start"
echo "[INFO] Seeds: $SEEDS"
echo "[INFO] MC_PRESET: $MC_PRESET"
echo "[INFO] Summary CSV: $SUMMARY_CSV"

for seed in $SEEDS; do
  RUN_LOG="runs/logs/microgrid_seed${seed}_${STAMP}.log"
  echo "[INFO] Running seed=$seed, log=$RUN_LOG"

  MC_PRESET="$MC_PRESET" "$SCRIPT_DIR/train_microgrid.sh" \
    --seed "$seed" \
    --data_seed "$seed" \
    --split_seed "$seed" \
    "${EXTRA_ARGS[@]}" | tee "$RUN_LOG"

  python3 - "$RUN_LOG" "$SUMMARY_CSV" "$seed" <<'PY'
import csv
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
seed = sys.argv[3]
text = log_path.read_text(encoding="utf-8", errors="ignore")


def pick(pattern, default="nan"):
    m = re.findall(pattern, text)
    return m[-1] if m else default

policy_obj = pick(r"Policy Obj mean/median:\s*([0-9eE+\-.]+)\s*/\s*([0-9eE+\-.]+)", default=("nan", "nan"))
solver_obj = pick(r"Solver Obj mean/median:\s*([0-9eE+\-.]+)\s*/\s*([0-9eE+\-.]+)", default=("nan", "nan"))
gap = pick(r"Gap% mean/median:\s*([0-9eE+\-.]+)\s*/\s*([0-9eE+\-.]+)", default=("nan", "nan"))
speedup = pick(r"Speedup mean/median \(solver/policy\):\s*([0-9eE+\-.]+)\s*/\s*([0-9eE+\-.]+)", default=("nan", "nan"))
infeasible = pick(r"Number of infeasible solutions:\s*([0-9]+)")
mc_imp = pick(r"MC no-proj improvements accepted:\s*([0-9]+)")

with summary_csv.open("a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        seed,
        policy_obj[0], policy_obj[1],
        solver_obj[0], solver_obj[1],
        gap[0], gap[1],
        speedup[0], speedup[1],
        infeasible,
        mc_imp,
        str(log_path),
    ])
PY

done

python3 - "$SUMMARY_CSV" <<'PY'
import csv
import math
import statistics as st
import sys

path = sys.argv[1]
rows = []
with open(path, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

if not rows:
    print("[WARN] No rows in summary CSV")
    sys.exit(0)


def vals(key):
    out = []
    for row in rows:
        try:
            v = float(row[key])
            if math.isfinite(v):
                out.append(v)
        except Exception:
            pass
    return out

for key, name in [
    ("policy_obj_mean", "Policy Obj mean"),
    ("gap_mean", "Gap% mean"),
    ("speedup_mean", "Speedup mean"),
    ("infeasible", "Infeasible count"),
    ("mc_improvements", "MC improvements"),
]:
    x = vals(key)
    if not x:
        continue
    mu = st.mean(x)
    sd = st.pstdev(x) if len(x) > 1 else 0.0
    print(f"[SUMMARY] {name}: mean={mu:.4f}, std={sd:.4f}, n={len(x)}")

print(f"[SUMMARY] CSV saved: {path}")
PY
