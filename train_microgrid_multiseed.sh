#!/usr/bin/env bash
set -euo pipefail

# Batch runner for multi-seed microgrid experiments.
# Usage examples:
#   train_microgrid_multiseed.sh
#   SEEDS="42 43 44" MC_PRESET=mc_strong train_microgrid_multiseed.sh
#   SEEDS="42 43 44 45 46" train_microgrid_multiseed.sh --policy_mc_noise 0.009
#   TAIL_GRID=1 train_microgrid_multiseed.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="${SEEDS:-45 46 47 48 49 50 51 52 53 54}"
MC_PRESET="${MC_PRESET:-mc_strong}"
TAIL_GRID="${TAIL_GRID:-0}"

# Base tail-suppression parameters used when TAIL_GRID=0.
BASE_TAIL_WEIGHT="${BASE_TAIL_WEIGHT:-0.03}"
BASE_TAIL_TOPK_RATIO="${BASE_TAIL_TOPK_RATIO:-0.10}"
BASE_TAIL_SHED_WEIGHT="${BASE_TAIL_SHED_WEIGHT:-0.50}"

# Small grid (comma-separated tuples: tail_weight,topk_ratio,shed_weight)
TAIL_GRID_CONFIGS="${TAIL_GRID_CONFIGS:-0.02,0.10,0.30 0.03,0.10,0.50 0.04,0.10,0.70 0.03,0.08,0.50 0.03,0.12,0.50}"

mkdir -p runs/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="runs/logs/microgrid_multiseed_${STAMP}.csv"
GRID_CSV="runs/logs/microgrid_tailgrid_${STAMP}.csv"

EXTRA_ARGS=("$@")

echo "tail_weight,tail_topk_ratio,tail_shed_weight,seed,policy_obj_mean,policy_obj_median,solver_obj_mean,solver_obj_median,gap_mean,gap_median,speedup_mean,speedup_median,infeasible,mc_improvements,log_file" > "$SUMMARY_CSV"

echo "[INFO] Multi-seed batch start"
echo "[INFO] Seeds: $SEEDS"
echo "[INFO] MC_PRESET: $MC_PRESET"
echo "[INFO] TAIL_GRID: $TAIL_GRID"
echo "[INFO] Summary CSV: $SUMMARY_CSV"

run_one_seed() {
    local seed="$1"
    local tail_weight="$2"
    local topk_ratio="$3"
    local shed_weight="$4"
    local run_tag="$5"

    local RUN_LOG="runs/logs/microgrid_seed${seed}_${run_tag}_${STAMP}.log"
    echo "[INFO] Running seed=$seed, tail=($tail_weight,$topk_ratio,$shed_weight), log=$RUN_LOG"

  MC_PRESET="$MC_PRESET" "$SCRIPT_DIR/train_microgrid.sh" \
    --seed "$seed" \
    --data_seed "$seed" \
    --split_seed "$seed" \
        --policy_tail_weight "$tail_weight" \
        --policy_tail_topk_ratio "$topk_ratio" \
        --policy_tail_shed_weight "$shed_weight" \
    "${EXTRA_ARGS[@]}" | tee "$RUN_LOG"

    python3 - "$RUN_LOG" "$SUMMARY_CSV" "$seed" "$tail_weight" "$topk_ratio" "$shed_weight" <<'PY'
import csv
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
seed = sys.argv[3]
tail_weight = sys.argv[4]
topk_ratio = sys.argv[5]
shed_weight = sys.argv[6]
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
        tail_weight,
        topk_ratio,
        shed_weight,
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
}

if [[ "$TAIL_GRID" == "1" ]]; then
    echo "[INFO] Running tail grid configs: $TAIL_GRID_CONFIGS"
    for cfg in $TAIL_GRID_CONFIGS; do
        IFS=',' read -r tail_weight topk_ratio shed_weight <<< "$cfg"
        tag="tw${tail_weight}_tk${topk_ratio}_sw${shed_weight}"
        tag="${tag//./p}"
        for seed in $SEEDS; do
            run_one_seed "$seed" "$tail_weight" "$topk_ratio" "$shed_weight" "$tag"
        done
    done
else
    for seed in $SEEDS; do
        run_one_seed "$seed" "$BASE_TAIL_WEIGHT" "$BASE_TAIL_TOPK_RATIO" "$BASE_TAIL_SHED_WEIGHT" "base"
    done
fi

python3 - "$SUMMARY_CSV" "$GRID_CSV" <<'PY'
import csv
import math
import statistics as st
import sys
from collections import defaultdict

path = sys.argv[1]
grid_out = sys.argv[2]
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


def vals_filtered(key, subset):
    out = []
    for row in subset:
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

groups = defaultdict(list)
for row in rows:
    g = (row["tail_weight"], row["tail_topk_ratio"], row["tail_shed_weight"])
    groups[g].append(row)

with open(grid_out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "tail_weight", "tail_topk_ratio", "tail_shed_weight", "n",
        "policy_obj_mean_avg", "policy_obj_mean_std",
        "gap_mean_avg", "gap_mean_std",
        "speedup_mean_avg", "speedup_mean_std",
        "infeasible_avg", "mc_improvements_avg",
    ])
    for g, subset in sorted(groups.items()):
        p = vals_filtered("policy_obj_mean", subset)
        gap = vals_filtered("gap_mean", subset)
        sp = vals_filtered("speedup_mean", subset)
        inf = vals_filtered("infeasible", subset)
        mc = vals_filtered("mc_improvements", subset)
        w.writerow([
            g[0], g[1], g[2], len(subset),
            f"{st.mean(p):.6f}" if p else "nan",
            f"{(st.pstdev(p) if len(p) > 1 else 0.0):.6f}" if p else "nan",
            f"{st.mean(gap):.6f}" if gap else "nan",
            f"{(st.pstdev(gap) if len(gap) > 1 else 0.0):.6f}" if gap else "nan",
            f"{st.mean(sp):.6f}" if sp else "nan",
            f"{(st.pstdev(sp) if len(sp) > 1 else 0.0):.6f}" if sp else "nan",
            f"{st.mean(inf):.6f}" if inf else "nan",
            f"{st.mean(mc):.6f}" if mc else "nan",
        ])

print(f"[SUMMARY] Tail-grid CSV saved: {grid_out}")
PY
