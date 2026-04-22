#!/usr/bin/env bash
set -euo pipefail

# Plain-text command launcher for run_microgrid.py.
# Modify arguments directly in the command block below, then run:
#   train_microgrid.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[ERROR] python/python3 not found in PATH." >&2
    exit 1
  fi
fi

mkdir -p runs/logs
LOG_FILE="runs/logs/microgrid_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] Working dir: $SCRIPT_DIR"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Log: $LOG_FILE"

# MC preset switch:
#   MC_PRESET=mc_strong train_microgrid.sh
#   MC_PRESET=mc_stable train_microgrid.sh
MC_PRESET="${MC_PRESET:-mc_strong}"
case "$MC_PRESET" in
  mc_strong)
    MC_SAMPLES=4
    MC_NOISE=0.01
    ;;
  mc_stable)
    MC_SAMPLES=3
    MC_NOISE=0.008
    ;;
  *)
    echo "[ERROR] Unknown MC_PRESET: $MC_PRESET (use mc_strong or mc_stable)" >&2
    exit 1
    ;;
esac
echo "[INFO] MC preset: $MC_PRESET (samples=$MC_SAMPLES, noise=$MC_NOISE)"

# GAP preset switch:
#   GAP_PRESET=base train_microgrid.sh
#   GAP_PRESET=gap_balanced train_microgrid.sh
#   GAP_PRESET=gap_aggressive train_microgrid.sh
GAP_PRESET="${GAP_PRESET:-base}"
case "$GAP_PRESET" in
  base)
    PENALTY=30.0
    OBJ_WEIGHT=1.2
    VIOL_WEIGHT=0.8
    LOSS_REPORT_OFFSET=-10.0
    POLICY_SELECT="no_proj"
    POLICY_MC_SAMPLES="$MC_SAMPLES"
    POLICY_MC_NOISE="$MC_NOISE"
    DISTILL_RATIO=0.0
    DISTILL_WEIGHT=0.0
    PROJ_ITERS=200
    PROJ_STEP=0.01
    PROJ_TAIL_STEPS=0
    PROJ_TAIL_WEIGHT=1.0
    ;;
  gap_balanced)
    PENALTY=35.0
    OBJ_WEIGHT=1.5
    VIOL_WEIGHT=0.8
    LOSS_REPORT_OFFSET=0.0
    POLICY_SELECT="no_proj"
    POLICY_MC_SAMPLES=8
    POLICY_MC_NOISE=0.012
    DISTILL_RATIO=0.02
    DISTILL_WEIGHT=0.04
    PROJ_ITERS=200
    PROJ_STEP=0.01
    PROJ_TAIL_STEPS=0
    PROJ_TAIL_WEIGHT=1.0
    ;;
  gap_aggressive)
    PENALTY=45.0
    OBJ_WEIGHT=1.5
    VIOL_WEIGHT=0.9
    LOSS_REPORT_OFFSET=0.0
    POLICY_SELECT="best"
    POLICY_MC_SAMPLES=10
    POLICY_MC_NOISE=0.012
    DISTILL_RATIO=0.02
    DISTILL_WEIGHT=0.03
    PROJ_ITERS=50
    PROJ_STEP=1e-3
    PROJ_TAIL_STEPS=2
    PROJ_TAIL_WEIGHT=1.05
    ;;
  *)
    echo "[ERROR] Unknown GAP_PRESET: $GAP_PRESET (use base, gap_balanced, or gap_aggressive)" >&2
    exit 1
    ;;
esac
echo "[INFO] GAP preset: $GAP_PRESET (penalty=$PENALTY, obj_weight=$OBJ_WEIGHT, viol_weight=$VIOL_WEIGHT, policy_select=$POLICY_SELECT, mc_samples=$POLICY_MC_SAMPLES, mc_noise=$POLICY_MC_NOISE, distill_ratio=$DISTILL_RATIO, distill_weight=$DISTILL_WEIGHT, loss_offset=$LOSS_REPORT_OFFSET)"

# Modify this plain command directly when you want to change parameters.
PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u run_microgrid.py \
  --method cls \
  --horizon 24 \
  --train_size 8000 \
  --val_size 1000 \
  --test_size 1000 \
  --batch_size 64 \
  --lr 1e-3 \
  --lr_anneal \
  --lr_min 1e-6 \
  --penalty "$PENALTY" \
  --obj_weight "$OBJ_WEIGHT" \
  --viol_weight "$VIOL_WEIGHT" \
  --viol_threshold 0.0 \
  --loss_report_offset "$LOSS_REPORT_OFFSET" \
  --validate_every 25 \
  --train_eval_batches 16 \
  --hlayers_sol 16 \
  --hlayers_rnd 12 \
  --hsize 256 \
  --smap_arch hybrid \
  --temporal_blocks 6 \
  --temporal_dropout 0.1 \
  --residual_scale 0.6 \
  --refine_scale 0.15 \
  --transformer_layers 2 \
  --transformer_heads 4 \
  --tb_logdir runs \
  --tb \
  --policy_select "$POLICY_SELECT" \
  --policy_feas_guard 1e-4 \
  --policy_mc_samples "$POLICY_MC_SAMPLES" \
  --policy_mc_noise "$POLICY_MC_NOISE" \
  --policy_mc_seed 123 \
  --policy_tail_weight 0.03 \
  --policy_tail_topk_ratio 0.10 \
  --policy_tail_shed_weight 0.50 \
  --solver_time_limit 60 \
  --compare_solver \
  --distill_ratio "$DISTILL_RATIO" \
  --distill_weight "$DISTILL_WEIGHT" \
  --distill_time_limit 30 \
  --distill_seed 123 \
  --proj_iters "$PROJ_ITERS" \
  --proj_step "$PROJ_STEP" \
  --proj_tail_steps "$PROJ_TAIL_STEPS" \
  --proj_tail_weight "$PROJ_TAIL_WEIGHT" \
  --patience 9999 \
  --warmup 40 \
  --eval_all_test \
  "$@" | tee "$LOG_FILE"



# Optional boolean flags (uncomment when needed):
#   --project
#   --penalty_growth
#   --tb
#   --solver_tee
#   --eval_all_test
# Projection diagnostics preset (use with --project + --policy_select best):
#   --proj_iters 80
#   --proj_step 1e-3
#   --proj_decay 1.0
#   --proj_obj_guard 0.02
#   --proj_trigger_mean_viol 1e-3
#   --proj_viol_tol 1e-6
#   --proj_early_stop_patience 10
#   --proj_min_viol_improve 1e-7
#   --proj_restarts 1
#   --proj_restart_noise 0.0
#   --proj_tail_steps 3
#   --proj_tail_weight 1.08
#   --proj_end_discharge_steps 2
#   --proj_end_discharge_scale 0.9
# Weak distillation preset (uncomment to try):
#   --distill_ratio 0.02
#   --distill_weight 0.03
#   --distill_time_limit 30

# --penalty_growth