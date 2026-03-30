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
  --penalty 30.0 \
  --obj_weight 1.2 \
  --viol_weight 0.8 \
  --viol_threshold 0.0 \
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
  --policy_select no_proj \
  --policy_feas_guard 1e-4 \
  --solver_time_limit 60 \
  --compare_solver \
  --distill_ratio 0.0 \
  --distill_weight 0.0 \
  --distill_time_limit 30 \
  --distill_seed 123 \
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