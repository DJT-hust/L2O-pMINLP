#!/usr/bin/env bash
set -euo pipefail

# Plain-text command launcher for run_multidc.py.
# Modify arguments directly in the command block below, then run:
#   train_multidc.sh

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
LOG_FILE="runs/logs/multidc_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] Working dir: $SCRIPT_DIR"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Log: $LOG_FILE"

# Mode preset:
#   MODE_PRESET=compare train_multidc.sh
#   MODE_PRESET=policy_only train_multidc.sh
#   MODE_PRESET=solver_only train_multidc.sh
MODE_PRESET="${MODE_PRESET:-policy_only}"
case "$MODE_PRESET" in
  compare)
    METHOD="cls"
    COMPARE_FLAG="--compare_solver"
    ;;
  policy_only)
    METHOD="cls"
    COMPARE_FLAG=""
    ;;
  solver_only)
    METHOD="solver"
    COMPARE_FLAG=""
    ;;
  *)
    echo "[ERROR] Unknown MODE_PRESET: $MODE_PRESET (use compare, policy_only, solver_only)" >&2
    exit 1
    ;;
esac

# Size preset:
#   SIZE_PRESET=tiny train_multidc.sh
#   SIZE_PRESET=dev train_multidc.sh
#   SIZE_PRESET=full train_multidc.sh
SIZE_PRESET="${SIZE_PRESET:-full}"
case "$SIZE_PRESET" in
  tiny)
    TRAIN_SIZE=640
    VAL_SIZE=160
    TEST_SIZE=160
    BATCH_SIZE=16
    EVAL_SAMPLES=5
    ;;
  dev)
    TRAIN_SIZE=5120
    VAL_SIZE=1280
    TEST_SIZE=1280
    BATCH_SIZE=32
    EVAL_SAMPLES=20
    ;;
  full)
    TRAIN_SIZE=30000
    VAL_SIZE=5000
    TEST_SIZE=5000
    BATCH_SIZE=512
    EVAL_SAMPLES=50
    ;;
  *)
    echo "[ERROR] Unknown SIZE_PRESET: $SIZE_PRESET (use tiny, dev, full)" >&2
    exit 1
    ;;
esac

echo "[INFO] MODE_PRESET=$MODE_PRESET, SIZE_PRESET=$SIZE_PRESET"
echo "[INFO] method=$METHOD, train/val/test=$TRAIN_SIZE/$VAL_SIZE/$TEST_SIZE, batch=$BATCH_SIZE"

# Policy architecture preset:
#   SMAP_ARCH=mlp train_multidc.sh
#   SMAP_ARCH=lstm train_multidc.sh
#   SMAP_ARCH=rnn train_multidc.sh
#   SMAP_ARCH=tcn train_multidc.sh
# Head/Residual switches:
#   SMAP_HEAD_MODE=single train_multidc.sh
#   SMAP_HEAD_MODE=dual train_multidc.sh
#   SMAP_RESIDUAL=1 SMAP_RESIDUAL_SCALE=0.3 train_multidc.sh
# Constraint-aware feature injection:
#   CONSTRAINT_FEAT_INJECT=1 train_multidc.sh
# LSTM-only knobs (ignored by mlp):
#   LSTM_LAYERS=2 LSTM_DROPOUT=0.1 SMAP_ARCH=lstm train_multidc.sh
# RNN-only knobs:
#   RNN_LAYERS=2 RNN_DROPOUT=0.1 SMAP_ARCH=rnn train_multidc.sh
# TCN-only knobs:
#   TCN_BLOCKS=4 TCN_DROPOUT=0.1 SMAP_ARCH=tcn train_multidc.sh
SMAP_ARCH="${SMAP_ARCH:-mlp}"
SMAP_HEAD_MODE="${SMAP_HEAD_MODE:-single}"
SMAP_RESIDUAL="${SMAP_RESIDUAL:-0}"
SMAP_RESIDUAL_SCALE="${SMAP_RESIDUAL_SCALE:-0.3}"
CONSTRAINT_FEAT_INJECT="${CONSTRAINT_FEAT_INJECT:-0}"
LSTM_LAYERS="${LSTM_LAYERS:-2}"
LSTM_DROPOUT="${LSTM_DROPOUT:-0.1}"
RNN_LAYERS="${RNN_LAYERS:-2}"
RNN_DROPOUT="${RNN_DROPOUT:-0.1}"
TCN_BLOCKS="${TCN_BLOCKS:-4}"
TCN_DROPOUT="${TCN_DROPOUT:-0.1}"
echo "[INFO] smap_arch=$SMAP_ARCH, head_mode=$SMAP_HEAD_MODE, residual=$SMAP_RESIDUAL, residual_scale=$SMAP_RESIDUAL_SCALE, constraint_feat_inject=$CONSTRAINT_FEAT_INJECT, lstm_layers=$LSTM_LAYERS, lstm_dropout=$LSTM_DROPOUT, rnn_layers=$RNN_LAYERS, rnn_dropout=$RNN_DROPOUT, tcn_blocks=$TCN_BLOCKS, tcn_dropout=$TCN_DROPOUT"

# CUDA selection:
#   CUDA_SLOT=auto train_multidc.sh   # mlp/lstm->0, rnn/tcn->1
#   CUDA_SLOT=0 train_multidc.sh
#   CUDA_SLOT=1 train_multidc.sh
CUDA_SLOT="${CUDA_SLOT:-auto}"
if [[ "$CUDA_SLOT" == "auto" ]]; then
  case "$SMAP_ARCH" in
    mlp|lstm)
      CUDA_VISIBLE_DEVICES=0
      ;;
    rnn|tcn)
      CUDA_VISIBLE_DEVICES=1
      ;;
    *)
      echo "[ERROR] Unknown SMAP_ARCH for auto CUDA mapping: $SMAP_ARCH" >&2
      exit 1
      ;;
  esac
else
  CUDA_VISIBLE_DEVICES="$CUDA_SLOT"
fi
export CUDA_VISIBLE_DEVICES
echo "[INFO] CUDA_SLOT=$CUDA_SLOT, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Model-scale preset:
#   MODEL_PRESET=base train_multidc.sh
#   MODEL_PRESET=large train_multidc.sh
#   MODEL_PRESET=solver5m train_multidc.sh
# Solver time limit convention:
#   SOLVER_TIME_LIMIT<=0 means unlimited (no TimeLimit).
MODEL_PRESET="${MODEL_PRESET:-base}"
case "$MODEL_PRESET" in
  base)
    HORIZON=288
    NUM_DC=3
    NUM_REGIONS=3
    NUM_JOBS=4
    DC_NODES_PER_DC=100
    # Keep per-DC aggregate compute near historical base (3.0).
    CPU_PER_NODE=0.03
    SOLVER_TIME_LIMIT=0
    ;;
  large)
    HORIZON=288
    NUM_DC=6
    NUM_REGIONS=6
    NUM_JOBS=16
    DC_NODES_PER_DC=8
    CPU_PER_NODE=0.8
    SOLVER_TIME_LIMIT=-1
    ;;
  solver5m)
    # Heavier MINLP intended to push solver close to ~5 minutes per sample.
    HORIZON=288
    NUM_DC=8
    NUM_REGIONS=8
    NUM_JOBS=32
    DC_NODES_PER_DC=12
    CPU_PER_NODE=0.7
    SOLVER_TIME_LIMIT=0
    # Calibration mode usually runs solver only with one sample.
    EVAL_SAMPLES="${EVAL_SAMPLES:-1}"
    ;;
  *)
    echo "[ERROR] Unknown MODEL_PRESET: $MODEL_PRESET (use base, large, solver5m)" >&2
    exit 1
    ;;
esac

# Node densification:
# Keep per-DC aggregate compute approximately unchanged while increasing node count.
# Example:
#   NODE_DENSIFY=20 train_multidc.sh
# This multiplies DC_NODES_PER_DC by 20 and divides CPU_PER_NODE by 20.
NODE_DENSIFY="${NODE_DENSIFY:-1}"
if ! [[ "$NODE_DENSIFY" =~ ^[0-9]+$ ]] || [[ "$NODE_DENSIFY" -lt 1 ]]; then
  echo "[ERROR] NODE_DENSIFY must be a positive integer." >&2
  exit 1
fi

if [[ "$NODE_DENSIFY" -gt 1 ]]; then
  DC_NODES_PER_DC=$((DC_NODES_PER_DC * NODE_DENSIFY))
  CPU_PER_NODE=$(awk "BEGIN{printf \"%.8f\", $CPU_PER_NODE / $NODE_DENSIFY}")
fi

echo "[INFO] MODEL_PRESET=$MODEL_PRESET, horizon=$HORIZON, dc/reg/jobs=$NUM_DC/$NUM_REGIONS/$NUM_JOBS, dc_nodes_per_dc=$DC_NODES_PER_DC, cpu_per_node=$CPU_PER_NODE, node_densify=$NODE_DENSIFY"

# Penalty preset:
#   PEN_PRESET=light train_multidc.sh
#   PEN_PRESET=base train_multidc.sh
#   PEN_PRESET=strong train_multidc.sh
#   PEN_PRESET=gap_tuned train_multidc.sh
PEN_PRESET="${PEN_PRESET:-base}"
case "$PEN_PRESET" in
  light)
    PENALTY=10
    EQ_WEIGHT=2.0
    OBJ_WEIGHT=1.0
    VIOL_WEIGHT=1.0
    ;;
  base)
    # Validated policy-only baseline from recent robustness runs.
    PENALTY=55
    EQ_WEIGHT=1.5
    OBJ_WEIGHT=1.5
    VIOL_WEIGHT=1.1
    ;;
  strong)
    PENALTY=100
    EQ_WEIGHT=2.0
    OBJ_WEIGHT=1.0
    VIOL_WEIGHT=1.0
    ;;
  gap_tuned)
    # Use when feasibility is already near-zero violation and you want smaller policy-solver gap.
    PENALTY=30
    EQ_WEIGHT=1.2
    OBJ_WEIGHT=2.0
    VIOL_WEIGHT=0.8
    ;;
  *)
    echo "[ERROR] Unknown PEN_PRESET: $PEN_PRESET (use light, base, strong, gap_tuned)" >&2
    exit 1
    ;;
esac

echo "[INFO] PEN_PRESET=$PEN_PRESET, penalty=$PENALTY, eq_w=$EQ_WEIGHT, obj_w=$OBJ_WEIGHT, viol_w=$VIOL_WEIGHT"

# Epochs override:
#   EPOCHS=50 train_multidc.sh
EPOCHS="${EPOCHS:-600}"
echo "[INFO] epochs=$EPOCHS"

# Training stability knobs:
#   LR=1e-3 PATIENCE=80 WARMUP=20 VALIDATE_EVERY=20 train_multidc.sh
LR="${LR:-3e-4}"
PATIENCE="${PATIENCE:-140}"
WARMUP="${WARMUP:-20}"
VALIDATE_EVERY="${VALIDATE_EVERY:-20}"
echo "[INFO] lr=$LR, patience=$PATIENCE, warmup=$WARMUP, validate_every=$VALIDATE_EVERY"

# Modify this plain command directly when you want to change parameters.
PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u run_multidc.py \
  --method "$METHOD" \
  --seed 50 \
  --data_seed 17 \
  --split_seed 42 \
  --horizon "$HORIZON" \
  --num_dc "$NUM_DC" \
  --num_regions "$NUM_REGIONS" \
  --num_jobs "$NUM_JOBS" \
  --dc_nodes_per_dc "$DC_NODES_PER_DC" \
  --cpu_per_node "$CPU_PER_NODE" \
  --train_size "$TRAIN_SIZE" \
  --val_size "$VAL_SIZE" \
  --test_size "$TEST_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --penalty "$PENALTY" \
  --eq_weight "$EQ_WEIGHT" \
  --obj_weight "$OBJ_WEIGHT" \
  --viol_weight "$VIOL_WEIGHT" \
  --hlayers_sol 6 \
  --hlayers_rnd 4 \
  --hsize 128 \
  --smap_arch "$SMAP_ARCH" \
  --smap_head_mode "$SMAP_HEAD_MODE" \
  --smap_residual "$SMAP_RESIDUAL" \
  --smap_residual_scale "$SMAP_RESIDUAL_SCALE" \
  --constraint_feat_inject "$CONSTRAINT_FEAT_INJECT" \
  --lstm_layers "$LSTM_LAYERS" \
  --lstm_dropout "$LSTM_DROPOUT" \
  --rnn_layers "$RNN_LAYERS" \
  --rnn_dropout "$RNN_DROPOUT" \
  --tcn_blocks "$TCN_BLOCKS" \
  --tcn_dropout "$TCN_DROPOUT" \
  --patience 9999 \
  --epochs "$EPOCHS" \
  --warmup "$WARMUP" \
  --validate_every "$VALIDATE_EVERY" \
  --train_eval_batches 8 \
  --solver_time_limit "$SOLVER_TIME_LIMIT" \
  --eval_samples "$EVAL_SAMPLES" \
  --tb_logdir runs \
  $COMPARE_FLAG \
  "$@" | tee "$LOG_FILE"

# Optional boolean flags (append at command line when needed):
#   --tb
#   --penalty_growth
#   --solver_tee
