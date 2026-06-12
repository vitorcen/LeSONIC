#!/usr/bin/env bash
# Finetune StarVLA (QwenPI_v3, frozen Qwen3.5-4B VLM) on the SONIC LAFAN flow3
# token dataset — the A/B baseline against GR00T N1.7 (same data, same sample
# budget, same frozen WBC). doc/sonic_starvla_swap_brainstorm.html §11.
#
#   bash scripts/starvla_sonic_finetune.sh             # full A/B run (6000 steps)
#   SMOKE=1 bash scripts/starvla_sonic_finetune.sh     # ~500-step smoke gate
#   RESUME=1 bash scripts/starvla_sonic_finetune.sh    # resume latest ckpt
#   BATCH=2 MAX_STEPS=12000 ...                        # OOM fallback (same samples)
#
# Runs LOCALLY (single 4090-24G) in the starvla_eval_qwen35 conda env
# (transformers 5.2.0 for Qwen3.5; deepspeed installed). No `conda activate` —
# full env-binary paths.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# starVLA engine: LeSONIC-nested submodule = vitorcen/StarVLA fork (branch starVLA_dev), self-contained.
STARVLA_DIR="${STARVLA_DIR:-$REPO_ROOT/dependencies/starVLA}"
ENV_BIN="${ENV_BIN:-$HOME/miniconda3/envs/starvla_eval_qwen35/bin}"
CONFIG_SRC="${CONFIG:-$REPO_ROOT/scripts/starvla/configs/sonic_qwen3_5_4b_pi_v3.yaml}"
CONFIG_SRC="$(readlink -f "$CONFIG_SRC")"   # absolutize: trainer launches with cwd=STARVLA_DIR
[[ -f "$CONFIG_SRC" ]] || { echo "[ft] config not found: ${CONFIG:-<default>}" >&2; exit 2; }
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasets}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/outputs/starvla}"
MAX_STEPS="${MAX_STEPS:-6000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
KEEP="${KEEP:-3}"
[[ "${SMOKE:-0}" == "1" ]] && MAX_STEPS=500 && SAVE_INTERVAL=500

# Deploy the data_registry kit into the starVLA repo (registry.py only scans
# examples/*/train_files/data_registry). Symlink = single source of truth here.
KIT="$STARVLA_DIR/examples/UNITREE_G1_SONIC/train_files"
mkdir -p "$KIT"
ln -sfn "$REPO_ROOT/scripts/starvla/data_registry" "$KIT/data_registry"
# QwenPI_CE head (framework.name=QwenPI_CE) ships in the vitorcen/StarVLA fork
# checkout at starVLA/model/framework/VLM4A/QwenPI_CE.py — no runtime deploy needed.

# Pre-run GPU hygiene (feedback-pre-run-gpu-check): refuse if >2GB already used.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if (( USED > 2000 )); then
  echo "[ft] GPU already has ${USED}MiB in use — clean up stale processes first." >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=8.9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

cd "$STARVLA_DIR"
OVERRIDES="--datasets.vla_data.data_root_dir $DATA_ROOT --run_root_dir $RUN_ROOT"
OVERRIDES="$OVERRIDES --trainer.max_train_steps $MAX_STEPS --trainer.save_interval $SAVE_INTERVAL"
OVERRIDES="$OVERRIDES --trainer.keep_last_checkpoints $KEEP"
[[ -n "${BATCH:-}" ]] && OVERRIDES="$OVERRIDES --datasets.vla_data.per_device_batch_size $BATCH"
[[ -n "${RUN_ID:-}" ]] && OVERRIDES="$OVERRIDES --run_id $RUN_ID"
[[ "${RESUME:-0}" == "1" ]] && OVERRIDES="$OVERRIDES --trainer.is_resume True"

echo "[ft] config=$CONFIG_SRC max_steps=$MAX_STEPS bs=${BATCH:-4(cfg)} out=$RUN_ROOT"
"$ENV_BIN/accelerate" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  --main_process_port "${MAIN_PORT:-29531}" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG_SRC" \
  $OVERRIDES
echo "[ft] exit $?"
