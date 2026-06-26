#!/usr/bin/env bash
# Finetune GR00T N1.7-3B on the SONIC token dataset (architecture-side VLA, Stage C derisk).
# Teaches GR00T to emit the 64-dim SONIC motion_token from (ego_view, prompt); SONIC WBC
# decodes it to balanced joints. Single RTX 4090 (24GB) recipe — see skill gr00t-4090-finetune.
#
#   bash scripts/gr00t_sonic_finetune.sh           # derisk run (MAX_STEPS=2000)
#   SMOKE=1 bash scripts/gr00t_sonic_finetune.sh   # 2-step smoke: validate load + VRAM fit
#   MAX_STEPS=8000 bash scripts/gr00t_sonic_finetune.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_DIR="$REPO_ROOT/dependencies/Isaac-GR00T"
DATASET="${DATASET:-$REPO_ROOT/datasets/sonic_vla_lerobot}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/gr00t_sonic_derisk}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_LIMIT="${SAVE_LIMIT:-3}"   # rolling temporaries for crash recovery; stage ckpts kept via keep/
# Phase-1 robustness (doc/sonic_robustness_retrain.html §5-A): additive Gaussian noise on the
# normalized state (input-noise augmentation). STATE_NOISE_STD=0.08 widens the closed-loop basin
# so OOD reset/WBC-drift proprio stops collapsing the SONIC token chunk into a frozen "hold pose".
STATE_NOISE_STD="${STATE_NOISE_STD:-0.0}"
STATE_NOISE_PROB="${STATE_NOISE_PROB:-0.5}"
# Tier-1 stand->onset robustness: zero the proprio state (STATE_DROPOUT_PROB) and/or drop the text
# prompt (PROMPT_DROPOUT_PROB) in train mode -> breaks state-dominance so the prompt can win at a
# cold stand, and trains a null-prompt branch for inference-time classifier-free guidance.
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.2}"
PROMPT_DROPOUT_PROB="${PROMPT_DROPOUT_PROB:-0.0}"
[[ "${SMOKE:-0}" == "1" ]] && MAX_STEPS=2 && SAVE_STEPS=1000

# --- 4090 (24GB) gotcha fixes (skill gr00t-4090-finetune) ---
export COMPILE_ACTION_HEAD_DISABLE=1               # torch.compile inductor crash
export DATALOADER_NUM_WORKERS=0                    # forked-worker CUDA corruption
export PIPELINE_OVERLAP_DISABLE=1                  # non_blocking H2D corruption
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GR00T_OPTIM="${GR00T_OPTIM:-adafactor}"     # no momentum state (adamw OOMs 24GB)
export GR00T_GRAD_CKPT="${GR00T_GRAD_CKPT:-1}"     # checkpoint activations (use_reentrant=False)
# DO NOT set HF_HUB_OFFLINE=1 (Qwen3VL tokenizer does one model_info() call).

cd "$GR00T_DIR"
echo "[ft] dataset=$DATASET  base=$BASE_MODEL  max_steps=$MAX_STEPS  out=$OUT_DIR"
echo "[ft] micro-batch=1 (global=4 / grad_accum=4 / 1 GPU); freeze VLM, train DiT head."
echo "[ft] state_noise_std=$STATE_NOISE_STD prob=$STATE_NOISE_PROB state_dropout=$STATE_DROPOUT_PROB prompt_dropout=$PROMPT_DROPOUT_PROB (0=off; Tier-1 robustness aug)."

.venv/bin/python gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET" \
    --embodiment-tag unitree_g1_sonic \
    --modality-config-path gr00t/configs/data/embodiment_configs.py \
    --global-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --dataloader-num-workers 0 \
    --num-gpus 1 \
    --state-noise-std "$STATE_NOISE_STD" \
    --state-noise-prob "$STATE_NOISE_PROB" \
    --state-dropout-prob "$STATE_DROPOUT_PROB" \
    --prompt-dropout-prob "$PROMPT_DROPOUT_PROB" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit "$SAVE_LIMIT" \
    --max-steps "$MAX_STEPS" \
    --output-dir "$OUT_DIR"

echo "[ft] exit $?"
