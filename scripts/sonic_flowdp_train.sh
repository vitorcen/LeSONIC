#!/usr/bin/env bash
# Train the FlowDP head (conv-UNet + rectified-flow, from LeIsaac/FlowHeads) on the
# BonesSeed 7-motion SONIC token dataset.
#
# Motion selection rides in observation.state: state(53) = joint(43) + gravity(3) +
# motion_onehot(7). FlowDP / Diffusion-Policy has NO language path, so the one-hot is
# how the single model tells "kick" from "dance". Action(78) = motion_token(64) +
# left_hand(7) + right_hand(7); the serve adapter returns only the 64-dim token.
#
#   bash scripts/sonic_flowdp_train.sh            # full run (detached recommended)
#   STEPS=10 SAVE_FREQ=10 bash scripts/sonic_flowdp_train.sh   # smoke (build + 1 ckpt)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOWHEADS="${FLOWHEADS:-$REPO_ROOT/dependencies/FlowHeads}"
PY="${PY:-$HOME/miniconda3/envs/lerobot-v044/bin/python}"

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/sonic_vla_flowdp}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/flowdp_sonic}"

# Epoch-aware budget (batch 64 -> ~60 steps/epoch on the 3815-frame BonesSeed set).
# The flowdp sweet spot is a few epochs (LeIsaac flowdp PickOrange peaked ~4.3 ep;
# GR00T SONIC ~8 ep), NOT hundreds — so save densely over a low-epoch range and let
# the open-loop similarity sweep pick the best. 2400 steps ~= 40 ep, save every 2 ep.
STEPS="${STEPS:-2400}"
SAVE_FREQ="${SAVE_FREQ:-120}"           # ~2 epochs/ckpt -> 20 ckpts across 2..40 ep
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FLOW_STEPS="${FLOW_STEPS:-10}"          # Euler NFE at sampling; sweep 1/2/4/8 at serve
HORIZON="${HORIZON:-40}"                # mirror GR00T action_horizon (divisible by 8 for UNet)
N_OBS_STEPS="${N_OBS_STEPS:-1}"         # single-frame obs (matches the live injector)
N_ACTION_STEPS="${N_ACTION_STEPS:-32}"  # <= horizon - n_obs_steps + 1
RESIZE="${RESIZE:-[240,320]}"           # ego cam looks at the ground; half-res is plenty

[ -d "$DATASET_ROOT" ] || { echo "[train] missing dataset $DATASET_ROOT (run sonic_flowdp_build_dataset.py)"; exit 1; }
[ -d "$FLOWHEADS/flowdp" ] || { echo "[train] missing FlowHeads at $FLOWHEADS"; exit 1; }

echo "[train] FlowDP-SONIC | steps=$STEPS batch=$BATCH_SIZE horizon=$HORIZON n_obs=$N_OBS_STEPS flow_steps=$FLOW_STEPS resize=$RESIZE"
echo "[train] dataset=$DATASET_ROOT  out=$OUTPUT_DIR"

cd "$FLOWHEADS"
export PYTHONPATH="$FLOWHEADS:${PYTHONPATH:-}"
export HF_HUB_DISABLE_XET=1
# Proprio-state noise augmentation for closed-loop / real-robot robustness (default off).
# SONIC state(53)=joint43+gravity3+onehot7. Noise ONLY the 43 joints: isotropic Gaussian on
# projected_gravity (a ~unit vector) would leave its manifold (review P1), and the onehot is
# the motion selector — both must stay clean. (Gravity-as-small-rotation aug = future work.)
export FLOWDP_STATE_NOISE="${FLOWDP_STATE_NOISE:-0}"
export FLOWDP_STATE_NOISE_DIMS="${FLOWDP_STATE_NOISE_DIMS:-43}"
echo "[train] proprio aug: FLOWDP_STATE_NOISE=$FLOWDP_STATE_NOISE dims=$FLOWDP_STATE_NOISE_DIMS"

# RESUME=1: continue the latest checkpoint (used by the auto-resume watchdog after a
# kernel-6.17 heap-corruption crash). lerobot rebuilds optimizer + fast-forwards the
# scheduler from <ckpt>/train_config.json.
LAST_CFG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
if [ "${RESUME:-0}" = "1" ] && [ -f "$LAST_CFG" ]; then
  echo "[train] RESUME from $LAST_CFG"
  exec "$PY" -m flowdp.train --config_path="$LAST_CFG" --resume=true
fi

exec "$PY" -m flowdp.train \
  --policy.type=flowdp \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.n_obs_steps="$N_OBS_STEPS" \
  --policy.horizon="$HORIZON" \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --policy.resize_shape="$RESIZE" \
  --policy.num_inference_steps="$FLOW_STEPS" \
  --dataset.repo_id=local/sonic_vla_flowdp \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --output_dir="$OUTPUT_DIR" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --num_workers="$NUM_WORKERS" \
  --wandb.enable=false
