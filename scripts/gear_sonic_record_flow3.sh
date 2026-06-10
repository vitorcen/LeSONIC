#!/usr/bin/env bash
# Record SONIC motion-tokens for the flow3 (LAFAN hand-picked windows) VLA dataset.
#
# Unlike gear_sonic_record.sh (7 in-dist demos), these windows pass the PHYSICAL-validity screen
# (never fall) but TRIP the strict adaptive deviation terminations on fast strikes/spins. So we
# record under the PHYS regime — the 4 strict deviation terms set to null — otherwise strict would
# cut the episode short (e.g. fight_combat_combo_kicks terminates @31% under strict but never
# falls). motion_time_out stays on so each window plays once. Disabling terms does NOT change the
# WBC policy forward pass (only episode reset), so the tapped tokens are identical to deployment.
#
# Prefix windows (block_pushkick_shove, fierce_swings) are recorded only up to their fall_step
# (the trackable prefix), with honestly-relabeled prompts.
#
#   bash scripts/gear_sonic_record_flow3.sh            # all flow3 windows
#   bash scripts/gear_sonic_record_flow3.sh run_circle # one window
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"
PKL="${PKL:-data/seg_flow3_all.pkl}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/datasets/sonic_vla_raw_flow3}"
CAM_RES="${CAM_RES:-[480,640]}"
SMOKE="${SMOKE:-0}"
export DISPLAY="${DISPLAY:-:0}"

# key | max_steps (window_steps for full, fall_step for prefix) | prompt
declare -A STEPS=(
  [fight_combat_combo_kicks]=749 [run_jog_backward]=749 [run_sprint_backpedal]=749
  [run_circle]=749 [dance_moonwalk]=999 [dance_spin_stepback_clap]=999
  [fight_block_pushkick_shove]=542 [fight_fierce_swings]=241
)
declare -A PROMPT=(
  [fight_combat_combo_kicks]="combat strikes and combo kicks"
  [run_jog_backward]="jog forward then run backward"
  [run_sprint_backpedal]="sprint back and forth then backpedal"
  [run_circle]="run in a circle"
  [dance_moonwalk]="moonwalk"
  [dance_spin_stepback_clap]="spin, step back, and clap"
  [fight_block_pushkick_shove]="block and push-kick"
  [fight_fierce_swings]="fierce swings"
)
ORDER=(fight_combat_combo_kicks run_jog_backward run_sprint_backpedal run_circle \
       dance_moonwalk dance_spin_stepback_clap fight_block_pushkick_shove fight_fierce_swings)

SEL="${1:-all}"
[[ "$SEL" == "all" ]] && TARGETS=("${ORDER[@]}") || TARGETS=("$SEL")

cd "$WBC_DIR"
[[ -f "$PKL" ]] || PKL="$WBC_DIR/$PKL"
[[ -f "$PKL" ]] || { echo "[rec-flow3] pkl missing: $PKL"; exit 1; }
REC_TARGET="gear_sonic.data.vla_token_recorder.VlaTokenRecorder"

for k in "${TARGETS[@]}"; do
  steps="${STEPS[$k]:-}"; [[ -z "$steps" ]] && { echo "[rec-flow3] unknown window '$k'"; continue; }
  [[ "$SMOKE" == "1" ]] && steps=30
  tag="000"
  echo "[rec-flow3] window=$k steps=$steps prompt='${PROMPT[$k]}' -> $OUT_DIR/$k/episode_$tag.npz"
  conda run --no-capture-output -n "$ENV_NAME" python gear_sonic/eval_agent_trl.py \
      +checkpoint="$CKPT" \
      +headless=True \
      ++num_envs=1 \
      ++manager_env.config.enable_cameras=True \
      "++manager_env.config.cameras.camera_resolution=$CAM_RES" \
      ++manager_env.observations.policy.enable_corruption=False \
      ++manager_env.observations.tokenizer.enable_corruption=False \
      ++manager_env.commands.motion.use_paired_motions=True \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=$PKL" \
      "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=$k" \
      ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
      ++manager_env.terminations.anchor_pos=null \
      ++manager_env.terminations.ee_body_pos=null \
      ++manager_env.terminations.anchor_ori_full=null \
      ++manager_env.terminations.foot_pos_xyz=null \
      "++eval_callbacks=[vla_recorder]" \
      "++callbacks.vla_recorder._target_=$REC_TARGET" \
      "++callbacks.vla_recorder.output_dir=$OUT_DIR" \
      "++callbacks.vla_recorder.motion_key=$k" \
      "++callbacks.vla_recorder.prompt='${PROMPT[$k]}'" \
      "++callbacks.vla_recorder.max_steps=$steps" \
      "++callbacks.vla_recorder.episode_tag=$tag" \
      || { echo "[rec-flow3] FAILED on $k"; exit 1; }
done
echo "[rec-flow3] done -> $OUT_DIR"
