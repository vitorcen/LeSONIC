#!/usr/bin/env bash
# Record SONIC motion-tokens for the P3 same-domain corpus (15 contiguous fight/run runs,
# flow3-EXCLUDED, WBC-trackable). Manifest-driven (outputs/p3_runs_manifest.csv).
#
# Same phys regime as gear_sonic_record_flow3.sh: the 4 strict adaptive deviation terminations
# nulled (fast strikes/spins trip strict but never fall), motion_time_out left on so each run
# plays once. Disabling terms does NOT change the WBC forward pass — tapped tokens == deployment.
#
#   bash scripts/gear_sonic_record_p3.sh                 # all 15 runs
#   bash scripts/gear_sonic_record_p3.sh fight_p3run07   # one run
#   SMOKE=1 bash scripts/gear_sonic_record_p3.sh fight_p3run07   # 30-step smoke
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"
PKL="${PKL:-data/seg_p3_runs_all.pkl}"
MANIFEST="${MANIFEST:-$REPO_ROOT/MaskBeT/outputs/p3_runs_manifest.csv}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/datasets/sonic_vla_raw_p3}"
CAM_RES="${CAM_RES:-[480,640]}"
SMOKE="${SMOKE:-0}"
export DISPLAY="${DISPLAY:-:0}"

[[ -f "$MANIFEST" ]] || { echo "[rec-p3] manifest missing: $MANIFEST"; exit 1; }

# parse manifest (skip header): key,family,fs,fe,frames,steps,prompt(may be quoted)
declare -A STEPS PROMPT
ORDER=()
while IFS= read -r line; do
  key="${line%%,*}"
  rest="${line#*,}"                       # family,fs,fe,frames,steps,prompt
  steps="$(echo "$rest" | cut -d, -f5)"
  prompt="$(echo "$line" | sed -E 's/\r$//; s/^([^,]*,){6}//; s/"//g')"
  STEPS["$key"]="$steps"; PROMPT["$key"]="$prompt"; ORDER+=("$key")
done < <(tail -n +2 "$MANIFEST")

SEL="${1:-all}"
[[ "$SEL" == "all" ]] && TARGETS=("${ORDER[@]}") || TARGETS=("$SEL")

cd "$WBC_DIR"
[[ -f "$PKL" ]] || PKL="$WBC_DIR/$PKL"
[[ -f "$PKL" ]] || { echo "[rec-p3] pkl missing: $PKL"; exit 1; }
REC_TARGET="gear_sonic.data.vla_token_recorder.VlaTokenRecorder"

for k in "${TARGETS[@]}"; do
  steps="${STEPS[$k]:-}"; [[ -z "$steps" ]] && { echo "[rec-p3] unknown run '$k'"; continue; }
  [[ "$SMOKE" == "1" ]] && steps=30
  tag="000"
  echo "[rec-p3] run=$k steps=$steps prompt='${PROMPT[$k]}' -> $OUT_DIR/$k/episode_$tag.npz"
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
      || { echo "[rec-p3] FAILED on $k"; exit 1; }
done
echo "[rec-p3] done -> $OUT_DIR"
