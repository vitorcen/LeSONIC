#!/usr/bin/env bash
# flow3 LOOPING demo — concatenated GR00T-predicted tokens of several LAFAN windows injected
# offline (no server, no GUI freeze) into the SONIC WBC, looping forever. Builds the timeline npz
# from $SEQ (ordered short-names) then launches the Isaac viewer.
#
#   bash scripts/gear_sonic_flow3.sh          # default loop: fight1 run3 fight2 run2 dance1 run1
#   SEQ=combat,circle,moonwalk bash scripts/gear_sonic_flow3.sh   # custom order
#
# Three-clock handling (else the env resets at the FIRST window's reference length, cutting the
# loop short): time_out=null + long episode_length_s + SONIC_NO_REF_RESAMPLE=1 (freeze reference
# clock) + the 4 deviation terminations nulled (open-loop drift must not reset).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"
PKL="${PKL:-data/seg_flow3_all.pkl}"
PRED_DIR="${PRED_DIR:-$REPO_ROOT/datasets/sonic_vla_pred_flow3}"
BLEND="${BLEND:-0}"
export DISPLAY="${DISPLAY:-:0}"
export SONIC_NO_REF_RESAMPLE=1   # freeze the reference motion clock (no resample/teleport at end)

# short-name -> window key (must match gear_sonic_inject.sh / the recorded windows)
declare -A KEY=(
  [combat]=fight_combat_combo_kicks [block]=fight_block_pushkick_shove [fierce]=fight_fierce_swings
  [jogback]=run_jog_backward [sprint]=run_sprint_backpedal [circle]=run_circle
  [moonwalk]=dance_moonwalk [spinclap]=dance_spin_stepback_clap
)
# default = user's flow3 order: fight1 run3 fight2 run2 dance1 run1
SEQ="${SEQ:-block,circle,combat,sprint,moonwalk,jogback}"

IFS=',' read -ra SHORTS <<< "$SEQ"
KEYS=()
for s in "${SHORTS[@]}"; do
  k="${KEY[$s]:-}"; [[ -z "$k" ]] && { echo "[flow3] unknown skill '$s'"; exit 2; }
  [[ -f "$PRED_DIR/$k.npz" ]] || { echo "[flow3] tokens missing: $PRED_DIR/$k.npz"; exit 1; }
  KEYS+=("$k")
done

LOOP_NPZ="$PRED_DIR/_flow3_loop.npz"
echo "[flow3] building timeline: ${SHORTS[*]}"
python3 "$REPO_ROOT/scripts/build_flow3_sequence.py" \
  --pred-dir "$PRED_DIR" --keys "${KEYS[@]}" --out "$LOOP_NPZ" --blend "$BLEND" || exit 1

# episode long enough for many loops; injector loops the timeline internally
EP_LEN_S="${EP_LEN_S:-36000}"
REF_KEY="${KEYS[0]}"   # any loaded reference (robot is token-driven; reference is cosmetic)

cd "$WBC_DIR"
echo "[flow3] launching looping demo (offline inject, no server). Viewer: F=free C=close V=ref R=reset."
conda run --no-capture-output -n "$ENV_NAME" python gear_sonic/eval_agent_trl.py \
    +checkpoint="$CKPT" \
    +headless=False \
    ++num_envs=1 \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    ++manager_env.commands.motion.use_paired_motions=True \
    "++manager_env.commands.motion.debug_vis=${DEBUG_VIS:-False}" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$PKL" \
    "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=$REF_KEY" \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
    ++manager_env.terminations.anchor_pos=null \
    ++manager_env.terminations.ee_body_pos=null \
    ++manager_env.terminations.anchor_ori_full=null \
    ++manager_env.terminations.foot_pos_xyz=null \
    ++manager_env.terminations.time_out=null \
    "++manager_env.config.episode_length_s=$EP_LEN_S" \
    "++eval_callbacks=[vla_injector]" \
    "++callbacks.vla_injector._target_=gear_sonic.data.vla_token_injector.VlaTokenInjector" \
    "++callbacks.vla_injector.token_npz=$LOOP_NPZ" \
    "++callbacks.vla_injector.loop=True"

echo "[flow3] viewer closed."
