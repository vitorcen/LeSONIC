#!/usr/bin/env bash
# ⚠️ This is the WBC PHYSICS TRACKABILITY SCREEN, not a visual preview. It runs the frozen SONIC
# WBC physically tracking the reference — if the WBC can't balance the motion (e.g. dance/jumps)
# the robot FALLS, and that fall is the useful signal (= segment not WBC-trackable). To just SEE
# a motion without falling (to pick windows), use scripts/gear_sonic_replay_window.sh (kinematic).
# ---
# Preview the RAW reference motion of ANY robot_filtered key via pure SONIC WBC tracking
# (NO GR00T, NO server, NO VLA) — to VISUALLY VERIFY what a segment actually contains before
# trusting its auto-assigned skill label. This is what you watch to confirm a cut is clean.
#
#   PKL=data/fight_segments_robot.pkl bash scripts/gear_sonic_preview_motion.sh fight_seg020
#   PKL=data/run_segments_robot.pkl   bash scripts/gear_sonic_preview_motion.sh run_seg006
#
# Viewer: C = snap close to robot, F = free cam, V = ref skeleton, R = reset. Grey Studio light,
# no marker balls. RELAX=1 plays the whole segment without deviation-reset.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"
PKL="${PKL:?set PKL=data/<fight|run>_segments_robot.pkl}"
KEY="${1:-${KEY:?pass a motion key, e.g. fight_seg020}}"
RELAX="${RELAX:-1}"
# Play the clip ONCE then HOLD the last frame (SONIC_PLAY_ONCE) — looping makes it impossible to
# read off the time boundary. Press R in the viewer to replay. Set SONIC_PLAY_ONCE=0 to loop.
export SONIC_PLAY_ONCE="${SONIC_PLAY_ONCE:-1}"
EP_LEN_S="${EP_LEN_S:-600}"   # long episode so gym time-truncation doesn't cut a long clip
unset HEADLESS
export DISPLAY="${DISPLAY:-:0}"

cd "$WBC_DIR"
[[ -f "$PKL" ]] || { echo "[preview] PKL missing: $PKL"; exit 1; }
echo "[preview] RAW WBC tracking: key=$KEY  pkl=$PKL  (no VLA, no server)"
echo "  Viewer: C=close cam, F=free, V=ref skeleton, R=reset."

# For a PREVIEW we never want a reset: DISABLE all tracking terminations entirely (set term=null,
# not just threshold=99 — the *_adaptive terms ignore the static threshold and recompute their own,
# so the robot still terminates on the hard parts of a long clip -> reset -> looks like a loop).
RELAX_ARGS=()
[[ "$RELAX" == "1" ]] && RELAX_ARGS=(
  ++manager_env.terminations.anchor_pos=null
  ++manager_env.terminations.ee_body_pos=null
  ++manager_env.terminations.anchor_ori_full=null
  ++manager_env.terminations.foot_pos_xyz=null
)

conda run --no-capture-output -n "$ENV_NAME" python gear_sonic/eval_agent_trl.py \
    +checkpoint="$CKPT" \
    +headless=False \
    ++num_envs=1 \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    ++manager_env.commands.motion.use_paired_motions=True \
    "++manager_env.commands.motion.debug_vis=${DEBUG_VIS:-False}" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$PKL" \
    "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=$KEY" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
    ++manager_env.terminations.time_out=null \
    "++manager_env.config.episode_length_s=$EP_LEN_S" \
    "${RELAX_ARGS[@]}"

echo "[preview] viewer closed."
