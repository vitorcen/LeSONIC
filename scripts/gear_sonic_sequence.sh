#!/usr/bin/env bash
# SEQUENCE multiple motions in ONE running session — no GUI restart, no server restart.
# A live GR00T server drives SONIC's WBC; this script switches the PROMPT over time so the
# robot does e.g. squat -> walk -> kick back-to-back, for a single smooth recording.
#   prompt[t] -> GR00T (ZMQ) -> 64-d token -> SONIC decode -> G1 moves.
# One-shot motions (kick/walk/jump) get a per-segment bootstrap; each seam gets a short settle.
#
# HONEST: like the single-motion demos, this runs with deviation/fall terminations relaxed
# (RELAX) and time_out disabled so the full multi-motion sequence plays without reset. It is a
# prompt-sequencing ORCHESTRATION over a memorized 7-skill model, not compound-prompt
# understanding and not a fall-resistance claim. See the model card scope.
#
# Prereq: start the GR00T server FIRST (in the Isaac-GR00T venv):
#   cd dependencies/Isaac-GR00T && COMPILE_ACTION_HEAD_DISABLE=1 .venv/bin/python \
#     -m gr00t.eval.run_gr00t_server --model_path <ckpt-8000> \
#     --embodiment_tag unitree_g1_sonic --port 5555
#
# Usage:
#   bash scripts/gear_sonic_sequence.sh list          # list named demo flows
#   bash scripts/gear_sonic_sequence.sh @flow1         # named flow (sonic_demo_flows.json)
#   bash scripts/gear_sonic_sequence.sh @flow2         # action -> walk -> action -> walk ... loop
#   bash scripts/gear_sonic_sequence.sh squat,walk,kick            # ad-hoc seq
#   bash scripts/gear_sonic_sequence.sh "squat:150,macarena:200"   # per-seg steps
#   LOOPS=30 STEPS=240 SETTLE=40 bash scripts/gear_sonic_sequence.sh @flow2
# names: dance | lunge | macarena | kick | squat | jump | walk
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"
PKL="${PKL:-data/demo_robot_filtered.pkl}"
HOST="${GR00T_HOST:-127.0.0.1}"
PORT="${GR00T_PORT:-5555}"
ACTION_HORIZON="${ACTION_HORIZON:-40}"
CAM_RES="${CAM_RES:-[480,640]}"
# Viewer auto-follow offset relative to the robot root (pelvis, ~0.74m). Default is a CLOSE,
# near-level FULL-BODY 3/4 view (vs the far god-view eye=[4.5,0,4]): eye at ~chest height and
# lookat at body centre so the legs/feet stay in frame (the old eye=1.6/lookat=0.8 looked DOWN
# at the chest and cropped the legs). The camera tracks the robot and keeps it framed.
# Tune live without editing code:  CAM_EYE="x,y,z" CAM_LOOKAT="x,y,z" bash ... @flow2
#   - raise/lower the 3rd number of CAM_EYE to tilt the view; lower CAM_LOOKAT's 3rd to see feet.
# In the running viewer press C to re-snap close anytime (SONIC_CAM_OFFSET tunes that eye offset).
CAM_EYE="${CAM_EYE:-2.2,2.2,0.5}"
CAM_LOOKAT="${CAM_LOOKAT:-0.0,0.0,-0.1}"
export SONIC_CAM_OFFSET="${SONIC_CAM_OFFSET:-1.6,1.6,0.4}"
# The loaded reference is only env scaffolding here (robot is driven by injected tokens). Stop the
# reference-clip wrap-around from re-RSI-ing (teleporting) the robot back to origin every ~8.5s —
# that teleport is the "flash back to origin" AND it prevents walking to a new spot and acting there.
export SONIC_NO_REF_RESAMPLE="${SONIC_NO_REF_RESAMPLE:-1}"
STEPS="${STEPS:-200}"                 # default control steps per segment
SETTLE="${SETTLE:-40}"                # neutral/freeze steps at each seam (smooth transition)
BOOTSTRAP_STEPS="${BOOTSTRAP_STEPS:-80}"
PRED_DIR="${PRED_DIR:-$REPO_ROOT/datasets/sonic_vla_pred_8k_final}"
NEUTRAL_NPZ="${NEUTRAL_NPZ:-}"        # optional standing-token clip for settle windows
VIS="${VIS:-0}"                       # 1 = show green ref skeleton (misleading for a sequence)
export DISPLAY="${DISPLAY:-:0}"

case "${HEADLESS:-False}" in 1|True|true|yes) HL=True;; *) HL=False;; esac
unset HEADLESS

# Same short-name -> motion key map as gear_sonic_live.sh (for the env's initial reset pose).
declare -A KEY=(
  [dance]=dance_in_da_party_001__A464  [lunge]=forward_lunge_R_001__A359_M
  [macarena]=macarena_001__A545        [kick]=neutral_kick_R_001__A543
  [squat]=squat_001__A359              [jump]=tired_one_leg_jumping_R_001__A359
  [walk]=walking_quip_360_R_002__A428
  [guard]=fight_seg000 [jab]=fight_seg020 [combo]=fight_seg050
  [turn]=run_seg001 [jog]=run_seg006 [runfast]=run_seg017 [fight]=lafan_fight_15s
  # flow3 windows — set PKL=data/seg_flow3_all.pkl PRED_DIR=datasets/sonic_vla_pred_flow3
  [combat]=fight_combat_combo_kicks [block]=fight_block_pushkick_shove [fierce]=fight_fierce_swings
  [jogback]=run_jog_backward [sprint]=run_sprint_backpedal [circle]=run_circle
  [moonwalk]=dance_moonwalk [spinclap]=dance_spin_stepback_clap
)

SEQ="${1:-${MOTION:-squat,lunge,macarena}}"
FLOWS_JSON="$REPO_ROOT/scripts/sonic_demo_flows.json"

# `list` -> show named demo flows and exit.
if [[ "$SEQ" == "list" ]]; then
  python3 "$REPO_ROOT/scripts/gr00t_build_sequence.py" --list --flows-file "$FLOWS_JSON"; exit 0
fi

# `@name` -> a named demo flow (sonic_demo_flows.json); otherwise an ad-hoc comma seq.
if [[ "$SEQ" == @* ]]; then
  FLOW="${SEQ#@}"
  BUILD_SRC=(--flow "$FLOW" --flows-file "$FLOWS_JSON")
  FIRST="$(python3 -c "import json;print(json.load(open('$FLOWS_JSON'))['$FLOW']['segments'][0]['motion'])" 2>/dev/null)"
  [[ -z "$FIRST" ]] && { echo "[seq] unknown flow '$FLOW' — try: bash scripts/gear_sonic_sequence.sh list"; exit 2; }
else
  BUILD_SRC=(--seq "$SEQ")
  FIRST="${SEQ%%,*}"; FIRST="${FIRST%%:*}"     # first motion name (strip :steps)
fi
FIRST_KEY="${KEY[$FIRST]:-}"
[[ -z "$FIRST_KEY" ]] && { echo "[seq] unknown first motion '$FIRST' (dance|lunge|macarena|kick|squat|jump|walk)"; exit 2; }

# 1) Build the timeline JSON (pure python, no GPU).
TL="${TIMELINE_JSON:-/tmp/sonic_seq_$$.json}"
NEUTRAL_ARG=(); [[ -n "$NEUTRAL_NPZ" ]] && NEUTRAL_ARG=(--neutral-npz "$NEUTRAL_NPZ")
python3 "$REPO_ROOT/scripts/gr00t_build_sequence.py" \
    "${BUILD_SRC[@]}" --pred-dir "$PRED_DIR" --out "$TL" \
    --steps "$STEPS" --settle "$SETTLE" --bootstrap-steps "$BOOTSTRAP_STEPS" \
    "${NEUTRAL_ARG[@]}" || exit 1

# Control runs at decimation*sim_dt = 4*0.005 = 0.02s (50Hz), and IsaacLab time-truncates each
# episode at config.episode_length_s (default 10s = 500 steps) REGARDLESS of time_out. The
# injector loops the timeline modulo-style (each segment keeps its step budget), so we want the
# episode to span MANY cycles -> the sequence runs continuously with no reset snap for viewing/
# recording. Set episode_length_s = one-cycle seconds * LOOPS.
TOTAL_STEPS="$(python3 -c "import json;print(json.load(open('$TL'))['total_steps'])")"
LOOPS="${LOOPS:-30}"
EP_LEN_S="$(python3 -c "print(round($TOTAL_STEPS*0.02*$LOOPS, 1))")"

cd "$WBC_DIR"
echo "[seq] sequence='$SEQ'  timeline=$TL  GR00T server=$HOST:$PORT  init-pose=$FIRST_KEY"
echo "  one cycle ${TOTAL_STEPS} steps (~$(python3 -c "print(round($TOTAL_STEPS*0.02,1))")s) x ${LOOPS} loops -> episode_length_s=${EP_LEN_S}"
echo "  Isaac viewer: F=free cam, R=reset.  Prompt switches over time (watch the [vla_live] >>> segment logs)."
echo "  NOTE: RELAX + time_out disabled so the whole sequence plays without reset (demo scope)."

# 2) Relax deviation/fall terminations AND disable the motion time_out so the multi-motion
#    sequence (longer than any single loaded clip) plays continuously without a mid-run reset.
RELAX_ARGS=(
  ++manager_env.terminations.anchor_pos.params.threshold=99
  ++manager_env.terminations.ee_body_pos.params.threshold=99
  ++manager_env.terminations.anchor_ori_full.params.threshold=99
  ++manager_env.terminations.foot_pos_xyz.params.threshold=99
  ++manager_env.terminations.time_out=null
  "++manager_env.config.episode_length_s=$EP_LEN_S"
)

# Reference/target markers (yellow goal spheres + red body/feet markers) track the LOADED
# reference motion — which for a sequence is just the init clip (squat, stationary, loops every
# ~8.5s). But the robot is driven by INJECTED tokens (walk/etc.) and moves away, so the markers
# don't match the moving robot (yellow stays put + "resets" on the clip loop; red lags the feet).
# Hide them by default for a clean demo; VIS=1 shows them (only meaningful for single-motion debug).
if [[ "$HL" == "False" && "$VIS" == "1" ]]; then
  VIS_ARGS=(++manager_env.commands.motion.debug_vis=True)
else
  VIS_ARGS=(++manager_env.commands.motion.debug_vis=False)
fi

conda run --no-capture-output -n "$ENV_NAME" python gear_sonic/eval_agent_trl.py \
    +checkpoint="$CKPT" \
    +headless="$HL" \
    ++num_envs=1 \
    ++manager_env.config.enable_cameras=True \
    ++manager_env.config.terrain_type=plane \
    "++manager_env.config.cameras.camera_resolution=$CAM_RES" \
    "++manager_env.config.viewer.eye=[$CAM_EYE]" \
    "++manager_env.config.viewer.lookat=[$CAM_LOOKAT]" \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    ++manager_env.commands.motion.use_paired_motions=True \
    "${VIS_ARGS[@]}" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$PKL" \
    "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=$FIRST_KEY" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
    "${RELAX_ARGS[@]}" \
    "++eval_callbacks=[vla_live]" \
    "++callbacks.vla_live._target_=gear_sonic.data.vla_live_injector.VlaLiveInjector" \
    "++callbacks.vla_live.host=$HOST" \
    "++callbacks.vla_live.port=$PORT" \
    "++callbacks.vla_live.action_horizon=$ACTION_HORIZON" \
    "++callbacks.vla_live.timeline_json=$TL"

echo "[seq] viewer closed."
