#!/usr/bin/env bash
# KINEMATIC replay of a LAFAN segment via SONIC's OWN run_replay (train_agent_trl.py ++replay=True):
# the robot is written directly to the reference pose each frame (write_joint_state_to_sim +
# write_root_state_to_sim) — NO policy, NO physics balance, so it NEVER falls. This is the right
# tool to SEE a motion and pick clean windows. (The WBC physics tracking — gear_sonic_screen_window.sh
# — is a different question: "can the frozen WBC physically track this?", where falling = not trackable.)
#
#   CLIP=dance START=8 END=12 bash scripts/gear_sonic_replay_window.sh   # 4s window, plays once
#   CLIP=fight START=0 END=9999 bash scripts/gear_sonic_replay_window.sh # whole clip, plays once
# Viewer: C=close cam, F=free, R=reset/replay. ++replay_loop_num=True to loop.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CLIP="${CLIP:-fight}"
PKL="${PKL:-$WBC_DIR/data/${CLIP}_full_robot.pkl}"
KEY="${KEY:-${CLIP}_full}"
START="${START:?set START=<seconds>}"
END="${END:?set END=<seconds>}"
TMP="${TMP:-/tmp/sonic_replay_window.pkl}"
LOOP="${LOOP:-False}"   # False = play once then hold; True = loop
unset HEADLESS
export DISPLAY="${DISPLAY:-:0}"
export WANDB_MODE=disabled   # train_agent_trl inits logging before the replay branch; no auth in replay

[[ -f "$PKL" ]] || PKL="$WBC_DIR/$PKL"          # allow a relative path (resolve against WBC dir)
[[ -f "$PKL" ]] || { echo "[replay] clip missing: $PKL (CLIP=fight|run|dance|jumps)"; exit 1; }
conda run --no-capture-output -n "$ENV_NAME" python "$REPO_ROOT/scripts/cut_motion_window.py" \
    --input "$PKL" --key "$KEY" --start_s "$START" --end_s "$END" \
    --output "$TMP" --out_key window || exit 1

echo "[replay] KINEMATIC replay (no policy, no falling): $CLIP [$START s, $END s]  loop=$LOOP"
echo "  Viewer: C=close cam, F=free, R=reset/replay."
cd "$WBC_DIR"
conda run --no-capture-output -n "$ENV_NAME" python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    headless=False \
    num_envs=1 \
    ++replay=True \
    "++replay_loop_num=$LOOP" \
    ++manager_env.commands.motion.use_paired_motions=True \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$TMP" \
    "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=window" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy"

echo "[replay] viewer closed."
