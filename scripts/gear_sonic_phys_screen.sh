#!/usr/bin/env bash
# PHYSICAL-VALIDITY screen: "did the WBC-tracked robot ACTUALLY FALL?" — the second tier below
# the strict trackability gate (gear_sonic_screen.sh). The strict gate terminates on instantaneous
# safety-envelope trips (fast strikes momentarily exceed the 0.15m hand/foot or 0.20m ankle
# threshold) even when local-pose tracking is in-distribution and the robot never falls. Here we
# DISABLE the 4 strict deviation terminations (so the episode is not reset on a trip) and let the
# frozen WBC physically track each window to the end, logging per-env min root height + max torso
# tilt. Verdict UPRIGHT (never fell) means the recorded token stream over that window is physically
# valid even though the strict gate killed it.  NOT a substitute for the strict gate — a second
# label (3-model review 2026-06-10).
#
#   PKL=data/seg_flow3_all.pkl bash scripts/gear_sonic_phys_screen.sh
#   ROOT_Z_FALL=0.6 TILT_FALL_DEG=40 ... overridable thresholds.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"
PKL="${PKL:?set PKL=data/<segments>.pkl}"
KEYS="${KEYS:-}"
OUT="${OUT:-/tmp/sonic_screen/flow3_phys}"
ROOT_Z_FALL="${ROOT_Z_FALL:-0.6}"
TILT_FALL_DEG="${TILT_FALL_DEG:-40}"
SMPL_STUB="${SMPL_STUB:-/tmp/smpl_stub}"

cd "$WBC_DIR"
[[ -f "$PKL" ]] || PKL="$WBC_DIR/$PKL"
[[ -f "$PKL" ]] || { echo "[phys] pkl missing: $PKL"; exit 1; }
mapfile -t ALLKEYS < <(conda run --no-capture-output -n "$ENV_NAME" python -c \
  "import joblib; print('\n'.join(joblib.load('$PKL').keys()))")
if [[ -z "$KEYS" ]]; then KEYS="$(IFS=,; echo "${ALLKEYS[*]}")"; NKEYS="${#ALLKEYS[@]}";
else NKEYS="$(awk -F, '{print NF}' <<<"$KEYS")"; fi
NENV="${NENV:-$NKEYS}"
mkdir -p "$OUT"
unset HEADLESS

echo "[phys] PHYSICAL-validity screen (strict deviation terms OFF, WBC still tracks): pkl=$PKL"
echo "  keys=$KEYS  num_envs=$NENV  fall: root_z<$ROOT_Z_FALL or tilt>${TILT_FALL_DEG}deg  out=$OUT"

PYTHONPATH="$SMPL_STUB:${PYTHONPATH:-}" conda run --no-capture-output -n "$ENV_NAME" \
  python gear_sonic/eval_agent_trl.py \
    +checkpoint="$CKPT" \
    +headless=True \
    "++num_envs=$NENV" \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    ++manager_env.commands.motion.use_paired_motions=True \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$PKL" \
    "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[$KEYS]" \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
    ++manager_env.terminations.anchor_pos=null \
    ++manager_env.terminations.ee_body_pos=null \
    ++manager_env.terminations.anchor_ori_full=null \
    ++manager_env.terminations.foot_pos_xyz=null \
    "++eval_callbacks=[phys_valid]" \
    "++callbacks.phys_valid._target_=gear_sonic.data.phys_valid_screen.PhysValidScreen" \
    "++callbacks.phys_valid.output_dir=$OUT" \
    "++callbacks.phys_valid.root_z_fall=$ROOT_Z_FALL" \
    "++callbacks.phys_valid.tilt_fall_deg=$TILT_FALL_DEG" 2>&1 | grep -iE "phys_valid|error|traceback" | grep -v "DEBUG" | tail -20

echo "[phys] === verdict (UPRIGHT = never fell = token stream physically valid) ==="
python3 - "$OUT/phys_valid.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{'key':38s} {'z_min':6s} {'tilt_max':8s} {'verdict':14s} {'trackable':9s}")
nup=0
for k,v in d.items():
    up = not v["fell"]; nup += up
    verdict = "UPRIGHT" if up else f"FELL@{v['fall_step']}"
    print(f"{k:38s} {v['min_root_z']:6.2f} {v['max_tilt_deg']:8.1f} {verdict:14s} {v['trackable_frac']:.2f}")
print(f"--- {nup}/{len(d)} UPRIGHT (never fell) ---")
PY
