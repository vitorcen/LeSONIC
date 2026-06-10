#!/usr/bin/env bash
# STRICT WBC trackability screen (the go/no-go gate before recording tokens / training the VLA).
# Runs the frozen SONIC WBC physically tracking each reference key under the release's OWN strict
# adaptive terminations (base_adaptive_strict_ori_foot_xyz) — NO RELAX, NO time_out override.
# A key that the WBC cannot balance TERMINATES -> that segment is NOT trackable -> drop it.
# This is "can the frozen WBC do this motion?", which is upstream of "can GR00T emit its token".
#
#   PKL=data/seg_flow3_all.pkl KEYS=fight_throwing_swings,run_circle bash scripts/gear_sonic_screen.sh
#   # KEYS empty -> screen ALL keys in the pkl. NENV defaults to the number of keys.
#
# Reads eval/all_metrics_dict from metrics_eval.json and prints a per-key PASS/FAIL table
# (PASS = not terminated AND progress >= PROG_MIN, default 0.95).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"
PKL="${PKL:?set PKL=data/<segments>.pkl}"
KEYS="${KEYS:-}"                     # comma-separated; empty = all keys in the pkl
OUT="${OUT:-/tmp/sonic_screen/flow3}"
PROG_MIN="${PROG_MIN:-0.95}"
SMPL_STUB="${SMPL_STUB:-/tmp/smpl_stub}"
export SMOKE="${SMOKE:-0}"

cd "$WBC_DIR"
[[ -f "$PKL" ]] || PKL="$WBC_DIR/$PKL"
[[ -f "$PKL" ]] || { echo "[screen] pkl missing: $PKL"; exit 1; }

# default filter = every key in the pkl; default num_envs = key count
mapfile -t ALLKEYS < <(conda run --no-capture-output -n "$ENV_NAME" python -c \
  "import joblib,sys; print('\n'.join(joblib.load('$PKL').keys()))")
if [[ -z "$KEYS" ]]; then
  KEYS="$(IFS=,; echo "${ALLKEYS[*]}")"
  NKEYS="${#ALLKEYS[@]}"
else
  NKEYS="$(awk -F, '{print NF}' <<<"$KEYS")"
fi
NENV="${NENV:-$NKEYS}"
mkdir -p "$OUT"
unset HEADLESS

echo "[screen] STRICT WBC trackability (no RELAX): pkl=$PKL"
echo "  keys=$KEYS  num_envs=$NENV  prog_min=$PROG_MIN  out=$OUT"

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
    "++eval_callbacks=[im_eval]" \
    "++eval_output_dir=$OUT" 2>&1 | tail -3

echo "[screen] === per-key verdict (PASS = not terminated and progress>=$PROG_MIN) ==="
python3 - "$OUT/metrics_eval.json" "$PROG_MIN" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); pm = float(sys.argv[2])
d = m["eval/all_metrics_dict"]
keys = d["motion_keys"]; term = d.get("terminated", [False]*len(keys))
prog = d.get("progress", [1.0]*len(keys)); mg = d.get("mpjpe_g", [float('nan')]*len(keys))
print(f"{'key':40s} {'term':5s} {'prog':6s} {'mpjpe_g(mm)':11s} verdict")
npass=0
for i,k in enumerate(keys):
    t=bool(term[i]); p=float(prog[i]); g=float(mg[i]) if i<len(mg) else float('nan')
    ok = (not t) and p>=pm
    npass += ok
    print(f"{k:40s} {str(t):5s} {p:6.3f} {g:11.1f} {'PASS' if ok else 'FAIL'}")
print(f"--- {npass}/{len(keys)} PASS ---")
PY
