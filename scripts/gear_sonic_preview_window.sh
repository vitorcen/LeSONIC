#!/usr/bin/env bash
# Preview a [START,END]-second window of a full LAFAN clip via pure SONIC WBC tracking.
# Hand-pick a clean sub-action by seconds instead of trusting the auto velocity-valley cut.
#
#   CLIP=fight START=3.0 END=6.0 bash scripts/gear_sonic_preview_window.sh
#   CLIP=run   START=12  END=15  bash scripts/gear_sonic_preview_window.sh
#   (CLIP=fight|run picks data/<clip>_full_robot.pkl; or set PKL+KEY for any clip.)
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CLIP="${CLIP:-fight}"
PKL="${PKL:-$WBC_DIR/data/${CLIP}_full_robot.pkl}"
KEY="${KEY:-${CLIP}_full}"
START="${START:?set START=<seconds>}"
END="${END:?set END=<seconds>}"
TMP="${TMP:-/tmp/sonic_window_robot.pkl}"

[[ -f "$PKL" ]] || PKL="$WBC_DIR/$PKL"   # allow a relative path (resolve against the WBC dir)
[[ -f "$PKL" ]] || { echo "[window] full clip missing: $PKL (CLIP=fight|run|dance|jumps)"; exit 1; }
conda run --no-capture-output -n "$ENV_NAME" python "$REPO_ROOT/scripts/cut_motion_window.py" \
    --input "$PKL" --key "$KEY" --start_s "$START" --end_s "$END" \
    --output "$TMP" --out_key window || exit 1

echo "[window] previewing $CLIP [$START s, $END s] via WBC tracking ..."
PKL="$TMP" KEY=window bash "$REPO_ROOT/scripts/gear_sonic_preview_motion.sh"
