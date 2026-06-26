#!/usr/bin/env bash
# Phase-1 two-tier eval funnel for a GR00T-SONIC training dir (doc/sonic_robustness_retrain.html).
#
#   Tier-1  open-loop token-MSE  — cheap, scans ALL ckpts (no Isaac sim). Confirms state-noise didn't
#           wreck token capability + reveals the sweet-spot band. NOT the selector (ckpt-8000 had
#           MSE 0.0011 yet froze closed-loop).
#   Tier-2  closed-loop kick-basin — the DECISIVE metric, run only on the sweet-band candidates.
#
#   bash scripts/sonic_funnel_eval.sh outputs/gr00t_sonic_noise08
#   CAND_MIN_STEP=4000 bash scripts/sonic_funnel_eval.sh <out_dir>        # band = steps >= 4000
#   CAND="checkpoint-6400 checkpoint-7200 checkpoint-8000" bash scripts/sonic_funnel_eval.sh <out_dir>
#   SKIP_OL=1 bash scripts/sonic_funnel_eval.sh <out_dir>                 # tier-2 only
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:?usage: sonic_funnel_eval.sh <training_out_dir>}"
OUT_DIR="$(cd "$OUT_DIR" 2>/dev/null && pwd || echo "$OUT_DIR")"   # absolute: ckpt paths are read by tools that cd elsewhere
GR00T_DIR="$REPO_ROOT/dependencies/Isaac-GR00T"
DATASET="${DATASET:-$REPO_ROOT/datasets/sonic_vla_lerobot}"
OL_TSV="${OL_TSV:-/tmp/sonic_ol_mse.tsv}"
CAND_MIN_STEP="${CAND_MIN_STEP:-4000}"   # tier-2 candidate band: keep undertrained ckpts out of the slow sweep
AH="${AH:-40}"

ckpts=$(ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
[[ -z "$ckpts" ]] && { echo "[funnel] no checkpoints in $OUT_DIR"; exit 1; }

# ---------- Tier 1: open-loop token-MSE (all ckpts, no sim) ----------
if [[ "${SKIP_OL:-0}" != "1" ]]; then
  rm -f "$OL_TSV"
  echo "[funnel] Tier-1 open-loop token-MSE over all ckpts -> $OL_TSV"
  for ck in $ckpts; do
    [[ -f "$ck/model.safetensors.index.json" || -f "$ck/model.safetensors" ]] || continue
    echo "[funnel]   OL-MSE $(basename "$ck") ..."
    ( cd "$GR00T_DIR" && COMPILE_ACTION_HEAD_DISABLE=1 .venv/bin/python \
        "$REPO_ROOT/scripts/sonic_gr00t_openloop_mse.py" \
        --model_path "$ck" --dataset_path "$DATASET" --action_horizon "$AH" --tsv "$OL_TSV" ) \
      2>>/tmp/sonic_ol_mse.err | grep -E "MACRO|^    " || echo "      (OL-MSE failed; see /tmp/sonic_ol_mse.err)"
  done
  echo; echo "===== Tier-1 open-loop token-MSE (raw units; baseline ckpt-8000 ~0.0011) ====="
  [[ -f "$OL_TSV" ]] && column -t "$OL_TSV"
  echo "  NOTE: do NOT pick the global min (=overtrained). Take the knee, in the ~6-8ep band."
fi

# ---------- Tier 2: closed-loop kick-basin (candidate band) ----------
if [[ -n "${CAND:-}" ]]; then
  cand_list="$CAND"
else
  cand_list=""
  for ck in $ckpts; do
    s="$(basename "$ck" | sed 's/checkpoint-//')"
    [[ "$s" =~ ^[0-9]+$ && "$s" -ge "$CAND_MIN_STEP" ]] && cand_list+="$(basename "$ck") "
  done
fi
echo; echo "[funnel] Tier-2 closed-loop kick-basin on candidates: ${cand_list:-<none>}"
[[ -z "$cand_list" ]] && { echo "[funnel] no tier-2 candidates (raise/lower CAND_MIN_STEP or set CAND)"; exit 0; }

# build a one-off out dir containing only the candidate ckpts via the existing basin sweep
# (the sweep iterates checkpoint-* in a dir; we pass a glob restricted to candidates).
for name in $cand_list; do
  RESULT="/tmp/kick_basin_${name}.tsv" \
    bash "$REPO_ROOT/scripts/sonic_kick_basin_sweep.sh" "$OUT_DIR" "$name" || true
done
echo; echo "===== Tier-2 closed-loop kick-basin (baseline ckpt-8000: rfoot_max 0.038, exc 0.67) ====="
echo -e "ckpt\trfoot_max\trfoot_rng\tjoint_exc\trootz_min\tfrac_kick"
grep -hvE '^ckpt' /tmp/kick_basin_checkpoint-*.tsv 2>/dev/null | sort -t- -k2 -n | column -t
echo "[funnel] best = highest rfoot_max/frac_kick with the 4 working motions intact."
