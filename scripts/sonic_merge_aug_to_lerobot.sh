#!/usr/bin/env bash
# Merge the physically-correct augmentation rollouts (sonic_vla_raw_aug, LAUNCH-filtered) with the
# clean base 7-motion recordings (sonic_vla_raw) into ONE combined raw dir, then convert to a
# UNITREE_G1_SONIC LeRobot v2.1 dataset for GR00T finetune. Non-LAUNCH aug episodes are excluded
# (read from the record-driver TSV) so only physically-valid transitions enter training.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_RAW="$REPO_ROOT/datasets/sonic_vla_raw"
AUG_RAW="$REPO_ROOT/datasets/sonic_vla_raw_aug"
COMB="$REPO_ROOT/datasets/sonic_vla_raw_aug_combined"
OUT="${OUT:-$REPO_ROOT/datasets/sonic_vla_lerobot_aug}"
TSV="${TSV:-/tmp/aug_record.tsv}"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
MOTIONS=(dance_in_da_party_001__A464 forward_lunge_R_001__A359_M macarena_001__A545 \
         neutral_kick_R_001__A543 squat_001__A359 tired_one_leg_jumping_R_001__A359 \
         walking_quip_360_R_002__A428)

rm -rf "$COMB"; mkdir -p "$COMB"
echo "=== merge: base 7 motions + LAUNCH-filtered aug ==="
nbase=0; naug=0; ndrop=0
for k in "${MOTIONS[@]}"; do
  mkdir -p "$COMB/$k"
  # base full-length episodes (clean GT-tracking recordings -> the sustain backbone)
  for ep in "$BASE_RAW/$k"/episode_*.npz; do
    [ -f "$ep" ] || continue
    ln -sf "$ep" "$COMB/$k/$(basename "$ep")"; nbase=$((nbase+1))
  done
  # aug transition episodes — keep only LAUNCH (per TSV verdict col); DROP kick-as-source.
  # kick momentum (rfoot~0.9) bleeds into the target onset (high foot-lift contaminates walk/dance/
  # macarena) and 1s isn't enough to settle it out -> all X_from_kick are bad samples. Exclude them.
  for ep in "$AUG_RAW/$k"/episode_*.npz; do
    [ -f "$ep" ] || continue
    tag="$(basename "$ep" .npz)"; tag="${tag#episode_}"          # e.g. aug_dance
    src="${tag#aug_}"
    if [ "$src" = "kick" ]; then
      echo "  drop $k/$tag (kick-source: momentum rfoot~0.9 contaminates onset)"; ndrop=$((ndrop+1)); continue
    fi
    verdict=$(awk -F'\t' -v s="$src" -v key="$k" '$4==key && $3==s {print $11}' "$TSV" 2>/dev/null | tail -1)
    if [ "$verdict" = "LAUNCH" ]; then
      ln -sf "$ep" "$COMB/$k/$(basename "$ep")"; naug=$((naug+1))
    else
      echo "  drop $k/$tag (verdict=${verdict:-MISSING})"; ndrop=$((ndrop+1))
    fi
  done
  echo "  $k: $(ls "$COMB/$k"/*.npz 2>/dev/null | wc -l) episodes"
done
# stand episodes (prompt="stand"): teach natural standing so the model stops emitting OOD motions at
# idle + VLA_STAND_MODEL works again. Only in AUG_RAW (no base). Gated by INCLUDE_STAND=1.
nstand=0
if [ "${INCLUDE_STAND:-1}" = "1" ] && [ -d "$AUG_RAW/stand_g1" ]; then
  mkdir -p "$COMB/stand_g1"
  for ep in "$AUG_RAW/stand_g1"/episode_*.npz; do
    [ -f "$ep" ] || continue
    ln -sf "$ep" "$COMB/stand_g1/$(basename "$ep")"; nstand=$((nstand+1))
  done
  echo "  stand_g1: $nstand episodes (prompt=stand)"
fi
echo "merged: base=$nbase aug_kept=$naug stand=$nstand aug_dropped=$ndrop -> $COMB"

echo "=== convert -> LeRobot v2.1 ($OUT) ==="
cd "$WBC_DIR"
PYTHONPATH="$WBC_DIR" .venv_data_collection/bin/python "$REPO_ROOT/scripts/sonic_vla_npz_to_lerobot.py" \
  --raw-dir "$COMB" --out "$OUT"
echo "=== done. frame summary ==="
python3 - "$OUT" <<'PY'
import sys,glob,pandas as pd,numpy as np,json,os
out=sys.argv[1]
pq=sorted(glob.glob(f"{out}/data/**/*.parquet",recursive=True))
tot=sum(len(pd.read_parquet(p)) for p in pq)
print(f"episodes={len(pq)} total_frames={tot}")
PY
