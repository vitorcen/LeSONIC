#!/usr/bin/env bash
# P5 — closed-loop A/B: does the open-loop held-out gain (mix −22.3% vs flow3-only) translate to
# a closed-loop fidelity gain? Drives the G1 with LIVE tokens from BOTH best ckpts over all 8 flow3
# segments (phys_valid, RELAX-off, plane — IDENTICAL setup to P1b + the GT-token baseline), then
# scores each against the GT-token replay (outputs/p1b_gt_baseline_plane.csv).
#
# Why Δtilt-vs-GT and not survival: §5.3.1 showed survival saturates in RELAX mode (everything
# stays up) AND the WBC base itself falls on block/fierce even replaying GT tokens — so raw tilt is
# confounded by the motion's own dynamics. The meaningful closed-loop metric is the UNDER-execution
# gap |live_tilt − gt_tilt|: how much smaller the model makes the motion than the reference (the
# smoothing/central-bin-collapse signature). Lower |Δtilt| = tracks the reference amplitude better.
#
#   bash scripts/sonic_p5_closed_loop_ab.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIX_CKPT="${MIX_CKPT:-$REPO_ROOT/dependencies/MaskBeT/outputs/p3_matrix/mix_s2/ckpt_011500.pt}"
FLOW3_CKPT="${FLOW3_CKPT:-$REPO_ROOT/dependencies/MaskBeT/outputs/p3_matrix/flow3_s2/ckpt_005000.pt}"
GT_CSV="${GT_CSV:-$REPO_ROOT/dependencies/MaskBeT/outputs/p1b_gt_baseline_plane.csv}"
OUT_CSV="${OUT_CSV:-$REPO_ROOT/dependencies/MaskBeT/outputs/p5_closed_loop_ab.csv}"
ROUNDS="${ROUNDS:-1}"            # expected+snap is deterministic -> 1 round (matches P1b/GT baseline)

run_arm() { # name ckpt
  local name=$1 ckpt=$2 out="/tmp/p5_$1"
  echo "[p5] === arm $name ckpt=$(basename "$ckpt") ==="
  [[ -f "$out/survival_summary.csv" ]] && { echo "[p5] $name done, skip"; return 0; }
  # the P1b harness starts its own MaskBeT server on :5557; kill any stale one between arms (diff ckpt)
  pkill -9 -f "serve_maskbet_sonic.py" 2>/dev/null; sleep 3
  MASKBET_CKPT="$ckpt" OUT_ROOT="$out" ROUNDS="$ROUNDS" DECODE=expected \
    SERVER_LOG="/tmp/p5_${name}_server.log" \
    bash "$REPO_ROOT/scripts/sonic_p1b_survival.sh"
}

run_arm mix    "$MIX_CKPT"
run_arm flow3  "$FLOW3_CKPT"
pkill -9 -f "serve_maskbet_sonic.py" 2>/dev/null

# join both arms with the GT baseline -> per-segment A/B on |Δtilt vs GT| + survival
"$HOME/miniconda3/bin/python" - "$GT_CSV" "$OUT_CSV" <<'PY'
import csv, json, sys
gt_csv, out_csv = sys.argv[1:3]
# shortname (harness) -> GT-baseline segment name
SHORT2GT = {"moonwalk":"dance_moonwalk","spinclap":"dance_spin_stepback_clap",
            "block":"fight_block_pushkick_shove","combat":"fight_combat_combo_kicks",
            "fierce":"fight_fierce_swings","circle":"run_circle",
            "jogback":"run_jog_backward","sprint":"run_sprint_backpedal"}
gt = {r["segment"]: r for r in csv.DictReader(open(gt_csv))}

def load(arm):
    out = {}
    with open(f"/tmp/p5_{arm}/survival_summary.csv") as f:
        for r in csv.DictReader(f):
            out[r["segment"]] = r
    return out
mix, flow3 = load("mix"), load("flow3")

rows = []
for short, gtname in SHORT2GT.items():
    g = gt.get(gtname); m = mix.get(short); fl = flow3.get(short)
    if not (g and m and fl):
        continue
    gtt = float(g["gt_tilt"]); mt = float(m["max_tilt"]); ft = float(fl["max_tilt"])
    md, fd = abs(mt - gtt), abs(ft - gtt)
    winner = "mix" if md < fd - 0.5 else ("flow3" if fd < md - 0.5 else "tie")
    rows.append({
        "segment": short, "family": gtname.split("_")[0], "openloop_mse": m["openloop_mse"],
        "gt_tilt": gtt, "mix_tilt": mt, "flow3_tilt": ft,
        "mix_abs_dtilt": round(md, 1), "flow3_abs_dtilt": round(fd, 1),
        "closer_to_gt": winner,
        "mix_fell": m["fell"], "flow3_fell": fl["fell"], "gt_fell": g["gt_fell"],
        "mix_minz": m["min_root_z"], "flow3_minz": fl["min_root_z"],
    })
# sort by open-loop MSE (easy motifs first)
rows.sort(key=lambda r: float(r["openloop_mse"]) if r["openloop_mse"] else 9)
cols = list(rows[0].keys())
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

# headline: mean |Δtilt| per arm (under-execution gap), and per-family
import statistics as st
def mean_abs(arm_key): return round(st.mean(float(r[arm_key]) for r in rows), 2)
print("=== P5 closed-loop A/B (mix_s2 vs flow3_s2, |Δtilt vs GT| = under-execution gap) ===")
for r in rows:
    print(f"  {r['segment']:9s} mse={r['openloop_mse']:>6} gt={r['gt_tilt']:>5} "
          f"mix={r['mix_tilt']:>5}(Δ{r['mix_abs_dtilt']:>4}) flow3={r['flow3_tilt']:>5}(Δ{r['flow3_abs_dtilt']:>4}) "
          f"-> {r['closer_to_gt']}")
print(f"  MEAN |Δtilt|: mix={mean_abs('mix_abs_dtilt')}  flow3={mean_abs('flow3_abs_dtilt')}  "
      f"(lower=better tracking)")
wins = {k: sum(r['closer_to_gt'] == k for r in rows) for k in ('mix', 'flow3', 'tie')}
print(f"  closer-to-GT count: {wins}")
print(f"  wrote {out_csv}")
PY
echo "[p5] DONE"