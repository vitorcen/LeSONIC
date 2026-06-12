#!/usr/bin/env bash
# P1b — closed-loop survival SCREEN for live MaskBeT (the open-loop->closed-loop calibration).
# Answers: (1) is closed-loop usable? (2) which open-loop metric predicts survival?
#
# Method: per flow3 segment, drive the G1 with LIVE MaskBeT tokens (vla_live injector + the
# MaskBeT ZMQ server) while the phys_valid callback measures the REAL physical fall (min root
# height < 0.6m or torso tilt > 40deg) — NOT the strict envelope trip (which overstates failure
# on fast strikes; see phys_valid_screen.py / 2026-06-10 review). The 4 strict deviation
# terminations are nulled so the robot plays each window to the end and phys_valid logs fall_step.
# Decode = expected+snap (P1 held-out validated it beats argmax 8/8 folds).
#
# Pairs with the GT upper bound: scripts/gear_sonic_phys_screen.sh (same phys_valid, recorded
# tokens). Live-vs-GT fall_step gap = closed-loop degradation. Correlate live fall_step against
# the per-segment open-loop window-MSE (design 5.2) to calibrate the objective.
#
#   bash scripts/sonic_p1b_survival.sh                 # all 8 segments x ROUNDS, expected+snap
#   ROUNDS=3 DECODE=expected bash scripts/sonic_p1b_survival.sh
#   SEGS="circle,combat" bash scripts/sonic_p1b_survival.sh   # subset
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC_DIR="$REPO_ROOT/dependencies/GR00T-WholeBodyControl"
ENV_NAME="${ENV_NAME:-isaaclab}"
CKPT="${CKPT:-sonic_release/last.pt}"               # WBC ckpt (frozen base), not MaskBeT
PKL="${PKL:-data/seg_flow3_all.pkl}"
PORT="${MASKBET_PORT:-5557}"
MASKBET_CKPT="${MASKBET_CKPT:-$REPO_ROOT/MaskBeT/outputs/flow3/ckpt_006000.pt}"
DECODE="${DECODE:-expected}"                        # P1-validated: expected+snap > argmax
ROUNDS="${ROUNDS:-3}"
CAP="${CAP:-1200}"                                  # max_render safety cap (> any window len)
OUT_ROOT="${OUT_ROOT:-/tmp/sonic_p1b}"
ENV_BIN="${ENV_BIN:-$HOME/miniconda3/envs/starvla_eval_qwen35/bin}"
SERVER_LOG="${SERVER_LOG:-/tmp/maskbet_p1b_server.log}"

# segment shortname -> (PKL key, prompt string, prompt_index, open-loop window-MSE from design 5.2)
declare -A SEGKEY=(
  [moonwalk]=dance_moonwalk [spinclap]=dance_spin_stepback_clap
  [block]=fight_block_pushkick_shove [combat]=fight_combat_combo_kicks
  [fierce]=fight_fierce_swings [circle]=run_circle
  [jogback]=run_jog_backward [sprint]=run_sprint_backpedal )
declare -A SEGPROMPT=(
  [moonwalk]="moonwalk" [spinclap]="spin, step back, and clap"
  [block]="block and push-kick" [combat]="combat strikes and combo kicks"
  [fierce]="fierce swings" [circle]="run in a circle"
  [jogback]="jog forward then run backward" [sprint]="sprint back and forth then backpedal" )
declare -A SEGMSE=(  # per-segment open-loop window-MSE (cross-motif held-out best, design 5.2)
  [moonwalk]=0.0441 [spinclap]=0.0505 [block]=0.0350 [combat]=0.0533
  [fierce]=0.0638 [circle]=0.0076 [jogback]=0.0110 [sprint]=0.0250 )

SEGS="${SEGS:-moonwalk,spinclap,block,combat,fierce,circle,jogback,sprint}"
IFS=',' read -ra SEGLIST <<< "$SEGS"
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/survival_summary.csv"
echo "segment,prompt_idx,openloop_mse,round,fell,fall_step,window_steps,survival_frac,min_root_z,max_tilt" > "$SUMMARY"

# 1) bring up MaskBeT server (decode=expected+snap), GPF-retry — reuse demo's robust launcher path
export SONIC_MASKBET_DECODE="$DECODE" SONIC_MASKBET_TEMP="${SONIC_MASKBET_TEMP:-1.0}"
start_server() {
  ps -eo cmd | grep -q "[s]erve_maskbet_sonic.py.*--port $PORT" && return 0
  for a in 1 2 3 4 5 6; do
    ( cd "$REPO_ROOT" && MASKBET_DIR="$REPO_ROOT/MaskBeT" nohup "$ENV_BIN/python" \
        "$REPO_ROOT/scripts/serve_maskbet_sonic.py" --ckpt "$MASKBET_CKPT" --port "$PORT" \
        --decode "$DECODE" > "$SERVER_LOG" 2>&1 & )
    for _ in $(seq 1 24); do
      grep -q SERVE_READY "$SERVER_LOG" 2>/dev/null && { echo "[p1b] server ready (decode=$DECODE)"; return 0; }
      ps -eo cmd | grep -q "[s]erve_maskbet_sonic.py.*--port $PORT" || break
      sleep 2
    done
    echo "[p1b] server load attempt $a died (GPF?) — relaunching"
  done
  echo "[p1b] server failed; log:"; tail -15 "$SERVER_LOG"; return 1
}
start_server || exit 1

run_seg() { # shortname round
  local seg=$1 rnd=$2 key=${SEGKEY[$seg]} prompt=${SEGPROMPT[$seg]}
  local out="$OUT_ROOT/${seg}_r${rnd}"; mkdir -p "$out"
  # resume: skip Isaac if this (seg,round) already produced a result (GPF-retry friendly)
  if [[ -f "$out/phys_valid.json" ]]; then echo "[p1b] $seg r$rnd already done — skip"; return 0; fi
  local TL=/tmp/p1b_${seg}_$$.json
  # single-segment timeline (per-seg bootstrap to ENTER the move, then live MaskBeT)
  python3 "$REPO_ROOT/scripts/gr00t_build_sequence.py" --seq "$seg" \
      --pred-dir "$REPO_ROOT/datasets/sonic_vla_pred_starvla_ce" --out "$TL" \
      --steps 400 --settle 0 --bootstrap-steps 80 >/dev/null 2>&1 || { echo "[p1b] timeline build failed $seg"; return 1; }
  echo "[p1b] === $seg round $rnd (key=$key prompt='$prompt' mse=${SEGMSE[$seg]}) ==="
  # SINGLE composing callback: the eval loop's all(cb.eval_step ...) short-circuits, so two
  # callbacks where vla_live returns False never reach phys_valid. VlaLivePhysProbe runs both.
  ( cd "$WBC_DIR" && GR00T_PORT=$PORT PYTHONPATH="$REPO_ROOT/scripts:${PYTHONPATH:-}" \
    conda run --no-capture-output -n "$ENV_NAME" python gear_sonic/eval_agent_trl.py \
      +checkpoint="$CKPT" +headless=True ++num_envs=1 +max_render_steps=$CAP \
      ++manager_env.config.enable_cameras=True ++manager_env.config.terrain_type=plane \
      ++manager_env.observations.policy.enable_corruption=False \
      ++manager_env.observations.tokenizer.enable_corruption=False \
      ++manager_env.commands.motion.use_paired_motions=True \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=$PKL" \
      "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=$key" \
      ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
      ++manager_env.terminations.anchor_pos=null \
      ++manager_env.terminations.ee_body_pos=null \
      ++manager_env.terminations.anchor_ori_full=null \
      ++manager_env.terminations.foot_pos_xyz=null \
      "++eval_callbacks=[probe]" \
      "++callbacks.probe._target_=vla_live_phys_probe.VlaLivePhysProbe" \
      "++callbacks.probe.host=127.0.0.1" "++callbacks.probe.port=$PORT" \
      "++callbacks.probe.action_horizon=40" "++callbacks.probe.timeline_json=$TL" \
      "++callbacks.probe.output_dir=$out" \
      "++callbacks.probe.root_z_fall=0.6" "++callbacks.probe.tilt_fall_deg=40" \
      2>&1 | grep -iE "phys_valid|vla_live|probe|RESET|error|traceback" | grep -v DEBUG | tail -25 )
  [[ -f "$out/phys_valid.json" ]] && echo "[p1b] $seg r$rnd done -> $(python3 -c "import json;v=next(iter(json.load(open('$out/phys_valid.json')).values()));print('fell',v['fell'],'fall_step',v['fall_step'],'frac',round(v['trackable_frac'],2))")" \
                                  || echo "[p1b] $seg r$rnd produced NO json (GPF?) — will retry on resume"
}

for seg in "${SEGLIST[@]}"; do
  for r in $(seq 1 "$ROUNDS"); do run_seg "$seg" "$r"; done
done
# aggregate ALL per-run JSONs -> summary (resume-clean: rebuilt from disk, not appended live)
python3 - "$OUT_ROOT" "$SUMMARY" <<'PY'
import json,glob,os,sys
root,summ=sys.argv[1:3]
MSE={"moonwalk":0.0441,"spinclap":0.0505,"block":0.0350,"combat":0.0533,
     "fierce":0.0638,"circle":0.0076,"jogback":0.0110,"sprint":0.0250}
rows=[]
for d in sorted(glob.glob(os.path.join(root,"*_r*"))):
    jp=os.path.join(d,"phys_valid.json")
    if not os.path.isfile(jp): continue
    name=os.path.basename(d); seg,rnd=name.rsplit("_r",1)
    v=next(iter(json.load(open(jp)).values()))
    rows.append((seg,rnd,MSE.get(seg,""),int(v["fell"]),v["fall_step"],
                 v["window_steps"],round(v["trackable_frac"],3),
                 round(v["min_root_z"],3),round(v["max_tilt_deg"],1)))
rows.sort(key=lambda r:(MSE.get(r[0],9),r[1]))
with open(summ,"w") as f:
    f.write("segment,round,openloop_mse,fell,fall_step,window_steps,survival_frac,min_root_z,max_tilt\n")
    for r in rows: f.write(",".join(str(x) for x in r)+"\n")
print(f"[p1b] aggregated {len(rows)} runs -> {summ}")
PY
echo "[p1b] === survival summary (sorted by open-loop MSE) ==="; cat "$SUMMARY"
