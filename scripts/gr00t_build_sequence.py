#!/usr/bin/env python3
"""Build a VlaLiveInjector timeline JSON from a motion sequence — ad-hoc (--seq) or a named
demo flow (--flow, from sonic_demo_flows.json).

A sequence is played back-to-back in ONE running session (no GUI/server restart) by swapping the
prompt over time; the injector loops it modulo-style. Motions that cannot self-start from rest
(kick/walk/jump — the memoryless single-frame policy stalls on the ambiguous standing pose) get a
bootstrap window that replays their open-loop dump tokens to enter the move before handing back to
live GR00T. ``full_bootstrap`` replays the WHOLE segment from the dump (continuous locomotion,
e.g. a walk-through between actions). A short ``settle`` window at each seam returns the robot
toward a neutral anchor so the recording is smooth.

Pure stdlib — runs under any python3, no GPU. The injector consumes the emitted JSON via
``++callbacks.vla_live.timeline_json=<path>``.

Usage:
  # named flow (the framework):
  python scripts/gr00t_build_sequence.py --flow flow2 --pred-dir datasets/sonic_vla_pred_8k_final --out /tmp/seq.json
  python scripts/gr00t_build_sequence.py --list                      # show available flows
  # ad-hoc:
  python scripts/gr00t_build_sequence.py --seq squat,walk,kick --pred-dir ... --out ...
  python scripts/gr00t_build_sequence.py --seq "squat:150,walk:250" --pred-dir ... --out ...
"""

from __future__ import annotations

import argparse
import json
import os

# Short name -> (robot_filtered motion key, prompt). Mirrors gear_sonic_live.sh exactly.
KEY = {
    "dance": "dance_in_da_party_001__A464",
    "lunge": "forward_lunge_R_001__A359_M",
    "macarena": "macarena_001__A545",
    "kick": "neutral_kick_R_001__A543",
    "squat": "squat_001__A359",
    "jump": "tired_one_leg_jumping_R_001__A359",
    "walk": "walking_quip_360_R_002__A428",
    # LAFAN skill segments (trained into outputs/gr00t_sonic_skills); use PKL=data/skill_demo_robot.pkl
    "guard": "fight_seg000", "jab": "fight_seg020", "combo": "fight_seg050",
    "turn": "run_seg001", "jog": "run_seg006", "runfast": "run_seg017",
    "fight": "lafan_fight_15s",
    # flow3 hand-picked LAFAN windows (trained into outputs/gr00t_sonic_flow3);
    # use PKL=data/seg_flow3_all.pkl PRED_DIR=datasets/sonic_vla_pred_flow3
    "combat": "fight_combat_combo_kicks", "block": "fight_block_pushkick_shove",
    "fierce": "fight_fierce_swings", "jogback": "run_jog_backward",
    "sprint": "run_sprint_backpedal", "circle": "run_circle",
    "moonwalk": "dance_moonwalk", "spinclap": "dance_spin_stepback_clap",
}
PROMPT = {
    "dance": "dance",
    "lunge": "do a forward lunge",
    "macarena": "dance the macarena",
    "kick": "kick",
    "squat": "squat",
    "jump": "jump on one leg",
    "walk": "walk and turn around",
    # prompts MUST match the trained task strings exactly (sonic_vla_lerobot_skills)
    "guard": "hold a fighting guard", "jab": "throw a quick jab",
    "combo": "throw a punch combo advancing",
    "turn": "turn around in place", "jog": "jog forward", "runfast": "run forward fast",
    "fight": "throw punches",
    # flow3 prompts MUST match the trained task strings exactly (sonic_vla_lerobot_flow3)
    "combat": "combat strikes and combo kicks", "block": "block and push-kick",
    "fierce": "fierce swings", "jogback": "jog forward then run backward",
    "sprint": "sprint back and forth then backpedal", "circle": "run in a circle",
    "moonwalk": "moonwalk", "spinclap": "spin, step back, and clap",
}
# Motions that do NOT self-start from rest (need a bootstrap window). From the closed-loop scan:
# squat/lunge/dance/macarena self-sustain; kick/walk/jump stall -> bootstrap.
# LAFAN skills: punches + locomotion all one-shot from rest -> bootstrap into the move.
# flow3 windows all start mid-action -> all one-shot.
ONESHOT = {"kick", "walk", "jump",
           "jab", "combo", "turn", "jog", "runfast", "fight",
           "combat", "block", "fierce", "jogback", "sprint", "circle", "moonwalk", "spinclap"}

FLOWS_FILE_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonic_demo_flows.json")


def parse_seq_item(item: str, default_steps: int) -> dict:
    """Parse 'name' or 'name:steps' -> item dict {motion, steps, full_bootstrap}."""
    if ":" in item:
        name, steps = item.split(":", 1)
        return {"motion": name.strip(), "steps": int(steps)}
    return {"motion": item.strip(), "steps": default_steps}


def load_flow(flows_file: str, name: str) -> list:
    """Return a flow's segment items [{motion, steps, full_bootstrap?}, ...]."""
    with open(flows_file) as fh:
        flows = json.load(fh)
    if name not in flows or name.startswith("_"):
        avail = [k for k in flows if not k.startswith("_")]
        raise SystemExit(f"unknown flow '{name}'. available: {', '.join(avail)}")
    return flows[name]["segments"]


def build_segments(items: list, pred_dir: str, settle: int, bootstrap_steps: int) -> list:
    """Turn motion items into injector timeline segments."""
    segments = []
    for i, it in enumerate(items):
        name = it["motion"]
        if name not in KEY:
            raise SystemExit(f"unknown motion '{name}'. choose from: {', '.join(KEY)}")
        # JSON `steps` is the PURE motion budget (= P x full-clip length; what a percentage means).
        # The seam settle is OVERHEAD added ON TOP, so the rendered motion length (live window, or
        # full_bootstrap replay) always equals `steps` exactly — percentages are honoured 1:1.
        steps = int(it["steps"])
        se = settle if (i > 0 and settle > 0) else 0
        seg = {"prompt": PROMPT[name], "steps": steps + se}
        # First segment starts from the env's reset pose (no settle needed); later seams settle.
        if se:
            seg["settle_steps"] = se
        if name in ONESHOT:
            npz = os.path.join(pred_dir, f"{KEY[name]}.npz")
            if not os.path.isfile(npz):
                raise SystemExit(f"bootstrap dump missing for one-shot '{name}': {npz}\n"
                                 f"  run scripts/gr00t_dump_pred_tokens.py first.")
            seg["bootstrap_npz"] = npz
            # full_bootstrap: replay the WHOLE segment from the dump (continuous locomotion);
            # else just a short window to enter the motion, then hand to live GR00T.
            seg["bootstrap_steps"] = steps if it.get("full_bootstrap") else bootstrap_steps
        segments.append(seg)
    return segments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", help="comma list, e.g. 'squat,walk,kick' or 'squat:150,kick:180'")
    ap.add_argument("--flow", help="named demo flow from --flows-file (e.g. flow1, flow2)")
    ap.add_argument("--flows-file", default=FLOWS_FILE_DEFAULT, help="demo-flow registry JSON")
    ap.add_argument("--list", action="store_true", help="list available flows and exit")
    ap.add_argument("--pred-dir", default="datasets/sonic_vla_pred_8k_final",
                    help="dir of <key>.npz GR00T dump tokens (for bootstrap of one-shot motions)")
    ap.add_argument("--out", help="output timeline JSON path")
    ap.add_argument("--steps", type=int, default=200, help="default control steps per segment")
    ap.add_argument("--settle", type=int, default=40, help="neutral/freeze steps at each seam")
    ap.add_argument("--bootstrap-steps", type=int, default=80,
                    help="bootstrap window for one-shot motions (unless full_bootstrap)")
    ap.add_argument("--neutral-npz", default="",
                    help="optional standing/neutral token npz injected during settle windows")
    args = ap.parse_args()

    if args.list:
        with open(args.flows_file) as fh:
            flows = json.load(fh)
        print(f"demo flows in {args.flows_file}:")
        for name, f in flows.items():
            if name.startswith("_"):
                continue
            seq = " -> ".join(s["motion"] for s in f["segments"])
            print(f"  {name:8s} {seq}\n           {f.get('desc', '')}")
        return

    if not args.out:
        raise SystemExit("--out is required (unless --list)")
    if bool(args.seq) == bool(args.flow):
        raise SystemExit("give exactly one of --seq or --flow")

    if args.flow:
        items = load_flow(args.flows_file, args.flow)
    else:
        items = [parse_seq_item(x, args.steps) for x in args.seq.split(",") if x.strip()]

    segments = build_segments(items, args.pred_dir, args.settle, args.bootstrap_steps)

    total = sum(s["steps"] for s in segments)
    spec = {"segments": segments, "total_steps": total}
    if args.neutral_npz:
        spec["neutral_npz"] = args.neutral_npz

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(spec, fh, indent=2)

    src = f"flow '{args.flow}'" if args.flow else "seq"
    print(f"[seq] wrote {args.out} from {src}: {len(segments)} segments, ~{total} steps "
          f"(~{total / 50:.0f}s @ 50Hz control)")
    for i, (it, seg) in enumerate(zip(items, segments)):
        tag = []
        if "bootstrap_steps" in seg:
            tag.append("full-bootstrap" if it.get("full_bootstrap") else f"bootstrap={seg['bootstrap_steps']}")
        if "settle_steps" in seg:
            tag.append(f"settle={seg['settle_steps']}")
        print(f"  {i}: {it['motion']:9s} steps={seg['steps']:4d} prompt={seg['prompt']!r} "
              f"{' '.join(tag)}")


if __name__ == "__main__":
    main()
