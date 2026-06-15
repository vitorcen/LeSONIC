#!/usr/bin/env python3
"""Open-loop token-MSE SWEEP for StarVLA SONIC ckpts — LeSONIC leaderboard standard.

Mirror of sonic_flowdp_openloop_eval.py (the FlowDP side) for StarVLA frameworks
(QwenGR00T_N17 / QwenPI_v3 / QwenPI_CE). Open-loop teacher forcing: feed each
motion's (ego_view, prompt, state) through the TRAINING dataloader (identical
transforms: 448 resize, state min_max, action identity), stride by the action
horizon, collect predicted chunks, and score against the GT motion_token in the
raw FSQ grid (k/16) — directly comparable to the GR00T open_loop_eval anchor.

Scoring (feedback-lesonic-similarity-leaderboard-standard):
  * macro-MSE (PRIMARY, asc): per-motion mse64 then EQUAL-weight mean — fair
    across the 10x length spread (kick ~165f vs macarena ~1375f).
  * frame-MSE: token-weighted (sum SE / total tokens) — comparable to GR00T 0.0011.
  * template baseline = per-motion-mean token (the bar to beat); skill = template/model;
    beat = #motions with model < its own template.

Sweeps every steps_*.pt under <run>/checkpoints, appends a CSV, rebuilds a
markdown leaderboard. Runs ON THE BOX (free GPU after training); set STARVLA_DIR
and pass --config (the deployed box config). Reference anchors (raw FSQ): GR00T
N1.7 frame ~0.0011, per-motion-mean ~0.039, FlowDP-SONIC frame ~0.00059.

    STARVLA_DIR=/root/autodl-tmp/starVLA \
    /root/autodl-tmp/envs/starvla/bin/python scripts/sonic_starvla_openloop_eval.py \
        --run  /root/autodl-tmp/starvla-outputs/sonic_qwen3_5_4b_gr00t_v2 \
        --config /root/autodl-tmp/starVLA/examples/UNITREE_G1_SONIC/train_files/configs/sonic_qwen3_5_4b_gr00t_v2.yaml \
        --csv  <run>/openloop_eval.csv --leaderboard <run>/leaderboard.md --epoch-steps 953.75
"""

from __future__ import annotations

import argparse
import csv as csvmod
import glob
import json
import os
import re
import sys

import numpy as np

STARVLA_DIR = os.environ.get("STARVLA_DIR", "/root/autodl-tmp/starVLA")
GRID = 16.0  # FSQ grid: tokens sit on k/16


def find_inner_dataset(ds):
    """Unwrap mixture/concat wrappers until we find the dataset with all_steps."""
    seen = [ds]
    while not hasattr(seen[-1], "all_steps"):
        cur = seen[-1]
        for attr in ("datasets", "dataset"):
            nxt = getattr(cur, attr, None)
            if nxt is not None:
                seen.append(nxt[0] if isinstance(nxt, (list, tuple)) else nxt)
                break
        else:
            raise RuntimeError(f"cannot find all_steps under {type(ds)}: {[type(s) for s in seen]}")
    return seen[-1]


def step_of(ckpt: str) -> int:
    m = re.search(r"steps_(\d+)_", os.path.basename(ckpt))
    return int(m.group(1)) if m else -1


def prompt_key(prompt: str, traj_id: int) -> str:
    """Stable short key per motion (just for display); MSE is what ranks."""
    p = (prompt or "").strip().lower()
    table = {
        "dance": "dance", "do a forward lunge": "lunge", "dance the macarena": "macarena",
        "kick": "kick", "squat": "squat", "jump on one leg": "jump",
        "walk and turn around": "walk",
    }
    return table.get(p, f"traj{traj_id}")


def score_ckpt(ckpt, cfg, action_horizon, batch):
    """Return per-motion metrics dict + aggregates for one ckpt."""
    import torch
    from starVLA.model.framework.base_framework import baseframework

    AH = action_horizon
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    ds = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    inner = find_inner_dataset(ds)

    chunk_starts: dict[int, list[tuple[int, int]]] = {}
    traj_len: dict[int, int] = {}
    for pos, (traj_id, base_index) in enumerate(inner.all_steps):
        traj_id, base_index = int(traj_id), int(base_index)
        traj_len[traj_id] = max(traj_len.get(traj_id, 0), base_index + 1)
        if base_index % AH == 0:
            chunk_starts.setdefault(traj_id, []).append((base_index, pos))

    framework = baseframework.from_pretrained(ckpt)
    framework = framework.to(torch.bfloat16).cuda().eval()

    per = {}
    for traj_id in sorted(chunk_starts):
        starts = sorted(chunk_starts[traj_id])
        T = traj_len[traj_id]
        preds, gts = [], []
        prompt = None
        for i in range(0, len(starts), batch):
            grp = starts[i:i + batch]
            examples = []
            for _, pos in grp:
                s = inner[pos]
                ex = {"image": s["image"], "lang": s["lang"]}
                if "state" in s:
                    ex["state"] = np.asarray(s["state"], dtype=np.float32)
                examples.append(ex)
                gts.append(np.asarray(s["action"], dtype=np.float32))  # (AH,78) identity raw
                prompt = s["lang"]
            out = framework.predict_action(examples=examples)
            pred = np.asarray(out["normalized_actions"], dtype=np.float32)  # (B,AH,78) identity raw
            preds.extend(pred)
        pred = np.concatenate(preds, axis=0)[:T]
        gt = np.concatenate(gts, axis=0)[:T]
        tok_p, tok_g = pred[:, :64], gt[:, :64]
        tmpl = tok_g.mean(axis=0, keepdims=True)             # per-motion-mean token baseline
        snapped = np.round(tok_p * GRID) / GRID
        key = prompt_key(prompt, traj_id)
        per[key] = {
            "T": int(T),
            "mse64": float(np.mean((tok_p - tok_g) ** 2)),
            "mse64_snapped": float(np.mean((snapped - tok_g) ** 2)),
            "template": float(np.mean((tmpl - tok_g) ** 2)),
            "bin_acc": float(np.mean(np.round(tok_p * GRID) == np.round(tok_g * GRID))),
            "offgrid": float(np.mean(np.abs(tok_p * GRID - np.round(tok_p * GRID)) / GRID)),
        }
    del framework
    torch.cuda.empty_cache()

    keys = list(per)
    macro = float(np.mean([per[k]["mse64"] for k in keys]))
    macro_tmpl = float(np.mean([per[k]["template"] for k in keys]))
    tot = sum(per[k]["T"] for k in keys)
    frame = float(sum(per[k]["mse64"] * per[k]["T"] for k in keys) / tot)
    frame_tmpl = float(sum(per[k]["template"] * per[k]["T"] for k in keys) / tot)
    beat = sum(1 for k in keys if per[k]["mse64"] < per[k]["template"])
    agg = {
        "macro_mse": macro, "frame_mse": frame,
        "macro_template": macro_tmpl, "frame_template": frame_tmpl,
        "skill": (macro_tmpl / macro) if macro > 0 else 0.0,
        "beat": beat, "n_motions": len(keys),
        "bin_acc": float(np.mean([per[k]["bin_acc"] for k in keys])),
        "snapped_macro": float(np.mean([per[k]["mse64_snapped"] for k in keys])),
    }
    return per, agg


def build_leaderboard(csv_path, lb_path, epoch_steps):
    rows = []
    with open(csv_path) as f:
        for r in csvmod.DictReader(f):
            try:
                r["macro_mse"] = float(r["macro_mse"])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["macro_mse"])
    with open(lb_path, "w") as f:
        f.write("# StarVLA-Qwen3.5-4B-GR00T_v2 — SONIC BonesSeed open-loop leaderboard\n\n")
        f.write("Sorted by macro-MSE (asc). Raw FSQ grid; GR00T N1.7 frame ~0.0011, "
                "per-motion-mean template ~0.039, FlowDP-SONIC frame ~0.00059.\n\n")
        f.write("| rank | step | epoch | macro-MSE | frame-MSE | skill | beat | bin_acc |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(rows, 1):
            ep = f"{int(r['step'])/epoch_steps:.1f}" if epoch_steps else "-"
            f.write(f"| {i} | {r['step']} | {ep} | {float(r['macro_mse']):.5f} | "
                    f"{float(r['frame_mse']):.5f} | {float(r['skill']):.1f}x | "
                    f"{r['beat']}/{r['n_motions']} | {float(r['bin_acc']):.3f} |\n")
    print(f"[lb] wrote {lb_path} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run dir (sweeps <run>/checkpoints/steps_*.pt)")
    ap.add_argument("--ckpt", help="single ckpt (overrides --run sweep)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--leaderboard")
    ap.add_argument("--epoch-steps", type=float, default=953.75, help="frames/global_batch (3815/4)")
    ap.add_argument("--action_horizon", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--leaderboard-from-csv", action="store_true")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)

    sys.path.insert(0, STARVLA_DIR)
    os.chdir(STARVLA_DIR)
    from omegaconf import OmegaConf
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29598")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    cfg = OmegaConf.load(args.config)

    csv_path = args.csv or (os.path.join(args.run, "openloop_eval.csv") if args.run else None)
    lb_path = args.leaderboard or (os.path.join(args.run, "leaderboard.md") if args.run else None)

    if args.leaderboard_from_csv:
        build_leaderboard(csv_path, lb_path, args.epoch_steps)
        return 0

    if args.ckpt:
        ckpts = [os.path.abspath(args.ckpt)]
    else:
        ckpts = sorted(glob.glob(os.path.join(args.run, "checkpoints", "steps_*_pytorch_model.pt")),
                       key=step_of)
    if not ckpts:
        print("[eval] no ckpts found"); return 1

    done = set()
    if csv_path and os.path.exists(csv_path) and args.skip_existing:
        with open(csv_path) as f:
            for r in csvmod.DictReader(f):
                done.add(int(r["step"]))

    fields = ["step", "macro_mse", "frame_mse", "macro_template", "frame_template",
              "skill", "beat", "n_motions", "bin_acc", "snapped_macro"]
    write_header = not (csv_path and os.path.exists(csv_path))
    for ckpt in ckpts:
        st = step_of(ckpt)
        if st in done:
            print(f"[eval] skip step {st} (in csv)"); continue
        per, agg = score_ckpt(ckpt, cfg, args.action_horizon, args.batch)
        print(f"[eval] step {st}: macro={agg['macro_mse']:.5f} frame={agg['frame_mse']:.5f} "
              f"skill={agg['skill']:.1f}x beat={agg['beat']}/{agg['n_motions']} "
              f"bin_acc={agg['bin_acc']:.3f}  per-motion=" +
              " ".join(f"{k}:{per[k]['mse64']:.4f}" for k in per))
        if csv_path:
            with open(csv_path, "a", newline="") as f:
                w = csvmod.DictWriter(f, fieldnames=fields)
                if write_header:
                    w.writeheader(); write_header = False
                w.writerow({"step": st, **{k: agg[k] for k in fields if k != "step"}})
    if lb_path and csv_path and os.path.exists(csv_path):
        build_leaderboard(csv_path, lb_path, args.epoch_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
