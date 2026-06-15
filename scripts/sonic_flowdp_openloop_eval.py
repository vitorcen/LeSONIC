#!/usr/bin/env python3
"""Open-loop token-MSE eval for the FlowDP SONIC head (the "similarity to the original
motion" score). Mirrors the GR00T open_loop_eval used on this same task:

  for each of the 7 motions, teacher-force the recorded observation at frame t, predict
  the action chunk, and MSE the predicted motion_token (un-normalized, raw FSQ-grid
  units) against the recorded GT token. Report per-motion + average, plus two template
  baselines so "is it actually learning vs predicting a mean" is answerable:

    per-motion-mean : predict that motion's mean token for every frame  (the bar to beat)
    global-mean     : predict the global mean token for every frame

  Reference numbers on this task (GR00T N1.7, raw token space): best ~0.0011,
  per-motion-mean ~0.039, global-mean ~0.048.

Usage (lerobot-v044 env, FlowHeads on PYTHONPATH):
    python scripts/sonic_flowdp_openloop_eval.py \
        --ckpt outputs/flowdp_sonic/checkpoints/001000/pretrained_model
    # sweep a whole run:
    python scripts/sonic_flowdp_openloop_eval.py --sweep outputs/flowdp_sonic --csv eval.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET = os.path.join(REPO_ROOT, "datasets/sonic_vla_flowdp")
STRIDE_DEFAULT = 8     # predict every STRIDE frames (chunks overlap GT windows)


def _load_policy(ckpt, flow_steps=None):
    import torch
    from flowdp.modeling_flowdp import FlowDPPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.utils.constants import ACTION
    pol = FlowDPPolicy.from_pretrained(ckpt).to("cuda").eval()
    if flow_steps is not None:
        pol.diffusion.num_inference_steps = int(flow_steps)
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json")
    amin = amax = None
    for step in pre.steps:
        stats = getattr(step, "stats", None) or getattr(step, "_stats", None)
        if isinstance(stats, dict) and ACTION in stats:
            s = stats[ACTION]
            to = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            amin, amax = to(s["min"]).astype(np.float32), to(s["max"]).astype(np.float32)
    assert amin is not None, "no ACTION stats in preprocessor"
    return torch, pol, pre, amin, amax


def evaluate(ckpt, flow_steps=None, stride=STRIDE_DEFAULT):
    torch, pol, pre, amin, amax = _load_policy(ckpt, flow_steps)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local/sonic_vla_flowdp", root=DATASET)
    H = pol.config.n_action_steps
    denom = np.where(amax != amin, amax - amin, 1e-8)

    # GT tokens (raw) per frame + episode boundaries. v3.0 consolidates ALL episodes into
    # one parquet, distinguished by the episode_index column (frame order == the
    # LeRobotDataset global frame index), so split on episode_index, not per-file.
    import pandas as pd
    parquet = sorted(glob.glob(os.path.join(DATASET, "data", "*", "*.parquet")))
    tasks_map = {}
    tp = os.path.join(DATASET, "meta", "tasks.parquet")
    if os.path.exists(tp):
        tdf = pd.read_parquet(tp)
        tasks_map = {int(v): str(k) for k, v in tdf["task_index"].items()}
    df = pd.concat([pd.read_parquet(p) for p in parquet]).reset_index(drop=True)
    gt_tokens = np.stack(df["action"].values)[:, :64].astype(np.float32)  # (total_frames,64) raw
    ep_ids = df["episode_index"].values
    starts, ends, ep_tasks = [], [], []
    for e in sorted(pd.unique(ep_ids)):
        idx = np.where(ep_ids == e)[0]
        starts.append(int(idx[0])); ends.append(int(idx[-1]) + 1)
        ep_tasks.append(tasks_map.get(int(df["task_index"].iloc[idx[0]]), f"ep{int(e)}"))

    per_motion, base_pm, base_gm = {}, {}, {}
    global_mean_tok = gt_tokens.mean(0)  # (64,)

    se_model, se_pm, se_gm, n_tok = 0.0, 0.0, 0.0, 0
    for ei, (a, b) in enumerate(zip(starts, ends)):
        T = b - a
        task = ep_tasks[ei]
        motion_mean_tok = gt_tokens[a:b].mean(0)  # (64,)
        m_se = pm_se = gm_se = 0.0
        m_n = 0
        for t in range(0, T - 1, stride):
            frame = ds[a + t]
            obs = {"observation.state": frame["observation.state"],
                   "observation.images.ego_view": frame["observation.images.ego_view"]}
            pol.reset()
            batch = pre(obs)
            with torch.no_grad():
                chunk = pol.predict_action_chunk(batch)        # (1,H,78) normalized
            chunk = chunk.detach().float().cpu().numpy()[0]     # (H,78)
            pred_tok = ((chunk + 1.0) / 2.0 * denom + amin)[:, :64]  # (H,64) raw
            h = min(H, T - t)
            gt = gt_tokens[a + t: a + t + h]                    # (h,64)
            p = pred_tok[:h]
            m_se += float(((p - gt) ** 2).sum())
            pm_se += float(((motion_mean_tok[None] - gt) ** 2).sum())
            gm_se += float(((global_mean_tok[None] - gt) ** 2).sum())
            m_n += gt.size
        per_motion[task] = m_se / max(m_n, 1)
        base_pm[task] = pm_se / max(m_n, 1)
        base_gm[task] = gm_se / max(m_n, 1)
        se_model += m_se; se_pm += pm_se; se_gm += gm_se; n_tok += m_n

    # macro = each motion equal weight (fair across 10x length spread, kick 165 vs
    # macarena 1375); frame = token-weighted (comparable to GR00T's 0.0011).
    macro = float(np.mean(list(per_motion.values())))
    macro_tmpl = float(np.mean(list(base_pm.values())))
    n_beat = sum(1 for k in per_motion if per_motion[k] < base_pm[k])
    return {
        "ckpt": ckpt,
        "flow_steps": pol.diffusion.num_inference_steps,
        "mse_macro": macro,                       # PRIMARY rank key (per-motion equal weight)
        "mse_macro_template": macro_tmpl,
        "skill": macro_tmpl / macro if macro > 0 else 0.0,  # x better than template
        "n_beat": n_beat,                          # motions beating their template /7
        "mse_frame": se_model / n_tok,             # token-weighted (GR00T-comparable)
        "mse_per_motion_mean": se_pm / n_tok,
        "mse_global_mean": se_gm / n_tok,
        "per_motion": per_motion,
        "base_per_motion_mean": base_pm,
    }


def _step_of(ckpt_dir):
    import re
    m = re.search(r"/(\d+)/pretrained_model", ckpt_dir)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="path to .../<step>/pretrained_model")
    ap.add_argument("--sweep", help="run dir; evaluate every checkpoints/<step>/pretrained_model")
    ap.add_argument("--flow-steps", type=int, default=None)
    ap.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--leaderboard", default=None, help="write a ranked markdown leaderboard here")
    ap.add_argument("--leaderboard-from-csv", default=None,
                    help="skip eval; just (re)build the markdown leaderboard from this CSV")
    ap.add_argument("--epoch-steps", type=float, default=None,
                    help="steps per epoch (for the epoch column); flowdp-sonic b64 ~= 60")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip ckpts whose step is already in --csv (watcher idempotency)")
    args = ap.parse_args()

    # CSV-only leaderboard rebuild (cheap path for the watcher)
    if args.leaderboard_from_csv:
        import csv as _csv
        rows = []
        with open(args.leaderboard_from_csv) as f:
            for d in _csv.DictReader(f):
                rows.append((int(d["step"]), {
                    "mse_macro": float(d["mse_macro"]), "mse_frame": float(d["mse_frame"]),
                    "skill": float(d["skill"]), "n_beat": int(d["n_beat"]),
                    "mse_macro_template": float(d["mse_macro_template"])}))
        ranked = sorted(rows, key=lambda x: x[1]["mse_macro"])
        _write_leaderboard(args.leaderboard or "leaderboard.md", ranked, args.epoch_steps)
        print(f"[eval] leaderboard ({len(ranked)} ckpts) -> {args.leaderboard}")
        return

    done_steps = set()
    if args.skip_existing and args.csv and os.path.exists(args.csv):
        import csv as _csv
        with open(args.csv) as f:
            for d in _csv.DictReader(f):
                done_steps.add(int(d["step"]))

    targets = []
    if args.sweep:
        targets = sorted(glob.glob(os.path.join(args.sweep, "checkpoints", "[0-9]*", "pretrained_model")),
                         key=_step_of)
    elif args.ckpt:
        targets = [args.ckpt]
    else:
        ap.error("need --ckpt or --sweep")

    rows = []
    for ck in targets:
        step = _step_of(ck)
        if step in done_steps:
            print(f"[eval] step={step} already scored, skip", flush=True)
            continue
        r = evaluate(os.path.abspath(ck), args.flow_steps, args.stride)
        ep = f"{step/args.epoch_steps:.1f}" if args.epoch_steps else "?"
        print(f"[eval] step={step:>6} ep={ep:>5} macro={r['mse_macro']:.5f} "
              f"frame={r['mse_frame']:.5f} skill={r['skill']:.2f}x beat={r['n_beat']}/7 "
              f"(tmpl macro={r['mse_macro_template']:.5f})", flush=True)
        for k, v in r["per_motion"].items():
            print(f"          {k:24s} {v:.5f}  (tmpl {r['base_per_motion_mean'][k]:.5f})")
        rows.append((step, r))
        if args.csv:
            new = not os.path.exists(args.csv)
            with open(args.csv, "a") as f:
                if new:
                    f.write("step,epoch,mse_macro,mse_frame,skill,n_beat,mse_macro_template\n")
                f.write(f"{step},{ep},{r['mse_macro']:.6f},{r['mse_frame']:.6f},"
                        f"{r['skill']:.3f},{r['n_beat']},{r['mse_macro_template']:.6f}\n")

    if rows:
        ranked = sorted(rows, key=lambda x: x[1]["mse_macro"])   # PRIMARY: macro-MSE ascending
        best = ranked[0]
        print(f"\n[eval] BEST: step={best[0]} macro={best[1]['mse_macro']:.5f} "
              f"frame={best[1]['mse_frame']:.5f} skill={best[1]['skill']:.2f}x")
        if args.leaderboard:
            _write_leaderboard(args.leaderboard, ranked, args.epoch_steps)
            print(f"[eval] leaderboard -> {args.leaderboard}")


def _write_leaderboard(path, ranked, epoch_steps):
    """ranked = [(step, result)] already sorted by mse_macro ascending."""
    lines = [
        "# FlowDP-SONIC open-loop similarity leaderboard",
        "",
        "Open-loop token-MSE vs the recorded BonesSeed motions (raw FSQ-grid units, "
        "teacher-forced). **Rank = macro-MSE** (7 motions equal weight). `frame` = "
        "token-weighted (comparable to GR00T N1.7's 0.0011). `skill` = template/model "
        "(>1 = beats the per-motion-mean template). `beat` = motions under their template /7.",
        "",
        "| rank | step | epoch | macro-MSE | frame-MSE | skill | beat | template(macro) |",
        "| ---- | ---- | ----- | --------- | --------- | ----- | ---- | --------------- |",
    ]
    for i, (step, r) in enumerate(ranked, 1):
        ep = f"{step/epoch_steps:.1f}" if epoch_steps else "?"
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, str(i))
        lines.append(f"| {medal} | {step} | {ep} | **{r['mse_macro']:.5f}** | "
                     f"{r['mse_frame']:.5f} | {r['skill']:.2f}x | {r['n_beat']}/7 | "
                     f"{r['mse_macro_template']:.5f} |")
    lines += ["",
              "**Reference (same task, raw token space):** GR00T N1.7 finetune best ≈ "
              "**0.0011** (frame); per-motion-mean template ≈ 0.039; global-mean ≈ 0.048.",
              ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
