#!/usr/bin/env python3
"""Open-loop SONIC token-MSE for a GR00T-SONIC checkpoint (funnel tier-1, cheap coarse filter).

Mirrors gr00t.eval.open_loop_eval.evaluate_single_trajectory with modality_keys=["motion_token"]
so the number is the *pure* token-MSE in raw FSQ units — directly comparable to the ckpt-8000
baseline (~0.0011). NO Isaac sim: loads the policy locally and predicts stride-action_horizon
chunks from each motion's GT (ego_view, prompt, state).

IMPORTANT (see doc/sonic_robustness_retrain.html): this is a COARSE filter only. ckpt-8000 had
OL-MSE 0.0011 yet froze in closed loop — OL-MSE cannot see the frozen-chunk collapse. Use it to
(a) confirm state-noise training didn't destroy token capability and (b) find the sweet-spot band;
the closed-loop kick-basin sweep is the decisive selector.

Run in the Isaac-GR00T venv:
    cd dependencies/Isaac-GR00T
    COMPILE_ACTION_HEAD_DISABLE=1 .venv/bin/python ../../scripts/sonic_gr00t_openloop_mse.py \
        --model_path ../../outputs/gr00t_sonic_noise08/checkpoint-6400 \
        --dataset_path ../../datasets/sonic_vla_lerobot
"""
from __future__ import annotations

import argparse
import logging
import os
import re

import numpy as np

# The canonical evaluate_single_trajectory always plots (plt.tight_layout + savefig); matplotlib's
# legend best-position search crashes on this data ("input operand has more dimensions ...").
# We only need the MSE number — neuter plotting so the eval never dies in rendering.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
_plt.tight_layout = lambda *a, **k: None
_plt.savefig = lambda *a, **k: None

# dataset episode order = convert order (mirrors gr00t_dump_pred_tokens.TRAJ_TO_KEY)
TRAJ_TO_KEY = [
    "dance", "lunge", "macarena", "kick", "squat", "jump", "walk",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--action_horizon", type=int, default=40)
    ap.add_argument("--steps", type=int, default=100000, help="cap; auto-clamped to each traj length")
    ap.add_argument("--traj_ids", type=int, nargs="+", default=list(range(7)))
    ap.add_argument("--tsv", default=None, help="append one result row to this TSV")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)  # quiet the per-step inference spam

    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.open_loop_eval import evaluate_single_trajectory
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    step_m = re.search(r"checkpoint-(\d+)", args.model_path)
    step = step_m.group(1) if step_m else os.path.basename(args.model_path.rstrip("/"))

    emb = EmbodimentTag.resolve("unitree_g1_sonic")
    policy = Gr00tPolicy(
        embodiment_tag=emb,
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )

    per = {}
    for tid in args.traj_ids:
        mse, _mae = evaluate_single_trajectory(
            policy, loader, tid, emb,
            modality_keys=["motion_token"],     # PURE token-MSE (comparable to 0.0011 baseline)
            steps=args.steps, action_horizon=args.action_horizon,
            save_plot_path=(f"/tmp/ol_mse_{step}_traj{tid}.jpeg" if os.environ.get("OL_MSE_PLOT") else None),
        )
        key = TRAJ_TO_KEY[tid] if tid < len(TRAJ_TO_KEY) else f"traj{tid}"
        per[key] = float(mse)

    macro = float(np.mean(list(per.values())))
    order = [TRAJ_TO_KEY[t] for t in args.traj_ids if t < len(TRAJ_TO_KEY)]
    print(f"\n[ol-mse] ckpt={step}  MACRO token-MSE={macro:.5f}")
    for k in order:
        print(f"    {k:9s} {per[k]:.5f}")

    if args.tsv:
        new = not os.path.exists(args.tsv)
        with open(args.tsv, "a") as f:
            if new:
                f.write("ckpt\tmacro_mse\t" + "\t".join(order) + "\n")
            f.write(f"{step}\t{macro:.5f}\t" + "\t".join(f"{per[k]:.5f}" for k in order) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
