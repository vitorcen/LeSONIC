#!/usr/bin/env python3
"""Dump FlowDP open-loop predicted motion-tokens per BonesSeed motion, in the npz format
the SONIC injector replays (`motion_token (T,64) f32` + `motion_key` + `prompt`), keyed by
the robot_filtered motion key so `gear_sonic_inject.sh` / sequence full_bootstrap can play them.

FlowDP's open-loop token-MSE is ~0.00059 (≈ GT), so replaying these drives the WBC faithfully —
the deployable path, since live closed-loop is brittle (the single-trajectory flow field
explodes off-manifold; see leaderboard / live finding).

Usage (lerobot-v044 env, FlowHeads on PYTHONPATH):
    python scripts/sonic_flowdp_dump_tokens.py \
        --ckpt outputs/flowdp_sonic/checkpoints/006000/pretrained_model \
        --out datasets/sonic_vla_pred_flowdp
"""
from __future__ import annotations
import argparse, os, glob
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET = os.path.join(REPO_ROOT, "datasets/sonic_vla_flowdp")

# task string -> robot_filtered motion key (mirrors gear_sonic_inject.sh KEY map)
TASK2KEY = {
    "dance": "dance_in_da_party_001__A464",
    "do a forward lunge": "forward_lunge_R_001__A359_M",
    "dance the macarena": "macarena_001__A545",
    "kick": "neutral_kick_R_001__A543",
    "squat": "squat_001__A359",
    "jump on one leg": "tired_one_leg_jumping_R_001__A359",
    "walk and turn around": "walking_quip_360_R_002__A428",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "datasets/sonic_vla_pred_flowdp"))
    ap.add_argument("--flow-steps", type=int, default=None)
    args = ap.parse_args()

    import torch, pandas as pd, flowdp  # noqa: F401
    from flowdp.modeling_flowdp import FlowDPPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION

    pol = FlowDPPolicy.from_pretrained(args.ckpt).to("cuda").eval()
    if args.flow_steps:
        pol.diffusion.num_inference_steps = int(args.flow_steps)
    pre = PolicyProcessorPipeline.from_pretrained(args.ckpt, config_filename="policy_preprocessor.json")
    amin = amax = None
    for step in pre.steps:
        s = getattr(step, "stats", None) or getattr(step, "_stats", None)
        if isinstance(s, dict) and ACTION in s:
            to = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            amin, amax = to(s[ACTION]["min"]).astype(np.float32), to(s[ACTION]["max"]).astype(np.float32)
    denom = np.where(amax != amin, amax - amin, 1e-8)
    H = pol.config.n_action_steps

    ds = LeRobotDataset(repo_id="local/sonic_vla_flowdp", root=DATASET)
    parquet = sorted(glob.glob(os.path.join(DATASET, "data", "*", "*.parquet")))
    df = pd.concat([pd.read_parquet(p) for p in parquet]).reset_index(drop=True)
    tdf = pd.read_parquet(os.path.join(DATASET, "meta", "tasks.parquet"))
    ti2task = {int(v): str(k) for k, v in tdf["task_index"].items()}
    ep_ids = df["episode_index"].values

    os.makedirs(args.out, exist_ok=True)
    for e in sorted(pd.unique(ep_ids)):
        idx = np.where(ep_ids == e)[0]
        a, b = int(idx[0]), int(idx[-1]) + 1
        T = b - a
        task = ti2task[int(df["task_index"].iloc[a])]
        key = TASK2KEY.get(task, f"ep{int(e)}")
        toks = np.zeros((T, 64), dtype=np.float32)
        t = 0
        while t < T:
            fr = ds[a + t]
            obs = {"observation.state": fr["observation.state"],
                   "observation.images.ego_view": fr["observation.images.ego_view"]}
            pol.reset()
            with torch.no_grad():
                ch = pol.predict_action_chunk(pre(obs)).detach().float().cpu().numpy()[0]  # (H,78) norm
            ch = np.clip(ch, -1.0, 1.0)
            un = (ch + 1.0) / 2.0 * denom + amin                      # (H,78) raw
            h = min(H, T - t)
            toks[t:t + h] = un[:h, :64]
            t += h
        np.savez(os.path.join(args.out, f"{key}.npz"),
                 motion_token=toks, motion_key=np.array(key), prompt=np.array(task))
        print(f"[dump] {task:24s} -> {key}.npz  ({T},64) tok range [{toks.min():.3f},{toks.max():.3f}]", flush=True)
    print(f"[dump] DONE -> {args.out}")


if __name__ == "__main__":
    main()
