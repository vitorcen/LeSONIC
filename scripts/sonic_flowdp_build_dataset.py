#!/usr/bin/env python3
"""Build a Diffusion-Policy-friendly LeRobot dataset for the FlowDP SONIC head.

The recorded BonesSeed dataset (`datasets/sonic_vla_lerobot`) is a GR00T-shaped
LeRobot v2.1 set: many `observation.*` / `action.*` / `teleop.*` keys. lerobot's
DiffusionPolicy / FlowDP backbone wants exactly ONE `observation.state`, ONE
`action`, and one image. It also has **no language path**, so the only thing that
tells "kick" from "dance" must ride in `observation.state`.

This converter rewrites the parquets (NO video re-encode — the ego_view mp4 files
are copied verbatim) into:

    observation.state   (53) = joint_state(43) + projected_gravity(3) + motion_onehot(7)
    action              (78) = motion_token(64) + left_hand_joints(7) + right_hand_joints(7)
    observation.images.ego_view  (480,640,3 video)  -- unchanged, referenced by path

`motion_onehot` is one-hot(task_index, 7); task_index is constant per episode
(ep_index == task_index here). The serve adapter maps prompt -> task_index via the
SAME meta/tasks.jsonl so train/serve agree.

Stats:
  - observation.state / action : recomputed (numpy) over all frames -> MIN_MAX works.
  - observation.images.ego_view : injected ImageNet mean/std (the ResNet18 backbone
    is ImageNet-pretrained, so this is the *correct* MEAN_STD, and needs no video
    decode). Same constants for every episode.

Usage:
    python scripts/sonic_flowdp_build_dataset.py \
        --src datasets/sonic_vla_lerobot --dst datasets/sonic_vla_flowdp
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

IMAGENET_MEAN = [[[0.485]], [[0.456]], [[0.406]]]  # (3,1,1)
IMAGENET_STD = [[[0.229]], [[0.224]], [[0.225]]]
IMG_MIN = [[[0.0]], [[0.0]], [[0.0]]]
IMG_MAX = [[[1.0]], [[1.0]], [[1.0]]]

N_MOTIONS = 7
IMG_KEY = "observation.images.ego_view"
KEEP_ID_COLS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]


def feat_stats_global(arr: np.ndarray) -> dict:
    """stats.json schema: mean/std/min/max/q01/q99 (lists, per-dim)."""
    return {
        "mean": arr.mean(0).tolist(),
        "std": arr.std(0).tolist(),
        "min": arr.min(0).tolist(),
        "max": arr.max(0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def feat_stats_episode(arr: np.ndarray) -> dict:
    """episodes_stats.jsonl schema: min/max/mean/std/count."""
    return {
        "min": arr.min(0).tolist(),
        "max": arr.max(0).tolist(),
        "mean": arr.mean(0).tolist(),
        "std": arr.std(0).tolist(),
        "count": [int(arr.shape[0])],
    }


def img_stats_global() -> dict:
    return {"mean": IMAGENET_MEAN, "std": IMAGENET_STD, "min": IMG_MIN, "max": IMG_MAX,
            "q01": IMG_MIN, "q99": IMG_MAX}


def img_stats_episode(count: int) -> dict:
    return {"min": IMG_MIN, "max": IMG_MAX, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
            "count": [int(count)]}


def stack(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in df[col].values])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/sonic_vla_lerobot")
    ap.add_argument("--dst", default="datasets/sonic_vla_flowdp")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if os.path.exists(dst):
        sys.exit(f"[build] dst already exists: {dst} (rm -rf it first)")

    print(f"[build] copy {src} -> {dst} (videos preserved, no re-encode)")
    shutil.copytree(src, dst)

    # --- rewrite each episode parquet ---
    data_root = os.path.join(dst, "data")
    parquets = []
    for ch in sorted(os.listdir(data_root)):
        chd = os.path.join(data_root, ch)
        if not os.path.isdir(chd):
            continue
        for fn in sorted(os.listdir(chd)):
            if fn.endswith(".parquet"):
                parquets.append(os.path.join(chd, fn))
    print(f"[build] {len(parquets)} episode parquets")

    all_state, all_action = [], []
    ep_state, ep_action, ep_index_order = {}, {}, {}
    for p in parquets:
        df = pd.read_parquet(p)
        n = len(df)
        ti = int(df["task_index"].iloc[0])
        assert (df["task_index"] == ti).all(), f"{p}: mixed task_index"
        assert 0 <= ti < N_MOTIONS, f"{p}: task_index={ti} out of [0,{N_MOTIONS})"
        ei = int(df["episode_index"].iloc[0])

        joint = stack(df, "observation.state")               # (n,43)
        grav = stack(df, "observation.projected_gravity")    # (n,3)
        onehot = np.zeros((n, N_MOTIONS), dtype=np.float64)
        onehot[:, ti] = 1.0
        state = np.concatenate([joint, grav, onehot], axis=1)  # (n,53)

        tok = stack(df, "action.motion_token")               # (n,64)
        lh = stack(df, "teleop.left_hand_joints")            # (n,7)
        rh = stack(df, "teleop.right_hand_joints")           # (n,7)
        action = np.concatenate([tok, lh, rh], axis=1)        # (n,78)

        out = pd.DataFrame({
            "observation.state": list(state.astype(np.float32)),
            "action": list(action.astype(np.float32)),
        })
        for c in KEEP_ID_COLS:
            out[c] = df[c].values
        out.to_parquet(p, index=False)

        all_state.append(state)
        all_action.append(action)
        ep_state[ei] = feat_stats_episode(state)
        ep_action[ei] = feat_stats_episode(action)
        ep_index_order[ei] = n
        print(f"  ep{ei} task{ti} n={n}  state{state.shape} action{action.shape}")

    all_state = np.concatenate(all_state, 0)
    all_action = np.concatenate(all_action, 0)

    # --- info.json: keep only the surviving features ---
    info_p = os.path.join(dst, "meta", "info.json")
    info = json.load(open(info_p))
    old = info["features"]
    new_feats = {IMG_KEY: old[IMG_KEY]}
    new_feats["observation.state"] = {
        "dtype": "float32", "shape": [53],
        "names": [f"joint_{i}" for i in range(43)]
                 + ["gravity_x", "gravity_y", "gravity_z"]
                 + [f"motion_onehot_{i}" for i in range(N_MOTIONS)],
    }
    new_feats["action"] = {
        "dtype": "float32", "shape": [78],
        "names": [f"motion_token_{i}" for i in range(64)]
                 + [f"left_hand_{i}" for i in range(7)]
                 + [f"right_hand_{i}" for i in range(7)],
    }
    for c in KEEP_ID_COLS:
        new_feats[c] = old[c]
    info["features"] = new_feats
    json.dump(info, open(info_p, "w"), indent=4)
    print(f"[build] info.json features -> {list(new_feats)}")

    # --- stats.json (global, aggregated) ---
    stats_p = os.path.join(dst, "meta", "stats.json")
    old_stats = json.load(open(stats_p))
    new_stats = {
        IMG_KEY: img_stats_global(),
        "observation.state": feat_stats_global(all_state),
        "action": feat_stats_global(all_action),
        "timestamp": old_stats["timestamp"],
    }
    json.dump(new_stats, open(stats_p, "w"), indent=4)
    print(f"[build] stats.json -> {list(new_stats)}")

    # --- episodes_stats.jsonl (per episode) ---
    es_p = os.path.join(dst, "meta", "episodes_stats.jsonl")
    old_es = {}
    with open(es_p) as f:
        for line in f:
            e = json.loads(line)
            old_es[int(e["episode_index"])] = e
    with open(es_p, "w") as f:
        for ei in sorted(ep_state):
            n = ep_index_order[ei]
            stats = {
                IMG_KEY: img_stats_episode(n),
                "observation.state": ep_state[ei],
                "action": ep_action[ei],
                "timestamp": old_es[ei]["stats"]["timestamp"],
            }
            f.write(json.dumps({"episode_index": ei, "stats": stats}) + "\n")
    print(f"[build] episodes_stats.jsonl rewritten ({len(ep_state)} eps)")

    # modality.json references dropped keys; DP ignores it. Remove to avoid confusion.
    mod = os.path.join(dst, "meta", "modality.json")
    if os.path.exists(mod):
        os.remove(mod)
    for junk in ("relative_stats.json",):
        jp = os.path.join(dst, "meta", junk)
        if os.path.exists(jp):
            os.remove(jp)

    print(f"[build] DONE -> {dst}")
    print(f"[build] state(53)=joint43+gravity3+onehot7 | action(78)=token64+lh7+rh7 | "
          f"frames={all_state.shape[0]}")


if __name__ == "__main__":
    main()
