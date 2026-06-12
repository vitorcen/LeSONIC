#!/usr/bin/env python3
"""Dump StarVLA's PREDICTED SONIC motion_token per LAFAN flow3 window + A/B metrics.

Mirror of gr00t_dump_pred_tokens.py for the StarVLA side of the A/B
(doc/sonic_starvla_swap_brainstorm.html §11). Open-loop teacher forcing: feed
each window's (ego_view, prompt, state) THROUGH THE TRAINING DATALOADER
(identical transforms: 448 resize, state min_max, action identity), stride by
the 40-step horizon, collect predicted chunks, save per-window npz compatible
with gear_sonic_inject.sh, and print the MSE table against the GT tokens.

Conventions match the GR00T anchor (open_loop_eval): chunk-stride sampling,
MSE over all 78 action dims (motion_token 64 + zero hand 14) AND token-only 64.

Run in the starvla_eval_qwen35 env:
    ~/miniconda3/envs/starvla_eval_qwen35/bin/python \
        scripts/starvla_dump_pred_tokens.py \
        --ckpt outputs/starvla/sonic_qwen3_5_4b_pi_v3/checkpoints/steps_6000_pytorch_model.pt \
        --out datasets/sonic_vla_pred_starvla
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STARVLA_DIR = os.environ.get("STARVLA_DIR",
                             os.path.join(REPO_ROOT, "dependencies", "starVLA"))
# QwenPI_CE head ships in the vitorcen/StarVLA fork checkout (no runtime deploy).

# prompt -> output npz key (must match datasets/sonic_vla_pred_flow3 names so the
# injector / build_flow3_sequence.py work unchanged on the StarVLA dumps)
PROMPT_TO_KEY = {
    "moonwalk": "dance_moonwalk",
    "spin, step back, and clap": "dance_spin_stepback_clap",
    "block and push-kick": "fight_block_pushkick_shove",
    "combat strikes and combo kicks": "fight_combat_combo_kicks",
    "fierce swings": "fight_fierce_swings",
    "run in a circle": "run_circle",
    "jog forward then run backward": "run_jog_backward",
    "sprint back and forth then backpedal": "run_sprint_backpedal",
}

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="steps_<N>_pytorch_model.pt under <run_dir>/checkpoints/")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "scripts/starvla/configs/sonic_qwen3_5_4b_pi_v3.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--action_horizon", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8, help="chunk-starts per forward")
    args = ap.parse_args()
    # absolutize before the chdir below
    args.ckpt = os.path.abspath(args.ckpt)
    args.config = os.path.abspath(args.config)
    args.out = os.path.abspath(args.out)

    sys.path.insert(0, STARVLA_DIR)
    os.chdir(STARVLA_DIR)

    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)

    # registry needs the deployed examples/UNITREE_G1_SONIC symlink (launcher creates it)
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29598")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.model.framework.base_framework import baseframework

    ds = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    inner = find_inner_dataset(ds)
    AH = args.action_horizon

    # group chunk-start positions by trajectory (do NOT trust global index order)
    chunk_starts: dict[int, list[tuple[int, int]]] = {}  # traj -> [(base_index, ds_pos)]
    traj_len: dict[int, int] = {}
    for pos, (traj_id, base_index) in enumerate(inner.all_steps):
        traj_id, base_index = int(traj_id), int(base_index)
        traj_len[traj_id] = max(traj_len.get(traj_id, 0), base_index + 1)
        if base_index % AH == 0:
            chunk_starts.setdefault(traj_id, []).append((base_index, pos))

    print(f"[dump] {len(traj_len)} trajectories, lens={[traj_len[t] for t in sorted(traj_len)]}")

    framework = baseframework.from_pretrained(args.ckpt)
    framework = framework.to(torch.bfloat16).cuda().eval()

    os.makedirs(args.out, exist_ok=True)
    metrics = {}
    for traj_id in sorted(chunk_starts):
        starts = sorted(chunk_starts[traj_id])
        T = traj_len[traj_id]
        preds, gts = [], []
        prompt = None
        for i in range(0, len(starts), args.batch):
            batch = starts[i:i + args.batch]
            examples = []
            for _, pos in batch:
                s = inner[pos]
                ex = {"image": s["image"], "lang": s["lang"]}
                if "state" in s:
                    ex["state"] = np.asarray(s["state"], dtype=np.float32)
                if "state_history" in s:  # P0+History models ([STATE_HIST] prompt)
                    ex["state_history"] = np.asarray(s["state_history"], dtype=np.float32)
                examples.append(ex)
                gts.append(np.asarray(s["action"], dtype=np.float32))  # (AH, 78) identity raw
                prompt = s["lang"]
            out = framework.predict_action(examples=examples)
            pred = np.asarray(out["normalized_actions"], dtype=np.float32)  # (B, AH, 78) identity raw
            preds.extend(pred)
        pred = np.concatenate(preds, axis=0)[:T]   # (T, 78)
        gt = np.concatenate(gts, axis=0)[:T]       # (T, 78)

        tok_p, tok_g = pred[:, :64], gt[:, :64]
        snapped = np.round(tok_p * GRID) / GRID
        m = {
            "T": int(T),
            "mse78": float(np.mean((pred - gt) ** 2)),
            "mse64": float(np.mean((tok_p - tok_g) ** 2)),
            "mse64_snapped": float(np.mean((snapped - tok_g) ** 2)),
            "bin_acc": float(np.mean(np.round(tok_p * GRID) == np.round(tok_g * GRID))),
            "offgrid_mean": float(np.mean(np.abs(tok_p * GRID - np.round(tok_p * GRID)) / GRID)),
        }
        key = PROMPT_TO_KEY.get(prompt, f"traj{traj_id}")
        metrics[key] = m
        np.savez_compressed(os.path.join(args.out, f"{key}.npz"),
                            motion_token=tok_p.astype(np.float32),
                            motion_key=key, prompt=str(prompt))
        print(f"[dump] traj {traj_id} ({key}): T={T} mse78={m['mse78']:.5f} mse64={m['mse64']:.5f} "
              f"snapped={m['mse64_snapped']:.5f} bin_acc={m['bin_acc']:.3f}")

    avg = {k: float(np.mean([m[k] for m in metrics.values()]))
           for k in ("mse78", "mse64", "mse64_snapped", "bin_acc", "offgrid_mean")}
    metrics["_avg"] = avg
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[dump] AVG mse78={avg['mse78']:.5f} mse64={avg['mse64']:.5f} "
          f"snapped={avg['mse64_snapped']:.5f} bin_acc={avg['bin_acc']:.3f} "
          f"offgrid={avg['offgrid_mean']:.5f}")
    print(f"[dump] done -> {args.out} (GR00T flow3 anchor: mse avg 0.00257, all windows <= 0.004)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
