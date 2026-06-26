#!/usr/bin/env python3
"""P0 prompt-selectivity + state-substitution probe (offline, no Isaac/WBC).

Codex review's #1 missing experiment: from a SINGLE frozen observation, query all 7 training
prompts and measure whether the predicted token chunk actually changes with the prompt, and
whether each prompt's chunk best-matches its OWN motion's GT chunk. Isolates the policy from
WBC/rollout/FSQ. Answers: is the freeze a prompt-conditioning collapse (state-dominant mapping),
a state-OOD problem, or downstream?

Run in the Isaac-GR00T venv:
    cd dependencies/Isaac-GR00T
    COMPILE_ACTION_HEAD_DISABLE=1 .venv/bin/python ../../scripts/sonic_prompt_selectivity.py \
        --model_path ../../outputs/gr00t_sonic_8k/checkpoint-8000 \
        --dataset_path ../../datasets/sonic_vla_lerobot
"""
from __future__ import annotations
import argparse, logging
from copy import deepcopy
import numpy as np

# dataset episode order -> (key, trained prompt)   (mirrors gr00t_build_sequence.PROMPT)
TRAJ = [
    ("dance", "dance"), ("lunge", "forward lunge"), ("macarena", "macarena"),
    ("kick", "kick"), ("squat", "squat"), ("jump", "jump on one leg"),
    ("walk", "walk and turn around"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--action_horizon", type=int, default=40)
    ap.add_argument("--obs_traj", type=int, default=3, help="which traj's frame-0 obs to hold fixed (3=kick)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    emb = EmbodimentTag.resolve("unitree_g1_sonic")
    policy = Gr00tPolicy(embodiment_tag=emb, model_path=args.model_path,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    loader = LeRobotEpisodeLoader(args.dataset_path, modality_configs=policy.get_modality_config(),
                                  video_backend="torchcodec", video_backend_kwargs=None)
    mod = loader.modality_configs
    mod_no_action = deepcopy(mod); mod_no_action.pop("action")
    lang_key = mod["language"].modality_keys[0]
    AH = args.action_horizon

    def query(traj_id, prompt_override):
        traj = loader[traj_id]
        dp = extract_step_data(traj, 0, mod_no_action, emb)
        obs = {}
        for k, v in dp.states.items(): obs[f"state.{k}"] = v
        for k, v in dp.images.items(): obs[f"video.{k}"] = np.array(v)
        obs[lang_key] = prompt_override   # parse_observation_gr00t wraps a str as [[str]]
        parsed = parse_observation_gr00t(obs, mod)
        chunk, _ = policy.get_action(parsed)
        mt = np.asarray(parse_action_gr00t(chunk)["action.motion_token"]).reshape(AH, -1)
        return mt

    def gt_chunk(traj_id):
        traj = loader[traj_id]
        col = np.vstack([a for a in traj["action.motion_token"]])[:AH]
        return col

    # GT chunk per motion (frame-0 target) + its intra-drift (dynamic vs hold)
    gts = {TRAJ[t][0]: gt_chunk(t) for t in range(7)}
    print("=== GT frame-0 chunk intra-step drift (dynamic motion if high) ===")
    for k, g in gts.items():
        print(f"  {k:9s} intra-drift={np.linalg.norm(np.diff(g,axis=0),axis=1).mean():.3f}  std={g.std():.3f}")

    # Hold the obs of args.obs_traj frame-0; sweep all 7 prompts.
    obs_name = TRAJ[args.obs_traj][0]
    print(f"\n=== prompt-selectivity: FIXED obs = {obs_name} frame-0, sweep 7 prompts ===")
    preds = {}
    for t in range(7):
        key, prm = TRAJ[t]
        mt = query(args.obs_traj, prm)
        preds[key] = mt
        intra = np.linalg.norm(np.diff(mt, axis=0), axis=1).mean()
        # which GT motion is this pred closest to?
        dists = {k: float(np.mean((mt - g) ** 2)) for k, g in gts.items()}
        best = min(dists, key=dists.get)
        print(f"  prompt={prm!r:22s} -> intra-drift={intra:.3f}  closest-GT={best:8s}  "
              f"MSE-vs-own({key})={dists[key]:.4f}  MSE-vs-kick={dists['kick']:.4f}")

    # pairwise: how different are the chunks across prompts? (low => prompt ignored)
    keys = list(preds)
    P = np.stack([preds[k].reshape(-1) for k in keys])
    pd = np.sqrt(((P[:, None] - P[None]) ** 2).mean(-1))
    offdiag = pd[~np.eye(len(keys), dtype=bool)]
    print(f"\n  pairwise RMS between prompts: mean={offdiag.mean():.4f} max={offdiag.max():.4f} "
          f"(near 0 => prompt IGNORED / collapsed)")
    print(f"  for reference, GT chunks pairwise RMS: ", end="")
    G = np.stack([gts[k].reshape(-1) for k in keys])
    gd = np.sqrt(((G[:, None] - G[None]) ** 2).mean(-1))
    print(f"mean={gd[~np.eye(len(keys),dtype=bool)].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
