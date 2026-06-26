#!/usr/bin/env python3
"""Build a RESAMPLED + STANDING-STITCHED UNITREE_G1_SONIC LeRobot dataset for cold-standing init.

Root cause (.memory/sonic-closeloop-freeze-rootcause.md): from the dead-frozen standing idle the
single-frame policy emits a hold token — most standing frames teach "standing->hold", and kick/jump
clips have NO standing head at all (no "standing->begin-motion" example). The earlier resample fixed
WALK (it has a natural standing head we trimmed) but not kick/jump, and even walk fails from the
DEAD-frozen idle (the deepest zero-velocity basin).

This v2 implements the user's plan — harvest macarena's natural standing front (real mocap stand-sway,
~3s, the longest clean standing in the corpus) and use it three ways:
  1. STAND motion: macarena[:STAND_LEN] as its own episode with prompt "stand" → a REAL standing
     motion the model can produce (natural micro-motion, not a dead-frozen injected token). The idle
     can then use the "stand" prompt through the model, avoiding the deepest freeze basin.
  2. kick/jump PREFIX: prepend macarena[:PREFIX_LEN] standing to the kick/jump clips → gives them the
     missing "standing -> motion onset" mapping with valid 40-horizons (proven to work for walk).
     Same standing distribution as the STAND motion, so "stand -> kick" is a natural learned seam.
  3. TRIM the wasted standing fronts (macarena ~3s, walk ~2s) so onsets dominate + no 6s dead wait.
Episode-level only (action.motion_token uses delta_indices=range(40) -> per-frame dup corrupts the
chunk target). No tiny clips (lerobot/ffmpeg hangs after many).

Run in the data-collection venv (cd dependencies/GR00T-WholeBodyControl, PYTHONPATH=$PWD):
    PYTHONPATH=$PWD .venv_data_collection/bin/python ../../scripts/sonic_vla_resample_build.py \
        --raw-dir ../../datasets/sonic_vla_raw --out ../../datasets/sonic_vla_lerobot_resampled2
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np

DIR = {"dance": "dance_in_da_party_001__A464", "lunge": "forward_lunge_R_001__A359_M",
       "macarena": "macarena_001__A545", "kick": "neutral_kick_R_001__A543",
       "squat": "squat_001__A359", "jump": "tired_one_leg_jumping_R_001__A359",
       "walk": "walking_quip_360_R_002__A428"}
STAND_LEN = 150     # macarena front -> "stand" motion (~3s natural standing micro-motion)
# Tier-1 (resampled3): ONSET-AT-STEP-0. v2's prefix=25 created an "unobservable countdown" — nearly
# identical standing obs mapped to chunks whose onset sat 15-25 steps in the FUTURE, so under
# receding-horizon execution + state-dominance the model predicts an all-hold chunk and never starts.
# PREFIX_LEN=6 puts the motion onset at chunk-steps 1..6 for the prefix frames (obs≈stand -> MOVE NOW);
# the counterfactual "stand->hold" comes from the STAND episodes. walk now gets the same treatment.
PREFIX_LEN = 6      # macarena standing prepended to fragile motions. MUST be < action_horizon(40).
FRONT_TRIM = {"macarena": 150, "walk": 100, "jump": 19}   # cut these clips' standing/weak fronts
PREFIX = {"kick", "jump", "walk"}                          # get a short macarena-standing onset prefix
REPEAT = {"stand": 3, "kick": 6, "jump": 4, "walk": 4, "dance": 1, "lunge": 1, "squat": 1, "macarena": 1}

_DTYPE = {"float64": np.float64, "float32": np.float32, "int64": np.int64, "int32": np.int32, "bool": np.bool_}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true", default=True)
    args = ap.parse_args()

    from gear_sonic.data.exporter import Gr00tDataExporter
    from gear_sonic.data.features_sonic_vla import get_features_sonic_vla, get_modality_config_sonic_vla
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.utils.data_collection.transforms import compute_projected_gravity

    robot_model = instantiate_g1_robot_model()
    features = get_features_sonic_vla(robot_model)
    modality = get_modality_config_sonic_vla(robot_model)
    state_dim = int(np.prod(features["observation.state"]["shape"]))
    video_key = "observation.images.ego_view"

    def load(label):
        d = np.load(os.path.join(args.raw_dir, DIR[label], "episode_000.npz"), allow_pickle=True)
        return (d["motion_token"].astype(np.float64), d["joint_pos"].astype(np.float64),
                d["root_quat"].astype(np.float64), d.get("ego_rgb", None), str(d["prompt"]))

    mac_tok, mac_j, mac_q, mac_rgb, _ = load("macarena")
    def mac_front(n):  # macarena standing front, all fields
        rgb = mac_rgb[:n] if mac_rgb is not None else None
        return mac_tok[:n], mac_j[:n], mac_q[:n], rgb

    # build the episode list: (label, prompt, token, joints, quat, rgb)
    episodes = []
    # 1) STAND motion from macarena front
    st, sj, sq, srgb = mac_front(STAND_LEN)
    for _ in range(REPEAT["stand"]):
        episodes.append(("stand", "stand", st, sj, sq, srgb))
    # 2/3) per motion: front-trim and/or macarena-standing prefix
    pf_tok, pf_j, pf_q, pf_rgb = mac_front(PREFIX_LEN)
    for label in ["dance", "lunge", "macarena", "kick", "squat", "jump", "walk"]:
        tok, j, q, rgb, prompt = load(label)
        a = FRONT_TRIM.get(label, 0)
        tok, j, q = tok[a:], j[a:], (q[a:])
        rgb = rgb[a:] if rgb is not None else None
        if label in PREFIX:                       # prepend macarena standing -> teaches standing->onset
            tok = np.concatenate([pf_tok, tok], 0); j = np.concatenate([pf_j, j], 0)
            q = np.concatenate([pf_q, q], 0)
            rgb = np.concatenate([pf_rgb, rgb], 0) if (rgb is not None and pf_rgb is not None) else rgb
        for _ in range(REPEAT.get(label, 1)):
            episodes.append((label, prompt, tok, j, q, rgb))

    exporter = Gr00tDataExporter.create(save_root=args.out, fps=args.fps, features=features,
                                        modality_config=modality, task=episodes[0][1],
                                        overwrite_existing=args.overwrite)

    def fit(v):
        v = np.asarray(v).reshape(-1)
        if v.shape[0] == state_dim: return v
        o = np.zeros(state_dim, v.dtype); o[:min(state_dim, v.shape[0])] = v[:min(state_dim, v.shape[0])]; return o

    nfr = 0
    plan = {}
    for label, prompt, tok, j, q, rgb in episodes:
        T = len(tok)
        for t in range(T):
            frame = {}
            for key, spec in features.items():
                if key == video_key:
                    frame[key] = rgb[t] if rgb is not None else np.zeros(tuple(spec["shape"]), np.uint8)
                elif key == "action.motion_token": frame[key] = tok[t]
                elif key in ("observation.state", "action.wbc"): frame[key] = fit(j[t])
                elif key == "observation.root_orientation": frame[key] = q[t]
                elif key == "observation.projected_gravity":
                    frame[key] = compute_projected_gravity(q[t]).astype(np.float64)
                else: frame[key] = np.zeros(tuple(spec["shape"]), _DTYPE.get(spec["dtype"], np.float32))
            frame["task"] = prompt
            exporter.add_frame(frame)
        exporter.save_episode()
        nfr += T; plan[label] = plan.get(label, 0) + T

    print("[resample2] per-motion frames:", {k: f"{v}({100*v/nfr:.0f}%)" for k, v in plan.items()})
    print(f"[resample2] DONE: {len(episodes)} episodes, {nfr} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
