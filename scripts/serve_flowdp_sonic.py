#!/usr/bin/env python3
"""ZMQ inference server for the FlowDP SONIC token producer (conv-UNet + flow head).

Speaks EXACTLY the gr00t PolicyServer wire that ``gear_sonic/data/vla_live_injector.py``
already talks (msgpack-numpy REQ/REP, {"endpoint": "ping"|"get_action",
"data": {"observation": ...}}), so the whole Isaac side (gear_sonic_sequence.sh /
live_demo @flow2 / LAFAN.ipynb live cells) runs unchanged — just point GR00T_PORT here.

FlowDP / Diffusion-Policy has NO language path. Motion selection rides in
observation.state: state(53) = joint(43, RAW column order) + projected_gravity(3) +
motion_onehot(7). The server maps the incoming prompt -> task_index (via the dataset's
meta/tasks.parquet) -> one-hot, so each flow2 segment's prompt picks the right motion.

Obs (raw values; normalization happens via the policy's own saved preprocessor):
    video.ego_view (1,1,480,640,3) u8 ; state.<group> each (1,1,d) f32 RAW ;
    language {"annotation.human.task_description": [[prompt]]}
Reply: {"action.motion_token": (1, n_action_steps, 64) f32}  (the 64-dim token slice
of the 78-dim action; left/right hand tail is trained but dropped here — the SONIC
decoder only consumes motion_token).

Run in the lerobot-v044 env (FlowHeads on PYTHONPATH):
    PYTHONPATH=dependencies/FlowHeads ~/miniconda3/envs/lerobot-v044/bin/python \
        scripts/serve_flowdp_sonic.py \
        --ckpt outputs/flowdp_sonic/checkpoints/015000/pretrained_model --port 5557
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# RAW observation.state column slices (from the BonesSeed meta/modality.json). The
# injector sends each group separately; we scatter them back into the 43-vec in the
# SAME order the FlowDP dataset was built (observation.state was copied verbatim from
# the source parquet, which is this raw order — NOT the permuted starvla key order).
_RAW_SLICES = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15),
               "left_arm": (15, 22), "left_hand": (22, 29),
               "right_arm": (29, 36), "right_hand": (36, 43)}
N_MOTIONS = 7

# prompt -> robot_filtered motion key (for SONIC_GT_DIR replay-capture mode)
_TASK2KEY = {"dance": "dance_in_da_party_001__A464",
             "do a forward lunge": "forward_lunge_R_001__A359_M",
             "dance the macarena": "macarena_001__A545",
             "kick": "neutral_kick_R_001__A543", "squat": "squat_001__A359",
             "jump on one leg": "tired_one_leg_jumping_R_001__A359",
             "walk and turn around": "walking_quip_360_R_002__A428"}


def load_prompt_to_index(tasks_path: str) -> dict:
    """prompt-string -> task_index. Handles the v3.0 `tasks.parquet` (task string is
    the row index, `task_index` is the column) and the legacy v2.1 `tasks.jsonl`."""
    if tasks_path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(tasks_path)
        return {str(k): int(v) for k, v in df["task_index"].items()}
    m = {}
    with open(tasks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                m[d["task"]] = int(d["task_index"])
    return m


class FlowDPSonicPolicy:
    def __init__(self, ckpt: str, tasks_path: str, flow_steps: int | None = None,
                 seed: int | None = None):
        import torch
        from flowdp.modeling_flowdp import FlowDPPolicy  # noqa: F401 registers "flowdp"
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.utils.constants import ACTION

        self.torch = torch
        self.ACTION = ACTION
        # Optional deterministic flow prior (reproducible chunks). FlowDP, like DP,
        # samples a fresh Gaussian prior per replan; without a seed each 32-frame chunk
        # is independently stochastic (possible boundary jitter in live chaining).
        self.gen = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
        self.pol = FlowDPPolicy.from_pretrained(ckpt).to("cuda").eval()
        if flow_steps is not None:                       # Euler-NFE override at serve
            self.pol.diffusion.num_inference_steps = int(flow_steps)
            self.pol.config.num_inference_steps = int(flow_steps)
        self.pre = PolicyProcessorPipeline.from_pretrained(
            ckpt, config_filename="policy_preprocessor.json")

        # action MIN_MAX stats straight from the model's own normalizer step, so the
        # un-normalization is byte-identical to training (no external stats file).
        amin = amax = None
        for step in self.pre.steps:
            stats = getattr(step, "stats", None) or getattr(step, "_stats", None)
            if isinstance(stats, dict) and ACTION in stats:
                s = stats[ACTION]
                amin = np.asarray(_to_np(s["min"]), dtype=np.float32)
                amax = np.asarray(_to_np(s["max"]), dtype=np.float32)
        if amin is None:
            raise RuntimeError("could not find ACTION min/max in the preprocessor stats")
        self.amin, self.amax = amin, amax

        # normalized (strip+lower) prompt -> index, so minor whitespace/case drift is tolerated
        self.p2i = {str(k).strip().lower(): v
                    for k, v in load_prompt_to_index(tasks_path).items()}
        # GT-replay capture mode (SONIC_GT_DIR): return recorded tokens (robot does the
        # motion correctly, no fall) while the obs capture logs the real live distribution
        # across ALL motions -> calibrate per-dim drift without watching it fall.
        self._gt_dir = os.environ.get("SONIC_GT_DIR")
        self._gt_cache, self._gt_ctr, self._gt_last = {}, 0, None
        self._H = self.pol.config.n_action_steps
        print(f"[serve] flowdp up: {ckpt} "
              f"({torch.cuda.memory_allocated()/1e9:.1f}G, flow_steps="
              f"{self.pol.diffusion.num_inference_steps}, motions={list(self.p2i)})",
              flush=True)

    def _onehot(self, prompt: str) -> np.ndarray:
        # Fail closed: the model NEVER saw a zero one-hot in training (every frame has
        # exactly one bit set), and prompt is the only motion-disambiguating signal —
        # serving a zero selector = undefined / motion-blind. Reject unknown prompts.
        idx = self.p2i.get(str(prompt).strip().lower())
        if idx is None:
            raise ValueError(f"unknown prompt {prompt!r}; valid motions = {sorted(self.p2i)}")
        oh = np.zeros(N_MOTIONS, dtype=np.float32)
        oh[idx] = 1.0
        return oh

    def get_action(self, observation: dict, options=None):
        torch = self.torch
        # --- image: (480,640,3) u8 -> CHW float [0,1] (lerobot video-decode convention) ---
        ev = np.asarray(observation["video"]["ego_view"])
        rgb = ev.reshape(-1, *ev.shape[-3:])[0].astype(np.float32) / 255.0   # (480,640,3)
        img = np.transpose(rgb, (2, 0, 1))                                   # (3,480,640)

        # --- state(53): scatter groups -> raw43, + gravity3 + motion_onehot7 ---
        st = observation["state"]
        raw43 = np.zeros(43, dtype=np.float32)
        for g, (a, b) in _RAW_SLICES.items():
            raw43[a:b] = np.asarray(st[g], dtype=np.float32).reshape(-1)[: b - a]
        # CRITICAL: the recorded dataset's observation.projected_gravity is all-zeros
        # (gravity was not captured during the SONIC recording), so its MIN_MAX stats are
        # degenerate (min==max==0). lerobot's degenerate-dim handling maps min->-1 but blows
        # up ANY other value as 2*x/eps-1 (~1e8). At deploy the injector sends the REAL
        # gravity (z≈-1) -> normalized to ~-2e8 -> the model explodes (|chunk|~5, robot falls).
        # The model never learned from gravity (constant in training), so feed the training
        # constant (zeros) -> normalizes to the same -1 the model trained on. (Verified: this
        # alone takes 47/50 live frames from |chunk|~4.4 explode to 0/50, all |chunk|~1.07.)
        pg3 = np.zeros(3, dtype=np.float32)  # was: np.asarray(st["projected_gravity"])[:3]
        lang = observation["language"]["annotation.human.task_description"]
        prompt = lang[0][0] if isinstance(lang, (list, tuple)) else str(lang)
        oh = self._onehot(str(prompt))
        state53 = np.concatenate([raw43, pg3, oh]).astype(np.float32)

        # Live-drift capture (SONIC_CAPTURE_LOG=path): append the raw state the deploy
        # obs actually carries, so we can measure the recorded-vs-live distribution shift
        # offline and calibrate the proprio aug to the REAL drift (not a guessed sigma).
        cap = os.environ.get("SONIC_CAPTURE_LOG")
        if cap:
            with open(cap, "a") as f:
                f.write(json.dumps({"prompt": str(prompt),
                                    "raw43": raw43.tolist(), "pg3": pg3.tolist()}) + "\n")

        if self._gt_dir:  # GT-replay: return recorded tokens, advance per-prompt cursor
            key = _TASK2KEY.get(str(prompt).strip().lower())
            if key not in self._gt_cache:
                self._gt_cache[key] = np.load(os.path.join(self._gt_dir, f"{key}.npz"))["motion_token"]
            if str(prompt) != self._gt_last:
                self._gt_ctr, self._gt_last = 0, str(prompt)
            gt = self._gt_cache[key]
            s = min(self._gt_ctr * self._H, max(0, len(gt) - self._H))
            chunk = gt[s:s + self._H][None].astype(np.float32)         # (1,H,64)
            self._gt_ctr += 1
            return {"action.motion_token": chunk}

        obs = {
            "observation.state": torch.from_numpy(state53),
            "observation.images.ego_view": torch.from_numpy(img),
        }
        self.pol.reset()
        batch = self.pre(obs)                                # normalize + batch + device
        noise = None
        if self.gen is not None:
            adim = self.pol.config.action_feature.shape[0]
            noise = torch.randn((1, self.pol.config.horizon, adim),
                                generator=self.gen, device="cuda")
        with torch.no_grad():
            chunk = self.pol.predict_action_chunk(batch, noise=noise)  # (1, n_action_steps, 78) norm
        chunk = chunk.detach().float().cpu().numpy()
        # Clamp to the valid MIN_MAX range. The flow ODE is unbounded, and on live Isaac
        # obs (single-frame, mildly OOD vs the recorded trajectory) the integrated chunk
        # can overshoot to ±4; un-normalized that would land far off the FSQ grid. The
        # token grid maps to exactly [-1,1] under MIN_MAX, so clamping there pins any
        # overshoot back onto valid grid values (cf. FlowHeads `action_clamp`).
        cmn, cmx = float(chunk.min()), float(chunk.max())
        if cmn < -1.5 or cmx > 1.5:
            print(f"[serve] flow overshoot [{cmn:.2f},{cmx:.2f}] -> clamp [-1,1]", flush=True)
        chunk = np.clip(chunk, -1.0, 1.0)
        # MIN_MAX un-normalize: (x+1)/2 * (max-min) + min  (eps for degenerate dims, like lerobot)
        denom = np.where(self.amax != self.amin, self.amax - self.amin, 1e-8)
        act = (chunk + 1.0) / 2.0 * denom + self.amin        # (1, A, 78)
        tok = act[:, :, :64].astype(np.float32)
        # Snap to the FSQ lattice (levels=16 -> k/16 in [-0.5625, 0.5]); the SONIC decoder
        # was trained on-grid, so emit exact grid values and log the off-grid magnitude as
        # a closed-loop health signal (continuous flow output drifting off-grid = trouble).
        if os.environ.get("SONIC_SNAP", "1") == "1":
            snapped = np.clip(np.round(tok * 16.0) / 16.0, -0.5625, 0.5)
            self._offgrid = float(np.abs(snapped - np.clip(tok, -0.5625, 0.5)).mean())
            tok = snapped.astype(np.float32)
        return {"action.motion_token": tok}


def _to_np(x):
    import numpy as _np
    try:
        return x.detach().cpu().numpy() if hasattr(x, "detach") else _np.asarray(x)
    except Exception:
        return _np.asarray(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to .../checkpoints/<step>/pretrained_model")
    ap.add_argument("--tasks", default=os.path.join(
        REPO_ROOT, "datasets/sonic_vla_flowdp/meta/tasks.parquet"))
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--flow-steps", type=int, default=None,
                    help="override Euler NFE at serve (sweep 1/2/4/8); default = trained value")
    ap.add_argument("--seed", type=int, default=None,
                    help="deterministic flow prior for reproducible chunks (default: stochastic)")
    args = ap.parse_args()

    import msgpack_numpy as mnp
    import zmq

    policy = FlowDPSonicPolicy(os.path.abspath(args.ckpt), os.path.abspath(args.tasks),
                               flow_steps=args.flow_steps, seed=args.seed)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"SERVE_READY tcp://0.0.0.0:{args.port}", flush=True)
    while True:
        msg = sock.recv()
        try:
            req = mnp.unpackb(msg, raw=False)
            ep = req.get("endpoint")
            if ep == "ping":
                sock.send(mnp.packb({"ok": True}))
            elif ep == "get_action":
                data = req.get("data") or {}
                action = policy.get_action(**data)
                sock.send(mnp.packb(action))
            elif ep == "kill":
                sock.send(mnp.packb({"ok": True}))
                break
            else:
                sock.send(mnp.packb({"error": f"unknown endpoint {ep!r}"}))
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            try:
                sock.send(mnp.packb({"error": tb}))
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
