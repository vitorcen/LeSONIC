#!/usr/bin/env python3
"""ZMQ inference server for the StarVLA SONIC token producer (QwenPI_CE / PI_v3).

Speaks EXACTLY the gr00t PolicyServer wire protocol that
``gear_sonic/data/vla_live_injector.py`` already talks (msgpack-numpy REQ/REP,
{"endpoint": "ping"|"get_action", "data": {"observation": ...}}), so the whole
Isaac side (gear_sonic_sequence.sh / live_demo flows / LAFAN.ipynb live cells)
works unchanged — just point GR00T_PORT at this server.

Obs (raw values, normalization happens HERE — mirrors the training transform):
    video.ego_view (1,1,480,640,3) u8 ; state.<group> each (1,1,d) f32 RAW;
    language {"annotation.human.task_description": [[prompt]]}
Reply: {"action.motion_token": (1, action_horizon, 64) f32}  — exact grid values
(QwenPI_CE argmax decodes onto k/16; snap-to-grid in the injector is a no-op).

Run in the starvla_eval_qwen35 env:
    ~/miniconda3/envs/starvla_eval_qwen35/bin/python scripts/serve_starvla_sonic.py \
        --ckpt outputs/starvla/sonic_qwen3_5_4b_ce/checkpoints/steps_6000_pytorch_model.pt \
        --port 5556
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STARVLA_DIR = os.environ.get("STARVLA_DIR",
                             os.path.join(REPO_ROOT, "dependencies", "starVLA"))
# QwenPI_CE head ships in the vitorcen/StarVLA fork checkout (no runtime deploy).

# training state key order (data_config.UnitreeG1SonicConfig.state_keys)
_STATE_GROUPS = ["left_leg", "right_leg", "waist", "left_arm", "right_arm",
                 "left_hand", "right_hand"]  # slices of observation.state(43)
# group -> (start, end) inside the 43-vec, from meta/modality.json
_STATE_SLICES = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15),
                 "left_arm": (15, 22), "left_hand": (22, 29),
                 "right_arm": (29, 36), "right_hand": (36, 43)}


class MinMax:
    """min_max normalization identical to the training StateActionTransform."""

    def __init__(self, stats_json: str):
        s = json.load(open(stats_json))
        self.smin = np.asarray(s["observation.state"]["min"], dtype=np.float32)
        self.smax = np.asarray(s["observation.state"]["max"], dtype=np.float32)
        self.pmin = np.asarray(s["observation.projected_gravity"]["min"], dtype=np.float32)
        self.pmax = np.asarray(s["observation.projected_gravity"]["max"], dtype=np.float32)

    @staticmethod
    def _norm(x, mn, mx):
        out = np.zeros_like(x, dtype=np.float32)
        mask = mn != mx
        out[mask] = 2.0 * (x[mask] - mn[mask]) / (mx[mask] - mn[mask]) - 1.0
        return out

    def __call__(self, state43: np.ndarray, pg3: np.ndarray) -> np.ndarray:
        # state43 arrives concatenated in _STATE_GROUPS order, but smin/smax are
        # indexed by RAW observation.state column and _STATE_SLICES is a PERMUTATION
        # (left_hand=22:29 sits before right_arm=29:36 in raw, the reverse of the
        # _STATE_GROUPS order). Scatter each group back to its raw column slice
        # FIRST, normalize the raw 43-vec, then re-slice in training key order.
        # The old concat-then-normalize double-permuted: it fed right_arm through
        # left_hand's stats and pushed left_hand through right_arm's degenerate
        # (all-zero) stats -> the right_arm proprio channel came out constant 0.
        raw = np.zeros(43, dtype=np.float32)
        ofs = 0
        for g in _STATE_GROUPS:
            a, b = _STATE_SLICES[g]
            raw[a:b] = state43[ofs:ofs + (b - a)]
            ofs += b - a
        n43 = self._norm(raw, self.smin, self.smax)
        npg = self._norm(pg3.astype(np.float32), self.pmin, self.pmax)
        groups = [n43[a:b] for g, (a, b) in
                  ((g, _STATE_SLICES[g]) for g in _STATE_GROUPS)]
        return np.concatenate(groups + [npg]).astype(np.float32)  # (46,)


class SonicStarVLAPolicy:
    def __init__(self, ckpt: str, stats_json: str, img_size: int = 448,
                 proprio_history: int = 0):
        from PIL import Image  # noqa: F401 (validated import)
        sys.path.insert(0, STARVLA_DIR)
        cwd = os.getcwd()
        os.chdir(STARVLA_DIR)  # framework code resolves repo-relative assets
        import torch
        from starVLA.model.framework.base_framework import baseframework
        self.torch = torch
        fw = baseframework.from_pretrained(os.path.abspath(ckpt))
        self.fw = fw.to(torch.bfloat16).cuda().eval()
        os.chdir(cwd)
        torch.cuda.empty_cache()
        self.norm = MinMax(stats_json)
        self.img_size = img_size
        self.proprio_history = proprio_history
        self._state_buf: list = []  # rolling buffer for proprio history
        print(f"[serve] model up: {os.path.basename(ckpt)} "
              f"({torch.cuda.memory_allocated()/1e9:.1f}G, "
              f"proprio_history={proprio_history})", flush=True)

    def get_action(self, observation: dict, options=None):
        from PIL import Image
        rgb = np.asarray(observation["video"]["ego_view"]).reshape(-1, *np.asarray(
            observation["video"]["ego_view"]).shape[-3:])[0].astype(np.uint8)
        img = Image.fromarray(rgb).resize((self.img_size, self.img_size))  # = _pack_sample(448)

        st = observation["state"]
        state43 = np.concatenate([np.asarray(st[g]).reshape(-1) for g in _STATE_GROUPS])
        pg3 = np.asarray(st["projected_gravity"]).reshape(-1)[:3]
        state46 = self.norm(state43, pg3)  # (46,) like training sample["state"]

        # Maintain rolling buffer for proprio history
        self._state_buf.append(state46.copy())
        if len(self._state_buf) > self.proprio_history:
            self._state_buf = self._state_buf[-self.proprio_history:]

        lang = observation["language"]["annotation.human.task_description"]
        prompt = lang[0][0] if isinstance(lang, (list, tuple)) else str(lang)

        ex = {"image": [img], "lang": str(prompt),
              "state": state46[None].astype(np.float32)}
        # Build state_history if model expects it (numpy, like training)
        if self.proprio_history > 0 and len(self._state_buf) >= self.proprio_history:
            hist = np.stack(self._state_buf[-self.proprio_history:])  # (K, 46)
            ex["state_history"] = hist[None].astype(np.float32)  # (1,K,46)
        out = self.fw.predict_action(examples=[ex])
        act = np.asarray(out["normalized_actions"], dtype=np.float32)  # (1,horizon,78)
        return {"action.motion_token": act[:, :, :64]}                 # (1,horizon,64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", default=os.path.join(
        REPO_ROOT, "datasets/sonic_vla_lerobot_flow3/meta/stats.json"))
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--proprio-history", type=int, default=0,
                    help="Number of past frames for proprio history (K)")
    args = ap.parse_args()

    import msgpack_numpy as mnp
    import zmq

    policy = SonicStarVLAPolicy(os.path.abspath(args.ckpt), os.path.abspath(args.stats),
                                proprio_history=args.proprio_history)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"SERVE_READY tcp://0.0.0.0:{args.port}", flush=True)  # print, not logging
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
