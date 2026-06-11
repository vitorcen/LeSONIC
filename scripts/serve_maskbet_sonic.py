#!/usr/bin/env python3
"""ZMQ inference server for the MaskBeT SONIC token producer (Route B, from-scratch 25M).

Speaks the SAME gr00t PolicyServer wire as ``serve_starvla_sonic.py`` /
``gear_sonic/data/vla_live_injector.py`` (msgpack-numpy REQ/REP,
{"endpoint": "ping"|"get_action", "data": {"observation": ...}}), so the whole Isaac side
(gear_sonic_sequence.sh / live demo) works unchanged — just point GR00T_PORT at this server.

Differences from the StarVLA server (which wraps a 4B VLM):
  * MaskBeT is a 25M from-scratch masked transformer (pure torch state_dict), loaded here.
  * Conditioning is prompt-id + K-frame proprio history, NOT image+text → so the language
    string is mapped to a task_index via meta/tasks.jsonl (the 8 LAFAN flow3 prompts).
  * Decode default = iterative argmax (constructively on the k/16 grid → snap is a no-op).
    SONIC_MASKBET_DECODE=expected uses the conditional-mean readout (lower open-loop MSE
    but OFF-grid) and snaps back to the grid for the wire.

Reuses ``MinMax`` from serve_starvla_sonic (carries the state-permutation fix: scatter each
state group back to its raw column BEFORE normalizing, else right_arm proprio freezes).

Run in any env with zmq + torch (e.g. starvla_eval_qwen35; the base env's zmq is broken):
    .../bin/python scripts/serve_maskbet_sonic.py \
        --ckpt MaskBeT/outputs/flow3/ckpt_006000.pt --port 5557
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MASKBET_DIR = os.environ.get("MASKBET_DIR", os.path.join(REPO_ROOT, "MaskBeT"))

# reuse the MinMax normalizer (+ state-permutation fix) and the state group layout
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from serve_starvla_sonic import MinMax, _STATE_GROUPS  # noqa: E402


class SonicMaskBeTPolicy:
    def __init__(self, ckpt: str, stats_json: str, tasks_jsonl: str,
                 decode: str = "argmax", temp: float = 1.0):
        sys.path.insert(0, MASKBET_DIR)
        import torch
        from maskbet.config import MaskBeTConfig
        from maskbet.model import MaskBeT
        self.torch = torch
        self.cfg = MaskBeTConfig()
        self.model = MaskBeT(self.cfg).cuda().eval()
        blob = torch.load(os.path.abspath(ckpt), map_location="cuda")
        self.model.load_state_dict(blob["model"])
        self.norm = MinMax(stats_json)
        self.decode = decode
        self.temp = temp
        self.K = self.cfg.state_history
        self._state_buf: list = []

        # prompt string -> task_index (the 8 LAFAN flow3 prompts)
        self.prompt2id = {}
        for line in open(tasks_jsonl):
            o = json.loads(line)
            self.prompt2id[o["task"].strip().lower()] = int(o["task_index"])
        torch.cuda.empty_cache()
        print(f"[serve] MaskBeT up: {os.path.basename(ckpt)} "
              f"({torch.cuda.memory_allocated()/1e9:.2f}G, decode={decode}, K={self.K}, "
              f"step={blob.get('step','?')}, prompts={len(self.prompt2id)})", flush=True)

    def _prompt_id(self, prompt: str) -> int:
        p = str(prompt).strip().lower()
        if p in self.prompt2id:
            return self.prompt2id[p]
        match = difflib.get_close_matches(p, self.prompt2id, n=1, cutoff=0.0)
        return self.prompt2id[match[0]] if match else 0

    def get_action(self, observation: dict, options=None):
        torch = self.torch
        st = observation["state"]
        state43 = np.concatenate([np.asarray(st[g]).reshape(-1) for g in _STATE_GROUPS])
        pg3 = np.asarray(st["projected_gravity"]).reshape(-1)[:3]
        state46 = self.norm(state43, pg3)                       # (46,) like training

        self._state_buf.append(state46.copy())
        if len(self._state_buf) > self.K:
            self._state_buf = self._state_buf[-self.K:]
        while len(self._state_buf) < self.K:                    # pad at stream start
            self._state_buf.insert(0, self._state_buf[0].copy())
        hist = np.stack(self._state_buf[-self.K:])              # (K, 46)

        lang = observation["language"]["annotation.human.task_description"]
        prompt = lang[0][0] if isinstance(lang, (list, tuple)) else str(lang)
        pid = self._prompt_id(prompt)

        cond = {
            "prompt": torch.tensor([pid], dtype=torch.long, device="cuda"),
            "state": torch.from_numpy(hist[None].astype(np.float32)).cuda(),  # (1,K,46)
        }
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            act = self.model.predict(cond, mode=self.decode, temp=self.temp)  # (1,40,78)
        act = act.float().cpu().numpy()
        if self.decode == "expected":                          # off-grid -> snap to k/16
            act = np.round(act * self.cfg.grid) / self.cfg.grid
        return {"action.motion_token": act[:, :, :64].astype(np.float32)}     # (1,40,64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", default=os.path.join(
        REPO_ROOT, "datasets/sonic_vla_lerobot_flow3/meta/stats.json"))
    ap.add_argument("--tasks", default=os.path.join(
        REPO_ROOT, "datasets/sonic_vla_lerobot_flow3/meta/tasks.jsonl"))
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--decode", default=os.environ.get("SONIC_MASKBET_DECODE", "argmax"),
                    choices=["argmax", "sample", "expected"])
    ap.add_argument("--temp", type=float, default=float(os.environ.get("SONIC_MASKBET_TEMP", 1.0)))
    args = ap.parse_args()

    import msgpack_numpy as mnp
    import zmq

    policy = SonicMaskBeTPolicy(os.path.abspath(args.ckpt), os.path.abspath(args.stats),
                                os.path.abspath(args.tasks), decode=args.decode, temp=args.temp)

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
                sock.send(mnp.packb(policy.get_action(**(req.get("data") or {}))))
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
