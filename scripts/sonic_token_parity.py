#!/usr/bin/env python3
"""SONIC token-pipeline parity check (P3 encoder parity gate).

Compares freshly-recorded motion tokens (VlaTokenRecorder npz) against a reference
token stream (the flow3 LeRobot parquets, or any prior npz) frame-by-frame on the
1/16 FSQ grid. A 100% bit-exact match proves the tokenize pipeline is reproducible
and the WBC ckpt is the exact version that produced the reference — the hard gate
that makes tokenizing NEW same-domain motion (LAFAN windows) trustworthy.

The recorder taps the actor's emitted token, whose encoder input is the REFERENCE
motion's future-frame kinematics (command_multi_future_*), not the achieved robot
state — so the forward pass is deterministic and parity is expected to be exact, not
merely >99%. Anything below 100% means a real drift (ckpt version / joint ordering /
config), not sim noise.

  # one segment vs its flow3 parquet (episode index)
  python scripts/sonic_token_parity.py --npz /tmp/sonic_parity_raw/run_circle/episode_000.npz \
      --parquet datasets/sonic_vla_lerobot_flow3 --episode 5
  # npz vs npz
  python scripts/sonic_token_parity.py --npz new.npz --ref-npz old.npz
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

GRID = 16.0  # token value = k / 16, k in [-16, 16]


def _load_npz_tokens(path: str) -> np.ndarray:
    return np.load(path)["motion_token"].astype(np.float64)


def _load_parquet_tokens(root: str, episode: int) -> np.ndarray:
    import pandas as pd

    files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquets under {root}")
    df = pd.read_parquet(files[episode])
    return np.stack(df["action.motion_token"].values).astype(np.float64)


def compare(rec: np.ndarray, ref: np.ndarray) -> dict:
    T = min(len(rec), len(ref))
    a, b = rec[:T], ref[:T]
    ka, kb = np.rint(a * GRID).astype(int), np.rint(b * GRID).astype(int)
    return {
        "rec_len": len(rec),
        "ref_len": len(ref),
        "compare_T": T,
        "per_dim_exact": float((ka == kb).mean()),
        "per_frame_exact": float((ka == kb).all(axis=1).mean()),
        "max_abs_diff": float(np.abs(a - b).max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="freshly-recorded VlaTokenRecorder npz")
    ap.add_argument("--parquet", help="flow3 LeRobot dataset root (use with --episode)")
    ap.add_argument("--episode", type=int, help="episode index in the parquet dataset")
    ap.add_argument("--ref-npz", help="reference npz instead of parquet")
    ap.add_argument("--pass-threshold", type=float, default=1.0,
                    help="per-dim exact-match fraction required to PASS (default 1.0 = bit-exact)")
    args = ap.parse_args()

    rec = _load_npz_tokens(args.npz)
    if args.ref_npz:
        ref = _load_npz_tokens(args.ref_npz)
    elif args.parquet is not None and args.episode is not None:
        ref = _load_parquet_tokens(args.parquet, args.episode)
    else:
        ap.error("provide either --ref-npz or (--parquet and --episode)")

    r = compare(rec, ref)
    ok = r["per_dim_exact"] >= args.pass_threshold and r["rec_len"] == r["ref_len"]
    print(f"rec={r['rec_len']} ref={r['ref_len']} (T={r['compare_T']})")
    print(f"per-dim exact bin match: {r['per_dim_exact']*100:.3f}%")
    print(f"per-frame all-64 match:  {r['per_frame_exact']*100:.3f}%")
    print(f"max abs token diff:      {r['max_abs_diff']:.4f}  (1 bin = {1/GRID:.4f})")
    print("PARITY PASS" if ok else "PARITY FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
