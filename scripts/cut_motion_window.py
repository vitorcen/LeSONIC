#!/usr/bin/env python3
"""Cut a [start_s, end_s] time window out of one robot_filtered motion key -> a 1-key pkl.

Used to hand-pick a clean sub-action from a full clip by seconds (instead of trusting the
auto velocity-valley cut). Frame index = second * stored_fps.

  python scripts/cut_motion_window.py --input data/fight_full_robot.pkl --key fight_full \
      --start_s 3.0 --end_s 6.0 --output /tmp/window_robot.pkl --out_key window
"""
import argparse
import joblib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--start_s", type=float, required=True)
    ap.add_argument("--end_s", type=float, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--out_key", default="window")
    a = ap.parse_args()

    d = joblib.load(a.input)
    e = d[a.key]
    fps = int(e["fps"])
    n = len(e["dof"])
    i0 = max(0, int(round(a.start_s * fps)))
    i1 = min(n, int(round(a.end_s * fps)))
    if i1 <= i0:
        raise SystemExit(f"empty window: start={a.start_s}s end={a.end_s}s -> frames [{i0},{i1}) of {n}")

    out = {}
    for k, v in e.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[i0:i1]
        else:
            out[k] = v  # scalars like fps
    joblib.dump({a.out_key: out}, a.output)
    print(f"[cut] {a.key} [{a.start_s}s,{a.end_s}s] = frames [{i0},{i1}) ({(i1-i0)/fps:.2f}s) "
          f"-> {a.output} (key={a.out_key}, fps={fps})")


if __name__ == "__main__":
    main()
