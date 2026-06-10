#!/usr/bin/env python3
"""Fast, no-GPU preview of a segmented motion dataset: render a CONTACT SHEET of per-segment
"fingerprints" so you can eyeball which cuts are turns / forward-runs / idle / in-place punches
WITHOUT booting Isaac per segment.

Each card shows, for one segment:
  - top-down root path (start = green dot, end = red dot) -> translation & where it goes
  - heading (yaw) vs time                                 -> turns (ramps) vs straight (flat)
  - title with duration, horiz travel (m), yaw change (deg), arm-motion std (rough)

A "clean turn-in-place" = tiny path + big yaw ramp. "Forward run" = long straight path + flat yaw.
"In-place punch" = tiny path + flat yaw + high arm motion. Multi-action = path bends + yaw jumps.

Usage (any env with numpy+matplotlib+scipy):
  python scripts/preview_segments_fingerprint.py \
    --pkl dependencies/GR00T-WholeBodyControl/data/fight_segments_robot.pkl \
    --out /tmp/fight_fingerprints.png
"""
import argparse
import math

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--fps", type=int, default=30, help="source fps of the stored frames")
    a = ap.parse_args()

    d = joblib.load(a.pkl)
    keys = sorted(d.keys())
    n = len(keys)
    cols = a.cols
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4), facecolor="white")
    axes = np.atleast_2d(axes)

    for i, k in enumerate(keys):
        e = d[k]
        r, c = divmod(i, cols)
        ax = axes[r][c]
        xy = e["root_trans_offset"][:, :2]
        yaw = np.degrees(R.from_quat(e["root_rot"]).as_euler("xyz")[:, 2])
        yaw = np.unwrap(np.radians(yaw))
        yaw = np.degrees(yaw)
        dur = len(xy) / a.fps
        trav = float(np.linalg.norm(xy[-1] - xy[0]))
        dyaw = float(yaw[-1] - yaw[0])
        arm = float(np.mean(np.std(e["dof"][:, 15:], axis=0)))
        # path (centered)
        p = xy - xy[0]
        ax.plot(p[:, 0], p[:, 1], "-", color="#2563eb", lw=1)
        ax.plot(0, 0, "o", color="#16a34a", ms=4)
        ax.plot(p[-1, 0], p[-1, 1], "o", color="#dc2626", ms=4)
        ax.set_aspect("equal")
        m = max(0.3, np.abs(p).max() * 1.1)
        ax.set_xlim(-m, m); ax.set_ylim(-m, m)
        ax.set_xticks([]); ax.set_yticks([])
        # yaw inset (twin) as text since space is tight
        short = k.split("_")[-1]
        ax.set_title(f"{short}\n{dur:.1f}s tr{trav:.1f} yaw{dyaw:+.0f}\narm{arm:.2f}", fontsize=7)
        ax.set_facecolor("white")
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    plt.savefig(a.out, dpi=110, facecolor="white")
    print(f"[fingerprint] {n} segments -> {a.out}")
    print("  legend: green=start red=end | tr=horiz travel(m) yaw=heading change(deg) arm=arm-joint motion")
    print("  turn-in-place: tiny path + big |yaw|  |  forward run: long straight path + small |yaw|")
    print("  in-place punch: tiny path + small |yaw| + high arm  |  multi-action: bent path / big yaw")


if __name__ == "__main__":
    main()
