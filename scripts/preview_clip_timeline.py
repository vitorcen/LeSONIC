#!/usr/bin/env python3
"""Timeline fingerprint of a FULL clip (x-axis = seconds) to help hand-pick [start,end] windows.
Plots, vs time: horizontal speed (m/s), cumulative heading change (deg), arm-joint motion.
Turns show as heading ramps; runs as speed humps; strikes/dance as arm-motion spikes.

  python scripts/preview_clip_timeline.py --pkl data/fight_full_robot.pkl --key fight_full \
      --out /tmp/fight_timeline.png
"""
import argparse
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    e = joblib.load(a.pkl)[a.key]
    fps = int(e["fps"])
    xy = e["root_trans_offset"][:, :2]
    t = np.arange(len(xy)) / fps
    speed = np.concatenate([[0], np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps])  # m/s
    yaw = np.degrees(np.unwrap(R.from_quat(e["root_rot"]).as_euler("xyz")[:, 2]))
    yaw = yaw - yaw[0]
    arm = np.concatenate([[0], np.linalg.norm(np.diff(e["dof"][:, 15:], axis=0), axis=1) * fps])
    # light smoothing
    def smooth(x, w=max(3, fps // 5)):
        return np.convolve(x, np.ones(w) / w, mode="same")
    fig, ax = plt.subplots(3, 1, figsize=(min(24, len(t) / fps * 0.8 + 4), 6), sharex=True, facecolor="white")
    ax[0].plot(t, smooth(speed), color="#2563eb"); ax[0].set_ylabel("speed m/s"); ax[0].set_facecolor("white")
    ax[1].plot(t, yaw, color="#dc2626"); ax[1].set_ylabel("heading deg"); ax[1].set_facecolor("white")
    ax[2].plot(t, smooth(arm), color="#16a34a"); ax[2].set_ylabel("arm motion"); ax[2].set_facecolor("white")
    ax[2].set_xlabel("seconds")
    for x in ax:
        x.grid(True, alpha=0.3)
        for s in range(0, int(t[-1]) + 1, 5):
            x.axvline(s, color="#cccccc", lw=0.4)
    ax[0].set_title(f"{a.key}  ({t[-1]:.0f}s)  — pick [start,end]s windows: turns=heading ramp, runs=speed hump, strikes=arm spike")
    plt.tight_layout()
    plt.savefig(a.out, dpi=100, facecolor="white")
    print(f"[timeline] {a.key} {t[-1]:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
