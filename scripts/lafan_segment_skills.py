#!/usr/bin/env python3
"""Segment a MimicKit G1 motion clip into coherent skill segments at velocity valleys
(NOT fixed time), and emit a multi-key SONIC robot_filtered pkl for the strict WBC screen.

Why velocity valleys: a coherent action (a jab combo, a turn, a stride) is bounded by moments
where the body momentarily settles (low joint+root speed). Cutting there yields segments whose
start/end poses are relatively stable -> better RSI init and cleaner inter-skill chaining than
arbitrary time slices.

Pipeline:
  full clip -> activity signal (joint angular speed + root linear speed, smoothed)
            -> deep valleys >= min_gap apart -> coherent segments (drop < min_dur)
            -> each segment -> robot_filtered entry (reuse convert_sequence) -> one pkl, one key/seg
            -> manifest.csv (key, frames, dur_s, horiz_travel_m) for picking/screening

Clustering into named skills is deliberately a SEPARATE later step, run on the WBC-trackable
SURVIVORS only (no point clustering segments the WBC can't track).

Usage (isaaclab env, cwd = GR00T-WholeBodyControl):
  python <this> --input dependencies/MimicKit/data/motions/g1/lafan_fight1.pkl \
      --prefix fight --output data/fight_segments_robot.pkl --joint_order mj
"""
import argparse
import csv
import os
import sys

import joblib
import numpy as np
from scipy.spatial import transform

_WBC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "dependencies", "GR00T-WholeBodyControl")
sys.path.insert(0, _WBC)
from gear_sonic.data_process.convert_soma_csv_to_motion_lib import convert_sequence  # noqa: E402


def detect_segments(frames, fps, valley_pct, min_gap_s, min_dur_s):
    root = frames[:, 0:3]
    dof = frames[:, 6:35]
    dj = np.linalg.norm(np.diff(dof, axis=0), axis=1)
    dr = np.linalg.norm(np.diff(root, axis=0), axis=1)
    act = dj / (np.median(dj) + 1e-9) + dr / (np.median(dr) + 1e-9)
    w = max(3, fps // 4)
    acts = np.convolve(act, np.ones(w) / w, mode="same")
    thr = np.percentile(acts, valley_pct)
    min_gap = int(fps * min_gap_s)
    val = []
    for i in range(w, len(acts) - w):
        if acts[i] < thr and acts[i] == acts[i - w:i + w].min():
            if not val or i - val[-1] > min_gap:
                val.append(i)
    b = [0] + val + [frames.shape[0] - 1]
    segs = [(b[i], b[i + 1]) for i in range(len(b) - 1)]
    return [(a, c) for a, c in segs if (c - a) >= fps * min_dur_s]


def seg_to_entry(frames, a, c, fps, joint_order):
    sl = frames[a:c]
    root_pos = sl[:, 0:3]
    root_expmap = sl[:, 3:6]
    dof_pos = sl[:, 6:35].astype(np.float32)
    quat_xyzw = transform.Rotation.from_rotvec(root_expmap).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    n = sl.shape[0]
    body_pos = np.zeros((n, 14, 3), dtype=np.float32); body_pos[:, 0] = root_pos
    body_quat = np.zeros((n, 14, 4), dtype=np.float32); body_quat[:, 0] = quat_wxyz
    return convert_sequence({"joint_pos": dof_pos, "body_pos_w": body_pos,
                             "body_quat_w": body_quat, "joint_order": joint_order}, fps=fps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--prefix", required=True, help="motion_key prefix, e.g. 'fight'")
    p.add_argument("--output", required=True)
    p.add_argument("--joint_order", default="mj", choices=["mj", "il"])
    p.add_argument("--valley_pct", type=float, default=25.0)
    p.add_argument("--min_gap_s", type=float, default=1.2)
    p.add_argument("--min_dur_s", type=float, default=0.8)
    p.add_argument("--max_segs", type=int, default=0, help="0 = all")
    a = p.parse_args()

    m = joblib.load(a.input)
    frames = np.asarray(m["frames"], dtype=np.float64)
    fps = int(m["fps"])
    segs = detect_segments(frames, fps, a.valley_pct, a.min_gap_s, a.min_dur_s)
    if a.max_segs:
        segs = segs[:a.max_segs]

    out = {}
    rows = []
    root = frames[:, 0:3]
    for idx, (s, e) in enumerate(segs):
        key = f"{a.prefix}_seg{idx:03d}"
        out[key] = seg_to_entry(frames, s, e, fps, a.joint_order)
        travel = float(np.linalg.norm(root[e - 1, :2] - root[s, :2]))
        rows.append({"key": key, "frame_start": s, "frame_end": e,
                     "dur_s": round((e - s) / fps, 2), "horiz_travel_m": round(travel, 2)})

    joblib.dump(out, a.output)
    man = os.path.splitext(a.output)[0] + "_manifest.csv"
    with open(man, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(f"[segment] {a.input} -> {a.output}  ({len(out)} segments, fps={fps})")
    print(f"[segment] manifest -> {man}")
    print(f"[segment] durations {min(r['dur_s'] for r in rows):.1f}..{max(r['dur_s'] for r in rows):.1f}s")


if __name__ == "__main__":
    main()
