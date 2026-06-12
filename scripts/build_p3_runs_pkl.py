#!/usr/bin/env python3
"""Build the P3 same-domain corpus pkl: 15 contiguous recording runs sliced from the
fight/run full clips by frame ranges (outputs/p3_recording_runs.csv).

Each run is a flow3-source-EXCLUDED, WBC-trackable (tilt<=33, z>=0.35) contiguous span
of the LAFAN full clip, merged from the net-keep pieces. Recording these 15 runs (rather
than 121 windows) avoids the ~80-step token bootstrap eating short windows, and the runs
carry no flow3 eval frames (leakage-free by construction — adjacent net pieces only merged
when <=1 frame apart, so flow3 gaps break the merge).

  python scripts/build_p3_runs_pkl.py \
      --runs LeSONIC/MaskBeT/outputs/p3_recording_runs.csv \
      --data-dir LeSONIC/dependencies/GR00T-WholeBodyControl/data \
      --out LeSONIC/dependencies/GR00T-WholeBodyControl/data/seg_p3_runs_all.pkl \
      --meta LeSONIC/MaskBeT/outputs/p3_runs_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import joblib
import numpy as np

# family-level honest prompts — the merged runs span several auto-cut sub-motions with no
# ground-truth sub-labels, so per-family text condition (not fabricated per-window labels).
PROMPT = {
    "fight": "martial arts combat with strikes and kicks",
    "run": "running, jogging, and turning",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="p3_recording_runs.csv")
    ap.add_argument("--data-dir", required=True, help="dir with {fight,run}_full_robot.pkl")
    ap.add_argument("--out", required=True, help="output multi-key pkl")
    ap.add_argument("--meta", required=True, help="output manifest csv (key,family,fs,fe,frames,steps_50hz,prompt)")
    a = ap.parse_args()

    fulls = {}
    rows = list(csv.DictReader(open(a.runs)))
    out: dict = {}
    meta = []
    per_fam = {}
    for r in rows:
        fam = r["family"]
        if fam not in fulls:
            d = joblib.load(os.path.join(a.data_dir, f"{fam}_full_robot.pkl"))
            fulls[fam] = d[list(d)[0]]
        e = fulls[fam]
        n = len(e["dof"])
        fps = int(e["fps"])
        s, t = int(r["frame_start"]), int(r["frame_end"])
        idx = per_fam.get(fam, 0)
        per_fam[fam] = idx + 1
        key = f"{fam}_p3run{idx:02d}"
        seg = {}
        for k, v in e.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
                seg[k] = v[s:t].copy()
            else:
                seg[k] = v
        out[key] = seg
        steps = int(round((t - s) * 50.0 / fps))
        meta.append((key, fam, s, t, t - s, steps, PROMPT[fam]))

    joblib.dump(out, a.out)
    with open(a.meta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "family", "frame_start", "frame_end", "frames_30fps", "steps_50hz", "prompt"])
        w.writerows(meta)
    tot_f = sum(m[4] for m in meta)
    tot_s = sum(m[5] for m in meta)
    print(f"[build] {len(out)} runs -> {a.out}")
    print(f"[build] {tot_f} frames@30fps = {tot_s} steps@50Hz ({tot_s/50/60:.1f} min)")
    print(f"[build] manifest -> {a.meta}")


if __name__ == "__main__":
    main()
