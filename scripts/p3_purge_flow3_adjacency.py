#!/usr/bin/env python3
"""Purge flow3-eval *adjacency* from the P3 recording corpus (methodology tightening).

The P3 runs were sliced flow3-source-EXCLUDED, so they share ZERO frames with the flow3
eval region (disjoint source ranges -> no bit-identical token leakage). But the temporal
held-out protocol uses a 43-step (@50Hz) purge gap to defend against autocorrelation: a
training frame a few frames before an eval frame carries near-duplicate tokens. The original
P3 build left 3-4 runs starting only 0-6 source frames after a flow3 segment's eval tail,
i.e. INSIDE the protocol's own purge band. This script drops those leading frames so every
retained P3 frame is >= PURGE source frames away from any same-family flow3 eval region.

Source frames are at the full-clip native fps (30); recorded steps are at 50Hz. Recorded
step i of a run starting at source frame s has source frame ~= s + i*30/50. flow3 eval
region of a segment = [offset + frac*L, offset + L) in source frames (conservative: frac on
source directly, wider than the 50Hz-chunk image).

Builds a clean raw dir (symlink unaffected runs, write head-trimmed npz for affected) so the
downstream npz->lerobot converter reruns unchanged.

    python scripts/p3_purge_flow3_adjacency.py \
        --runs LeSONIC/MaskBeT/outputs/p3_recording_runs.csv \
        --raw  LeSONIC/datasets/sonic_vla_raw_p3 \
        --data-dir LeSONIC/dependencies/GR00T-WholeBodyControl/data \
        --out-raw LeSONIC/datasets/sonic_vla_raw_p3_clean \
        --out-csv LeSONIC/MaskBeT/outputs/p3_purge_trims.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import joblib
import numpy as np

# flow3 segment source pkls per family (same-family = can share source frames with a P3 run)
FLOW3_SEGS = {
    "fight": ["seg_fight_block_pushkick_shove.pkl", "seg_fight_combat_combo_kicks.pkl",
              "seg_fight_fierce_swings.pkl"],
    "run": ["seg_run_circle.pkl", "seg_run_jog_backward.pkl", "seg_run_sprint_backpedal.pkl"],
}
FRAC = 0.8          # flow3 temporal split
PURGE_50HZ = 43     # protocol purge gap in 50Hz steps
SRC_FPS, REC_FPS = 30.0, 50.0


def locate(full_dof: np.ndarray, seg_dof: np.ndarray) -> int | None:
    """Return the source-frame offset of seg inside the full clip (exact content match)."""
    hits = np.where(np.all(np.isclose(full_dof, seg_dof[0], atol=1e-9), axis=1))[0]
    L = len(seg_dof)
    for h in hits:
        if h + L <= len(full_dof) and np.allclose(full_dof[h:h + L], seg_dof, atol=1e-9):
            return int(h)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-raw", required=True)
    ap.add_argument("--out-csv", required=True)
    a = ap.parse_args()

    purge_src = math.ceil(PURGE_50HZ * SRC_FPS / REC_FPS)  # 43@50Hz -> 26 @30fps source

    # flow3 eval source ranges per family
    fulls = {f: joblib.load(os.path.join(a.data_dir, f"{f}_full_robot.pkl")) for f in FLOW3_SEGS}
    fulls = {k: v[list(v)[0]]["dof"] for k, v in fulls.items()}
    eval_ranges: dict[str, list] = {f: [] for f in FLOW3_SEGS}
    for fam, segs in FLOW3_SEGS.items():
        for sf in segs:
            d = joblib.load(os.path.join(a.data_dir, sf))
            sd = d[list(d)[0]]["dof"]
            off = locate(fulls[fam], sd)
            L = len(sd)
            ev_s, ev_e = int(off + FRAC * L), int(off + L)
            eval_ranges[fam].append((sf.replace("seg_", "").replace(".pkl", ""), ev_s, ev_e))
            print(f"[purge] flow3 {fam}/{sf}: src=[{off},{off + L}) eval=[{ev_s},{ev_e})")

    rows = list(csv.DictReader(open(a.runs)))
    per_fam: dict[str, int] = {}
    trims = []
    for r in rows:
        fam = r["family"]
        idx = per_fam.get(fam, 0)
        per_fam[fam] = idx + 1
        key = f"{fam}_p3run{idx:02d}"
        s, t = int(r["frame_start"]), int(r["frame_end"])
        # front-trim: smallest source >= any eval_end+purge that the run starts inside
        need_src = s
        culprit = ""
        for name, ev_s, ev_e in eval_ranges[fam]:
            thr = ev_e + purge_src
            # run starts after the eval region but within the purge band
            if ev_e <= s < thr and thr > need_src:
                need_src, culprit = thr, name
            # (sanity) overlap into an eval region would be a hard leak — assert none
            assert not (s < ev_e and t > ev_s and s < ev_s), \
                f"{key} source [{s},{t}) overlaps flow3 eval {name} [{ev_s},{ev_e}) — hard leak!"
        trim_steps = int(math.ceil((need_src - s) * REC_FPS / SRC_FPS)) if need_src > s else 0
        trims.append((key, fam, s, t, culprit, trim_steps))
        if trim_steps:
            print(f"[purge] {key}: starts {s}, within purge of {culprit} -> drop {trim_steps} "
                  f"head steps (new src start ~{s + trim_steps * SRC_FPS / REC_FPS:.0f} >= {need_src})")

    # build clean raw dir
    os.makedirs(a.out_raw, exist_ok=True)
    trim_map = {k: ts for k, _, _, _, _, ts in trims}
    n_trimmed = 0
    for key, ts in trim_map.items():
        src_dir = os.path.join(a.raw, key)
        dst_dir = os.path.join(a.out_raw, key)
        os.makedirs(dst_dir, exist_ok=True)
        src_npz = os.path.join(src_dir, "episode_000.npz")
        dst_npz = os.path.join(dst_dir, "episode_000.npz")
        if os.path.lexists(dst_npz):
            os.remove(dst_npz)
        if ts == 0:
            os.symlink(os.path.abspath(src_npz), dst_npz)  # unaffected: symlink, zero disk
        else:
            d = np.load(src_npz, allow_pickle=True)
            out = {}
            for kk in d.files:
                arr = d[kk]
                out[kk] = arr[ts:] if (arr.ndim >= 1 and arr.shape and arr.shape[0] > ts) else arr
            np.savez_compressed(dst_npz, **out)
            n_trimmed += 1
            print(f"[purge] wrote trimmed {key}: {d['motion_token'].shape[0]} -> "
                  f"{out['motion_token'].shape[0]} steps")

    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "family", "src_start", "src_end", "purge_culprit", "head_trim_steps"])
        w.writerows(trims)
    affected = sum(1 for _, _, _, _, _, ts in trims if ts)
    print(f"[purge] {affected} runs trimmed ({n_trimmed} npz rewritten), "
          f"{len(trims) - affected} symlinked -> {a.out_raw}")
    print(f"[purge] trims -> {a.out_csv}")


if __name__ == "__main__":
    main()
