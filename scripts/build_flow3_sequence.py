#!/usr/bin/env python3
"""Concatenate per-window predicted SONIC tokens into ONE looping flow3 timeline npz.

flow3 = a multi-skill demo: the GR00T-predicted token streams of several hand-picked LAFAN
windows are concatenated in a chosen order into a single (T_total, 64) token array. Injected
with loop=True (offline, no server) it drives the SONIC WBC through skill→skill→… forever.

  python scripts/build_flow3_sequence.py \
      --pred-dir datasets/sonic_vla_pred_flow3 \
      --keys fight_block_pushkick_shove run_circle fight_combat_combo_kicks \
             run_sprint_backpedal dance_moonwalk run_jog_backward \
      --out datasets/sonic_vla_pred_flow3/_flow3_loop.npz

Seams are direct token concatenation (abrupt at boundaries). The WBC holds balance across
them (NULLTERM, no reset); a brief wobble at a seam is expected. --blend N linearly cross-fades
the last/first N tokens of adjacent windows to soften seams (0 = hard cut).
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blend", type=int, default=0, help="cross-fade N frames at each seam")
    a = ap.parse_args()

    segs = []
    for k in a.keys:
        d = np.load(f"{a.pred_dir}/{k}.npz")
        t = d["motion_token"] if "motion_token" in d.files else d[d.files[0]]
        segs.append(np.asarray(t, dtype=np.float32))
        print(f"[flow3] {k}: {t.shape}")

    if a.blend > 0:
        out = segs[0]
        for nxt in segs[1:]:
            b = min(a.blend, len(out), len(nxt))
            w = np.linspace(0, 1, b, dtype=np.float32)[:, None]
            tail = out[-b:] * (1 - w) + nxt[:b] * w
            out = np.concatenate([out[:-b], tail, nxt[b:]], axis=0)
        timeline = out
    else:
        timeline = np.concatenate(segs, axis=0)

    np.savez(a.out, motion_token=timeline, motion_key="flow3_loop",
             prompt="flow3 skill loop", segment_keys=np.array(a.keys),
             segment_lengths=np.array([len(s) for s in segs]))
    print(f"[flow3] timeline {timeline.shape} ({len(timeline)/50:.1f}s @50Hz) -> {a.out}")


if __name__ == "__main__":
    main()
