#!/usr/bin/env python3
"""Build the full SONIC data-augmentation TRANSITION token set (physically-correct route).

For EVERY target motion, produce multiple `[1s other-motion source] ++ [trimmed target]` token
sequences. These concat token sequences are then replayed through the SONIC WBC (real physics) and
RECORDED into LeRobot training data — NOT spliced as raw frames (that was the old, non-physical
sonic_vla_resample_build.py that codex flagged). The WBC basin probe (10 probes, 9 LAUNCH) proved
the WBC can execute any motion's onset from arbitrary {static, momentum} states, so these rollouts
launch and yield valid (obs, token) pairs covering the deployment "switch-anytime" distribution.

Recipe (per the user):
  - 1s (50f) of ANOTHER motion in front = the momentum source. The 1s window is taken AFTER the
    source's own weak front (FRONT_TRIM) so it carries real momentum, not the source's stand-front.
  - macarena target: trim front 6s (300f); walk target: trim front 2.5s (125f); jump: 19f
    (these motions open with a long standing/weak section we don't want in the onset data).
  - Also one COLD-STAND variant per target (no source: WBC default-init stand -> target from f0).

Output: datasets/sonic_transition_samples/{target}_from_{source|coldstand}.npz
Each npz: motion_token (T,64), motion_key, switch_at, target, source, kind. Next step = rollout+record.
"""
import argparse, os
import numpy as np

GT = os.path.join(os.path.dirname(__file__), "..", "datasets", "sonic_vla_gt")
OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "datasets", "sonic_transition_samples")
K = {"dance": "dance_in_da_party_001__A464", "lunge": "forward_lunge_R_001__A359_M",
     "macarena": "macarena_001__A545", "kick": "neutral_kick_R_001__A543",
     "squat": "squat_001__A359", "jump": "tired_one_leg_jumping_R_001__A359",
     "walk": "walking_quip_360_R_002__A428"}
MOTIONS = list(K)
# front frames to drop (long stand/weak openers). User: macarena 6s, walk 2.5s; jump 19 (old build).
FRONT_TRIM = {"macarena": 300, "walk": 125, "jump": 19}


def tok(name):
    return np.load(f"{GT}/{K[name]}.npz", allow_pickle=True)["motion_token"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--src-len", type=int, default=50)    # 1s momentum prefix
    ap.add_argument("--tgt-len", type=int, default=250)   # 5s target (onset + sustain) after trim
    ap.add_argument("--cold", action="store_true", default=True, help="also emit a no-source cold-stand variant")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n = 0
    for tgt in MOTIONS:
        tt = FRONT_TRIM.get(tgt, 0)
        g = tok(tgt)[tt: tt + args.tgt_len]                # trimmed target (onset dominates)
        # cold-stand: WBC default init is a stable stand -> target straight from f0, no prefix.
        if args.cold:
            np.savez(f"{args.out}/{tgt}_from_coldstand.npz", motion_token=g.astype(np.float32),
                     motion_key=K[tgt], switch_at=0, target=tgt, source="cold", kind="aug_transition")
            n += 1
        for src in MOTIONS:
            if src == tgt:
                continue
            st = FRONT_TRIM.get(src, 0)                     # skip source's weak front -> real momentum
            s = tok(src)[st: st + args.src_len]
            seq = np.concatenate([s, g]).astype(np.float32)
            np.savez(f"{args.out}/{tgt}_from_{src}.npz", motion_token=seq, motion_key=K[tgt],
                     switch_at=len(s), target=tgt, source=src, kind="aug_transition")
            n += 1
        print(f"  target {tgt:9s} (trim {tt:3d}f, len {len(g)}) -> cold + {len(MOTIONS)-1} momentum sources")
    print(f"\n{n} augmentation transition sequences -> {args.out}")


if __name__ == "__main__":
    main()
