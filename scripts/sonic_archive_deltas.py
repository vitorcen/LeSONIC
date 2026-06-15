#!/usr/bin/env python3
"""Incremental head-delta extractor for StarVLA frozen-backbone ckpts (box-side).

Builds a frozen base ONCE (ckpt minus the trainable tensors, learned from two
ckpts' diff), then for each requested step extracts a GOLD-verified delta
(head + unfrozen layers = the only tensors that differ from base). Idempotent:
skips a step whose delta already exists. Designed to run on the training box in
parallel with training (CPU/RAM only — no GPU), so deltas are ready to pull as
checkpoints appear. Reconstruct with scripts/ckpt/merge_ckpt.py base delta out.

    python sonic_archive_deltas.py <run_dir> <base_path> 4800,5400,6000
"""
import os, sys, glob, torch


def load(p):
    return torch.load(p, map_location="cpu")


def ckpt_path(run, step):
    return os.path.join(run, "checkpoints", f"steps_{step}_pytorch_model.pt")


def delta_path(run, step):
    return os.path.join(run, "heads", f"steps_{step}_delta.pt")


def main():
    run, base_path, steps_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    steps = [int(s) for s in steps_csv.split(",") if s]
    os.makedirs(os.path.join(run, "heads"), exist_ok=True)

    # build base once: frozen = keys byte-identical across two different ckpts
    if not os.path.exists(base_path):
        existing = sorted(glob.glob(ckpt_path(run, "*").replace("steps_*", "steps_*")),
                          key=lambda p: int(p.split("steps_")[1].split("_")[0]))
        if len(existing) < 2:
            print("[base] need >=2 ckpts to build base; deferring"); return 0
        a, b = load(existing[-1]), load(existing[-2])
        train = {k for k in a if k not in b or not torch.equal(a[k], b[k])}
        base = {k: v.clone() for k, v in a.items() if k not in train}
        ffrac = sum(v.numel()*v.element_size() for v in base.values()) / \
                max(sum(v.numel()*v.element_size() for v in a.values()), 1)
        torch.save(base, base_path)
        print(f"[base] trainable={len(train)} frozen={len(base)} frozen_frac={ffrac:.3f} "
              f"-> {os.path.basename(base_path)} {os.path.getsize(base_path)/1e9:.1f}G")
        del a, b, base

    base = load(base_path)
    for step in steps:
        dp = delta_path(run, step)
        if os.path.exists(dp):
            print(f"[skip] step {step} delta exists"); continue
        cp = ckpt_path(run, step)
        if not os.path.exists(cp):
            print(f"[miss] step {step} ckpt gone (pruned?)"); continue
        sd = load(cp)
        delta = {k: sd[k].clone() for k in sd if k not in base or not torch.equal(sd[k], base[k])}
        recon = {**base, **delta}
        gold = set(recon) == set(sd) and all(torch.equal(recon[k], sd[k]) for k in sd)
        if gold and delta:
            torch.save(delta, dp)
            print(f"[ok ] step {step} delta={len(delta)}k "
                  f"{sum(t.numel()*t.element_size() for t in delta.values())/1e9:.2f}G GOLD✓")
        else:
            print(f"[FAIL] step {step} gold={gold} deltaN={len(delta)}")
        del sd, delta, recon
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
