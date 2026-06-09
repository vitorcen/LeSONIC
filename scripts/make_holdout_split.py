#!/usr/bin/env python3
"""Build a leave-one-motion-out (LOMO) held-out split of the SONIC-VLA LeRobot dataset.

P0 #2 go/no-go gate. The published model trained on all 7 motions (1 episode each) with NO
held-out split, so `MSE 0.0011` is train-set reconstruction, not generalization. This tool
carves a clean held-out dataset: it drops ONE motion entirely from training so that, after a
retrain on the remaining 6, an open-loop eval on the dropped motion measures *new-motion*
generalization. If held-out MSE ≈ train MSE → the model generalizes; if it explodes → the
0.0011 was memorization (as suspected), and everything downstream is adding rows to a lookup
table.

Produces a fully-valid LeRobot v2.1 dataset (contiguous episode/frame/task indices) with the
held-out episode removed. Video files are symlinked (no GB duplication); parquets are rewritten
to fix indices. stats.json / relative_stats.json are intentionally NOT copied so GR00T's
`generate_stats` recomputes normalization over the 6-episode subset (it only uses lowdim
features and regenerates when stats are absent — verified in gr00t/data/stats.py).

    python scripts/make_holdout_split.py <src_lerobot_dir> <dst_dir> --holdout-motion squat
    python scripts/make_holdout_split.py <src_lerobot_dir> <dst_dir> --holdout-episode 4

The dropped episode is reported so you can open-loop-eval it as the held-out probe.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

DATA_REL = "data/chunk-000/episode_{:06d}.parquet"
VIDEO_DIR_REL = "videos/chunk-000"


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--holdout-episode", type=int, help="episode_index to hold out")
    g.add_argument("--holdout-motion", type=str, help="task string (substring) to hold out")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    episodes = read_jsonl(src / "meta/episodes.jsonl")

    # resolve which episode to drop
    if args.holdout_episode is not None:
        holdout = args.holdout_episode
    else:
        matches = [e["episode_index"] for e in episodes
                   if any(args.holdout_motion.lower() in t.lower() for t in e["tasks"])]
        if len(matches) != 1:
            raise SystemExit(f"--holdout-motion {args.holdout_motion!r} matched {matches}; want exactly 1")
        holdout = matches[0]
    held = next(e for e in episodes if e["episode_index"] == holdout)
    kept = [e for e in episodes if e["episode_index"] != holdout]
    print(f"HOLD OUT episode {holdout} {held['tasks']} ({held['length']} frames)")
    print(f"KEEP {len(kept)} episodes: {[(e['episode_index'], e['tasks'][0]) for e in kept]}")

    if dst.exists():
        raise SystemExit(f"dst exists: {dst} (remove it first)")
    (dst / "data/chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)

    # old episode_index -> new contiguous id
    ep_remap = {e["episode_index"]: i for i, e in enumerate(kept)}

    # rebuild tasks.jsonl over kept episodes only; old task_index -> new
    tasks_src = read_jsonl(src / "meta/tasks.jsonl")
    task_str_by_idx = {t["task_index"]: t["task"] for t in tasks_src}
    kept_task_strs = []
    for e in kept:
        for t in e["tasks"]:
            if t not in kept_task_strs:
                kept_task_strs.append(t)
    new_task_idx = {s: i for i, s in enumerate(kept_task_strs)}
    write_jsonl(dst / "meta/tasks.jsonl",
                [{"task_index": i, "task": s} for s, i in new_task_idx.items()])
    old_task_idx_to_new = {oi: new_task_idx[s] for oi, s in task_str_by_idx.items()
                           if s in new_task_idx}

    # rewrite parquets with corrected episode_index, task_index, global index
    global_index = 0
    new_episodes, new_lengths = [], {}
    for e in kept:
        old_id = e["episode_index"]
        new_id = ep_remap[old_id]
        df = pd.read_parquet(src / DATA_REL.format(old_id))
        df["episode_index"] = new_id
        if "task_index" in df.columns:
            df["task_index"] = df["task_index"].map(lambda v: old_task_idx_to_new.get(int(v), 0))
        df["index"] = range(global_index, global_index + len(df))
        global_index += len(df)
        df.to_parquet(dst / DATA_REL.format(new_id))
        new_lengths[new_id] = len(df)
        ne = dict(e); ne["episode_index"] = new_id
        new_episodes.append(ne)

    write_jsonl(dst / "meta/episodes.jsonl", new_episodes)

    # episodes_stats.jsonl: keep per-episode stats, reindex episode_index
    es = read_jsonl(src / "meta/episodes_stats.jsonl")
    es_kept = []
    for row in es:
        if row["episode_index"] in ep_remap:
            r = dict(row); r["episode_index"] = ep_remap[row["episode_index"]]
            es_kept.append(r)
    es_kept.sort(key=lambda r: r["episode_index"])
    write_jsonl(dst / "meta/episodes_stats.jsonl", es_kept)

    # info.json: fix counts + splits
    info = json.loads((src / "meta/info.json").read_text())
    info["total_episodes"] = len(kept)
    info["total_frames"] = global_index
    info["total_tasks"] = len(kept_task_strs)
    info["total_videos"] = len(kept)
    info["splits"] = {"train": f"0:{len(kept)}"}
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))

    # modality.json verbatim; stats.json / relative_stats.json deliberately omitted
    shutil.copy2(src / "meta/modality.json", dst / "meta/modality.json")

    # symlink videos (reindexed names) — no GB duplication
    for old_id, new_id in ep_remap.items():
        for vkey_dir in (src / VIDEO_DIR_REL).iterdir():
            if not vkey_dir.is_dir():
                continue
            srcv = vkey_dir / f"episode_{old_id:06d}.mp4"
            if not srcv.exists():
                continue
            dstv_dir = dst / VIDEO_DIR_REL / vkey_dir.name
            dstv_dir.mkdir(parents=True, exist_ok=True)
            os.symlink(srcv, dstv_dir / f"episode_{new_id:06d}.mp4")

    print(f"\nDONE: {len(kept)} episodes / {global_index} frames -> {dst}")
    print(f"held-out probe = episode {holdout} {held['tasks']} "
          f"(eval this after retraining on the split to measure new-motion generalization)")
    print("note: stats.json omitted on purpose -> GR00T regenerates over the 6-episode subset.")


if __name__ == "__main__":
    main()
