#!/usr/bin/env python3
"""Re-save a GR00T checkpoint from fp32 → bf16 (CPU-only, no GPU).

P0 #3: published ckpt is stored fp32 (~12.6 GB / 3 shards) although training was bf16.
This casts every floating tensor to bfloat16 and rewrites the shard index, halving the
artifact to ~6.3 GB with no precision loss beyond the bf16 the model was already trained in.

Preserves the original 3-shard grouping (loader only trusts index.json) and copies all
non-weight files so the output dir is a complete, loadable checkpoint.

    python scripts/gr00t_resave_bf16.py <src_ckpt_dir> <dst_ckpt_dir>
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main(src: Path, dst: Path) -> None:
    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]  # tensor_name -> shard_filename

    # invert: shard_filename -> [tensor_name, ...], preserving original grouping
    shard_to_keys: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(name)

    dst.mkdir(parents=True, exist_ok=True)
    total_size = 0
    n_cast = 0
    for shard, keys in sorted(shard_to_keys.items()):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(src / shard, framework="pt", device="cpu") as f:
            for k in keys:
                t = f.get_tensor(k)
                if t.is_floating_point() and t.dtype != torch.bfloat16:
                    t = t.to(torch.bfloat16)
                    n_cast += 1
                tensors[k] = t.contiguous()
                total_size += t.numel() * t.element_size()
        save_file(tensors, str(dst / shard), metadata={"format": "pt"})
        print(f"  wrote {shard}: {len(keys)} tensors", flush=True)

    # rewrite index with corrected total_size (offsets are per-file, regenerated on save)
    index["metadata"]["total_size"] = total_size
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # copy every non-weight artifact so dst is a self-sufficient checkpoint
    for item in src.iterdir():
        if item.name.endswith(".safetensors") or item.name == "model.safetensors.index.json":
            continue
        if item.name.startswith("checkpoint-") or item.name == "keep":
            continue  # intermediate ckpts / hardlink dir — not part of the published weights
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        print(f"  copied {item.name}", flush=True)

    print(f"\nDONE: cast {n_cast} float tensors → bf16, total {total_size/1e9:.2f} GB → {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
