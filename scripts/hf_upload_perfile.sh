#!/usr/bin/env bash
# Per-file HF upload — the reliable path for >GB shards behind a mihomo TUN proxy.
# Memory: folder/upload-large-folder batch-commit hangs at xet commit_chunk under TUN;
# per-file + HF_HUB_DISABLE_XET=1 (standard LFS, fast pointer commit) lands in one shot.
# Big shards FIRST (overwrite same-named fp32), small files LAST so the bf16 index.json
# only flips once the bf16 shards are present.
set -uo pipefail

REPO="wsagi/GR00T-N1.7-G1-SONIC-BonesSeed"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/hf_upload"
cd "$DIR"

export HF_HUB_ENABLE_HF_TRANSFER=1
# NOTE: keep xet ENABLED. DISABLE_XET stalls the LFS-batch handshake under this TUN proxy
# (hangs at 0% before any bytes). xet transfers fine (~200MB/s); the only risk is the final
# per-file commit, which is small here. Transient ConnectTimeout on /api/repos/create =
# proxy connection table saturated by a previous attempt's sockets -> sleep lets it drain.

up() {  # up <path-in-repo> <timeout_s> <max_attempts>
    local f="$1" to="$2" max="$3" a
    for a in $(seq 1 "$max"); do
        if timeout "$to" hf upload "$REPO" "$f" "$f" --repo-type model; then
            echo "[OK] $f (attempt $a)"; sleep 8; return 0   # let sockets drain before next file
        fi
        echo "[retry $a/$max] $f hung/failed; draining sockets..."; sleep 15
    done
    echo "[FAIL] $f after $max attempts"; return 1
}

rc=0
# 1) big shards first, generous timeout
for s in model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
    up "$s" 360 12 || rc=1
done
# 2) small files / dirs last (index.json carries the bf16 total_size)
for f in config.json processor experiment_cfg training_args.bin wandb_config.json \
         model.safetensors.index.json README.md; do
    up "$f" 120 8 || rc=1
done

echo "=== DONE rc=$rc ==="
exit $rc
