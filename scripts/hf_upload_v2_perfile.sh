#!/usr/bin/env bash
# Per-file HF upload for the V2 (aug2-8000) checkpoint — reliable path for >GB shards behind a
# mihomo TUN proxy (folder batch-commit hangs at xet commit_chunk; per-file lands in one shot).
# Big shards first, then demo videos, then small config + README last.
set -uo pipefail

REPO="wsagi/GR00T-N1.7-G1-SONIC-BonesSeed-V2"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/hf_upload_v2"
cd "$DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1

up() {  # up <path-in-repo> <timeout_s> <max_attempts>
    local f="$1" to="$2" max="$3" a
    for a in $(seq 1 "$max"); do
        if timeout "$to" hf upload "$REPO" "$f" "$f" --repo-type model; then
            echo "[OK] $f (attempt $a)"; sleep 8; return 0
        fi
        echo "[retry $a/$max] $f hung/failed; draining sockets..."; sleep 15
    done
    echo "[FAIL] $f after $max attempts"; return 1
}

rc=0
for s in model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
    up "$s" 360 12 || rc=1
done
for f in videos/demo_livedemo.mp4 videos/demo_flow2.mp4; do
    up "$f" 240 8 || rc=1
done
for f in config.json processor_config.json statistics.json embodiment_id.json \
         training_args.bin wandb_config.json experiment_cfg \
         model.safetensors.index.json README.md; do
    up "$f" 120 8 || rc=1
done
echo "=== MODEL UPLOAD DONE rc=$rc ==="
exit $rc
