#!/usr/bin/env bash
# Stop the live demo: kill the Isaac viewer + GR00T inference server and free the GPU.
# (Bracket patterns avoid matching this script's own grep — see the pgrep self-match gotcha.)
set -uo pipefail

pids=$(ps -eo pid,cmd | grep -E '[e]val_agent_trl\.py|[g]ear_sonic_sequence\.sh|[r]un_gr00t_server' | awk '{print $1}')
if [[ -n "$pids" ]]; then
  echo "[stop] killing demo procs: $pids"
  kill -9 $pids 2>/dev/null || true
  sleep 3
else
  echo "[stop] no demo procs found."
fi

left=$(ps -eo cmd | grep -cE '[e]val_agent_trl\.py|[r]un_gr00t_server' || true)
echo "[stop] remaining demo procs: $left"
echo "[stop] GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
