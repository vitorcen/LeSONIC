#!/usr/bin/env bash
# Switch the RUNNING live SONIC demo to another motion — no viewer/server restart.
# The persistent viewer (started with VLA_PROMPT_FILE set, see BonesSeed.ipynb 5.2a) polls
# this file every few control steps and swaps the prompt live.
#   bash scripts/sonic_say.sh kick        # -> writes "kick"'s trained prompt string
#   bash scripts/sonic_say.sh "raw prompt string"   # or pass a raw prompt directly
# names: dance | lunge | macarena | kick | squat | jump | walk | (flow3) combat|block|... 
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="${VLA_PROMPT_FILE:-/tmp/sonic_live_prompt.txt}"
arg="${1:?usage: sonic_say.sh <motion-key|raw prompt>}"
# Map a known motion key -> its trained prompt string (single source of truth = build_sequence).
prompt="$(python3 - "$arg" <<PY
import sys
sys.path.insert(0, "$HERE")
try:
    from gr00t_build_sequence import PROMPT
except Exception:
    PROMPT = {}
a = sys.argv[1]
print(PROMPT.get(a, a))   # known key -> prompt; otherwise treat arg as a raw prompt
PY
)"
printf '%s' "$prompt" > "$PROMPT_FILE"
echo "[sonic_say] switched -> '$prompt'   (file=$PROMPT_FILE)"
