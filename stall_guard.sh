#!/bin/bash
# stall_guard.sh — kill a GPU stage that has hung.
#
# vLLM's EngineCore can segfault and leave the PARENT process alive, blocked
# forever on a dead worker. run_with_retry.sh only reacts to a non-zero EXIT, so
# a hang stalls the pipeline indefinitely (observed: 4.4h of zero progress).
# Both training and generation keep the GPU busy, so sustained 0% utilisation
# while a stage is running means it is wedged: kill it and let the watchdog retry.
set -u
STALL_MIN="${STALL_MIN:-20}"
POLL="${POLL:-60}"
LOG=/workspace/LLM_Crystal_CIF/logs/stall_guard.log
idle=0
while true; do
  pids=$(pgrep -f "code_FineTune.py|generate_cifs_vllm.py" | tr '\n' ' ')
  if [ -z "$pids" ]; then idle=0; sleep "$POLL"; continue; fi
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  if [ "${util:-0}" -le 2 ]; then
    idle=$((idle + POLL))
    if [ "$idle" -ge $((STALL_MIN * 60)) ]; then
      echo "[stall-guard $(date -u +%FT%TZ)] GPU 0% for ${STALL_MIN}m with pids [$pids] -> killing" >> "$LOG"
      for p in $pids; do kill -9 "$p" 2>/dev/null; done
      idle=0
    fi
  else
    idle=0
  fi
  sleep "$POLL"
done
