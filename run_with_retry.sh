#!/bin/bash
# ============================================================================
# run_with_retry.sh — re-invoke a RESUMABLE sweep until it completes (exit 0).
# ----------------------------------------------------------------------------
# Guards against transient mid-sweep deaths — notably vLLM exiting non-zero
# during CUDA shutdown *after* it already saved its output, which set -e/pipefail
# in the sweep otherwise treats as a fatal error. Each retry simply re-runs the
# sweep, which skips every stage whose output already exists, so it resumes where
# it left off. NOT a substitute for systemd/tmux if the whole host/session dies.
#
# Usage:  nohup ./run_with_retry.sh ./run_rank_sweep.sh > logs/x.log 2>&1 &
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
MAX="${RETRY_MAX:-20}"
for i in $(seq 1 "$MAX"); do
  echo "[watchdog $(date -u +%FT%TZ)] attempt $i/$MAX: $*"
  if "$@"; then
    echo "[watchdog $(date -u +%FT%TZ)] sweep completed cleanly on attempt $i"
    exit 0
  fi
  echo "[watchdog $(date -u +%FT%TZ)] sweep exited non-zero; resuming in 30s..."
  sleep 30
done
echo "[watchdog] gave up after $MAX attempts"; exit 1
