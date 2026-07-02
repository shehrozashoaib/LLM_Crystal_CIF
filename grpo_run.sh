#!/bin/bash
# grpo_run.sh — run the discrete GRPO with crash-resume. The script auto-resumes
# from the newest checkpoint, so each retry continues where the last left off.
set -u
cd /home/ubuntu/LLM_Crystal_CIF
for i in $(seq 1 "${RETRY_MAX:-10}"); do
  echo "[grpo $(date -u +%FT%TZ)] attempt $i" >> logs/grpo_watchdog.log
  PYTORCH_ALLOC_CONF=expandable_segments:True \
    /venv/py312/bin/python grpo_r32_from3000_discrete.py >> logs/grpo_r32_from3000_discrete.log 2>&1
  rc=$?
  echo "[grpo $(date -u +%FT%TZ)] exited rc=$rc" >> logs/grpo_watchdog.log
  [ "$rc" -eq 0 ] && { echo "[grpo $(date -u +%FT%TZ)] completed cleanly" >> logs/grpo_watchdog.log; break; }
  sleep 30
done
