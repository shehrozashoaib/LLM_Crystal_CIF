#!/bin/bash
# ============================================================================
# grpo_delayed_launch.sh — unattended GRPO kickoff.
# Sleeps ~7h (for the running r128 rank job to finish), then waits until the GPU
# is actually free (no train/gen/val/sweep process) before launching the discrete
# GRPO run, so it never collides with the rank sweep. Runs once (GRPO is not
# auto-resumable here); logs everything.
# ============================================================================
set -u
cd /home/ubuntu/LLM_Crystal_CIF
LAUNCH_LOG=logs/grpo_launcher.log
RUN_LOG=logs/grpo_r32_from3000_discrete.log
mkdir -p logs

echo "[launcher $(date -u +%FT%TZ)] sleeping 7h before GRPO..." >> "$LAUNCH_LOG"
sleep 25200   # 7 hours

echo "[launcher $(date -u +%FT%TZ)] awake; waiting for GPU to free (rank sweep to finish)..." >> "$LAUNCH_LOG"
while pgrep -f "code_FineTune.py|generate_cifs_vllm.py|cif_structure_validator_mp52.py|run_rank_sweep.sh|run_with_retry.sh" >/dev/null 2>&1; do
  sleep 300
done

echo "[launcher $(date -u +%FT%TZ)] GPU free -> launching discrete GRPO (r32, fork ckpt-3000, 1500 steps)." >> "$LAUNCH_LOG"
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /venv/py312/bin/python grpo_r32_from3000_discrete.py >> "$RUN_LOG" 2>&1
echo "[launcher $(date -u +%FT%TZ)] GRPO process exited with code $?." >> "$LAUNCH_LOG"
