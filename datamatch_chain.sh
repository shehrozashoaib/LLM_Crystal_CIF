#!/bin/bash
# datamatch_chain.sh — wait for the running phase-split sweep (k=3000) to finish
# and free the GPU, then run the data-matched run under the retry watchdog.
set -u
cd /home/ubuntu/LLM_Crystal_CIF
mkdir -p logs
echo "[datamatch-chain $(date -u +%FT%TZ)] waiting for phase-split sweep to finish..." >> logs/datamatch_chain.log
# wait for the phase-split watchdog to exit
if [ -f logs/phase_split.pid ]; then
  while kill -0 "$(cat logs/phase_split.pid)" 2>/dev/null; do sleep 300; done
fi
# and wait for any GPU-using stage to be gone
while pgrep -f "code_FineTune.py|generate_cifs_vllm.py|cif_structure_validator_mp52.py|run_phase_split_sweep.sh" >/dev/null 2>&1; do
  sleep 120
done
echo "[datamatch-chain $(date -u +%FT%TZ)] GPU free -> launching data-matched run." >> logs/datamatch_chain.log
./run_with_retry.sh ./run_datamatched.sh >> logs/datamatch_watchdog.log 2>&1
echo "[datamatch-chain $(date -u +%FT%TZ)] data-matched run finished (rc=$?)." >> logs/datamatch_chain.log
