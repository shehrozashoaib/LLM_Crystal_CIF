Partial vLLM generation for cks123456_k1500 (seed 123456), pushed early because the compute could stop.
Each shard = 1,000 consecutive MPTS-52 test materials x 10 CIFs (part_0000000 = rows 0-999, ...).
To resume elsewhere: gunzip the shards into
  generated/cks123456_k1500/generated_cifs_cks123456_k1500_10seq_maxtok3072_0_8095_shards/
copy gen_meta.json to generated/cks123456_k1500/, fetch the adapter from HF
(shehrozashoaib/LLM_Crystal_CIF/cks123456_k1500) into experiments/cks123456_k1500/final_model/,
and rerun training/run_curriculum_k_sweep.sh 1500 — finished shards are skipped
(per-request seeds make the remaining shards identical to an uninterrupted run).
