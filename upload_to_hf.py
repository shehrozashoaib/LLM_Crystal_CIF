#!/usr/bin/env python3
"""upload_to_hf.py — push completed experiment adapters to the HF model repo.

Mirrors the layout already used for the composition runs in
shehrozashoaib/LLM_Crystal_CIF:

    <run_name>/
        config.json
        training_stats.json
        model/      (adapter_config.json, adapter_model.safetensors, README.md)
        tokenizer/  (tokenizer.json, tokenizer_config.json, chat_template.jinja)

Locally those live under experiments/<run>/{config.json, training_stats.json,
final_model/model, final_model/tokenizer} — this script remaps final_model/* up
one level to match the repo layout.

Idempotent: a run already present on the repo (its adapter_model.safetensors
exists) is SKIPPED, so it's safe to re-run as more experiments finish (e.g. rank
seed 1234). Pass run names as args to upload a specific subset.

Usage:
    /venv/py312/bin/python upload_to_hf.py                 # all completed runs
    /venv/py312/bin/python upload_to_hf.py rank_r16_s1234  # one run
"""
import os
import sys
from huggingface_hub import HfApi

REPO_ID = "shehrozashoaib/LLM_Crystal_CIF"
ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(ROOT, "experiments")

# Completed GH200 experiments (composition already uploaded from A100).
DEFAULT_RUNS = [
    "curr_fwd_p1", "curr_fwd", "curr_rev_p1", "curr_rev",
    "rank_r16_s3407", "rank_r32_s3407", "rank_r64_s3407", "rank_r128_s3407",
]


def main():
    runs = sys.argv[1:] or DEFAULT_RUNS
    api = HfApi()
    existing = set(api.list_repo_files(REPO_ID))

    for run in runs:
        d = os.path.join(EXP, run)
        adapter = os.path.join(d, "final_model", "model", "adapter_model.safetensors")
        if not os.path.isfile(adapter):
            print(f"[skip] {run}: no adapter at {adapter} (not finished?)")
            continue
        if f"{run}/model/adapter_model.safetensors" in existing:
            print(f"[skip] {run}: already on {REPO_ID}")
            continue

        print(f"[upload] {run} -> {REPO_ID}/{run}/ ...")
        # model/ and tokenizer/ (remap final_model/* -> */)
        api.upload_folder(repo_id=REPO_ID, folder_path=os.path.join(d, "final_model", "model"),
                          path_in_repo=f"{run}/model",
                          commit_message=f"Add {run} adapter")
        api.upload_folder(repo_id=REPO_ID, folder_path=os.path.join(d, "final_model", "tokenizer"),
                          path_in_repo=f"{run}/tokenizer",
                          commit_message=f"Add {run} tokenizer")
        # the two top-level json files
        for fn in ("config.json", "training_stats.json"):
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                api.upload_file(repo_id=REPO_ID, path_or_fileobj=p,
                                path_in_repo=f"{run}/{fn}",
                                commit_message=f"Add {run}/{fn}")
        print(f"[done] {run}")

    print("All requested uploads complete.")


if __name__ == "__main__":
    main()
