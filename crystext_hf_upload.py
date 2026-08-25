#!/usr/bin/env python3
"""crystext_hf_upload.py — push the CrysText-reward GRPO adapter to the HF model repo.

Mirrors the layout the other runs use in shehrozashoaib/LLM_Crystal_CIF:

    grpo_crystext_reward/
        step2150/model/       (adapter_config.json, adapter_model.safetensors, chat_template.jinja)
        step2150/tokenizer/   (tokenizer.json, tokenizer_config.json, chat_template.jinja)
        config.json           (hyperparameters, provenance, measured results)

Weights only — the optimizer / scheduler / RNG state in the TRL checkpoint is not uploaded.

Idempotent: a step already present on the repo is SKIPPED.

Usage:
    HF_TOKEN=hf_xxx /venv/py312/bin/python crystext_hf_upload.py            # default: step 2150
    HF_TOKEN=hf_xxx /venv/py312/bin/python crystext_hf_upload.py --step 2100
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "shehrozashoaib/LLM_Crystal_CIF"
ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments" / "grpo_crystext_reward"
PREFIX = "grpo_crystext_reward"

MODEL_FILES = ["adapter_config.json", "adapter_model.safetensors", "chat_template.jinja"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]

# Provenance + what was actually measured. Kept in the repo so the adapter is not
# a bare weight blob: anyone downloading it can see how it was trained and how it scored.
CONFIG = {
    "run_name": "grpo_crystext_reward",
    "method": "GRPO (TRL 0.24) + Unsloth LoRA",
    "base_model": "unsloth/Qwen2.5-7B-Instruct",
    "start_model": "rank_r32_s3407 (this repo's SFT adapter, r=32, 4500 steps, 16-bit)",
    "reward": {
        "source": "https://github.com/truptimohanty/CrysText/blob/main/grpo_training.py",
        "note": "validate_structure() copied verbatim; verified bit-identical by "
                "test_crystext_reward_parity.py (180/180 cases)",
        "range": [-2.0, 3.0],
        "ladder": {
            "parses_with_pymatgen": 0.5,
            "structure_validity": 0.5,
            "reduced_formula_match": 0.5,
            "match_stol0.9_ltol0.7_ang20": 0.25,
            "match_stol0.7_ltol0.5_ang15": 0.25,
            "match_stol0.5_ltol0.3_ang10": 1.0,
            "exception_unparseable": -2.0,
        },
    },
    "hyperparams": {
        "learning_rate": 1e-6,
        "per_device_train_batch_size": 12,
        "gradient_accumulation_steps": 1,
        "num_generations": 6,
        "completions_per_step": 12,
        "crystals_per_step": 2,
        "temperature": 1.0,
        "top_p": 0.9,
        "beta_kl": 0.15,
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 0.1,
        "adam_betas": [0.9, 0.99],
        "max_grad_norm": 0.1,
        "max_completion_length": 3072,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "load_in_4bit": False,
        "precision": "bf16",
        "note": "hyperparameters and batch geometry are CrysText's; precision/rank are ours "
                "(they used a 4-bit mistral-7b base with r=16), because this continues our "
                "16-bit r=32 SFT checkpoint",
    },
    "data": {
        "train": "Data/source/mp_52_train.csv.gz (27,380 crystals)",
        "val": "Data/source/mp_52_val.csv.gz (5,000)",
        "test": "Data/source/mp_52_test.csv.gz (8,096)",
    },
    "prompt": {
        "system": "You are an expert in materials science and crystallography. "
                  "Return only one complete CIF file and nothing else.",
        "note": "GRPO adapters were optimised with THIS system message, which differs from the "
                "SFT/eval message used elsewhere in the repo. Prompt this adapter with it.",
    },
    "training": {
        "max_steps": 3000,
        "stopped_at_step": 2184,
        "stop_reason": "held-out match declining monotonically",
        "hardware": "NVIDIA RTX PRO 6000 Blackwell (sm120), torch 2.10.0+cu128",
    },
    "results_best_of_10_full_8096_mpts52_test": {
        "sft_start_point_rank_r32_s3407": {"match": 0.299, "strict_rms_median_A": 0.050},
        "step_150": {"match": 0.298, "strict_rms_median_A": 0.053,
                     "note": "best held-out checkpoint; weights lost (see repo README)"},
        "step_1150": {"match": 0.243, "strict_rms_median_A": 0.108,
                      "note": "weights lost (evicted by save_total_limit)"},
        "step_2150": {"match": None,
                      "note": "THIS adapter — the last surviving checkpoint. NOT evaluated on "
                              "the test set; the step-1150 trend implies it is worse than 24.3%"},
    },
    "code": "https://github.com/shehrozashoaib/LLM_Crystal_CIF (grpo_crystext_reward.py)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=2150)
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ck = EXP / "checkpoints" / f"checkpoint-{args.step}"
    if not ck.is_dir():
        raise SystemExit(f"no checkpoint at {ck}")

    label = f"step{args.step}"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    existing = set(api.list_repo_files(args.repo))
    target = f"{PREFIX}/{label}/model/adapter_model.safetensors"
    if target in existing:
        print(f"[skip] {target} already on {args.repo}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model").mkdir()
        (tmp / "tokenizer").mkdir()
        for f in MODEL_FILES:
            if (ck / f).exists():
                shutil.copy2(ck / f, tmp / "model" / f)
        for f in TOKENIZER_FILES:
            if (ck / f).exists():
                shutil.copy2(ck / f, tmp / "tokenizer" / f)

        cfg = dict(CONFIG, uploaded_checkpoint=f"checkpoint-{args.step}")
        (tmp / "config.json").write_text(json.dumps(cfg, indent=2))

        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(tmp)}  {p.stat().st_size / 1e6:.1f} MB")
        if args.dry_run:
            print("[dry-run] nothing uploaded")
            return

        print(f"[upload] {PREFIX}/{label} -> {args.repo}")
        api.upload_folder(
            repo_id=args.repo,
            folder_path=str(tmp / "model"),
            path_in_repo=f"{PREFIX}/{label}/model",
            commit_message=f"Add GRPO CrysText-reward adapter ({label})",
        )
        api.upload_folder(
            repo_id=args.repo,
            folder_path=str(tmp / "tokenizer"),
            path_in_repo=f"{PREFIX}/{label}/tokenizer",
            commit_message=f"Add GRPO CrysText-reward tokenizer ({label})",
        )
        api.upload_file(
            repo_id=args.repo,
            path_or_fileobj=str(tmp / "config.json"),
            path_in_repo=f"{PREFIX}/config.json",
            commit_message="Add GRPO CrysText-reward run config",
        )
    print("done")


if __name__ == "__main__":
    main()
