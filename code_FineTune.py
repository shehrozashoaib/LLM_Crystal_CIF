#!/usr/bin/env python3
"""
code_FineTune.py — SFT (Unsloth + Qwen2.5-7B-Instruct + LoRA)
=============================================================
Refactored for the controlled experiment sweep (experiment_framework.md):

  * Fully CLI-driven (train/val CSV, run name, LoRA rank, steps...) so one
    script drives every run in the composition / rank / curriculum sweeps.
  * PINNED STEPS: --max_steps is a HARD cap (default 4500). num_train_epochs is
    set high and is NOT the stopping criterion — this removes the step-count
    drift that confounded the published curriculum run (framework §3.3).
  * EARLY STOPPING REMOVED. load_best_model_at_end REMOVED. We keep the model at
    exactly max_steps, so every run is compared at identical optimizer steps.
  * No hardcoded secrets — WANDB_API_KEY is read from the environment only.

The prompt / chat-template formatting (system prompt, user content, EOS) is
kept byte-identical to the original so SFT and inference stay in parity.

Example:
    python code_FineTune.py \
        --train_csv Data/composition_sweep/train_mp20_50.csv \
        --val_csv   Data/composition_sweep/val_mp20_50.csv \
        --run_name  comp_mp20_50 \
        --max_steps 4500
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer
import wandb


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_csv", required=True, help="training CSV (material_id, instruction, input, output)")
    p.add_argument("--val_csv", default="", help="validation CSV; empty => no eval")
    p.add_argument("--run_name", required=True, help="experiment/output dir name + wandb run name")
    p.add_argument("--output_root", default="experiments", help="root dir for outputs")
    p.add_argument("--max_steps", type=int, default=4500, help="HARD cap on optimizer steps (pinned)")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--base_model", default="Qwen2.5-7B-Instruct")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--arch", choices=["auto", "a100", "gh200"], default="auto",
                   help="GPU architecture. Controls the attention backend: gh200 "
                        "(Hopper sm_90, aarch64) pins cuDNN SDPA to avoid the MATH-backend "
                        "OOM (see README_GH200_SETUP.md); a100 (Ampere sm_80) uses the stock "
                        "flash/SDPA path. 'auto' detects from compute capability. Set "
                        "explicitly to avoid running GH200-specific code on an A100.")
    p.add_argument("--init_adapter", default="",
                   help="path to an existing LoRA adapter dir to CONTINUE training "
                        "(curriculum phase 2 / continued-SFT). Empty => fresh LoRA on the "
                        "base model (default). When set, --lora_r/alpha/dropout are inherited "
                        "from the loaded adapter and ignored.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Architecture handling (A100 vs GH200) — see README_GH200_SETUP.md
# ----------------------------------------------------------------------------
def resolve_arch(choice):
    """Map --arch (auto|a100|gh200) to a concrete arch via compute capability."""
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        return "gh200" if major == 9 else "a100"   # 9.x = Hopper (GH200/H100)
    return "a100"


def attention_context(arch):
    """Attention backend per architecture.

    GH200 (sm_90, aarch64) ships no flash-attn/xformers, so Unsloth's SDPA falls
    back to the MATH backend; combined with padding-free masks + Qwen2.5 GQA that
    materializes the full N×N scores (~62 GB at micro_batch=8 -> OOM). Only cuDNN
    handles mask+GQA fused (~2.4 GB), and PyTorch's default priority reaches MATH
    before cuDNN, so we pin cuDNN FIRST with set_priority=True. A100 (sm_80) uses
    its stock flash/SDPA path and needs no pin.
    """
    import contextlib
    if arch == "gh200":
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
                           set_priority=True)
    return contextlib.nullcontext()


def main():
    args = parse_args()
    ARCH = resolve_arch(args.arch)
    _cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "cpu"
    print(f"🖥️  architecture: {ARCH}  (--arch {args.arch}; device capability {_cap})")

    HYPERPARAMS = {
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": False,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
    }

    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")  # no hardcoded secret

    EXPERIMENT_CONFIG = {
        "project": "cif-finetuning",
        "base_model": args.base_model,
        "run_name": args.run_name,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "arch": ARCH,
        "init_adapter": args.init_adapter or None,
        "method": "lora",
        "quantization": "16bit",
        "run_id": args.run_name,
        "hyperparams": HYPERPARAMS,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if torch.cuda.is_available() else 0,
        },
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    # --- Output dirs: {output_root}/{run_name}/... ---------------------------
    EXPERIMENT_DIR = Path(args.output_root) / args.run_name
    CHECKPOINTS_DIR = EXPERIMENT_DIR / "checkpoints"
    FINAL_MODEL_DIR = EXPERIMENT_DIR / "final_model"
    for d in [CHECKPOINTS_DIR, FINAL_MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_DIR / "config.json", "w") as f:
        json.dump(EXPERIMENT_CONFIG, f, indent=2)

    # --- Model ---------------------------------------------------------------
    if args.init_adapter:
        # CONTINUE training an existing LoRA adapter (curriculum phase 2 /
        # continued-SFT). Unsloth loads base+adapter as a trainable PEFT model
        # from the adapter dir (adapter_config.json -> base_model_name_or_path),
        # so the SAME LoRA weights carry forward — this is what makes the
        # forgetting probe meaningful. We must NOT call get_peft_model again.
        print(f"🔁 Continuing from existing adapter: {args.init_adapter}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.init_adapter,
            max_seq_length=HYPERPARAMS["max_seq_length"],
            dtype=None,
            load_in_4bit=HYPERPARAMS["load_in_4bit"],
        )
        FastLanguageModel.for_training(model)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=f"unsloth/{args.base_model}",
            max_seq_length=HYPERPARAMS["max_seq_length"],
            dtype=None,
            load_in_4bit=HYPERPARAMS["load_in_4bit"],
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=HYPERPARAMS["lora_r"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=HYPERPARAMS["lora_alpha"],
            lora_dropout=HYPERPARAMS["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )

    # Tokenizer for formatting (stock Qwen chat template — matches inference)
    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{args.base_model}")
    EOS_TOKEN = tokenizer.eos_token

    # --- Prompt formatting (UNCHANGED — parity with generate_cifs_qwen_chat) --
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            user_content = f"{instruction}\n\n{input}" if input else instruction
            messages = [
                {"role": "system", "content": "You are an expert in materials science and crystallography."},
                {"role": "user", "content": user_content},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            text = text + output + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    # --- Data ----------------------------------------------------------------
    dataset = Dataset.from_pandas(pd.read_csv(args.train_csv))
    dataset = dataset.map(formatting_prompts_func, batched=True)

    use_eval = bool(args.val_csv)
    val_dataset = None
    if use_eval:
        val_dataset = Dataset.from_pandas(pd.read_csv(args.val_csv))
        val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    dataset_size = len(dataset)
    effective_batch_size = (HYPERPARAMS["per_device_train_batch_size"]
                            * HYPERPARAMS["gradient_accumulation_steps"])
    steps_per_epoch = max(1, dataset_size // effective_batch_size)
    # Warmup / save / eval cadence are derived from the PINNED step budget,
    # not from epochs, so they are identical regardless of dataset size.
    warmup_steps = int(args.max_steps * HYPERPARAMS["warmup_ratio"])
    save_steps = max(args.max_steps // 6, 50)

    print("\n📈 Training configuration")
    print(f"  run_name           : {args.run_name}")
    print(f"  train_csv          : {args.train_csv}  ({dataset_size} rows)")
    print(f"  val_csv            : {args.val_csv or '(none — eval disabled)'}")
    print(f"  max_steps (PINNED) : {args.max_steps}")
    print(f"  lora_r / alpha     : {args.lora_r} / {args.lora_alpha}")
    print(f"  effective batch    : {effective_batch_size}")
    print(f"  steps_per_epoch    : {steps_per_epoch}  (~{args.max_steps/steps_per_epoch:.2f} epochs at this size)")
    print(f"  warmup / save      : {warmup_steps} / {save_steps}")
    print(f"  early stopping     : DISABLED   |  load_best_model_at_end: OFF")

    if WANDB_API_KEY:
        wandb.init(
            project=EXPERIMENT_CONFIG["project"],
            name=args.run_name,
            id=args.run_name,
            resume="allow",
            config=EXPERIMENT_CONFIG,
            tags=[args.base_model, "lora", f"lr_{args.learning_rate}", f"r{args.lora_r}"],
        )

    # --- Trainer (trl >= 0.24 API: SFTConfig holds the data/seq fields) ------
    training_args = SFTConfig(
        output_dir=str(CHECKPOINTS_DIR),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        per_device_train_batch_size=HYPERPARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=HYPERPARAMS["per_device_train_batch_size"],
        gradient_accumulation_steps=HYPERPARAMS["gradient_accumulation_steps"],
        eval_strategy="steps" if use_eval else "no",
        eval_steps=save_steps if use_eval else None,
        warmup_steps=warmup_steps,
        num_train_epochs=100,          # high — never binds; max_steps is the limit
        max_steps=args.max_steps,      # ← PINNED hard cap
        learning_rate=HYPERPARAMS["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_torch",
        report_to="wandb" if WANDB_API_KEY else "none",
        weight_decay=HYPERPARAMS["weight_decay"],
        lr_scheduler_type="cosine",
        seed=args.seed,
        gradient_checkpointing=True,
        run_name=args.run_name,
        # data / sequence fields (moved here from SFTTrainer in trl >= 0.24)
        dataset_text_field="text",
        max_length=HYPERPARAMS["max_seq_length"],
        packing=False,
        dataset_num_proc=2,
        # NOTE: no load_best_model_at_end / metric_for_best_model — we keep the
        # model at exactly max_steps for a fair matched-steps comparison.
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,          # was tokenizer= in old trl
        train_dataset=dataset,
        eval_dataset=val_dataset,            # None when eval disabled
        # NO EarlyStoppingCallback — pinned-step training.
        args=training_args,
    )

    print("\n" + "=" * 80 + "\n🚀 STARTING TRAINING (pinned to %d steps)\n" % args.max_steps + "=" * 80)
    try:
        # Resume from the latest checkpoint if one exists for this run.
        resume_ckpt = None
        ckpts = sorted([p for p in CHECKPOINTS_DIR.glob("checkpoint-*") if p.is_dir()],
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            resume_ckpt = str(ckpts[0])
            print(f"Resuming from {resume_ckpt}")
        # Attention backend is chosen by architecture (see attention_context):
        # GH200 pins cuDNN SDPA to avoid the MATH-backend OOM; A100 is a no-op.
        with attention_context(ARCH):
            trainer_stats = trainer.train(resume_from_checkpoint=resume_ckpt) if resume_ckpt else trainer.train()

        metrics = getattr(trainer_stats, "metrics", {}) or {}
        with open(EXPERIMENT_DIR / "training_stats.json", "w") as f:
            json.dump({
                "train_loss": getattr(trainer_stats, "training_loss", None),
                "train_runtime": metrics.get("train_runtime"),
                "train_samples_per_second": metrics.get("train_samples_per_second"),
                "epoch": metrics.get("epoch"),
                "max_steps": args.max_steps,
            }, f, indent=2)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

    print(f"\n💾 Saving final model to {FINAL_MODEL_DIR}")
    model.save_pretrained(FINAL_MODEL_DIR / "model")
    tokenizer.save_pretrained(FINAL_MODEL_DIR / "tokenizer")
    print("✅ Done.")


if __name__ == "__main__":
    main()
