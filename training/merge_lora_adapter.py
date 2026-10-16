#!/usr/bin/env python3
"""
training/merge_lora_adapter.py — fold an SFT LoRA adapter into its base weights
===============================================================================
Used to start GRPO from an SFT checkpoint with that checkpoint as the KL REFERENCE.

Why: with a PEFT policy, TRL/Unsloth GRPO compute the reference log-probs under
`model.disable_adapter()`. If GRPO simply continues the SFT adapter, the "reference"
is therefore the BASE model, not the SFT checkpoint. Merging the SFT adapter into the
weights and training a FRESH LoRA on top (GRPO_START_MODEL=<merged dir>) makes the
adapter-off model — the reference — exactly the frozen SFT checkpoint, and the policy
starts there too (a fresh LoRA's B is zero).

Merge is done in float32 on CPU and cast to bfloat16 once (one rounding, not two).
Writes model + Qwen tokenizer + merge_provenance.json (adapter sha256, base, scale).

    python training/merge_lora_adapter.py \
        --adapter experiments/cks25478_k0000_to2000/checkpoints/checkpoint-2000 \
        --out experiments/merged/cks25478_k0000_step2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (final_model/model or checkpoints/checkpoint-N)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="", help="override base (default: adapter_config base_model_name_or_path)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct", help="tokenizer saved alongside (the SFT one)")
    a = ap.parse_args()

    adapter = Path(a.adapter)
    cfg = json.load(open(adapter / "adapter_config.json"))
    base = a.base or cfg["base_model_name_or_path"]
    sha = hashlib.sha256(open(adapter / "adapter_model.safetensors", "rb").read()).hexdigest()[:16]
    print(f"base={base}  adapter={adapter} (sha {sha})  r={cfg['r']} alpha={cfg['lora_alpha']}")

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model = model.to(torch.bfloat16)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.tokenizer).save_pretrained(out)
    # transformers 4.57.2 bug: loading a tokenizer from a LOCAL dir whose config.json says
    # transformers_version <= 4.57.2 reads the config as a dict and then calls .model_type
    # on it (tokenization_utils_base.py) -> AttributeError, which breaks Unsloth's loader.
    # The field is metadata only; dropping it skips that branch.
    cfg_path = out / "config.json"
    mcfg = json.load(open(cfg_path)); mcfg.pop("transformers_version", None)
    json.dump(mcfg, open(cfg_path, "w"), indent=2)
    json.dump({"base": base, "adapter": str(adapter), "adapter_sha256": sha,
               "lora_r": cfg["r"], "lora_alpha": cfg["lora_alpha"],
               "merge": "fp32 on CPU, cast to bf16 once", "tokenizer": a.tokenizer},
              open(out / "merge_provenance.json", "w"), indent=2)
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
