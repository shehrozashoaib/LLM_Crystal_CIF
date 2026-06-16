#!/usr/bin/env python3
"""
generate_cifs_vllm.py — fast CIF generation with vLLM (drop-in for the HF script)
=================================================================================
Same CLI, same prompt format, and same output CSV columns as
generate_cifs_qwen_chat.py — but uses vLLM (paged attention + continuous
batching) instead of HuggingFace `model.generate`. On the full MPTS-52 test set
(8096 x 10) this turns a multi-day job into a few hours, because vLLM swaps a
finished sequence out the instant it hits EOS instead of locking the whole batch
to the slowest generation.

It loads the base model + the SFT LoRA adapter (no merge needed). The prompt is
built with the SAME chat template as training, so SFT/inference stay in parity.

Output CSV is byte-compatible with what cif_structure_validator_mp52.py expects:
the original test columns (material_id, instruction, input, output) plus
generation_1..N. Filename matches the HF script's convention so the orchestrator
path contract is unchanged.

Example:
    python generate_cifs_vllm.py \
        --model_dir experiments/comp_mp20_50/final_model/model \
        --tokenizer_dir experiments/comp_mp20_50/final_model/tokenizer \
        --csv_path mp_52_test_cifs_description_reduced_withmpids.csv \
        --output_dir generated/comp_mp20_50 \
        --start_ix 0 --stop_ix 8096 --ret_seqs 10 --max_new_tokens 3072
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"   # adapter's base_model_name_or_path


def add_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--model_dir", required=True, help="LoRA adapter dir (final_model/model)")
    p.add_argument("--tokenizer_dir", default="", help="tokenizer dir; defaults to model_dir then base")
    p.add_argument("--base_model", default="", help="override base model (else read from adapter_config.json)")
    p.add_argument("--csv_path", required=True)
    p.add_argument("--output_dir", default="generated_vllm")
    p.add_argument("--start_ix", type=int, default=0)
    p.add_argument("--stop_ix", type=int, default=8096)
    p.add_argument("--ret_seqs", type=int, default=10, help="CIFs per material (generation_1..N)")
    p.add_argument("--max_new_tokens", type=int, default=3072)
    p.add_argument("--max_model_len", type=int, default=4096, help="must match training max_seq_length")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_mem_util", type=float, default=0.90)
    p.add_argument("--chunk", type=int, default=1000, help="prompts per incremental save")
    # accepted for CLI compatibility with the HF script (vLLM batches internally):
    p.add_argument("--batch_size", type=int, default=None, help="ignored (vLLM continuous-batches)")
    return p


def resolve_base_model(model_dir: str, override: str) -> str:
    if override:
        return override
    cfg = Path(model_dir) / "adapter_config.json"
    if cfg.exists():
        import json
        b = json.load(open(cfg)).get("base_model_name_or_path")
        if b:
            return b
    return DEFAULT_BASE_MODEL


def make_prompt_text(tokenizer, instruction: str, inp: str) -> str:
    """Identical to training's formatting (without the appended target)."""
    user_content = f"{instruction}\n\n{inp}" if (inp is not None and str(inp).strip()) else instruction
    messages = [
        {"role": "system", "content": "You are an expert in materials science and crystallography."},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    args = add_args(argparse.ArgumentParser()).parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    base_model = resolve_base_model(args.model_dir, args.base_model)
    tok_dir = args.tokenizer_dir or (args.model_dir if Path(args.model_dir, "tokenizer_config.json").exists() else base_model)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    # ---- Data ----
    df = pd.read_csv(args.csv_path)
    stop_ix = min(args.stop_ix, len(df))
    sub = df.iloc[args.start_ix:stop_ix].reset_index(drop=True)
    for c in ("material_id", "output", "instruction", "input"):
        if c not in df.columns:
            print(f"[warn] input CSV missing column required downstream: {c}")

    print(f"CSV: {args.csv_path}  rows={len(df)}  range=[{args.start_ix},{stop_ix})")
    print(f"base={base_model}  adapter={args.model_dir}")
    print(f"ret_seqs={args.ret_seqs}  max_new_tokens={args.max_new_tokens}  max_model_len={args.max_model_len}")
    print(f"temperature={args.temperature}  top_p={args.top_p}  gpu_mem_util={args.gpu_mem_util}")

    # ---- vLLM engine (base + LoRA) ----
    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=64,                 # >= adapter r (32); safe headroom
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        seed=args.seed,
    )
    lora_req = LoRARequest("sft_adapter", 1, lora_path=args.model_dir)

    sampling = SamplingParams(
        n=args.ret_seqs,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    # ---- Output naming (matches the HF script's convention) ----
    md_parts = Path(args.model_dir).parts
    run_tag = Path(args.model_dir).parent.parent.name if len(md_parts) >= 3 else "model"
    output_name = (f"generated_cifs_{run_tag}_{args.ret_seqs}seq_"
                   f"maxtok{args.max_new_tokens}_{args.start_ix}_{stop_ix - 1}.csv")
    output_path = Path(args.output_dir) / output_name

    # ---- Generate in chunks, save incrementally ----
    rows_out = []
    truncated = 0
    total_gens = 0
    for c0 in range(0, len(sub), args.chunk):
        chunk = sub.iloc[c0:c0 + args.chunk]
        prompts = [make_prompt_text(tokenizer, str(r["instruction"]),
                                    str(r["input"]) if pd.notna(r["input"]) else "")
                   for _, r in chunk.iterrows()]
        outputs = llm.generate(prompts, sampling, lora_request=lora_req)
        for local_i, out in enumerate(outputs):
            row = chunk.iloc[local_i].to_dict()
            row["index"] = args.start_ix + c0 + local_i
            for j, comp in enumerate(out.outputs, start=1):
                row[f"generation_{j}"] = comp.text
                total_gens += 1
                if comp.finish_reason == "length":   # hit max_tokens, no EOS
                    truncated += 1
            rows_out.append(row)
        pd.DataFrame(rows_out).to_csv(output_path, index=False)
        print(f"  saved {len(rows_out)}/{len(sub)} materials -> {output_path}")

    # ---- Summary ----
    final = pd.DataFrame(rows_out)
    gen_cols = [c for c in final.columns if c.startswith("generation_")]
    print("\n--- validator compatibility ---")
    print(f"  material_id col: {'material_id' in final.columns}  output col: {'output' in final.columns}")
    print(f"  generation_* cols: {len(gen_cols)}")
    if total_gens:
        pct = 100.0 * truncated / total_gens
        print(f"  truncated (hit max_tokens, no EOS): {truncated}/{total_gens} ({pct:.1f}%)")
        if pct > 5:
            print("  [warn] >5% truncated — consider raising --max_new_tokens")
    print(f"\nDone -> {output_path}")


if __name__ == "__main__":
    main()
