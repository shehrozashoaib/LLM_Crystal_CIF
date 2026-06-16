#!/usr/bin/env python3
"""
Generate CIFs with the same prompt/data format used in training.

Fixes vs. prior version:
- Uses the stock Qwen2.5-7B-Instruct tokenizer (matches training) instead of a
  tokenizer from an unrelated GRPO experiment.
- Sets tokenizer.padding_side = "left" (required for correct batched generation
  with a causal LM).
- Slices newly generated tokens using the padded input length, which is correct
  under left-padding (the old prompt_lens-based slicing was wrong for batches
  containing prompts of different lengths).
- Explicit truncation max_length so prompts are never silently cut.

Example:
    python generate_cifs_qwen_chat.py \
        --csv_path Data/mp_52_test_cifs_description_reduced_withmpids.csv \
        --batch_size 8 \
        --start_ix 0 \
        --stop_ix 500 \
        --ret_seqs 8
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import AutoTokenizer


DEFAULT_MODEL_DIR = "experiments/grpo_from_repair_mix_sft_target_aligned_v4/final_model/model"
# Use the SAME tokenizer as training (stock Qwen2.5-7B-Instruct), not a tokenizer
# from an unrelated experiment. This guarantees identical chat template, special
# tokens, EOS id, and vocab behavior.
DEFAULT_TOKENIZER_DIR = "experiments/grpo_from_repair_mix_sft_target_aligned_v4/final_model/model"
DEFAULT_CSV_PATH = "Data/mp_52_test_cifs_description_reduced_withmpids.csv"
DEFAULT_OUTPUT_DIR = "generated_cifs_results_part_SFT_corrected"

# Sampling regime
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
 
# Generation budget.
# CIFs for larger unit cells easily exceed 512 tokens. 2048 is a safe default
# that accommodates Z=8 cells with tens of atom_site rows while still fitting
# comfortably under the 4096 max_seq_length used in training.
DEFAULT_MAX_NEW_TOKENS = 3072      # ← NOT 4096. See explanation below.

# Default number of generations per material (validator reads generation_1..N)
DEFAULT_RET_SEQS = 10

MAX_SEQ_LENGTH = 4096              # ← Keep at 4096, matches training
LOAD_IN_4BIT = False               # ← Keep False, matches training
 
 
def add_parser_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tokenizer_dir", type=str, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="At bs=1 no padding is used, which is safest. Raise if GPU memory allows.")
    parser.add_argument("--start_ix", type=int, default=0)
    parser.add_argument("--stop_ix", type=int, default=750)
    parser.add_argument("--ret_seqs", type=int, default=DEFAULT_RET_SEQS,
                        help="Number of CIFs to generate per material (populates generation_1..generation_N).")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Max tokens to generate per CIF. 512 truncates larger cells. Default 2048.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warn_on_truncation", action="store_true",
                        help="If set, print a warning whenever a generation hits the max_new_tokens cap.")
    return parser
 
 
def make_prompt_text(tokenizer, instruction: str, inp: str) -> str:
    """Identical to training's formatting_prompts_func (without the appended target)."""
    user_content = f"{instruction}\n\n{inp}" if (inp is not None and str(inp).strip()) else instruction
    messages = [
        {
            "role": "system",
            "content": "You are an expert in materials science and crystallography.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser = add_parser_args(parser)
    args = parser.parse_args()
 
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
 
    os.makedirs(args.output_dir, exist_ok=True)
 
    # ---- Tokenizer ----------------------------------------------------------
    tokenizer_dir = args.tokenizer_dir
    if tokenizer_dir != DEFAULT_TOKENIZER_DIR and not Path(tokenizer_dir).exists():
        print(f"[warn] Tokenizer dir '{tokenizer_dir}' not found; falling back to '{DEFAULT_TOKENIZER_DIR}'")
        tokenizer_dir = DEFAULT_TOKENIZER_DIR
 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
 
    # CRITICAL for batched generation with a causal LM (bs>1). Harmless at bs=1.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    # ---- Model --------------------------------------------------------------
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
 
    if torch.cuda.is_available():
        model.to("cuda")
 
    # ---- Data ---------------------------------------------------------------
    test_data = pd.read_csv(args.csv_path)
    stop_ix = min(args.stop_ix, len(test_data))
 
    required_input_cols = {"material_id", "output", "instruction", "input"}
    missing = required_input_cols - set(test_data.columns)
    if missing:
        print(f"[warn] Input CSV is missing columns required by the validator: {sorted(missing)}")
 
    print(f"CSV path: {args.csv_path}")
    print(f"Rows in CSV: {len(test_data)}")
    print(f"Processing range: [{args.start_ix}, {stop_ix})")
    print(f"Model dir: {args.model_dir}")
    print(f"Tokenizer dir: {tokenizer_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Return sequences (CIFs per material): {args.ret_seqs}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"padding_side: {tokenizer.padding_side}")
 
    # Leave headroom for new tokens so prompts are never silently truncated.
    prompt_max_length = max(1, MAX_SEQ_LENGTH - args.max_new_tokens)
    if prompt_max_length < 256:
        print(f"[warn] prompt_max_length is only {prompt_max_length} tokens. Consider lowering --max_new_tokens "
              f"or raising MAX_SEQ_LENGTH if your prompts are long.")
 
    generated_data = []
    truncation_hits = 0
    total_generations = 0
 
    output_name = (
        f"generated_cifs_"
        f"{Path(args.model_dir).parent.parent.name if len(Path(args.model_dir).parts) >= 3 else 'model'}_"
        f"{args.ret_seqs}seq_"
        f"maxtok{args.max_new_tokens}_"
        f"{args.start_ix}_{stop_ix - 1}.csv"
    )
    output_path = Path(args.output_dir) / output_name
 
    for i in tqdm(range(args.start_ix, stop_ix, args.batch_size)):
        batch_df = test_data.iloc[i:i + args.batch_size]
        batch_len = len(batch_df)
        if batch_len == 0:
            break
 
        prompts = [
            make_prompt_text(
                tokenizer=tokenizer,
                instruction=str(row["instruction"]),
                inp=str(row["input"]) if pd.notna(row["input"]) else "",
            )
            for _, row in batch_df.iterrows()
        ]
 
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_max_length,
        )
 
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
 
        # With left-padding, every sequence in the batch shares the same padded
        # input length, and newly generated tokens begin exactly at that offset.
        input_len = inputs["input_ids"].shape[1]
 
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.ret_seqs,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
 
        decoded_generations = []
        for seq_idx in range(outputs.shape[0]):
            gen_ids = outputs[seq_idx, input_len:]
 
            # Detect whether this generation stopped naturally (hit EOS) or
            # got cut off by the max_new_tokens budget. If no EOS is present
            # anywhere in gen_ids, the generation was truncated — the model
            # wanted to emit more CIF but wasn't allowed to.
            eos_id = tokenizer.eos_token_id
            hit_eos = (gen_ids == eos_id).any().item() if eos_id is not None else True
            if not hit_eos:
                truncation_hits += 1
                if args.warn_on_truncation:
                    print(f"[warn] Generation at batch offset seq_idx={seq_idx} hit max_new_tokens "
                          f"({args.max_new_tokens}) without emitting EOS — CIF likely truncated.")
            total_generations += 1
 
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            decoded_generations.append(text)
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
        # model.generate with num_return_sequences=N returns sequences grouped
        # by sample: [s0_g0, s0_g1, ..., s0_g{N-1}, s1_g0, s1_g1, ...].
        for batch_ix in range(batch_len):
            row_data = batch_df.iloc[batch_ix].to_dict()
            row_data["index"] = i + batch_ix
 
            start = batch_ix * args.ret_seqs
            end = start + args.ret_seqs
            batch_generations = decoded_generations[start:end]
 
            for gen_num, gen_text in enumerate(batch_generations, start=1):
                row_data[f"generation_{gen_num}"] = gen_text
 
            generated_data.append(copy.deepcopy(row_data))
 
        generated_df = pd.DataFrame(generated_data)
        generated_df.to_csv(output_path, index=False)
 
    # Final summary
    final_df = pd.DataFrame(generated_data)
    gen_cols = [c for c in final_df.columns if c.startswith("generation_")]
    print("\n--- Validator compatibility check ---")
    print(f"  has 'material_id' column: {'material_id' in final_df.columns}")
    print(f"  has 'output' (ground truth) column: {'output' in final_df.columns}")
    print(f"  generation_* columns found: {len(gen_cols)} -> {gen_cols}")
 
    print("\n--- Generation truncation report ---")
    if total_generations > 0:
        pct = 100.0 * truncation_hits / total_generations
        print(f"  Truncated (hit max_new_tokens without EOS): {truncation_hits}/{total_generations} ({pct:.1f}%)")
        if pct > 5:
            print(f"  [warn] >5% of generations were truncated. Consider raising --max_new_tokens.")
    print(f"\nDone. Final output written to: {output_path}")
 
 
if __name__ == "__main__":
    main()