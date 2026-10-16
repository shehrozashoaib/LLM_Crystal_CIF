#!/usr/bin/env python3
"""
training/grpo_continuous_base.py — GRPO with the CONTINUOUS reward, from the BASE model
========================================================================================
Planned run (seed 123456): fresh LoRA (r32/a64/dropout 0.05, 7 proj modules — same as SFT)
on unsloth/Qwen2.5-7B-Instruct, NO SFT. 10 completions per prompt, 2 prompts per optimizer
step (pdbs 1 x grad_accum 20 = 20 completions), pinned 4,500 steps, lr 5e-6, beta 0.15,
temperature 0.7, completions <= 3,072 tokens (= eval max_new_tokens). Prompts: the 24,000
MPTS-52 crystals of Data/curriculum_k_sweep_s123456/train_mp52_k0000.csv.gz (the SFT k=0 set).

Built from training/grpo_crystext_reward.py (infrastructure: env config, smoke mode, Qwen
tokenizer guard, binary val monitor) with the reward of training/grpo_r32_from3000_continuous.py.
Fixes vs. those scripts (see the audit in the commit message):
  * no early stopping and no best-model swap — the final model is exactly step MAX_STEPS
    (the old monitor would have stopped a base-model run at ~step 100);
  * one SEED everywhere: LoRA init, GRPOConfig seed + data_seed, val subset;
  * the reward scores the RAW completion, exactly what the eval validator parses
    (the old reward stripped ```cif fences/prose, so a fenced CIF earned +1.03 in training
    and failed at eval);
  * max_completion_length 3072 (the old continuous run really used 3372);
  * row ids come from the dataset (TRL passes them per completion), not a prompt->row map
    that collapsed the 412 duplicate prompts and val/train prompt collisions;
  * warmup = 5% of MAX_STEPS; loss settings passed explicitly; batch geometry asserted
    (Unsloth silently rewrites pdbs if it is not divisible by num_generations);
  * the val monitor is logging-only, on a fixed seeded subset, every VAL_EVERY steps.
"""
from __future__ import annotations

import os
import math
import re
import json
import hashlib
import random
from functools import lru_cache
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
import wandb

# ============================================================
# Environment and runtime toggles
# ============================================================
DEBUG_SYNC = os.environ.get("GRPO_DEBUG_SYNC", "0") == "1"
FORCE_MATH_SDPA = os.environ.get("GRPO_FORCE_MATH_SDPA", "0") == "1"
if DEBUG_SYNC:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

try:
    import transformers.utils.hub as _tf_hub
    if not hasattr(_tf_hub, "TRANSFORMERS_CACHE"):
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            _tf_hub.TRANSFORMERS_CACHE = HF_HUB_CACHE
        except Exception:
            _tf_hub.TRANSFORMERS_CACHE = os.path.expanduser(
                "~/.cache/huggingface/transformers"
            )
except Exception:
    pass

# ============================================================
# Paths and run identity
# ============================================================
SEED = int(os.environ.get("GRPO_SEED", "123456"))
START_MODEL_DIR = os.environ.get("GRPO_START_MODEL", "unsloth/Qwen2.5-7B-Instruct")
# Optional: continue an SFT LoRA adapter instead of a fresh LoRA on the base model.
START_ADAPTER = os.environ.get("GRPO_START_ADAPTER", "")
# KL reference: "base" = adapter disabled (base Qwen; TRL/Unsloth default for PEFT),
# "start" = a frozen copy of the starting policy (base + START_ADAPTER). For a base
# start the two are identical; for an SFT start only "start" anchors GRPO to the SFT model.
REF_MODE = os.environ.get("GRPO_REF", "start" if START_ADAPTER else "base")
if START_ADAPTER:
    if not (Path(START_ADAPTER) / "adapter_model.safetensors").exists():
        raise SystemExit(f"GRPO_START_ADAPTER has no adapter_model.safetensors: {START_ADAPTER}")
    _acfg = json.load(open(Path(START_ADAPTER) / "adapter_config.json"))
    if _acfg.get("base_model_name_or_path") != START_MODEL_DIR:
        raise SystemExit(f"adapter base {_acfg.get('base_model_name_or_path')} != {START_MODEL_DIR}")
assert REF_MODE in ("base", "start"), REF_MODE
SFT_MODEL_DIR = START_MODEL_DIR            # name kept for the helpers below
TOKENIZER_DIR = os.environ.get("GRPO_START_TOKENIZER", START_MODEL_DIR)

TRAIN_CSV = os.environ.get("GRPO_TRAIN_CSV", f"Data/curriculum_k_sweep_s{SEED}/train_mp52_k0000.csv.gz")
VAL_CSV = "Data/source/mp_52_val.csv.gz"   # monitor only — never trained on

RUN_ID = os.environ.get("GRPO_RUN_ID", f"grpo_cont_base_s{SEED}")
OUTPUT_DIR = os.environ.get("GRPO_OUTPUT_DIR", f"experiments/{RUN_ID}")
EXPERIMENT_DIR = Path(OUTPUT_DIR)
CHECKPOINTS_DIR = EXPERIMENT_DIR / "checkpoints"
FINAL_MODEL_DIR = EXPERIMENT_DIR / "final_model"
DEBUG_LOG_PATH = EXPERIMENT_DIR / "debug_batches.log"
REWARD_TRACE_PATH = EXPERIMENT_DIR / "reward_trace.jsonl"
REWARD_GROUP_TRACE_PATH = EXPERIMENT_DIR / "reward_group_trace.jsonl"
VAL_TRACE_PATH = EXPERIMENT_DIR / "val_trace.jsonl"

_ck = [c for c in CHECKPOINTS_DIR.glob("checkpoint-*") if c.name.split("-")[-1].isdigit()]
RESUME_CHECKPOINT = str(max(_ck, key=lambda c: int(c.name.split("-")[-1]))) if _ck else None
if RESUME_CHECKPOINT:
    print(f"RESUMING GRPO from {RESUME_CHECKPOINT}")

for d in [EXPERIMENT_DIR, CHECKPOINTS_DIR, FINAL_MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

# ============================================================
# Hyperparameters
# ============================================================
HYPERPARAMS = {
    "learning_rate": float(os.environ.get("GRPO_LR", "5e-6")),   # fresh LoRA: 10x the old 5e-7
    "per_device_train_batch_size": 1,     # counted in COMPLETIONS by TRL
    "gradient_accumulation_steps": 20,    # 20 completions/step = 2 prompts x 10 generations
    "num_train_epochs": 1,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "max_seq_length": 4096,
    "load_in_4bit": False,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
}

GRPO_KW = {
    "num_generations": 10,
    "max_completion_length": 3072,        # == eval max_new_tokens
    "temperature": 0.7,
    "top_p": 0.9,
    "beta": 0.15,
}

OPTIM_KW = {
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "max_grad_norm": 1.0,
}

# Explicit so a change of Unsloth/TRL defaults cannot silently change the objective.
LOSS_KW = {
    "loss_type": "bnpo",
    "scale_rewards": "group",
    "mask_truncated_completions": False,  # truncated CIFs are failures at eval too
    "epsilon": 0.2,
    "num_iterations": 1,
}

# Logging-only val monitor: fixed seeded subset, every VAL_EVERY steps.
VAL_CONFIG = {
    "n_val_samples": 20,
    "n_gens_per_sample": 4,
    "max_new_tokens": 3072,
    "temperature": 0.6,
    "top_p": 0.9,
}

SAVE_STEPS = int(os.environ.get("GRPO_SAVE_STEPS", "50"))
MAX_STEPS = int(os.environ.get("GRPO_MAX_STEPS", "4500"))
VAL_EVERY = int(os.environ.get("GRPO_VAL_EVERY", "250"))

HYPERPARAMS["per_device_train_batch_size"] = int(
    os.environ.get("GRPO_PDBS", HYPERPARAMS["per_device_train_batch_size"]))
HYPERPARAMS["gradient_accumulation_steps"] = int(
    os.environ.get("GRPO_GRAD_ACCUM", HYPERPARAMS["gradient_accumulation_steps"]))
GRPO_KW["num_generations"] = int(os.environ.get("GRPO_NUM_GEN", GRPO_KW["num_generations"]))

# GRPO_SMOKE=1 — few-step end-to-end wiring check (works on the 7B or a small model).
SMOKE = os.environ.get("GRPO_SMOKE", "0") == "1"
SMOKE_ROWS = int(os.environ.get("GRPO_SMOKE_ROWS", "8"))
if SMOKE:
    HYPERPARAMS["gradient_accumulation_steps"] = int(os.environ.get("GRPO_GRAD_ACCUM", "4"))
    GRPO_KW["num_generations"] = int(os.environ.get("GRPO_NUM_GEN", "2"))
    GRPO_KW["max_completion_length"] = int(os.environ.get("GRPO_SMOKE_MAXLEN", "512"))
    VAL_CONFIG.update(n_val_samples=2, n_gens_per_sample=2, max_new_tokens=GRPO_KW["max_completion_length"])
    SAVE_STEPS = int(os.environ.get("GRPO_SAVE_STEPS", "2"))
    MAX_STEPS = int(os.environ.get("GRPO_MAX_STEPS", "2"))
    VAL_EVERY = int(os.environ.get("GRPO_VAL_EVERY", "2"))
    print(f"[config] SMOKE MODE — {MAX_STEPS} steps on {SMOKE_ROWS} rows")

# ============================================================
# Model and tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)


def _configure_qwen_tokenizer(tok):
    """Qwen2.5-Instruct specifics.

    CrysText ran a Mistral base model and simply stopped on `tokenizer.eos_token`.
    Qwen2.5-Instruct is ChatML: turns end with <|im_end|>, and the tokenizer
    ships a *separate* pad token <|endoftext|>. Three things must hold or GRPO
    silently mis-scores:
      * eos = <|im_end|>, so generation stops at the end of the assistant turn
        instead of running to max_completion_length (which would clip CIFs and
        hand the reward function truncated text);
      * pad != eos, so padding is not confused with a real end-of-turn;
      * left padding, because we batch-generate from prompts.
    """
    vocab = tok.get_vocab()
    im_end = vocab.get("<|im_end|>")
    endoftext = vocab.get("<|endoftext|>")

    if im_end is not None and tok.eos_token_id != im_end:
        tok.eos_token = "<|im_end|>"
        print("[tokenizer] eos_token -> <|im_end|> (ChatML end-of-turn)")

    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>" if endoftext is not None else tok.eos_token
        print(f"[tokenizer] pad_token -> {tok.pad_token}")
    elif tok.pad_token_id == tok.eos_token_id and endoftext is not None:
        tok.pad_token = "<|endoftext|>"
        print("[tokenizer] pad_token was eos; moved to <|endoftext|>")

    tok.padding_side = "left"

    if tok.chat_template is None:
        raise SystemExit(
            "Tokenizer has no chat template. This trainer prompts Qwen2.5 in "
            "ChatML; re-save the tokenizer from the SFT run or point "
            "GRPO_START_TOKENIZER at Qwen/Qwen2.5-7B-Instruct."
        )

    print(f"[tokenizer] eos={tok.eos_token}({tok.eos_token_id}) "
          f"pad={tok.pad_token}({tok.pad_token_id}) padding_side={tok.padding_side}")
    return tok


tokenizer = _configure_qwen_tokenizer(tokenizer)

# Stop generation on either ChatML terminator (the SFT data ends turns with
# <|im_end|>; a stray <|endoftext|> should also terminate).
EOS_TOKEN_IDS = sorted({i for i in [tokenizer.eos_token_id,
                                    tokenizer.get_vocab().get("<|im_end|>"),
                                    tokenizer.get_vocab().get("<|endoftext|>")]
                        if i is not None})

model, _ = FastLanguageModel.from_pretrained(
    model_name=START_ADAPTER or START_MODEL_DIR,
    max_seq_length=HYPERPARAMS["max_seq_length"],
    dtype=None,
    load_in_4bit=HYPERPARAMS["load_in_4bit"],
)

if START_ADAPTER:
    # Unsloth loaded base + the SFT adapter as a PEFT model; continue THAT adapter.
    FastLanguageModel.for_training(model)
    print(f"[start] continuing SFT adapter {START_ADAPTER}")
else:
    model = FastLanguageModel.get_peft_model(
        model,
        r=HYPERPARAMS["lora_r"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=HYPERPARAMS["lora_alpha"],
        lora_dropout=HYPERPARAMS["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

if not hasattr(model, "warnings_issued") or getattr(model, "warnings_issued", None) is None:
    model.warnings_issued = {}

_n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert _n_train > 0, "no trainable parameters — the LoRA adapter is frozen"
print(f"[start] trainable parameters: {_n_train:,}")

ref_model = None
if REF_MODE == "start" and START_ADAPTER:
    # Frozen copy of the starting policy, loaded through Unsloth like the policy
    # (Unsloth's log-prob path expects its patched models).
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name=START_ADAPTER,
        max_seq_length=HYPERPARAMS["max_seq_length"],
        dtype=None,
        load_in_4bit=HYPERPARAMS["load_in_4bit"],
    )
    ref_model.eval()
    for _p in ref_model.parameters():
        _p.requires_grad_(False)
    print(f"[ref] KL reference = frozen {START_ADAPTER}")


# ============================================================
# Constants & helpers
# ============================================================
VALID_ELEMENTS = {
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S",
    "Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga",
    "Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd",
    "Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm",
    "Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os",
    "Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa",
    "U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
}

CELL_PATTERNS = {
    "a":     re.compile(r"_cell_length_a\s+([0-9eE+\-.]+)"),
    "b":     re.compile(r"_cell_length_b\s+([0-9eE+\-.]+)"),
    "c":     re.compile(r"_cell_length_c\s+([0-9eE+\-.]+)"),
    "alpha": re.compile(r"_cell_angle_alpha\s+([0-9eE+\-.]+)"),
    "beta":  re.compile(r"_cell_angle_beta\s+([0-9eE+\-.]+)"),
    "gamma": re.compile(r"_cell_angle_gamma\s+([0-9eE+\-.]+)"),
}

ATOM_HEADERS = [
    "_atom_site_type_symbol", "_atom_site_label",
    "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
]


def _soft_clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def strip_code_fences(text: str) -> str:
    t = text or ""
    t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
    t = re.sub(r"\n```$", "", t)
    return t.strip()

def extract_cif_block(text: str):
    t = strip_code_fences(text)
    if not t:
        return "none", ""
    starts = []
    for pat in [r"(^|\n)(data_[^\n]*)", r"(^|\n)(_cell_length_a[^\n]*)", r"(^|\n)(loop_[^\n]*)"]:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            starts.append(m.start(2))
    if not starts:
        return "raw", t
    start = min(starts)
    t2 = t[start:].strip()
    end = len(t2)
    for pat in [r"\n```", r"\n(?:assistant|user|system)\s*:", r"\nHere is", r"\nExplanation:"]:
        m = re.search(pat, t2, flags=re.IGNORECASE)
        if m:
            end = min(end, m.start())
    return "extracted", t2[:end].strip()

def _extract_atom_site_block(cif_text: str):
    lines = [line.rstrip() for line in (cif_text or "").splitlines()]
    for idx, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        headers = []
        j = idx + 1
        while j < len(lines) and lines[j].strip().startswith("_"):
            headers.append(lines[j].strip())
            j += 1
        if "_atom_site_type_symbol" not in headers:
            continue
        rows = []
        while j < len(lines):
            raw = lines[j].strip()
            if not raw or raw == "loop_" or raw.startswith("data_") or raw.startswith("_"):
                break
            rows.append(raw.split())
            j += 1
        return headers, rows
    return [], []

def _parse_formula_sum(cif_text: str):
    pats = [
        r"_chemical_formula_sum\s+'([^']+)'",
        r'_chemical_formula_sum\s+"([^"]+)"',
        r"_chemical_formula_sum\s+([^\n]+)",
    ]
    for pat in pats:
        m = re.search(pat, cif_text or "")
        if m:
            raw = m.group(1).strip()
            tokens = re.findall(r"([A-Z][a-z]?)([0-9.]+)?", raw)
            out = Counter()
            for sym, amt in tokens:
                out[sym] += float(amt) if amt else 1.0
            return out
    return Counter()

def _reduced_comp_key(counter: Counter):
    if not counter:
        return tuple()
    vals = [int(round(v * 1000)) for v in counter.values() if v > 0]
    if not vals:
        return tuple()
    g = vals[0]
    for v in vals[1:]:
        g = math.gcd(g, v)
    g = max(g, 1)
    return tuple(sorted((k, int(round(v * 1000)) // g) for k, v in counter.items() if v > 0))

def _build_lattice_matrix_torch(a, b, c, alpha, beta, gamma, device):
    ar, br, gr = [math.radians(x) for x in (alpha, beta, gamma)]
    va = torch.tensor([a, 0.0, 0.0], dtype=torch.float32, device=device)
    vb = torch.tensor([b * math.cos(gr), b * math.sin(gr), 0.0], dtype=torch.float32, device=device)
    cx = c * math.cos(br)
    cy_num = math.cos(ar) - math.cos(br) * math.cos(gr)
    cy_den = max(1e-8, math.sin(gr))
    cy = c * cy_num / cy_den
    cz_sq = max(c * c - cx * cx - cy * cy, 0.0)
    vc = torch.tensor([cx, cy, math.sqrt(cz_sq)], dtype=torch.float32, device=device)
    return torch.stack([va, vb, vc], dim=0)

def _distance_margin_gpu(frac, lattice, dist_cutoff, dist_margin):
    if frac.shape[0] <= 1:
        return 0.0
    delta = frac[:, None, :] - frac[None, :, :]
    delta = delta - torch.round(delta)
    cart = torch.matmul(delta, lattice)
    dist = torch.linalg.norm(cart, dim=-1)
    eye = torch.eye(dist.shape[0], device=dist.device, dtype=dist.dtype) * (dist_cutoff + 10.0)
    dmin = float((dist + eye).min().item())
    return _soft_clip01((dmin - dist_cutoff) / max(1e-8, dist_margin))

def _volume_margin_gpu(lattice, min_volume, vol_margin):
    vol = float(torch.abs(torch.det(lattice)).item())
    return _soft_clip01((vol - min_volume) / max(1e-8, vol_margin))

def _scaffold_score(ext: str):
    """LOGGING ONLY."""
    has_data = 1.0 if "data_" in (ext or "").lower() else 0.0
    n_cell = sum(1 for x in ["_cell_length_a","_cell_length_b","_cell_length_c",
                              "_cell_angle_alpha","_cell_angle_beta","_cell_angle_gamma"]
                 if x in (ext or "").lower())
    n_headers = sum(1 for x in ATOM_HEADERS if x in (ext or "").lower())
    n_rows = len(_extract_atom_site_block(ext)[1])
    vis = sum(1 for ch in (ext or "") if not ch.isspace()) / max(1, len(ext or "x"))
    s_nonempty = _soft_clip01(vis / 0.4)
    return {
        "visible_ratio": vis, "atom_rows_found": n_rows, "has_data": has_data,
        "s_scaffold": _soft_clip01(
            0.15*s_nonempty + 0.20*has_data + 0.30*_soft_clip01(n_cell/6.0)
            + 0.20*_soft_clip01(n_headers/5.0) + 0.15*_soft_clip01(n_rows/4.0)),
    }

def _completion_token_length(completion_ids):
    if completion_ids is None:
        return 0
    if hasattr(completion_ids, "shape") and len(completion_ids.shape) > 0:
        return int(completion_ids.shape[-1])
    if isinstance(completion_ids, (list, tuple)):
        return len(completion_ids)
    return 0


@lru_cache(maxsize=32768)
def _parse_target_cif_cached(cif_text: str):
    return _parse_cif_lightweight(cif_text)

def _parse_cif_lightweight(cif_text: str):
    if not cif_text:
        return None
    cell = {}
    try:
        for key, pat in CELL_PATTERNS.items():
            m = pat.search(cif_text)
            if not m:
                return None
            cell[key] = float(m.group(1))
        headers, rows = _extract_atom_site_block(cif_text)
        if not headers or not rows:
            return None
        idx = {h: i for i, h in enumerate(headers)}
        req = ["_atom_site_type_symbol","_atom_site_fract_x","_atom_site_fract_y","_atom_site_fract_z"]
        if any(r not in idx for r in req):
            return None
        occ_idx = idx.get("_atom_site_occupancy")
        species, frac, comp = [], [], Counter()
        occ_over_one = occ_nonpositive = 0
        for row in rows:
            if len(row) < len(headers):
                continue
            sym = row[idx["_atom_site_type_symbol"]]
            if sym not in VALID_ELEMENTS:
                return None
            occupancy = float(row[occ_idx]) if occ_idx is not None and occ_idx < len(row) else 1.0
            if occupancy > 1.0 + 1e-6: occ_over_one += 1
            if occupancy <= 1e-8: occ_nonpositive += 1
            xyz = [float(row[idx[f"_atom_site_fract_{c}"]]) for c in "xyz"]
            species.append(sym); frac.append(xyz); comp[sym] += occupancy
        if not frac:
            return None
        return {"cell": cell, "species": species, "frac": frac, "comp": comp,
                "comp_key": _reduced_comp_key(comp),
                "formula_key": _reduced_comp_key(_parse_formula_sum(cif_text)),
                "n_sites": len(frac), "occ_over_one": occ_over_one,
                "occ_nonpositive": occ_nonpositive}
    except Exception:
        return None


# ------------------------------------------------------------------
# Target-comparison helpers (logging + tiny tiebreaker)
# ------------------------------------------------------------------
def _normalized_counter(c):
    tot = sum(float(v) for v in c.values())
    return {k: float(v)/tot for k,v in c.items()} if tot > 0 else {}

def _exact_comp_score(g, t):
    return 1.0 if _reduced_comp_key(g) == _reduced_comp_key(t) else 0.0

def _element_set_score(g, t):
    gs = set(k for k,v in g.items() if v>0); ts = set(k for k,v in t.items() if v>0)
    inter = len(gs & ts); union = len(gs | ts)
    return inter/union if union else 0.0

def _stoich_score(g, t):
    gn, tn = _normalized_counter(g), _normalized_counter(t)
    keys = sorted(set(gn)|set(tn))
    if not keys: return 0.0
    return _soft_clip01(1.0 - 0.5*sum(abs(gn.get(k,0)-tn.get(k,0)) for k in keys))

def _space_group_number(cif_text: str):
    try:
        s = Structure.from_str(cif_text, fmt="cif")
        spg = s.get_space_group_info()
        return spg[1] if spg else None
    except Exception:
        return None

def _lattice_proxy_score(gc, tc):
    diffs = [abs(float(gc[k])-float(tc[k]))/max(1e-6,abs(float(tc[k])))
             for k in ["a","b","c","alpha","beta","gamma"]]
    return _soft_clip01(1.0 - sum(diffs)/max(1,len(diffs)))

def _site_count_score(gn, tn):
    return _soft_clip01(1.0 - min(1.0, abs(gn-tn)/max(1,tn)))


# ------------------------------------------------------------------
# StructureMatcher ladder — THE reward signal
# ------------------------------------------------------------------
_MATCHER_LOOSE = StructureMatcher(stol=0.7, ltol=0.5, angle_tol=15)
_MATCHER_MED   = StructureMatcher(stol=0.6, ltol=0.4, angle_tol=12)
_MATCHER_VAL   = StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10)

def _structure_match_ladder(gen_cif: str, tgt_cif: str):
    """
    Returns (scalar_reward, ladder_dict).
    scalar_reward is continuous in [-0.10, 1.00]:
      - val tier match:   0.85–1.00 (RMS-graded within tier)
      - med tier match:   0.55–0.70
      - loose tier match: 0.25–0.35
      - no match, but loose-RMS computable: -0.05 to +0.20
        (closer = less penalty — gives GRPO signal on "almost" cases)
      - fully unmatched:  -0.10
    """
    try:
        g = Structure.from_str(gen_cif, fmt="cif")
        t = Structure.from_str(tgt_cif, fmt="cif")
    except Exception:
        return -0.10, {"loose": 0.0, "med": 0.0, "val": 0.0, "rms": None}

    # Try val first (tightest)
    try:
        val_rms = _MATCHER_VAL.get_rms_dist(g, t)
    except Exception:
        val_rms = None

    try:
        med_rms = _MATCHER_MED.get_rms_dist(g, t)
    except Exception:
        med_rms = None

    try:
        loose_rms = _MATCHER_LOOSE.get_rms_dist(g, t)
    except Exception:
        loose_rms = None

    def _rms_bonus(rms_tuple, rms_ceiling):
        """Map RMS to a bonus in [0, 1]. Lower RMS = higher bonus."""
        if rms_tuple is None:
            return 0.0
        rms = rms_tuple[0] if isinstance(rms_tuple, tuple) else float(rms_tuple)
        # rms=0 → bonus=1, rms=ceiling → bonus=0
        return _soft_clip01(1.0 - (float(rms) / max(1e-8, rms_ceiling)))

    ladder = {
        "loose": 1.0 if loose_rms is not None else 0.0,
        "med":   1.0 if med_rms is not None else 0.0,
        "val":   1.0 if val_rms is not None else 0.0,
        "rms":   float(val_rms[0]) if val_rms is not None
                 else (float(med_rms[0]) if med_rms is not None
                 else (float(loose_rms[0]) if loose_rms is not None else None)),
    }

    # MONITOR ONLY (the training reward is _continuous_match_ladder below): binary
    # match at the eval tolerance (_MATCHER_VAL, stol 0.5), as the eval scores it.
    return (1.0 if val_rms is not None else 0.0), ladder


# ==================================================================
# CONTINUOUS reward ladder (training reward) — verbatim from
# training/grpo_r32_from3000_continuous.py; the binary ladder above is the monitor.
# ==================================================================
def _continuous_match_ladder(gen_cif: str, tgt_cif: str):
    """
    Returns (scalar_reward, ladder_dict).
    scalar_reward is continuous in [-0.10, 1.00]:
      - val tier match:   0.85–1.00 (RMS-graded within tier)
      - med tier match:   0.55–0.70
      - loose tier match: 0.25–0.35
      - no match, but loose-RMS computable: -0.05 to +0.20
        (closer = less penalty — gives GRPO signal on "almost" cases)
      - fully unmatched:  -0.10
    """
    try:
        g = Structure.from_str(gen_cif, fmt="cif")
        t = Structure.from_str(tgt_cif, fmt="cif")
    except Exception:
        return -0.10, {"loose": 0.0, "med": 0.0, "val": 0.0, "rms": None}

    # Try val first (tightest)
    try:
        val_rms = _MATCHER_VAL.get_rms_dist(g, t)
    except Exception:
        val_rms = None

    try:
        med_rms = _MATCHER_MED.get_rms_dist(g, t)
    except Exception:
        med_rms = None

    try:
        loose_rms = _MATCHER_LOOSE.get_rms_dist(g, t)
    except Exception:
        loose_rms = None

    def _rms_bonus(rms_tuple, rms_ceiling):
        """Map RMS to a bonus in [0, 1]. Lower RMS = higher bonus."""
        if rms_tuple is None:
            return 0.0
        rms = rms_tuple[0] if isinstance(rms_tuple, tuple) else float(rms_tuple)
        # rms=0 → bonus=1, rms=ceiling → bonus=0
        return _soft_clip01(1.0 - (float(rms) / max(1e-8, rms_ceiling)))

    ladder = {
        "loose": 1.0 if loose_rms is not None else 0.0,
        "med":   1.0 if med_rms is not None else 0.0,
        "val":   1.0 if val_rms is not None else 0.0,
        "rms":   float(val_rms[0]) if val_rms is not None
                 else (float(med_rms[0]) if med_rms is not None
                 else (float(loose_rms[0]) if loose_rms is not None else None)),
    }

    # Tier-graded continuous reward
    if val_rms is not None:
        # Validator match: 0.85 base + up to 0.15 for tight RMS
        bonus = _rms_bonus(val_rms, 0.5)  # val stol = 0.5
        return 0.85 + 0.15 * bonus, ladder

    if med_rms is not None:
        # Medium match: 0.55 base + up to 0.15 for tight RMS
        bonus = _rms_bonus(med_rms, 0.6)
        return 0.55 + 0.15 * bonus, ladder

    if loose_rms is not None:
        # Loose match: 0.25 base + up to 0.10 for tight RMS
        bonus = _rms_bonus(loose_rms, 0.7)
        return 0.25 + 0.10 * bonus, ladder

    # No match at any tolerance — try the loosest possible matcher for
    # a "how close were you" signal. Use a very permissive matcher.
    try:
        very_loose = StructureMatcher(stol=1.5, ltol=1.0, angle_tol=25)
        close_rms = very_loose.get_rms_dist(g, t)
        if close_rms is not None:
            # Partial credit for being in the ballpark: maps to [-0.05, +0.20]
            bonus = _rms_bonus(close_rms, 1.5)
            return -0.05 + 0.25 * bonus, {**ladder, "close_rms": float(close_rms[0])}
    except Exception:
        pass

    return -0.10, ladder


# ------------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------------
def _append_jsonl(path, records):
    if not records:
        return
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def _safe_std(values):
    return float(np.std(np.asarray(values, dtype=float))) if values else 0.0


# ============================================================
# Dataset construction
# ============================================================
def make_prompt_text(instruction: str, inp: str) -> str:
    user_content = (f"{instruction}\n\n{inp}" if (inp is not None and str(inp).strip())
                    else instruction)
    messages = [
        {"role": "system", "content": "You are an expert in materials science and crystallography."},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_rl_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    df["prompt"] = [make_prompt_text(i, x) for i, x in zip(df["instruction"], df["input"])]
    df["target_cif"] = df["output"]
    # TRL passes every extra column per completion, so the reward sees each
    # completion's OWN row (duplicate prompts with different targets stay apart).
    df["row_id"] = df["material_id"].astype(str)
    return Dataset.from_pandas(df[["prompt", "target_cif", "row_id"]], preserve_index=False)


train_ds = load_rl_dataset(TRAIN_CSV)
if SMOKE:
    train_ds = train_ds.select(range(min(SMOKE_ROWS, len(train_ds))))

# Fixed, seeded val subset for the logging-only monitor.
val_df = pd.read_csv(VAL_CSV)
val_df = val_df.sample(n=min(VAL_CONFIG["n_val_samples"], len(val_df)), random_state=SEED).reset_index(drop=True)

dataset_size = len(train_ds)
effective_batch_size = HYPERPARAMS["per_device_train_batch_size"] * HYPERPARAMS["gradient_accumulation_steps"]
prompts_per_step = effective_batch_size // GRPO_KW["num_generations"]
warmup_steps = int(round(HYPERPARAMS["warmup_ratio"] * MAX_STEPS))

print(f"Train prompts: {dataset_size}  ({TRAIN_CSV})")
print(f"Per step: {effective_batch_size} completions = {prompts_per_step} prompts x {GRPO_KW['num_generations']}")
print(f"Prompts seen over {MAX_STEPS} steps: {prompts_per_step * MAX_STEPS} ({prompts_per_step * MAX_STEPS / dataset_size:.2f} epochs)")
print(f"Warmup steps: {warmup_steps}   Save every: {SAVE_STEPS}   Val every: {VAL_EVERY}")


# ============================================================
# REWARD FUNCTION — continuous ladder on the RAW completion
# ============================================================
def cif_reward_batch(
    prompts, completions, target_cif=None, row_id=None, completion_ids=None, **kwargs,
):
    rewards = []
    component_rows = []
    batch_ids = list(row_id) if row_id is not None else ["unknown"] * len(completions)
    batch_hash = hashlib.sha1("||".join(batch_ids).encode()).hexdigest()[:12]
    max_cl = int(kwargs.get("max_completion_length", GRPO_KW["max_completion_length"]))

    # normalise completion texts
    completion_texts = []
    for i, c in enumerate(completions):
        text = ""
        if isinstance(c, str): text = c
        elif isinstance(c, dict): text = c.get("content","") or c.get("text","") or ""
        elif isinstance(c, list):
            parts = []
            for item in c:
                if isinstance(item, str): parts.append(item)
                elif isinstance(item, dict): parts.append(item.get("content","") or item.get("text","") or "")
            text = "".join(parts).strip()
        if (not text) and completion_ids is not None and i < len(completion_ids):
            try:
                ids = completion_ids[i]
                if hasattr(ids, "tolist"): ids = ids.tolist()
                text = tokenizer.decode(ids, skip_special_tokens=True)
            except Exception: pass
        completion_texts.append(text or "")

    for i, raw_text in enumerate(completion_texts):
        row_id = batch_ids[i] if i < len(batch_ids) else "unknown"
        token_len = (_completion_token_length(completion_ids[i])
                     if completion_ids is not None and i < len(completion_ids) else 0)
        is_clipped = token_len >= max_cl > 0
        # Score EXACTLY what the eval validator parses: the raw completion.
        mode, ext = "raw", raw_text.strip()
        scaffold = _scaffold_score(ext)

        extracted_parse_ok = 0.0
        gen_data = None
        parse_error = None

        try:
            gen_data = _parse_cif_lightweight(ext)
            if gen_data is None: raise ValueError("parse failed")
            extracted_parse_ok = 1.0
        except Exception as e:
            parse_error = repr(e)

        # NOT PARSEABLE
        if extracted_parse_ok == 0.0:
            r_total = -0.50
            rewards.append(float(r_total))
            component_rows.append({
                "batch_hash": batch_hash, "row_id": row_id, "sample_idx": i,
                "token_len": int(token_len), "is_clipped": bool(is_clipped),
                "parse_ok": 0.0, "parse_error": parse_error, "extraction_mode": mode,
                "s_scaffold": scaffold["s_scaffold"],
                "s_comp_exact": 0.0, "s_elem": 0.0, "s_stoich": 0.0,
                "s_spg": 0.0, "s_lattice": 0.0, "s_sites": 0.0,
                "match_loose": 0.0, "match_med": 0.0, "match_val": 0.0,
                "match_tier": "not_parseable", "r_match": 0.0, "tiebreaker": 0.0,
                "final_reward": float(r_total),
                "raw_completion_preview": raw_text[:400], "extracted_cif_preview": ext[:400],
                "match_rms": None,  # unparseable CIF -> no match (fixes UnboundLocalError)
            })
            continue

        # PARSEABLE → StructureMatcher
        r_match_scalar = 0.0
        match_ladder = {"loose": 0.0, "med": 0.0, "val": 0.0}
        s_comp_exact = s_elem = s_stoich = s_spg = s_lattice = s_sites = 0.0

        if target_cif is not None and i < len(target_cif):
            tgt = _parse_target_cif_cached(target_cif[i])
            r_match_scalar, match_ladder = _continuous_match_ladder(ext, target_cif[i])
            if tgt is not None:
                s_comp_exact = _exact_comp_score(gen_data["comp"], tgt["comp"])
                s_elem = _element_set_score(gen_data["comp"], tgt["comp"])
                s_stoich = _stoich_score(gen_data["comp"], tgt["comp"])
                s_lattice = _lattice_proxy_score(gen_data["cell"], tgt["cell"])
                s_sites = _site_count_score(gen_data["n_sites"], tgt["n_sites"])
                gen_spg = _space_group_number(ext)
                tgt_spg = _space_group_number(target_cif[i])
                if gen_spg is not None and tgt_spg is not None and int(gen_spg)==int(tgt_spg):
                    s_spg = 1.0

        # Reward assembly
        r_base = r_match_scalar
        if   r_match_scalar >= 0.85: match_tier = "val"
        elif r_match_scalar >= 0.55: match_tier = "med"
        elif r_match_scalar >= 0.25: match_tier = "loose"
        elif r_match_scalar >= -0.05: match_tier = "close"   # new tier
        else:                        match_tier = "no_match"

        tiebreaker = 0.03 * (0.30*s_comp_exact + 0.15*s_elem + 0.15*s_stoich
                             + 0.15*s_spg + 0.15*s_lattice + 0.10*s_sites)
        r_total = _clip(r_base + tiebreaker, -0.50, 1.05)
        rewards.append(float(r_total))

        component_rows.append({
            "batch_hash": batch_hash, "row_id": row_id, "sample_idx": i,
            "token_len": int(token_len), "is_clipped": bool(is_clipped),
            "parse_ok": extracted_parse_ok, "parse_error": parse_error,
            "extraction_mode": mode, "s_scaffold": scaffold["s_scaffold"],
            "s_comp_exact": s_comp_exact, "s_elem": s_elem, "s_stoich": s_stoich,
            "s_spg": s_spg, "s_lattice": s_lattice, "s_sites": s_sites,
            "match_loose": match_ladder["loose"], "match_med": match_ladder["med"],
            "match_val": match_ladder["val"], "match_tier": match_tier,
            "r_match": r_base, "tiebreaker": tiebreaker, "final_reward": float(r_total),
            "raw_completion_preview": raw_text[:400], "extracted_cif_preview": ext[:400],
        })

    _append_jsonl(REWARD_TRACE_PATH, component_rows)

    # group-level traces (training data — kept for debugging only)
    grouped = {}
    for row in component_rows:
        grouped.setdefault(row["row_id"], []).append(row)
    group_rows = []
    for row_id, rows in grouped.items():
        finals = [r["final_reward"] for r in rows]
        tiers = [r["match_tier"] for r in rows]
        group_rows.append({
            "batch_hash": batch_hash, "row_id": row_id,
            "num_samples": len(rows),
            "reward_mean": float(np.mean(finals)) if finals else 0.0,
            "reward_std": _safe_std(finals),
            "parse_mean": float(np.mean([r["parse_ok"] for r in rows])),
            "match_val_mean": float(np.mean([r["match_val"] for r in rows])),
            "n_val_match": sum(1 for t in tiers if t == "val"),
            "n_no_match": sum(1 for t in tiers if t == "no_match"),
            "n_not_parseable": sum(1 for t in tiers if t == "not_parseable"),
            "has_match_variance": len(set(tiers)) > 1,
        })
    _append_jsonl(REWARD_GROUP_TRACE_PATH, group_rows)
    return rewards


# ============================================================
# ★★★ VALIDATION CALLBACK — generates on VAL data ★★★
# ============================================================
from transformers import TrainerCallback

class GRPOValidationCallback(TrainerCallback):
    """
    LOGGING ONLY. Every check_interval steps: generate n_gens_per_sample completions
    for each prompt of a FIXED seeded val subset and grade them with StructureMatcher
    at the eval tolerance (stol 0.5 / ltol 0.3 / angle 10) on the RAW text, as the
    eval does. Never stops training and never replaces the final model.
    """
    def __init__(self, val_df, tokenizer, check_interval=250,
                 n_val_samples=20, n_gens_per_sample=4,
                 max_new_tokens=3072, temperature=0.6, top_p=0.9):
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.check_interval = check_interval
        self.n_val_samples = n_val_samples
        self.n_gens_per_sample = n_gens_per_sample
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.trainer_ref = None

    def set_trainer(self, trainer):
        self.trainer_ref = trainer

    def _generate_and_evaluate(self, model):
        """Generate on val samples and evaluate with StructureMatcher."""
        # Sample val prompts
        n = min(self.n_val_samples, len(self.val_df))
        sample_rows = self.val_df.head(n)          # fixed, seeded subset (see val_df)

        results = {
            "total_gens": 0, "val_matches": 0, "med_matches": 0,
            "loose_matches": 0, "no_matches": 0, "not_parseable": 0,
            "materials_with_match": 0, "total_materials": n,
        }

        device = next(model.parameters()).device

        was_training = model.training
        model.eval()

        for _, row in sample_rows.iterrows():
            prompt_text = make_prompt_text(str(row["instruction"]),
                                           str(row["input"]) if pd.notna(row.get("input")) else "")
            target_cif = str(row["output"])

            # Tokenize prompt
            inputs = self.tokenizer(
                prompt_text, return_tensors="pt", truncation=True,
                max_length=max(1, HYPERPARAMS["max_seq_length"] - self.max_new_tokens),
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            material_matched = False

            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        num_return_sequences=self.n_gens_per_sample,
                        use_cache=True,
                        eos_token_id=EOS_TOKEN_IDS,   # ChatML: <|im_end|> + <|endoftext|>
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                except Exception as e:
                    print(f"  [VAL] Generation failed: {e}")
                    results["not_parseable"] += self.n_gens_per_sample
                    results["total_gens"] += self.n_gens_per_sample
                    continue

            for seq_idx in range(outputs.shape[0]):
                gen_ids = outputs[seq_idx, input_len:]
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                ext = gen_text                      # RAW text, exactly as the eval parses it
                results["total_gens"] += 1
                results["fenced"] = results.get("fenced", 0) + int("```" in gen_text)

                # Binary ladder: 1.0 iff matched at the eval tolerance; ladder flags the looser tiers.
                r_match, ladder = _structure_match_ladder(ext, target_cif)

                if r_match >= 1.0:
                    results["val_matches"] += 1
                    material_matched = True          # same criterion as the eval's best-of-N
                elif ladder.get("med"):
                    results["med_matches"] += 1
                elif ladder.get("loose"):
                    results["loose_matches"] += 1
                else:
                    # Check if parseable
                    if _parse_cif_lightweight(ext) is None:
                        results["not_parseable"] += 1
                    else:
                        results["no_matches"] += 1

            if material_matched:
                results["materials_with_match"] += 1

            # Free memory
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if was_training:
            model.train()

        return results

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.check_interval != 0 or state.global_step == 0:
            return

        model = kwargs.get("model")
        if model is None and self.trainer_ref is not None:
            model = self.trainer_ref.model
        if model is None:
            print(f"\n[VAL step {state.global_step}] Could not access model, skipping validation")
            return

        print(f"\n{'='*60}")
        print(f"[VAL step {state.global_step}] Running validation on {self.n_val_samples} val samples...")
        print(f"{'='*60}")

        results = self._generate_and_evaluate(model)

        total = results["total_gens"]
        val_rate = results["val_matches"] / max(1, total)
        material_rate = results["materials_with_match"] / max(1, results["total_materials"])

        print(f"[VAL step {state.global_step}] VALIDATION RESULTS (on unseen val data):")
        print(f"  Per-generation:  val={results['val_matches']}/{total} ({val_rate:.1%})"
              f"  med={results['med_matches']}/{total}"
              f"  loose={results['loose_matches']}/{total}"
              f"  no_match={results['no_matches']}/{total}"
              f"  not_parseable={results['not_parseable']}/{total}")
        print(f"  Per-material:    {results['materials_with_match']}/{results['total_materials']}"
              f" ({material_rate:.1%}) have at least one match")
        print(f"{'='*60}")

        # Log to WandB
        if WANDB_API_KEY:
            wandb.log({
                "val/step": state.global_step,
                "val/per_gen_match_rate": val_rate,
                "val/per_material_match_rate": material_rate,
                "val/val_matches": results["val_matches"],
                "val/med_matches": results["med_matches"],
                "val/loose_matches": results["loose_matches"],
                "val/no_matches": results["no_matches"],
                "val/not_parseable": results["not_parseable"],
                "val/total_gens": total,
                "val/materials_with_match": results["materials_with_match"],
            }, step=state.global_step)

        # Log to file
        _append_jsonl(VAL_TRACE_PATH, [{
            "step": state.global_step,
            "val_rate": val_rate,
            "material_rate": material_rate,
            **results,
        }])

        return control


# ============================================================
# GRPO config and trainer
# ============================================================
if WANDB_API_KEY:
    wandb.init(
        project="cif-grpo-continuous-base",
        name=RUN_ID,
        id=RUN_ID,
        resume="allow",
        config={"hyperparams": HYPERPARAMS, "grpo": GRPO_KW, "loss": LOSS_KW, "optim": OPTIM_KW,
                "val_config": VAL_CONFIG, "start_model": START_MODEL_DIR, "train_csv": TRAIN_CSV,
                "seed": SEED, "max_steps": MAX_STEPS, "start_adapter": START_ADAPTER, "ref": REF_MODE},
    )

grpo_args = GRPOConfig(
    output_dir=str(CHECKPOINTS_DIR),
    max_steps=MAX_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    per_device_train_batch_size=HYPERPARAMS["per_device_train_batch_size"],
    per_device_eval_batch_size=HYPERPARAMS["per_device_train_batch_size"],
    gradient_accumulation_steps=HYPERPARAMS["gradient_accumulation_steps"],
    eval_strategy="no",
    num_train_epochs=HYPERPARAMS["num_train_epochs"],
    learning_rate=HYPERPARAMS["learning_rate"],
    warmup_steps=warmup_steps,
    lr_scheduler_type=OPTIM_KW["lr_scheduler_type"],
    weight_decay=HYPERPARAMS["weight_decay"],
    adam_beta1=OPTIM_KW["adam_beta1"],
    adam_beta2=OPTIM_KW["adam_beta2"],
    max_grad_norm=OPTIM_KW["max_grad_norm"],
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=5,
    optim=OPTIM_KW["optim"],
    seed=SEED,
    data_seed=SEED,
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    report_to="wandb" if WANDB_API_KEY else "none",
    num_generations=GRPO_KW["num_generations"],
    max_completion_length=GRPO_KW["max_completion_length"],
    temperature=GRPO_KW["temperature"],
    top_p=GRPO_KW["top_p"],
    beta=GRPO_KW["beta"],
    **LOSS_KW,
)

# Guard the geometry: Unsloth silently resets pdbs to num_generations when
# pdbs*grad_accum is not divisible by it, which would change the experiment.
assert grpo_args.per_device_train_batch_size == HYPERPARAMS["per_device_train_batch_size"], grpo_args.per_device_train_batch_size
assert grpo_args.gradient_accumulation_steps == HYPERPARAMS["gradient_accumulation_steps"]
assert grpo_args.generation_batch_size == effective_batch_size, grpo_args.generation_batch_size
assert grpo_args.generation_batch_size % grpo_args.num_generations == 0
assert grpo_args.max_completion_length == GRPO_KW["max_completion_length"]
for k, v in LOSS_KW.items():
    assert getattr(grpo_args, k) == v, (k, getattr(grpo_args, k), v)
assert tokenizer.eos_token == "<|im_end|>", tokenizer.eos_token
assert grpo_args.seed == SEED and grpo_args.data_seed == SEED

val_monitor = GRPOValidationCallback(
    val_df=val_df,
    tokenizer=tokenizer,
    check_interval=VAL_EVERY,
    n_val_samples=VAL_CONFIG["n_val_samples"],
    n_gens_per_sample=VAL_CONFIG["n_gens_per_sample"],
    max_new_tokens=VAL_CONFIG["max_new_tokens"],
    temperature=VAL_CONFIG["temperature"],
    top_p=VAL_CONFIG["top_p"],
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
    reward_funcs=[cif_reward_batch],
    callbacks=[val_monitor],
)
val_monitor.set_trainer(trainer)

if ref_model is not None:
    # TRL/Unsloth set ref_model=None for PEFT models and fall back to disable_adapter()
    # (= base Qwen). With ref_model set, reference log-probs come from this frozen copy.
    trainer.ref_model = ref_model
assert (trainer.ref_model is not None) == (REF_MODE == "start" and bool(START_ADAPTER))

print("=" * 70)
print("GRPO — CONTINUOUS reward, from the BASE model (fresh LoRA)")
print("=" * 70)
print(f"run id / output:     {RUN_ID} -> {EXPERIMENT_DIR}")
print(f"start model:         {START_MODEL_DIR}   (LoRA r{HYPERPARAMS['lora_r']} a{HYPERPARAMS['lora_alpha']})")
print(f"start adapter:       {START_ADAPTER or '(fresh LoRA)'}")
print(f"KL reference:        {'frozen start model' if trainer.ref_model is not None else 'base (adapter disabled)'}")
print(f"seed:                {SEED}")
print(f"train prompts:       {TRAIN_CSV} ({dataset_size})")
print(f"max_steps:           {MAX_STEPS}   save every {SAVE_STEPS}   val every {VAL_EVERY}")
print(f"per step:            {effective_batch_size} completions = {prompts_per_step} prompts x {GRPO_KW['num_generations']}")
print(f"lr / warmup / sched: {HYPERPARAMS['learning_rate']} / {warmup_steps} / {OPTIM_KW['lr_scheduler_type']}")
print(f"beta / temperature:  {GRPO_KW['beta']} / {GRPO_KW['temperature']}  top_p {GRPO_KW['top_p']}")
print(f"max_completion_len:  {GRPO_KW['max_completion_length']}")
print(f"loss:                {LOSS_KW}")
print("REWARD: continuous ladder on the RAW completion; -0.50 unparseable, -0.10 no match,")
print("        [-0.05,0.20] very loose, 0.25-0.35 loose, 0.55-0.70 med, 0.85-1.00 eval-tolerance")
print("        match, + <=0.03 composition/symmetry tiebreak")
print("=" * 70)

trainer_stats = trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT)

if trainer.state.global_step != MAX_STEPS:
    raise SystemExit(f"trained {trainer.state.global_step} steps, expected {MAX_STEPS}")
print("Saving final model (step %d)..." % trainer.state.global_step)
trainer.save_model(str(FINAL_MODEL_DIR / "model"))
tokenizer.save_pretrained(str(FINAL_MODEL_DIR / "tokenizer"))
with open(EXPERIMENT_DIR / "training_stats.json", "w") as f:
    json.dump({"global_step": trainer.state.global_step, "max_steps": MAX_STEPS, "seed": SEED,
               "start_model": START_MODEL_DIR, "start_adapter": START_ADAPTER or None, "ref": REF_MODE,
               "learning_rate": HYPERPARAMS["learning_rate"],
               "train_runtime": (getattr(trainer_stats, "metrics", {}) or {}).get("train_runtime"),
               "resumed_from": RESUME_CHECKPOINT}, f, indent=2)
print("Done.")
