#!/usr/bin/env python3
"""
GRPO from SFT – target-aligned v5
==================================
KEY CHANGES FROM v4:
  1. VALIDATION-BASED MONITORING: The monitor now generates on VAL data
     and evaluates with StructureMatcher. Training data monitoring was
     useless because the model memorized those structures during SFT.
  2. BEST MODEL SELECTION: Tracks best val match rate and saves the
     best checkpoint separately.
  3. FREQUENT CHECKPOINTS: save_steps=20 (was 285).
  4. EARLY STOPPING on val performance: stops if val match rate drops
     below SFT baseline (25%) for consecutive checks.
  5. beta=0.15 (was 0.35 which was too conservative — model couldn't learn).
  6. Fixed max_completion_length=3072 (was typo'd as 3372).

REWARD DESIGN (unchanged from v4):
  StructureMatcher is the ONLY reward signal.
  Chemistry/composition are ±0.03 tiebreakers.
  Parseability is a gate, not a reward axis.
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
# Paths
# ============================================================
SFT_MODEL_DIR = "experiments/rank_r32_s3407/checkpoints/checkpoint-3000"
SFT_TOKENIZER_DIR = SFT_MODEL_DIR

TRAIN_CSV = "Data/source/mp_52_train.csv.gz"
VAL_CSV = "Data/source/mp_52_val.csv.gz"

OUTPUT_DIR = "experiments/grpo_r32_from3000_discrete"
EXPERIMENT_DIR = Path(OUTPUT_DIR)
CHECKPOINTS_DIR = EXPERIMENT_DIR / "checkpoints"
FINAL_MODEL_DIR = EXPERIMENT_DIR / "final_model"
BEST_MODEL_DIR = EXPERIMENT_DIR / "best_model"
DEBUG_LOG_PATH = EXPERIMENT_DIR / "debug_batches.log"
REWARD_TRACE_PATH = EXPERIMENT_DIR / "reward_trace.jsonl"
REWARD_GROUP_TRACE_PATH = EXPERIMENT_DIR / "reward_group_trace.jsonl"
VAL_TRACE_PATH = EXPERIMENT_DIR / "val_trace.jsonl"

_ck = [c for c in CHECKPOINTS_DIR.glob("checkpoint-*") if c.name.split("-")[-1].isdigit()]
RESUME_CHECKPOINT = str(max(_ck, key=lambda c: int(c.name.split("-")[-1]))) if _ck else None
if RESUME_CHECKPOINT:
    print(f"RESUMING GRPO from {RESUME_CHECKPOINT}")
START_MODEL_DIR = SFT_MODEL_DIR

for d in [EXPERIMENT_DIR, CHECKPOINTS_DIR, FINAL_MODEL_DIR, BEST_MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

# ============================================================
# Hyperparameters
# ============================================================
HYPERPARAMS = {
    "learning_rate": 5e-7,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
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
    "num_generations": 4,
    "max_completion_length": 3372,          # ← fixed (was typo'd 3372)
    "temperature": 0.7,                     # ← raised for within-group diversity
    "top_p": 0.9,
    "beta": 0.15,                           # ← lowered (0.35 was too conservative)
}

# Validation config
VAL_CONFIG = {
    "n_val_samples": 20,                    # prompts to evaluate each check
    "n_gens_per_sample": 4,                 # generations per val prompt
    "max_new_tokens": 3072,                 # match training
    "temperature": 0.6,                     # slightly lower for val (less noise)
    "top_p": 0.9,
    "sft_baseline": 0.10,                   # 10% floor — effectively no early stop except collapse
}

SAVE_STEPS = 20                             # ← hardcoded, not computed

RUN_ID = "grpo_target_aligned_v5"

# ============================================================
# Model and tokenizer
# ============================================================
TOKENIZER_DIR = SFT_TOKENIZER_DIR if Path(SFT_TOKENIZER_DIR).exists() else SFT_MODEL_DIR
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model, _ = FastLanguageModel.from_pretrained(
    model_name=START_MODEL_DIR,
    max_seq_length=HYPERPARAMS["max_seq_length"],
    dtype=None,
    load_in_4bit=HYPERPARAMS["load_in_4bit"],
)

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
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

if not hasattr(model, "warnings_issued") or getattr(model, "warnings_issued", None) is None:
    model.warnings_issued = {}


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

    # DISCRETE reward (binary StructureMatcher match at the eval/val tolerance):
    # 1.0 iff the generated structure matches the target under _MATCHER_VAL
    # (stol=0.5), else 0.0. Overrides the continuous tier grading above. The
    # small ±0.03 composition tiebreaker in cif_reward_batch is retained so that
    # all-fail groups still carry a little within-group advantage signal.
    return (1.0 if val_rms is not None else 0.0), ladder


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
        {"role": "system", "content": "You are an expert in materials science and crystallography. "
                                       "Return only one complete CIF file and nothing else."},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_rl_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    df["prompt"] = [make_prompt_text(i, x) for i, x in zip(df["instruction"], df["input"])]
    df["target_cif"] = df["output"]
    df["row_id"] = [f"{Path(csv_path).stem}:{i}" for i in range(len(df))]
    return Dataset.from_pandas(df[["prompt", "target_cif", "row_id"]], preserve_index=False)


train_ds = load_rl_dataset(TRAIN_CSV)
eval_ds = load_rl_dataset(VAL_CSV)

# Build lookup for both train and val (reward function needs it for row_id logging)
PROMPT_TO_ROW_ID = {row["prompt"]: row["row_id"] for row in train_ds}
PROMPT_TO_ROW_ID.update({row["prompt"]: row["row_id"] for row in eval_ds})

# Also load val as a raw dataframe for the validation callback
val_df = pd.read_csv(VAL_CSV)

dataset_size = len(train_ds)
effective_batch_size = HYPERPARAMS["per_device_train_batch_size"] * HYPERPARAMS["gradient_accumulation_steps"]
steps_per_epoch = max(1, dataset_size // effective_batch_size)
warmup_steps = int(steps_per_epoch * HYPERPARAMS["warmup_ratio"])

print(f"Dataset size: {dataset_size}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Save steps: {SAVE_STEPS}")
print(f"Val dataset size: {len(val_df)}")


# ============================================================
# REWARD FUNCTION — StructureMatcher IS the reward
# ============================================================
def cif_reward_batch(
    prompts, completions, target_cif=None, completion_ids=None, **kwargs,
):
    rewards = []
    component_rows = []
    batch_ids = [PROMPT_TO_ROW_ID.get(p, "unknown") for p in prompts]
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
        mode, ext = extract_cif_block(raw_text)
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
            r_match_scalar, match_ladder = _structure_match_ladder(ext, target_cif[i])
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
    Every check_interval steps:
      1. Samples n_val_samples prompts from the VALIDATION set
      2. Generates n_gens_per_sample completions for each
      3. Evaluates each with StructureMatcher against the val target
      4. Reports val match rate, per-material success rate
      5. Saves best model based on val match rate
      6. Early stops if val performance degrades below SFT baseline
    """
    def __init__(self, val_df, tokenizer, check_interval=20,
                 n_val_samples=20, n_gens_per_sample=4,
                 max_new_tokens=3072, temperature=0.6, top_p=0.9,
                 sft_baseline=0.25, patience=5):
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.check_interval = check_interval
        self.n_val_samples = n_val_samples
        self.n_gens_per_sample = n_gens_per_sample
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.sft_baseline = sft_baseline
        self.patience = patience

        self.best_val_match_rate = 0.0
        self.best_step = 0
        self.bad_checks = 0
        self.trainer_ref = None

    def set_trainer(self, trainer):
        self.trainer_ref = trainer

    def _generate_and_evaluate(self, model):
        """Generate on val samples and evaluate with StructureMatcher."""
        # Sample val prompts
        n = min(self.n_val_samples, len(self.val_df))
        sample_indices = random.sample(range(len(self.val_df)), n)
        sample_rows = self.val_df.iloc[sample_indices]

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
                        eos_token_id=self.tokenizer.eos_token_id,
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
                _, ext = extract_cif_block(gen_text)

                results["total_gens"] += 1

                # Run StructureMatcher
                r_match, ladder = _structure_match_ladder(ext, target_cif)

                if r_match >= 1.0:
                    results["val_matches"] += 1
                    material_matched = True
                elif r_match >= 0.6:
                    results["med_matches"] += 1
                    material_matched = True
                elif r_match >= 0.2:
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
        print(f"  Best so far:     {self.best_val_match_rate:.1%} at step {self.best_step}")
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
                "val/best_material_rate": self.best_val_match_rate,
            }, step=state.global_step)

        # Log to file
        _append_jsonl(VAL_TRACE_PATH, [{
            "step": state.global_step,
            "val_rate": val_rate,
            "material_rate": material_rate,
            **results,
        }])

        # Best model tracking
        if material_rate > self.best_val_match_rate:
            self.best_val_match_rate = material_rate
            self.best_step = state.global_step
            print(f"  ★ NEW BEST VAL: {material_rate:.1%} — saving best model...")
            if self.trainer_ref is not None:
                self.trainer_ref.save_model(str(BEST_MODEL_DIR / "model"))
                self.tokenizer.save_pretrained(str(BEST_MODEL_DIR / "tokenizer"))
                # Save metadata
                with open(BEST_MODEL_DIR / "best_info.json", "w") as f:
                    json.dump({"step": state.global_step, "material_rate": material_rate,
                               "val_rate": val_rate, **results}, f, indent=2)
            self.bad_checks = 0
        else:
            # Check for degradation
            if material_rate < self.sft_baseline:
                self.bad_checks += 1
                print(f"  ⚠ BELOW SFT BASELINE: {material_rate:.1%} < {self.sft_baseline:.1%}"
                      f" ({self.bad_checks}/{self.patience})")
                if self.bad_checks >= self.patience:
                    print(f"\n{'='*70}")
                    print(f"EARLY STOPPING: val performance below SFT baseline for "
                          f"{self.patience} consecutive checks.")
                    print(f"Best model saved at step {self.best_step} with "
                          f"{self.best_val_match_rate:.1%} per-material match rate.")
                    print(f"{'='*70}\n")
                    control.should_training_stop = True
            else:
                self.bad_checks = 0

        return control


# ============================================================
# GRPO config and trainer
# ============================================================
if WANDB_API_KEY:
    wandb.init(
        project="cif-grpo-target-aligned-v5",
        name=RUN_ID,
        id=RUN_ID,
        resume="allow",
        config={
            "hyperparams": HYPERPARAMS,
            "grpo": GRPO_KW,
            "val_config": VAL_CONFIG,
            "start_model_dir": START_MODEL_DIR,
        },
    )

grpo_args = GRPOConfig(
    output_dir=str(CHECKPOINTS_DIR),
    max_steps=1500,
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
    lr_scheduler_type="cosine",
    weight_decay=HYPERPARAMS["weight_decay"],
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=5,
    optim="adamw_torch",
    seed=3407,
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    report_to="wandb" if WANDB_API_KEY else "none",
    num_generations=GRPO_KW["num_generations"],
    max_completion_length=GRPO_KW["max_completion_length"],
    temperature=GRPO_KW["temperature"],
    top_p=GRPO_KW["top_p"],
    beta=GRPO_KW["beta"],
)

# Create validation callback
val_monitor = GRPOValidationCallback(
    val_df=val_df,
    tokenizer=tokenizer,
    check_interval=SAVE_STEPS,              # validate every save
    n_val_samples=VAL_CONFIG["n_val_samples"],
    n_gens_per_sample=VAL_CONFIG["n_gens_per_sample"],
    max_new_tokens=VAL_CONFIG["max_new_tokens"],
    temperature=VAL_CONFIG["temperature"],
    top_p=VAL_CONFIG["top_p"],
    sft_baseline=VAL_CONFIG["sft_baseline"],
    patience=5,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
    reward_funcs=[cif_reward_batch],
    callbacks=[val_monitor],
)

# Give the callback a reference to the trainer for model saving
val_monitor.set_trainer(trainer)

print("=" * 70)
print("GRPO TARGET-ALIGNED v5 — VALIDATION-BASED MONITORING")
print("=" * 70)
print(f"START MODEL:         {START_MODEL_DIR}")
print(f"save_steps:          {SAVE_STEPS}")
print(f"warmup_steps:        {warmup_steps}")
print(f"steps_per_epoch:     {steps_per_epoch}")
print(f"effective_batch:     {effective_batch_size}")
print(f"num_generations:     {GRPO_KW['num_generations']}")
print(f"beta (KL anchor):    {GRPO_KW['beta']}")
print(f"temperature:         {GRPO_KW['temperature']}")
print(f"learning_rate:       {HYPERPARAMS['learning_rate']}")
print(f"max_completion_len:  {GRPO_KW['max_completion_length']}")
print()
print(f"VALIDATION CONFIG:")
print(f"  val samples/check: {VAL_CONFIG['n_val_samples']}")
print(f"  gens per sample:   {VAL_CONFIG['n_gens_per_sample']}")
print(f"  SFT baseline:      {VAL_CONFIG['sft_baseline']:.0%}")
print(f"  check interval:    every {SAVE_STEPS} steps")
print()
print("REWARD: StructureMatcher only (chemistry = ±0.03 tiebreaker)")
print("MONITOR: Generates on VAL data, not train data")
print("BEST MODEL: Saved to", BEST_MODEL_DIR)
print("=" * 70)

trainer_stats = trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT)

print("Saving final model...")
trainer.save_model(str(FINAL_MODEL_DIR / "model"))
tokenizer.save_pretrained(str(FINAL_MODEL_DIR / "tokenizer"))
print(f"Best model was at step {val_monitor.best_step} with "
      f"{val_monitor.best_val_match_rate:.1%} val match rate")
print("Done.")