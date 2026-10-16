#!/usr/bin/env python3
"""
release/grpo_publish.py — push a GRPO-from-SFT run's artifacts as they appear
=============================================================================
Companion to training/run_grpo_continuous_base.sh started from a merged SFT checkpoint
(training/merge_lora_adapter.py). Publishes each item once, as soon as it is complete:

  * the SFT start adapter (e.g. checkpoint-2000)  -> HF  <SFT_NAME>/model
  * the newest GRPO checkpoint every MIRROR_EVERY steps (adapter + optimizer + scheduler +
    rng + trainer_state, i.e. resumable)            -> HF  <RUN_ID>/checkpoint_latest/ (overwritten)
  * the final GRPO adapter + DONE.json / stats      -> HF  <RUN_ID>/{model,tokenizer,...}
  * the generation CSV (8,096 x 10)                 -> GitHub <RESULTS_DIR>/<RUN_ID>/predicted_cifs.csv.gz
  * the validation panel + traces + train_DONE.json -> GitHub <RESULTS_DIR>/<RUN_ID>/

GitHub pushes reuse release/curriculum_k_sweep_publish.py's publish-only clone (fetch,
re-apply, push, retry; never force). State: experiments/grpo_publish_state_<RUN_ID>.json.

    SEED=25478 RUN_ID=grpo_disc_sft2000_s25478 RESULTS_DIR=results/grpo_discrete_sft2000_s25478 \
    SFT_CKPT=experiments/cks25478_k0000_to2000/checkpoints/checkpoint-2000 SFT_NAME=cks25478_k0000_step2000 \
    python release/grpo_publish.py
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curriculum_k_sweep_publish as kp            # noqa: E402  (git_publish, settled, log, gen rows)

ROOT = kp.ROOT
SEED = kp.SEED
RUN = os.environ["RUN_ID"]
RESULTS_REL = Path(os.environ["RESULTS_DIR"])
SFT_CKPT = Path(os.environ["SFT_CKPT"])
SFT_NAME = os.environ["SFT_NAME"]
MIRROR_EVERY = int(os.environ.get("MIRROR_EVERY", "250"))
EXP = ROOT / "experiments" / RUN
STATE = ROOT / "experiments" / f"grpo_publish_state_{RUN}.json"
SFT_FILES = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json",
             "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
             "trainer_state.json"]


def load_state() -> dict:
    return json.load(open(STATE)) if STATE.exists() else {"done": [], "mirrored_step": 0}


def save_state(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp"); json.dump(st, open(tmp, "w"), indent=1); os.replace(tmp, STATE)


def hf_commit(ops, msg: str) -> None:
    HfApi().create_commit(repo_id=kp.HF_REPO, repo_type="model", operations=ops, commit_message=msg)
    kp.log(f"HF  <- {msg} ({len(ops)} files)")


def ckpt_complete(c: Path) -> bool:
    return all((c / f).exists() for f in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
                                          "trainer_state.json")) and kp.settled(c)


def push_sft(st: dict) -> None:
    if "sft" in st["done"] or not ckpt_complete(SFT_CKPT):
        return
    ops = [CommitOperationAdd(f"{SFT_NAME}/model/{f}", str(SFT_CKPT / f)) for f in SFT_FILES if (SFT_CKPT / f).exists()]
    hf_commit(ops, f"{SFT_NAME}: SFT adapter at {SFT_CKPT.name} (GRPO start point and KL reference, seed {SEED})")
    st["done"].append("sft"); save_state(st)


def mirror_checkpoint(st: dict) -> None:
    cks = [c for c in (EXP / "checkpoints").glob("checkpoint-*") if c.name.split("-")[-1].isdigit()]
    if not cks:
        return
    c = max(cks, key=lambda p: int(p.name.split("-")[-1])); step = int(c.name.split("-")[-1])
    if step % MIRROR_EVERY or step <= st["mirrored_step"] or not ckpt_complete(c):
        return
    ops = [CommitOperationAdd(f"{RUN}/checkpoint_latest/{f.name}", str(f)) for f in sorted(c.iterdir()) if f.is_file()]
    hf_commit(ops, f"{RUN}: resumable checkpoint at step {step}")
    st["mirrored_step"] = step; save_state(st)


def push_final(st: dict) -> None:
    if "hf_final" in st["done"] or not (EXP / "DONE.json").exists():
        return
    fm = EXP / "final_model"
    if not kp.settled(fm, EXP / "DONE.json"):
        return
    ops = []
    for sub in ("model", "tokenizer"):
        for f in sorted((fm / sub).rglob("*")):
            if f.is_file():
                ops.append(CommitOperationAdd(f"{RUN}/{sub}/{f.relative_to(fm / sub)}", str(f)))
    for meta in ("DONE.json", "training_stats.json"):
        ops.append(CommitOperationAdd(f"{RUN}/{meta}", str(EXP / meta)))
    hf_commit(ops, f"{RUN}: final GRPO adapter")
    st["done"].append("hf_final"); save_state(st)


def gen_csv() -> Path:
    return ROOT / "generated" / RUN / f"generated_cifs_{RUN}_{kp.RET_SEQS}seq_maxtok{kp.MAX_NEW_TOKENS}_0_{kp.TEST_N - 1}.csv"


def push_gen(st: dict) -> None:
    if "gen" in st["done"] or not kp.settled(gen_csv()):
        return
    import csv; csv.field_size_limit(10 ** 9)
    with open(gen_csv(), newline="") as fh:
        r = csv.reader(fh); next(r, None)
        if sum(1 for _ in r) != kp.TEST_N:
            return
    def w(pub: Path):
        out = pub / RESULTS_REL / RUN; out.mkdir(parents=True, exist_ok=True)
        with open(gen_csv(), "rb") as src, gzip.GzipFile(out / "predicted_cifs.csv.gz", "wb", mtime=0) as gz:
            shutil.copyfileobj(src, gz)
        shutil.copy2(ROOT / "generated" / RUN / "gen_meta.json", out / "gen_meta.json")
        return [(out / "predicted_cifs.csv.gz").relative_to(pub), (out / "gen_meta.json").relative_to(pub)]
    kp.git_publish(w, f"{RUN}: generated CIFs (8,096 x {kp.RET_SEQS}, seed {SEED})")
    st["done"].append("gen"); save_state(st)


def push_val(st: dict) -> None:
    if "val" in st["done"]:
        return
    try:
        sha = json.load(open(EXP / "DONE.json"))["adapter_sha256"]
    except Exception:
        return
    marker = EXP / f"validation_seed{SEED}" / "VALIDATED_FROM"
    rd = ROOT / RESULTS_REL / RUN
    if not (marker.exists() and marker.read_text().strip() == sha and list((rd / "validation").glob("*_summary.txt"))
            and kp.settled(rd)):
        return
    def w(pub: Path):
        out = pub / RESULTS_REL / RUN; out.mkdir(parents=True, exist_ok=True)
        if (out / "validation").exists():
            shutil.rmtree(out / "validation")
        shutil.copytree(rd / "validation", out / "validation")
        paths = [(out / "validation").relative_to(pub)]
        for f in sorted(rd.iterdir()):            # train_DONE.json, *_trace.jsonl.gz (not the CSV again)
            if f.is_file() and f.name != "predicted_cifs.csv.gz":
                shutil.copy2(f, out / f.name); paths.append((out / f.name).relative_to(pub))
        return paths
    kp.git_publish(w, f"{RUN}: validation panel + reward/val traces")
    st["done"].append("val"); save_state(st)


def main() -> None:
    st = load_state()
    kp.log(f"grpo publisher: run {RUN}, SFT {SFT_CKPT} -> {SFT_NAME}; state {st}")
    while "val" not in st["done"]:
        for step in (push_sft, mirror_checkpoint, push_final, push_gen, push_val):
            try:
                step(st)
            except Exception as e:
                kp.log(f"ERROR {step.__name__}: {type(e).__name__}: {str(e)[:300]}")
        time.sleep(kp.POLL)
    kp.log("all GRPO artifacts published")


if __name__ == "__main__":
    main()
