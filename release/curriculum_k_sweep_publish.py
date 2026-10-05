#!/usr/bin/env python3
"""
release/curriculum_k_sweep_publish.py — push k-sweep artifacts the moment they exist
=======================================================================
Companion to training/run_curriculum_k_sweep.sh. Polls experiments/cks<SEED>_k*/
and publishes each artifact once, as soon as it is complete:

  * adapter  -> HF  shehrozashoaib/LLM_Crystal_CIF  <run>/model, <run>/tokenizer,
                <run>/{config,training_stats,DONE}.json      (phase-1 runs too)
  * generation CSV -> GitHub  results/curriculum_k_sweep_s<SEED>/<run>/predicted_cifs.csv.gz
                              (+ gen_meta.json), before grading finishes
  * grading        -> GitHub  results/curriculum_k_sweep_s<SEED>/<run>/validation/
                              (+ train_DONE.json, train_p1_DONE.json)

Other agents push to the same GitHub repo, so pushes go through a dedicated
publish-only clone (PUBLISH_DIR), never the live working tree the sweep is
executing from. Each attempt starts from the latest origin/main, re-applies only
this run's files (all under new paths), commits and pushes; a rejected push is
retried from a fresh fetch. Nothing is ever force-pushed.

Published keys are recorded in experiments/ksweep_publish_state_s<SEED>.json, so
a restart never re-uploads. Exits once every k in the manifest is graded and
pushed.

    SEED=25478 python release/curriculum_k_sweep_publish.py
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

csv.field_size_limit(10 ** 9)

ROOT = Path(__file__).resolve().parent.parent
SEED = int(os.environ["SEED"])
HF_REPO = os.environ.get("HF_REPO", "shehrozashoaib/LLM_Crystal_CIF")
PUB = Path(os.environ.get("PUBLISH_DIR", "/workspace/LLM_Crystal_CIF_publish"))
BRANCH = os.environ.get("PUBLISH_BRANCH", "main")
TEST_N = int(os.environ.get("TEST_N", "8096"))
RET_SEQS = int(os.environ.get("RET_SEQS", "10"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "3072"))
POLL = int(os.environ.get("POLL", "120"))
SETTLE = 60          # seconds an artifact must sit unmodified before it is read
EXP = ROOT / "experiments"
GEN = ROOT / "generated"
RESULTS_REL = Path("results") / f"curriculum_k_sweep_s{SEED}"
STATE = EXP / f"cks_publish_state_s{SEED}.json"
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def log(msg: str) -> None:
    print(f"[publish {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def load_state() -> set:
    return set(json.load(open(STATE))) if STATE.exists() else set()


def save_state(done: set) -> None:
    tmp = STATE.with_suffix(".tmp")
    json.dump(sorted(done), open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def settled(*paths: Path) -> bool:
    now = time.time()
    for p in paths:
        if not p.exists():
            return False
        files = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p]
        if not files or any(now - f.stat().st_mtime < SETTLE for f in files):
            return False
    return True


# ---------------------------------------------------------------- readiness --
def train_ready(run_dir: Path) -> bool:
    """The launcher writes DONE.json only after training exits 0 and the adapter
    is saved; also require this seed and a settled adapter + tokenizer."""
    try:
        done = json.load(open(run_dir / "DONE.json"))
        st = json.load(open(run_dir / "training_stats.json"))
    except Exception:
        return False
    fm = run_dir / "final_model"
    return (done.get("seed") == SEED and st.get("global_step") == done.get("steps")
            and (fm / "model" / "adapter_model.safetensors").exists()
            and (fm / "tokenizer" / "tokenizer_config.json").exists()
            and settled(fm, run_dir / "DONE.json"))


def gen_csv(run: str) -> Path:
    return GEN / run / f"generated_cifs_{run}_{RET_SEQS}seq_maxtok{MAX_NEW_TOKENS}_0_{TEST_N - 1}.csv"


def gen_ready(run: str) -> bool:
    p = gen_csv(run)
    if not settled(p):
        return False
    with open(p, newline="") as fh:
        r = csv.reader(fh)
        next(r, None)
        return sum(1 for _ in r) == TEST_N


def val_dir(run: str) -> Path | None:
    """Collected results/<...>/<run>/validation/, once the launcher has graded THIS
    adapter (VALIDATED_FROM == DONE.json's adapter hash) and copied it over."""
    marker = EXP / run / f"validation_seed{SEED}" / "VALIDATED_FROM"
    try:
        sha = json.load(open(EXP / run / "DONE.json"))["adapter_sha256"]
    except Exception:
        return None
    d = ROOT / RESULTS_REL / run / "validation"
    if not (marker.exists() and marker.read_text().strip() == sha and list(d.glob("*_summary.txt"))):
        return None
    return d if settled(d.parent) else None


# ----------------------------------------------------------------------- HF --
def push_adapter(run: str) -> None:
    d = EXP / run
    ops = []
    for sub in ("model", "tokenizer"):
        for f in sorted((d / "final_model" / sub).rglob("*")):
            if f.is_file():
                ops.append(CommitOperationAdd(f"{run}/{sub}/{f.relative_to(d / 'final_model' / sub)}", str(f)))
    for meta in ("config.json", "training_stats.json", "DONE.json"):
        ops.append(CommitOperationAdd(f"{run}/{meta}", str(d / meta)))
    st = json.load(open(d / "training_stats.json"))
    HfApi().create_commit(
        repo_id=HF_REPO, repo_type="model", operations=ops,
        commit_message=f"{run}: LoRA adapter, {st['global_step']} steps, seed {SEED}")
    log(f"HF  <- {run} ({len(ops)} files)")


# ------------------------------------------------------------------- GitHub --
def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(PUB), *args], check=check, capture_output=True, text=True)


def ensure_clone() -> None:
    if (PUB / ".git").exists():
        return
    url = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"],
                         check=True, capture_output=True, text=True).stdout.strip()
    # Local clone hardlinks objects (fast, no 2 GB re-download); then point at GitHub.
    subprocess.run(["git", "clone", "--quiet", str(ROOT), str(PUB)], check=True)
    git("remote", "set-url", "origin", url)
    git("config", "user.name", "shehrozashoaib")
    git("config", "user.email", "shehrozashoaib@gmail.com")
    git("config", "core.autocrlf", "false")


def git_publish(write, message: str) -> None:
    """write(pub_root) materialises files and returns their repo-relative paths."""
    ensure_clone()
    for attempt in range(1, 11):
        git("fetch", "--quiet", "origin", BRANCH)
        # Publish-only clone: nothing here is ever unpushed except our own
        # previous failed attempt, which write() recreates. Safe to reset.
        git("checkout", "--quiet", "-B", BRANCH, f"origin/{BRANCH}")
        git("reset", "--quiet", "--hard", f"origin/{BRANCH}")
        paths = write(PUB)
        git("add", "--", *map(str, paths))
        if git("diff", "--cached", "--quiet", check=False).returncode == 0:
            log(f"git: nothing new for '{message}' (already upstream)")
            return
        git("commit", "--quiet", "-m", message + TRAILER)
        res = git("push", "--quiet", "origin", f"HEAD:{BRANCH}", check=False)
        if res.returncode == 0:
            log(f"git <- {message}")
            return
        log(f"git push rejected (attempt {attempt}): {res.stderr.strip()[:200]} — refetching")
        time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"could not push '{message}' after 10 attempts")


def write_generation(run: str):
    def w(pub: Path):
        out = pub / RESULTS_REL / run
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "predicted_cifs.csv.gz"
        with open(gen_csv(run), "rb") as src, gzip.GzipFile(dst, "wb", mtime=0) as gz:
            shutil.copyfileobj(src, gz)
        shutil.copy2(GEN / run / "gen_meta.json", out / "gen_meta.json")
        return [dst.relative_to(pub), (out / "gen_meta.json").relative_to(pub)]
    return w


def write_validation(run: str, vdir: Path):
    def w(pub: Path):
        out = pub / RESULTS_REL / run
        if (out / "validation").exists():
            shutil.rmtree(out / "validation")
        shutil.copytree(vdir, out / "validation")
        paths = [(out / "validation").relative_to(pub)]
        # predicted_cifs.csv.gz is NOT re-copied: it was pushed at generation time,
        # and the launcher's own gzip differs only in its header bytes.
        for f in ("train_DONE.json", "train_p1_DONE.json"):
            if (vdir.parent / f).exists():
                shutil.copy2(vdir.parent / f, out / f)
                paths.append((out / f).relative_to(pub))
        return paths
    return w


# --------------------------------------------------------------------- main --
def manifest_runs() -> list[str]:
    m = json.load(open(ROOT / "Data" / f"curriculum_k_sweep_s{SEED}" / "manifest.json"))
    return [f"cks{SEED}_{r['tag']}" for r in m["runs"].values()]


def main() -> None:
    finals = manifest_runs()
    done = load_state()
    log(f"seed {SEED}: watching {finals}; already published: {len(done)} items")
    while True:
        for d in sorted(EXP.glob(f"cks{SEED}_k*")):
            run = d.name
            try:
                if f"hf:{run}" not in done and train_ready(d):
                    push_adapter(run); done.add(f"hf:{run}"); save_state(done)
                if run in finals:
                    if f"gen:{run}" not in done and gen_ready(run):
                        git_publish(write_generation(run), f"{run}: generated CIFs (8,096 x {RET_SEQS}, seed {SEED})")
                        done.add(f"gen:{run}"); save_state(done)
                    vd = val_dir(run) if f"val:{run}" not in done else None
                    if vd is not None:
                        git_publish(write_validation(run, vd), f"{run}: validation panel")
                        done.add(f"val:{run}"); save_state(done)
            except Exception as e:      # network blips etc.: retry next poll
                log(f"ERROR {run}: {type(e).__name__}: {str(e)[:300]}")
        if all(f"val:{r}" in done and f"hf:{r}" in done for r in finals):
            log("all k published"); return
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
