import os, sys, json, pandas as pd
name, kind, budget = sys.argv[1], sys.argv[2], int(sys.argv[3])
REPO = "/workspace/LLM_Crystal_CIF"; BASE = "/workspace/models/Qwen/Qwen2.5-7B-Instruct"
AD = "/workspace/models/adapters/rank_r32_s3407"
R = f"/workspace/wsp/runs/{name}"
for d in ("work/edited", "practice", "requests"):
    os.makedirs(f"{R}/{d}", exist_ok=True)
cfg = {"kind": kind, "base_model": BASE, "adapter": f"{AD}/model" if kind == "adapter" else None, "gpu_mem_util": 0.40}
json.dump(cfg, open(f"{R}/config.json", "w"), indent=1)
json.dump({"budget": budget, "passes_used": 0, "code_runs": 0}, open(f"{R}/state.json", "w"), indent=1)
def link(src, dst):
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(src, dst)
if kind == "adapter":
    link(f"{AD}/model", f"{R}/model"); link(BASE, f"{R}/base_model"); link(BASE, f"{R}/tokenizer")
else:
    link(BASE, f"{R}/model"); link(BASE, f"{R}/tokenizer")
seen = set(pd.read_csv(f"{REPO}/Data/composition_sweep/train_mp20_00.csv.gz").material_id)
test = set(pd.read_csv(f"{REPO}/Data/source/mp_52_test.csv.gz").material_id)
stats = {}
for split in ("train", "val"):
    d = pd.read_csv(f"{REPO}/Data/source/mp_52_{split}.csv.gz")
    assert not (set(d.material_id) & test) and d.material_id.is_unique
    with open(f"{R}/practice/mpts52_{split}.jsonl", "w") as f:
        for r in d.itertuples():
            src = split if (kind == "base" or split == "val") else ("train_seen_by_adapter" if r.material_id in seen else "train_not_seen_by_adapter")
            stats[src] = stats.get(src, 0) + 1
            f.write(json.dumps({"id": r.material_id, "source": src, "instruction": r.instruction, "input": r.input,
                                "problem": f"{r.instruction}\n\n{r.input}", "answer": r.output}) + "\n")
json.dump(stats, open(f"{R}/practice/stats.json", "w"))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"{R}/tokenizer")
ex = pd.read_csv(f"{REPO}/Data/source/mp_52_val.csv.gz").iloc[0]
p = tok.apply_chat_template([{"role": "system", "content": "You are an expert in materials science and crystallography."},
                             {"role": "user", "content": f"{ex.instruction}\n\n{ex.input}"}], tokenize=False, add_generation_prompt=True)
open(f"{R}/work/eval_prompt_format.txt", "w").write(
    "The exact evaluation prompt (raw text) for one practice problem; the model's answer follows it directly and ends with <|im_end|>.\n"
    "To score a text with a teacher-forced pass, send prompt + answer as one raw text in \"texts\".\n"
    "-----BEGIN PROMPT-----\n" + p + "-----END PROMPT-----\n-----REFERENCE ANSWER (" + ex.material_id + ")-----\n" + ex.output + "\n")
for tool, script in (("run", "wsp_run.py"), ("fwd", "wsp_fwd.py")):
    open(f"{R}/{tool}", "w").write(f"#!/bin/bash\nWSP_RUN={R} exec /venv/py312/bin/python /workspace/wsp/harness/{script} \"$@\"\n")
    os.chmod(f"{R}/{tool}", 0o755)
print(name, kind, stats)
