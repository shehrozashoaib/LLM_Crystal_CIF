"""vLLM generation worker (runs in /venv/vllm). Same engine, prompt format and sampling as
evaluation/generate_cifs_vllm.py; per-request seeds; optional LoRA adapters (several at once)."""
import argparse, json, os, sys
sys.path.insert(0, "/workspace/LLM_Crystal_CIF/evaluation")
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from generate_cifs_vllm import make_prompt_text

ap = argparse.ArgumentParser()
ap.add_argument("--base_model", required=True); ap.add_argument("--tokenizer_dir", required=True)
ap.add_argument("--adapters", default="{}"); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--gpu_mem_util", type=float, default=0.40); ap.add_argument("--temperature", type=float, default=0.6)
ap.add_argument("--top_p", type=float, default=0.9); ap.add_argument("--max_model_len", type=int, default=4096)
ap.add_argument("--eager", type=int, default=1); ap.add_argument("--system_prompt", default="sft"); ap.add_argument("--engine_seed", type=int, default=42)
a = ap.parse_args()
adapters = json.loads(a.adapters)
reqs = [json.loads(l) for l in open(a.inp)]
tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
kw = dict(model=a.base_model, max_model_len=a.max_model_len, dtype="bfloat16",
          gpu_memory_utilization=a.gpu_mem_util, seed=a.engine_seed)
if a.eager:
    kw.update(enforce_eager=True)      # skip CUDA-graph capture: ~2 min faster start-up, same model math
if adapters:
    kw.update(enable_lora=True, max_lora_rank=128, max_loras=max(1, len(adapters)))
llm = LLM(**kw)
lreq = {n: LoRARequest(n, i + 1, lora_path=p) for i, (n, p) in enumerate(adapters.items())}
prompts, sps, lrs = [], [], []
for r in reqs:
    p = make_prompt_text(tok, r["instruction"], r.get("input", ""), system=a.system_prompt)
    plen = len(tok(p, add_special_tokens=False)["input_ids"])
    mt = max(1, min(int(r["max_tokens"]), a.max_model_len - plen - 1))
    prompts.append(p)
    sps.append(SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, max_tokens=mt, seed=int(r["seed"])))
    lrs.append(lreq[r["adapter"]] if r.get("adapter") else None)
outs = llm.generate(prompts, sps, lora_request=(lrs if adapters else None))
tmp = a.out + ".tmp"
with open(tmp, "w") as f:
    for r, p, o in zip(reqs, prompts, outs):
        c = o.outputs[0]
        f.write(json.dumps({"uid": r["uid"], "prompt_text": p, "text": c.text, "finish_reason": c.finish_reason,
                            "n_tokens": len(c.token_ids)}) + "\n")
os.replace(tmp, a.out)
print("GEN_WORKER_DONE", len(reqs))
