"""Teacher-forced reads with HF transformers (runs in /venv/py312, GPU). No generation, no gradients.
Layer convention: layer l in 1..28 is the block whose tensors are model.layers.{l-1}.*; hidden state l is the
residual stream after layer l (0 = embeddings); mlp of layer l = input of model.layers.{l-1}.mlp.down_proj."""
import argparse, json, os, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", required=True); ap.add_argument("--adapter", default="")
ap.add_argument("--tokenizer_dir", required=True); ap.add_argument("--job", required=True)
a = ap.parse_args()
job = json.load(open(a.job))
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
want_attn = bool(job.get("attn_layers"))
kw = dict(torch_dtype=torch.bfloat16, device_map="cuda")
if want_attn:
    kw["attn_implementation"] = "eager"
model = AutoModelForCausalLM.from_pretrained(a.model_dir, **kw)
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)     # unmerged LoRA, as vLLM applies it
model.eval()
NL = 28
mods = {n: m for n, m in model.named_modules()}
def down_proj(l):
    c = [m for n, m in mods.items() if n.endswith(f"layers.{l - 1}.mlp.down_proj")]
    assert len(c) == 1, (l, len(c)); return c[0]

out = job["out"]; os.makedirs(out, exist_ok=True)
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
PAD = tok.pad_token_id if tok.pad_token_id is not None else 151643


def run_batches(seqs, budget, mlp_layers, need_logits, attn):
    """yield (indices, lengths, outputs, mlp dict) for length-sorted padded batches"""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    cap = {}
    hooks = [down_proj(l).register_forward_pre_hook(lambda m, inp, l=l: cap.__setitem__(l, inp[0])) for l in mlp_layers]
    i = 0
    while i < len(order):
        j = i; L = 0
        while j < len(order) and max(L, len(seqs[order[j]])) * (j - i + 1) <= max(budget, len(seqs[order[j]])):
            L = max(L, len(seqs[order[j]])); j += 1
        idx = order[i:j]; i = j
        ids = torch.full((len(idx), L), PAD, dtype=torch.long); am = torch.zeros((len(idx), L), dtype=torch.long)
        for r, q in enumerate(idx):
            ids[r, :len(seqs[q])] = torch.tensor(seqs[q]); am[r, :len(seqs[q])] = 1
        cap.clear()
        o = model(input_ids=ids.cuda(), attention_mask=am.cuda(), output_hidden_states=True,
                  output_attentions=attn, use_cache=False)
        yield idx, [len(seqs[q]) for q in idx], ids, o, dict(cap)
        del o
    for h in hooks:
        h.remove()


if job["type"] == "teacher_forced":
    seqs = [enc(t) for t in job["texts"]]
    N, L = len(seqs), max(len(s) for s in seqs)
    hs, ml, at, k = job.get("hidden_states", []), job.get("mlp_act", []), job.get("attn_layers", []), int(job.get("topk", 0))
    lp = bool(job.get("token_logprobs", False))
    mm = lambda name, shape, dt: np.lib.format.open_memmap(os.path.join(out, name), mode="w+", dtype=dt, shape=shape)
    T = mm("tokens.npy", (N, L), np.int32); T[:] = -1
    for q, s in enumerate(seqs):
        T[q, :len(s)] = s
    LP = mm("token_logprobs.npy", (N, L), np.float32) if lp else None
    if lp: LP[:] = np.nan
    KI = mm("topk_ids.npy", (N, L, k), np.int32) if k else None
    KL = mm("topk_logprobs.npy", (N, L, k), np.float16) if k else None
    H = {l: mm(f"hidden_L{l}.npy", (N, L, 3584), np.float16) for l in hs}
    M = {l: mm(f"mlp_L{l}.npy", (N, L, 18944), np.float16) for l in ml}
    A = {l: mm(f"attn_L{l}.npy", (N, 28, L, L), np.float16) for l in at}
    budget = 6000 if (lp or k) else 16000
    for idx, lens, ids, o, cap in run_batches(seqs, budget, ml, lp or k, bool(at)):
        if lp or k:
            lsm = o.logits.float().log_softmax(-1)
            if lp:   # token_logprobs[i, t] = log p(token t | tokens < t); position 0 has no prediction
                g = lsm[:, :-1].gather(-1, ids[:, 1:].cuda().unsqueeze(-1)).squeeze(-1).cpu().numpy()
            if k:    # topk at position t = the model's prediction for token t+1
                tv, ti = lsm.topk(k, -1); tv, ti = tv.cpu().numpy(), ti.cpu().numpy()
            del lsm
        for r, q in enumerate(idx):
            n = lens[r]
            if lp: LP[q, 1:n] = g[r, :n - 1]
            if k: KI[q, :n] = ti[r, :n]; KL[q, :n] = tv[r, :n]
            for l in hs: H[l][q, :n] = o.hidden_states[l][r, :n].float().cpu().numpy()
            for l in ml: M[l][q, :n] = cap[l][r, :n].float().cpu().numpy()
            for l in at: A[l][q, :, :n, :n] = o.attentions[l - 1][r, :, :n, :n].float().cpu().numpy()
    for x in [T, LP, KI, KL, *H.values(), *M.values(), *A.values()]:
        if x is not None: x.flush()
    json.dump({"n_texts": N, "max_len": L, "lengths": [len(s) for s in seqs],
               "files": sorted(f for f in os.listdir(out) if f.endswith(".npy")),
               "notes": {"tokens.npy": "int32 [N,L], -1 = padding",
                         "token_logprobs.npy": "float32 [N,L]; [i,t] = log p(token t | tokens before t); nan at t=0 and padding",
                         "topk_*.npy": "[N,L,k]; position t holds the model's top-k predictions for token t+1",
                         "hidden_L{l}.npy": "float16 [N,L,3584] residual stream after layer l (0 = embeddings)",
                         "mlp_L{l}.npy": "float16 [N,L,18944] input of model.layers.{l-1}.mlp.down_proj",
                         "attn_L{l}.npy": "float16 [N,28 heads,L,L] attention probabilities of model.layers.{l-1}"}},
              open(os.path.join(out, "meta.json"), "w"), indent=1)
else:   # gen_acts: per generated answer, activations at the last prompt token and averaged over the answer
    S = job["samples"]; layers = job["layers"]; ml = [l for l in layers if l >= 1] if job.get("mlp") else []
    P = [enc(s["prompt_text"]) for s in S]; Ans = [enc(s["text"])[:8192] for s in S]
    seqs = [p + x for p, x in zip(P, Ans)]
    R = {f"hidden_L{l}_{w}": np.zeros((len(S), 3584), np.float32) for l in layers for w in ("prompt_last", "answer_mean")}
    R.update({f"mlp_L{l}_{w}": np.zeros((len(S), 18944), np.float32) for l in ml for w in ("prompt_last", "answer_mean")})
    for idx, lens, ids, o, cap in run_batches(seqs, 16000, ml, False, False):
        for r, q in enumerate(idx):
            p, n = len(P[q]), lens[r]
            for name, src in [(f"hidden_L{l}", o.hidden_states[l]) for l in layers] + [(f"mlp_L{l}", cap[l]) for l in ml]:
                R[name + "_prompt_last"][q] = src[r, p - 1].float().cpu().numpy()
                if n > p:
                    R[name + "_answer_mean"][q] = src[r, p:n].float().mean(0).cpu().numpy()
    for kname, v in R.items():
        np.save(os.path.join(out, kname + ".npy"), v)
    json.dump({"files": sorted(R), "samples": [{kk: s[kk] for kk in s if kk not in ("prompt_text", "text")} for s in S],
               "notes": "row i of every file = samples[i]; *_prompt_last = at the last token of the prompt, "
                        "*_answer_mean = mean over the answer's tokens; hidden_L{l} = residual after layer l "
                        "(0 = embeddings); mlp_L{l} = input of model.layers.{l-1}.mlp.down_proj"},
              open(os.path.join(out, "index.json"), "w"), indent=1)
print("TF_WORKER_DONE")
