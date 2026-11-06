#!/venv/py312/bin/python
"""Forward-pass tool (budgeted).   fwd submit <request.json> | fwd wait | fwd status
Port of the blog's Exp. 8 forward-pass tool: (a) GENERATION with the evaluation's protocol, graded;
(b) TEACHER-FORCED reads. A request that fails does not count against the budget."""
import os, sys, json, time, subprocess, hashlib, traceback, math, shutil, fcntl, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RUN = os.environ["WSP_RUN"]
REQ = os.path.join(RUN, "requests"); os.makedirs(REQ, exist_ok=True)
STATE, RUNNING, LAST, PENDING = (os.path.join(RUN, "state.json"), os.path.join(REQ, "RUNNING"),
                                 os.path.join(REQ, "last_result.txt"), os.path.join(REQ, "pending.json"))
PY, VPY = "/venv/py312/bin/python", "/venv/vllm/bin/python"
MAXTOK, MODEL_LEN, TF_BYTES_CAP, WORK_CAP = 3072, 4096, 12e9, 60e9
GEN_LIMIT, TF_TEXTS, TF_TOKENS = 512, 1024, 524288
ENV = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0", VLLM_ATTENTION_BACKEND="FLASH_ATTN",
           TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
ENV.pop("CUDA_VISIBLE_DEVICES", None)


class UserError(Exception):
    pass


def state(fn=None):
    with open(STATE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        s = json.load(f)
        if fn:
            fn(s); f.seek(0); f.truncate(); json.dump(s, f, indent=1)
        return s


def stage(msg):
    r = json.load(open(RUNNING)); r["stage"] = msg
    json.dump(r, open(RUNNING, "w"))


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except Exception:
        return False


def running():
    if not os.path.exists(RUNNING):
        return None
    try:
        r = json.load(open(RUNNING))
    except Exception:
        time.sleep(1); r = json.load(open(RUNNING))
    if r.get("pid") and not alive(r["pid"]):
        os.remove(RUNNING)
        open(LAST, "w").write("[forward pass FAILED (the driver died) and was NOT counted against the budget]")
        return None
    return r


def wait(maxs=540):
    t0 = time.time()
    while time.time() - t0 < maxs:
        if running() is None:
            print(open(LAST).read() if os.path.exists(LAST) else "[no forward pass has been run yet]"); return
        time.sleep(2)
    r = running()
    if r is None:
        print(open(LAST).read()); return
    print(f"[still running: stage '{r.get('stage')}', {time.time() - r['t0']:.0f} s elapsed. Run `fwd wait` again.]")


def submit(path):
    if running() is not None:
        sys.exit("[refused: a forward pass is still running; `fwd wait` for it first]")
    s = state()
    if s["passes_used"] >= s["budget"]:
        sys.exit(f"[refused: all {s['budget']} forward passes are used]")
    try:
        req = json.load(open(path))
    except Exception as e:
        sys.exit(f"[request is not valid JSON ({e}); not counted]")
    json.dump(req, open(PENDING, "w"))
    json.dump({"t0": time.time(), "stage": "starting", "pid": None}, open(RUNNING, "w"))
    log = open(os.path.join(REQ, "driver.log"), "a")
    p = subprocess.Popen([PY, os.path.abspath(__file__), "_drive"], stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True, env=ENV)
    r = json.load(open(RUNNING)); r["pid"] = p.pid; json.dump(r, open(RUNNING, "w"))
    wait()


# ------------------------------------------------------------------ driver
def sh(cmd, logf, tries, what, okfile=None):
    for t in range(tries):
        with open(logf, "a") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=lf, env=ENV)
        if rc == 0 and (okfile is None or os.path.exists(okfile)):
            return
        time.sleep(5)
    raise RuntimeError(f"{what} failed {tries}x (rc={rc}); see {logf}")


def practice():
    P = {}
    d = os.path.join(RUN, "practice")
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".jsonl"):
            for l in open(os.path.join(d, fn)):
                r = json.loads(l); P[r["id"]] = r
    return P


def du(path):
    return int(subprocess.check_output(["du", "-sb", path]).split()[0])


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def stats(rows, n):
    g = [r for r in rows if r["graded"]]
    out = {"generations": len(rows), "graded": len(g),
           "mean_tokens": round(mean([r["n_tokens"] for r in rows]), 1),
           "cut_at_max_tokens_share": round(mean([r["cut"] for r in rows]), 4)}
    if not g:
        return out
    def block(g):
        items = {}
        for r in g:
            items.setdefault(r["id"], []).append(r["match"])
        rm = sorted(r["rms"] for r in g if r["match"]); nm = sorted(r["near_miss_rms"] for r in g if r["near_miss_rms"] is not None)
        cats = {}
        for r in g:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        return {"n": len(g), "problems": len(items),
                "match_accuracy_per_generation": round(mean([r["match"] for r in g]), 4),
                f"best_of_{n}_per_problem": round(mean([int(any(v)) for v in items.values()]), 4),
                "valid_share": round(mean([r["valid"] for r in g]), 4),
                "parseable_share": round(mean([r["parseable"] for r in g]), 4),
                "composition_correct_share": round(mean([r["comp_ok"] for r in g]), 4),
                "space_group_correct_share": round(mean([r["spg_ok"] for r in g]), 4),
                "matched_rms_median": round(rm[len(rm) // 2], 4) if rm else None,
                "near_miss_rms_median_valid": round(nm[len(nm) // 2], 4) if nm else None,
                "outcomes": dict(sorted(cats.items(), key=lambda kv: -kv[1]))}
    out["all"] = block(g)
    srcs = sorted({r["source"] for r in g})
    if len(srcs) > 1:
        out["by_source"] = {s: block([r for r in g if r["source"] == s]) for s in srcs}
    return out


def paired(ro, re_, n):
    o = {(r["id"], r["sample"]): r["match"] for r in ro if r["graded"]}
    e = {(r["id"], r["sample"]): r["match"] for r in re_ if r["graded"]}
    ids = sorted({k[0] for k in o})
    if not ids:
        return None
    d, bo, be = [], [], []
    for i in ids:
        a = [o[(i, j)] for j in range(n)]; b = [e[(i, j)] for j in range(n)]
        d.append(mean(b) - mean(a)); bo.append(int(any(a))); be.append(int(any(b)))
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {"items": len(ids),
            "accuracy_difference_edited_minus_original": round(mean(d), 4), "standard_error": round(se, 4),
            "samples_wrong_to_right": sum(1 for k in o if not o[k] and e[k]),
            "samples_right_to_wrong": sum(1 for k in o if o[k] and not e[k]),
            f"best_of_{n}_original": round(mean(bo), 4), f"best_of_{n}_edited": round(mean(be), 4),
            "problems_gained_best_of_n": sum(1 for a, b in zip(bo, be) if b and not a),
            "problems_lost_best_of_n": sum(1 for a, b in zip(bo, be) if a and not b)}


def do_generate(req, k, C):
    from multiprocessing import Pool
    import grader
    m = req.get("model")
    if m not in ("original", "edited", "both"):
        raise UserError('"model" must be "original", "edited" or "both"')
    probs, prompts, n = list(req.get("problems", [])), list(req.get("prompts", [])), int(req.get("n", 1))
    mt = int(req.get("max_tokens", MAXTOK))
    if not (1 <= mt <= MAXTOK):
        raise UserError(f"max_tokens must be in 1..{MAXTOK} (the evaluation's limit)")
    tot = (len(probs) + len(prompts)) * n
    lim = GEN_LIMIT // 2 if m == "both" else GEN_LIMIT
    if n < 1 or tot < 1 or tot > lim:
        raise UserError(f"items x n = {tot}; must be 1..{lim}" + (" per model with \"both\"" if m == "both" else ""))
    P = practice()
    bad = [p for p in probs if p not in P]
    if bad:
        raise UserError(f"{len(bad)} unknown practice ids, e.g. {bad[:5]}")
    if len(set(probs)) != len(probs) or len(set(prompts)) != len(prompts):
        raise UserError("duplicate items in the request")
    act = req.get("activations")
    if act:
        if not all(isinstance(l, int) and 0 <= l <= 28 for l in act.get("layers", [])) or not act.get("layers"):
            raise UserError("activations.layers must be a non-empty list of layer indices 0..28")
        if len(act["layers"]) > 29:
            raise UserError("too many layers")
    variants = ["original", "edited"] if m == "both" else [m]
    ckpt = None
    if "edited" in variants:
        if not C.edited():
            raise UserError("there are no edits (save_edit) to run as the edited model")
        stage("building the edited checkpoint")
        ckpt, _ = C.build_ckpt()
    manifest = C.edit_manifest() if ckpt else {}
    items = [("problem", p) for p in probs] + [("prompt", q) for q in prompts]
    meta, by_var = {}, {v: [] for v in variants}
    for v in variants:
        for kind, it in items:
            key = it if kind == "problem" else "prompt:" + hashlib.sha256(it.encode()).hexdigest()[:12]
            for j in range(n):
                uid = f"{v}|{key}|{j}"
                seed = int(hashlib.sha256(f"{key}|{j}".encode()).hexdigest()[:8], 16) % (2 ** 31)
                if kind == "problem":
                    r = {"instruction": P[it]["instruction"], "input": P[it]["input"]}
                else:
                    r = {"instruction": it, "input": ""}
                r.update(uid=uid, seed=seed, max_tokens=mt, adapter=(v if C.KIND == "adapter" else None))
                by_var[v].append(r)
                meta[uid] = {"id": key, "sample": j, "model": v, "graded": kind == "problem",
                             "source": P[it]["source"] if kind == "problem" else "own_prompt",
                             "prompt": (P[it]["problem"] if kind == "problem" else it)}
    tmp = os.path.join(REQ, f"tmp_{k}"); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    glog = os.path.join(REQ, f"fwd_{k}_worker.log")
    res = {}
    def gen(tag, base_model, adapters, rows):
        inp, outp = os.path.join(tmp, f"{tag}.in.jsonl"), os.path.join(tmp, f"{tag}.out.jsonl")
        with open(inp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        stage(f"generating ({tag}, {len(rows)} answers)")
        sh([VPY, os.path.join(HERE, "gen_worker.py"), "--base_model", base_model, "--tokenizer_dir",
            os.path.join(RUN, "tokenizer"), "--adapters", json.dumps(adapters), "--inp", inp, "--out", outp,
            "--gpu_mem_util", str(C.CFG.get("gpu_mem_util", 0.40))], glog, 3, "generation", outp)
        for l in open(outp):
            r = json.loads(l); res[r["uid"]] = r
    if C.KIND == "adapter":
        ad = {v: (C.ADAPTER_DIR if v == "original" else ckpt) for v in variants}
        gen("+".join(variants), C.BASE_DIR, ad, [r for v in variants for r in by_var[v]])
    else:
        for v in variants:
            gen(v, C.BASE_DIR if v == "original" else ckpt, {}, by_var[v])
    stage("grading")
    uids = [u for u in meta if meta[u]["graded"]]
    with Pool(48) as pool:
        grades = pool.map(grader.grade_one, [(res[u]["text"], P[meta[u]["id"]]["answer"]) for u in uids], chunksize=2)
    G = dict(zip(uids, grades))
    rows = {v: [] for v in variants}
    for u, mm in meta.items():
        r = dict(uid=u, **mm, text=res[u]["text"], n_tokens=res[u]["n_tokens"],
                 finish_reason=res[u]["finish_reason"], cut=int(res[u]["finish_reason"] == "length"))
        r.update(G.get(u, {}))
        rows[mm["model"]].append(r)
    files = {}
    for v in variants:
        fn = os.path.join(C.WORK, f"gen_{k}.jsonl" if len(variants) == 1 else f"gen_{k}_{v}.jsonl")
        with open(fn, "w") as f:
            for r in rows[v]:
                f.write(json.dumps(r) + "\n")
        files[v] = fn
    if act:
        for v in variants:
            stage(f"activations ({v})")
            out = os.path.join(C.WORK, f"gen_{k}_acts" if len(variants) == 1 else f"gen_{k}_acts/{v}")
            job = {"type": "gen_acts", "layers": sorted(set(act["layers"])), "mlp": bool(act.get("mlp")), "out": out,
                   "samples": [{"uid": r["uid"], "id": r["id"], "sample": r["sample"], "graded": r["graded"],
                                "match": r.get("match"), "category": r.get("category"),
                                "prompt_text": res[r["uid"]]["prompt_text"], "text": r["text"]} for r in rows[v]]}
            run_tf(job, v, ckpt, C, tmp, glog)
            files[v + "_activations"] = out
    summ = {"pass": k, "type": "generate", "model": m, "n": n, "max_tokens": mt, "edit_manifest": manifest,
            "files": files, "results": {v: stats(rows[v], n) for v in variants}}
    if m == "both":
        summ["paired_comparison"] = paired(rows["original"], rows["edited"], n)
    shutil.rmtree(tmp, ignore_errors=True)
    return summ


def run_tf(job, variant, ckpt, C, tmp, log):
    jp = os.path.join(tmp, f"tfjob_{variant}.json"); json.dump(job, open(jp, "w"))
    if C.KIND == "adapter":
        md, ad = C.BASE_DIR, (C.ADAPTER_DIR if variant == "original" else ckpt)
    else:
        md, ad = (C.BASE_DIR if variant == "original" else ckpt), ""
    sh([PY, os.path.join(HERE, "tf_worker.py"), "--model_dir", md, "--adapter", ad, "--tokenizer_dir",
        os.path.join(RUN, "tokenizer"), "--job", jp], log, 2, "teacher-forced worker")


def do_tf(req, k, C):
    import numpy as np
    from transformers import AutoTokenizer
    m = req.get("model")
    if m not in ("original", "edited"):
        raise UserError('"model" must be "original" or "edited" for a teacher-forced pass')
    tok = AutoTokenizer.from_pretrained(os.path.join(RUN, "tokenizer"))
    if ("texts" in req) == ("chats" in req):
        raise UserError('give exactly one of "texts" or "chats"')
    texts = req["texts"] if "texts" in req else [tok.apply_chat_template(c, tokenize=False) for c in req["chats"]]
    if not texts or len(texts) > TF_TEXTS or not all(isinstance(t, str) and t for t in texts):
        raise UserError(f"1..{TF_TEXTS} non-empty texts required")
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    if max(lens) > MODEL_LEN:
        raise UserError(f"a text has {max(lens)} tokens; the limit is {MODEL_LEN} (the model's evaluation context)")
    if sum(lens) > TF_TOKENS:
        raise UserError(f"{sum(lens)} tokens in total; the limit is {TF_TOKENS}")
    ret = req.get("return", {})
    hs, ml, at, kk = ret.get("hidden_states", []), ret.get("mlp_act", []), ret.get("attn_layers", []), int(ret.get("topk", 0))
    if not all(isinstance(l, int) and 0 <= l <= 28 for l in hs) or not all(isinstance(l, int) and 1 <= l <= 28 for l in ml + at):
        raise UserError("hidden_states layers must be 0..28; mlp_act and attn_layers layers 1..28")
    if kk > 50 or kk < 0:
        raise UserError("topk is at most 50")
    if at and (len(texts) > 2 or max(lens) > 2048):
        raise UserError("attn_layers: at most 2 texts of at most 2,048 tokens")
    N, L = len(texts), max(lens)
    size = N * L * (4 + 4 * bool(ret.get("token_logprobs")) + kk * 6 + len(set(hs)) * 3584 * 2 + len(set(ml)) * 18944 * 2) \
        + len(set(at)) * N * 28 * L * L * 2
    if size > TF_BYTES_CAP:
        raise UserError(f"the result would be {size / 1e9:.1f} GB (arrays are right-padded to the longest text: "
                        f"{N} x {L} tokens); the limit is {TF_BYTES_CAP / 1e9:.0f} GB per pass. Ask for fewer layers / texts.")
    ckpt = None
    if m == "edited":
        if not C.edited():
            raise UserError("there are no edits (save_edit) to run as the edited model")
        stage("building the edited checkpoint")
        ckpt, _ = C.build_ckpt()
    out = os.path.join(C.WORK, f"fwd_{k}")
    shutil.rmtree(out, ignore_errors=True)
    tmp = os.path.join(REQ, f"tmp_{k}"); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    job = {"type": "teacher_forced", "texts": texts, "out": out, "token_logprobs": bool(ret.get("token_logprobs")),
           "topk": kk, "hidden_states": sorted(set(hs)), "mlp_act": sorted(set(ml)), "attn_layers": sorted(set(at))}
    stage(f"teacher-forced read ({N} texts, {sum(lens)} tokens)")
    run_tf(job, m, ckpt, C, tmp, os.path.join(REQ, f"fwd_{k}_worker.log"))
    shutil.rmtree(tmp, ignore_errors=True)
    summ = {"pass": k, "type": "teacher_forced", "model": m, "texts": N, "tokens": sum(lens), "max_len": L,
            "edit_manifest": C.edit_manifest() if ckpt else {}, "dir": out,
            "files": {f: list(np.load(os.path.join(out, f), mmap_mode="r").shape) for f in sorted(os.listdir(out)) if f.endswith(".npy")}}
    if job["token_logprobs"]:
        lp = np.load(os.path.join(out, "token_logprobs.npy"), mmap_mode="r")
        summ["mean_token_logprob"] = round(float(np.nanmean(lp)), 4)
    return summ


def drive():
    import wsp_common as C
    k = state()["passes_used"] + 1
    t0 = time.time(); req = None
    try:
        req = json.load(open(PENDING))
        if du(C.WORK) > WORK_CAP:
            raise UserError(f"your work folder holds more than {WORK_CAP / 1e9:.0f} GB; delete results you no longer need")
        if req.get("type") == "generate":
            summ = do_generate(req, k, C)
        elif req.get("type") == "teacher_forced":
            summ = do_tf(req, k, C)
        else:
            raise UserError('"type" must be "generate" or "teacher_forced"')
        summ["seconds"] = round(time.time() - t0)
        s = state(lambda s: s.__setitem__("passes_used", k))
        json.dump(req, open(os.path.join(REQ, f"fwd_{k}.json.claimed"), "w"))
        json.dump(summ, open(os.path.join(REQ, f"fwd_{k}.done.json"), "w"), indent=1)
        text = json.dumps(summ, indent=1) + f"\n[forward pass {k}/{s['budget']} used; {s['budget'] - k} left]"
    except Exception as e:
        msg = str(e) if isinstance(e, UserError) else traceback.format_exc()[-1500:]
        text = f"[forward pass FAILED and was NOT counted against the budget]\n{msg}"
        summ = {"failed": msg}
    open(LAST, "w").write(text)
    with open(os.path.join(RUN, "tool_calls.jsonl"), "a") as f:
        f.write(json.dumps({"kind": "forward", "t": time.time(), "seconds": round(time.time() - t0, 1),
                            "request": req, "summary": summ}) + "\n")
    if os.path.exists(RUNNING):
        os.remove(RUNNING)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["submit"] and len(a) == 2:
        submit(a[1])
    elif a[:1] == ["wait"]:
        wait()
    elif a[:1] == ["status"]:
        s = state(); r = running()
        print(json.dumps({"forward_passes_used": s["passes_used"], "budget": s["budget"], "code_runs_used": s["code_runs"],
                          "code_run_limit": 300, "running": r}, indent=1))
    elif a[:1] == ["_drive"]:
        drive()
    else:
        sys.exit(__doc__)
