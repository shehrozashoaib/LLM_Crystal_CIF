"""Work accounting for the two weight-editing agents: tokens from their transcripts (deduplicated by API
message id), tool calls, and the harness's own counters (forward passes, code runs, wall time)."""
import json, os, sys, glob
T = os.environ.get("WSP_TRANSCRIPT_DIR", ".")  # directory holding the Claude Code subagent transcripts (<agent id>.output)
RUNS = {"base": os.environ.get("WSP_BASE_AGENT_ID", ""), "adapter": os.environ.get("WSP_ADAPTER_AGENT_ID", "")}  # Claude Code subagent transcript ids
out = {}
for run, aid in RUNS.items():
    seen, tot, tools, turns, t0, t1 = {}, {}, {}, 0, None, None
    p = os.path.realpath(f"{T}/{aid}.output")
    for line in open(p):
        try:
            d = json.loads(line)
        except Exception:
            continue
        ts = d.get("timestamp")
        if ts:
            t0 = t0 or ts; t1 = ts
        m = d.get("message") or {}
        if d.get("type") == "assistant" and isinstance(m, dict):
            u = m.get("usage")
            if u and m.get("id"):
                seen[m["id"]] = u            # last record of a message id carries its final usage
            for c in m.get("content", []) if isinstance(m.get("content"), list) else []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    tools[c.get("name")] = tools.get(c.get("name"), 0) + 1
    for u in seen.values():
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"):
            tot[k] = tot.get(k, 0) + int(u.get(k) or 0)
    R = f"/workspace/wsp/runs/{run}"
    st = json.load(open(f"{R}/state.json"))
    calls = [json.loads(l) for l in open(f"{R}/tool_calls.jsonl")] if os.path.exists(f"{R}/tool_calls.jsonl") else []
    fw = [c for c in calls if c["kind"] == "forward" and "failed" not in c["summary"]]
    out[run] = {"api_messages": len(seen), "tokens": tot,
                "total_input_incl_cache": sum(v for k, v in tot.items() if k != "output_tokens"),
                "tool_calls": tools, "first": t0, "last": t1,
                "forward_passes_used": st["passes_used"], "budget": st["budget"], "code_runs": st["code_runs"],
                "passes_by_type": {t: sum(1 for c in fw if c["summary"].get("type") == t) for t in ("generate", "teacher_forced")},
                "failed_requests": sum(1 for c in calls if c["kind"] == "forward" and "failed" in c["summary"]),
                "generations_run": sum(v.get("generations", 0) for c in fw if c["summary"].get("type") == "generate" for v in c["summary"]["results"].values()),
                "gpu_pass_seconds": sum(c["seconds"] for c in fw), "kernel_seconds": round(sum(c["seconds"] for c in calls if c["kind"] == "python"))}
json.dump(out, open("/workspace/wsp/eval/agent_usage.json", "w"), indent=1)
print(json.dumps(out, indent=1))
