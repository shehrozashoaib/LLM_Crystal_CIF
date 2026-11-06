# Weight-space programming on the CIF models (2026-11-05/06)

An LLM agent was given direct read/write access to model weights and asked to improve
CIF generation **without any gradient descent or training**, following the recipe of the
blog post [Weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab; Experiment 8, the 50-forward-pass setting). The recipe was
kept as in the post; only the domain changed, from MATH-500 to composition + space group
-> CIF on MPTS-52.

Two independent runs:

| Run | What the agent could edit |
|---|---|
| `base` | the weights of the base model, Qwen2.5-7B-Instruct (no adapter) |
| `adapter` | the LoRA tensors of this repo's `rank_sweep/rank_r32_s3407` adapter (base frozen) |

**Result in one line:** neither edit gave a real gain on held-out test materials. The
base-model edit repaired much of the output format (parseable 9% -> 47%) but matched one
material of 100; the adapter edit left the adapter where it was (29% -> 30%, noise).

## Setup

| Item | Value |
|---|---|
| Editor model (the agent that changed the weights) | Claude Fable 5.1 (`claude-fable-5-1`), run as two Claude Code subagents (general-purpose agent type), one per run, launched in parallel by a Claude Code session on the same model |
| Edited base model | `unsloth/Qwen2.5-7B-Instruct` (28 layers, hidden 3584, bf16) |
| Edited adapter | `rank_r32_s3407` from [huggingface.co/shehrozashoaib/LLM_Crystal_CIF](https://huggingface.co/shehrozashoaib/LLM_Crystal_CIF): r=32, alpha=64, targets q/k/v/o/gate/up/down, 4,500 SFT steps on `Data/composition_sweep/train_mp20_00.csv.gz`; stored full-test best-of-10 = 29.9% |
| Budget per run (as in the post) | 50 forward passes, 300 code runs, 900 s per code run, 10,000-character output cut; no token limit on the editor (the post set none either) |
| Practice data given to the agent | MPTS-52 train (27,380) and val (5,000) only; the test set was never given |
| Hardware | 1x RTX PRO 6000 Blackwell 97 GB, vLLM 0.22, PyTorch 2.11 |
| Repo commit the runs used | `199afbe` |

### The prompt

- `prompts/TASK_base.md`, `prompts/TASK_adapter.md`: the full task specification each
  agent worked from. It is the post's Experiment 8 prompt (`exp8_math/runs/p50c/message.txt`
  in the post's logs) ported section by section: task, how it is measured, what counts,
  tools, how to work, explain every step, rules, how the session runs.
- `prompts/launch_message_base.md`, `prompts/launch_message_adapter.md`: the short message
  that started each subagent and pointed it at its `TASK.md`.

### Rules carried over from the post

- Tools: a persistent CPU-only Python kernel over the weights (`load`, `load_original`,
  `save_edit`, `drop_edit`, `edited`, `NAMES`, `tok`) and a budgeted forward-pass tool.
- A forward pass is either generation (at most 512 answers, or 256 + 256 for a paired
  original-vs-edited pass) or teacher-forced reads (at most 1,024 texts / 524,288 tokens,
  with log-probabilities, hidden states, MLP activations, attention on request).
- Forbidden: gradients or training of any kind, running the model outside the forward-pass
  tool, internet.
- `save_edit` refuses the final norm, the embedding and the output layer, and any edit
  that is a constant multiple of a whole tensor.
- The session runs until the passes are used; the agent submits its best edit; the
  evaluation is held out and run afterwards on the edit cast to bf16.

All 50 passes were kept (not halved) because they are sequential and adaptive: each pass
is designed from the results of the earlier ones.

## Held-out evaluation

100 materials drawn at random from the 8,096 MPTS-52 test materials with
`np.random.default_rng(42).choice(8096, 100, replace=False)` (ids and original row index in
`eval/mpts52_test100_seed42_with_index.csv`). Protocol identical to the repo's evaluations:
`evaluation/generate_cifs_vllm.py` (N=10, temperature 0.6, top_p 0.9, max_new_tokens 3072,
seed 42, system prompt `sft`) then `evaluation/cif_structure_validator_mp52.py`
(StructureMatcher stol=0.5, ltol=0.3, angle_tol=10).

| Condition | Best-of-10 | Per-generation match | Valid | Parseable | Composition correct | Space group correct |
|---|---|---|---|---|---|---|
| Base Qwen, unedited | 0% | 0.0% | 9.2% | 9.2% | 0.6% | 0.4% |
| Base Qwen, edited | 1% | 0.2% | 44.9% | 46.5% | 18.0% | 0.5% |
| Adapter `rank_r32_s3407`, unedited | 29% | 19.7% | 73.9% | 94.6% | 94.0% | 56.1% |
| Adapter `rank_r32_s3407`, edited | 30% | 20.2% | 74.7% | 94.5% | 93.3% | 56.2% |
| Stored `rank_sweep/rank_r32_s3407`, same 100 ids | 29% | 19.5% | | | | |

- Base, paired: 1 material gained, 0 lost (McNemar exact p = 1.0). The match is
  mp-1017439 in 2 of its 10 generations, both loose (RMS 0.42). Answers cut at the token
  limit fell from 16.7% to 9.3%.
- Adapter, paired: 2 gained, 1 lost (p = 1.0). Re-running the unedited adapter against
  the stored result on the same ids already moves 1 material each way.
- With 100 materials the standard error of a 29% score is about 4.5 points.
- `eval/comparison_on_test100.csv` has the four conditions next to every stored run in
  `results/` restricted to the same 100 ids (those range from 25% to 40%).

## What the agents changed

Full write-ups are in `editor_runs/<run>/final_answer.md`; `explanations.md` has the
step-by-step reasoning and `notes.md` the working notes.

**Base** (three tensors of `model.layers.27.mlp`):
- `down_proj.weight`: closed-form rank-178 key->value update that forces the fixed tokens
  of the pymatgen P1 CIF header and atom lines, plus a stop rule that ends the answer when
  the atom list is complete.
- `gate_proj.weight`, `up_proj.weight` (7 rows each): seven near-dead neurons rewired as
  gated circuits that write 60, 120 or rhombohedral cell angles when the lattice type
  calls for them (the original model always writes 90).
- On 3,584 held-out practice problems: 15 matches against 0 (+0.42 +/- 0.11 points per
  answer), all in cells of 5 atoms or fewer.

**Adapter** (one adapted matrix, the `lora_A`/`lora_B` pair of `layers.27.self_attn.o_proj`):
- Multiplies the direct digit-logit output of attention heads 21, 23, 24 by 7, then
  re-factorises to rank 32 by SVD; aimed at atoms placed on top of each other.
- On practice problems not used for any choice: -0.08 +/- 0.57 points. The agent reported
  that it found no edit that improves the adapter, and that the match rate is limited by
  what the model believes about cell and atom positions, not by one broken step.

## Work done by each editor agent

| | Base | Adapter | Post's 50-pass run |
|---|---|---|---|
| Forward passes | 50 of 50 (36 generation, 14 teacher-forced) | 50 of 50 (27 generation, 23 teacher-forced) | 50 |
| Code runs | 189 of 300 | 139 of 300 | 80 |
| Model turns | 256 | 189 | 158 |
| Output tokens | 454k | 332k | about 150k |
| Fresh (uncached) input tokens | 0.68M | 0.53M | about 0.26M |
| Input tokens including cache re-reads | 101M | 58.8M | 5.44M |
| Wall time | 4.4 h | 3.7 h | 20.3 h |
| Practice answers generated | 18,350 | 13,824 | not recorded |

The "including cache re-reads" row is not like-for-like: each agent here ran as one long
conversation that re-reads its history every turn, while the post's editor was restarted
with a handoff every 40 turns. Raw numbers are in `eval/agent_usage.json`.

## Deviations and caveats

- Isolation was by rule, not by sandbox. The agents' commands and kernel code were audited
  afterwards; nothing outside their run folders was read (no test set, no stored results).
- Self-reported rule slips: each agent once piped tool output through a formatting command
  outside the allowed list; the base agent applied the final norm and unembedding to
  recorded hidden states for offline checks (no new inputs were run).
- Harness faults: the last-layer hidden state was saved after the final norm, not before
  it as the task text said; one base-run pass failed on GPU memory and was not counted.
- Practice passes used vLLM in eager mode (faster start-up); the held-out evaluation used
  the repo's generation script unchanged. Base-model conditions use
  `eval/generate_cifs_vllm_base.py`, the same script with LoRA loading removed.
- The base model's tokenizer was used everywhere; the adapter's tokenizer config does not
  load in the vLLM environment (chat template and token ids are identical, only the pad
  token differs).
- Teacher-forced passes were capped at 12 GB of recorded arrays (the post: 6 GB) and
  4,096 tokens per text.
- `env/setup_vllm.sh` has CRLF line endings and fails as committed; a CR-stripped copy was
  run instead.
- **Edited weights are on Hugging Face, not in git** (repo policy: no weights in git):
  [huggingface.co/shehrozashoaib/LLM_Crystal_CIF/tree/main/weight_space_programming](https://huggingface.co/shehrozashoaib/LLM_Crystal_CIF/tree/main/weight_space_programming).
  `adapter_edited_rank_r32_s3407/model` is a complete drop-in adapter;
  `base_edited_qwen2.5_7b_instruct/edited_tensors.safetensors` holds the three edited base
  tensors in bf16 with `apply_edit.py` to write a full edited model. Both are exactly the
  weights that were evaluated here.

## Layout

```
prompts/            task specification and launch message for each run
harness/            the tools the agents used
  wsp_common.py       weight access, save_edit rules, checkpoint builder
  kernel_server.py    persistent CPU kernel        wsp_run.py   its client (./run)
  wsp_fwd.py          budgeted forward-pass tool (./fwd)
  gen_worker.py       vLLM generation worker       tf_worker.py teacher-forced worker
  grader.py           match / validity grading (validator functions copied verbatim)
  setup_run.py        creates a run folder          make_tasks.py writes TASK.md
eval/               held-out evaluation scripts, the 100 test ids, comparison table, usage
heldout_test100/    per condition: predicted_cifs.csv.gz, panel.json, validation/
editor_runs/<run>/  final_answer.md, explanations.md, notes.md
                    kernel_code_runs.jsonl.gz     every code run with its output
                    forward_passes/               result summary of each of the 50 passes
                    forward_pass_requests.tar.gz  the request behind each pass
```

Scripts contain the absolute paths of the machine they ran on (`/workspace/wsp/...`,
`/workspace/models/...`).
