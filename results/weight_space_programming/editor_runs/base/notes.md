# NOTES
- Problem sets: perm=np.random.RandomState(0).permutation(5000) over val; VID=[val[i]['id'] for i in perm]; S_k = VID[k*256:(k+1)*256].
- Pass 1: generate original, S0 x n=2, max_tokens 3072, activations all layers + mlp (req_01.json).

- Train order: tperm=np.random.RandomState(1).permutation(27380); TID. Pass 1 (TF, original): prompt+reference+<|im_end|> for first 428 of TID (those <=4000 tok), logprobs+top20.

- Pass 2 (gen, original, custom prompts): S1[:128] x {A one-shot+instruction, B instruction only} x n=2. Example = train mp-22408.

- Pass 3 (TF, original): P3ids = first 200 of TID not in pass-1 ids; prompt+reference truncated to 948 answer tokens; returns logprobs, top20, hidden 27/28, mlp_act 28.

- E1 = model.layers.27.mlp.down_proj.weight += sum_c beta_c d_c w_c^T over 30 template decision classes CLS (keys from pass-3 mlp_L28 at the position before each low-probability template token; w_c = C^-1 kbar_c/(kbar_c C^-1 kbar_c), C = second moment over answer positions of 160 fit texts + 0.01*trace/d ridge; d_c = unit(gamma*(U_t - 0.5*comp mean - 0.5*20-NN mean)); beta_c=(gapmax+8)*s85/sens, ' generated' x1.6); randomized rounding to bf16 (seed 0). Exact update saved in E1_delta.pt.
- Pass 4: generate edited (E1), S0 x n=2, max_tokens 3072.

- E2 (saved E2_delta.pt): same tensor; 178 valued classes (all rigid header tokens lines 0-25; atom-line start/label0/e44,e43,e42(mult '1'),e41,e40,e3,e2,e1; e0n = digits -10 & blank line -10 at non-final atom-line ends; stop = im_end +11, blank -10, digits -10 at final ends; sum_start digits -10), 135 protected classes with zero response; joint keys; vocabulary-whitened values (Sigma_U ridge 0.1*mean eig); margin 10 nats over the 98th-percentile gap.
- DS = [i for i in VID[2560:] if natoms<=10]. Pass 5: generate edited(E2) on S0[:128]+DS[:128], n=2.

- E3 (E3_delta.pt): E2 minus sum_start, minus forced line start after every atom (now only first atom: start0), plus startn (im_end +8; NL, NLNL, fences -10 after atom-line newline); single values constrain top-8 competitors + 6 NN individually; sym_end tried and dropped (ill-determined). Pass 6: generate edited(E3), S0[:128]+DS[:128], n=2.

- Pass 7 (TF, original): P7ids = next 1024 of TID (not in pass 1/3); prompt + header lines 0-8; hidden 12,18,24,27 + mlp 28.

- After a kernel restart: run work/setup_a.py, setup_b.py, setup_c.py, then CL,PROT=build_classes(); NV,NP,WJ,cond=solve_keys(CL,PROT); e3_values(CL); D3=assemble(CL,NV). Never run sklearn LDA on 18,944-dim data (it hit the 900 s limit and restarted the kernel at code run 69).

- Pass 8: generate edited(E3) on train P8 = TID[2000:] (unused) stratified fcc115/hex115/90x115/other95/bcc72, n=1, max_tokens 1100.

- Pass 9 (TF, original): prompt + E3-generated header lines 0-8 for pass-8 train problems (p9_texts.json); hidden 6,10,14,18,22,26,27 + mlp 28.

- E4 (E4_delta.pt) = E3 + lattice classes (lat alpha fcc -> '6' +17; lat gamma hex -> '1' +8; gamma after 60/60 -> '6' +6), metric C + within-position cov (Woodbury), keys from pass 9 (generation contexts) / pass 7. Pass 10: generate edited(E4) on S0[:128]+DS[:128], n=2.

- Current edit restored to E3 (E4 rejected: off-position noise of lattice keys). Pass 11 (TF original): 512 pass-9 texts + 300 pass-7 texts (P7ids[:300]); hidden 27 + mlp_act 27.

- Pass 12 (TF original): 220 full E3 generations from pass 8 (p12_texts.json, <=1000 tokens); hidden 27 + mlp_act 27.

- Gated lattice neurons v1 (neurons_v1.pt, neuron_design.pt): model.layers.26.mlp neurons 15029 (fcc@alpha->'6', slope 36, th1 .25), 2845 (hex@gamma->'1', slope 20, th1 .40), 17799 (60/60 ctx@gamma->'6', slope 12, th1 .40); gate a=60, th_pos .42. Pass 13: TF edited (neurons only, E3 dropped) on pass-9 texts + pass-12 texts.

- E5c = E3 + last-layer bias classes (alpha '6' +9, gamma '1' +4; E5c_last_delta.pt) + neurons v2 (neurons_v2.pt: fcc@alpha slope 24 th .5; hex@gamma slope 20 th .5; f60@gamma slope 24 th .4). PROT without ('pv',6,-2),('pv',8,-2). Pass 14: TF edited on the 812 pass-11 texts, topk 20.

- E5d: last layer = E3 classes + bias keys (alpha '6' +9, gamma '1' +4) solved with metric C + within-position cov of own-header alpha/gamma positions (solveS2); neurons A slope 36 th .5, B slope 20 th .5 (no C). Pass 15: TF edited on 512 own headers + 448 synthetic alpha=beta=60 variants; topk20 + hidden27 + mlp27.

- E5e: last layer E3 + alpha bias ('6' +9) (E5e_last_delta.pt); neurons_v4.pt: A fcc@alpha slope 24 th.5; B hex2@gamma slope 20 th.5; C c60@gamma v=24*unit6-24*unit1 th.5. Pass 16: TF edited on p15_texts.json (512 own + 448 synthetic 60/60), topk 20.

- E5f: last layer E3 + bias alpha '6' +7.6 + bias gamma '1' +4.0 (solveS2 keys); neurons_v5.pt: A fcc@alpha slope 24 th .5, B hex2@gamma slope 20 th .5. Pass 17: TF edited on p15_texts.json, topk 20 + mlp_act 28.

- E5g: last layer E3 + bias alpha '6' +7.0 + bias gamma '1' +4.0 + rule g60 '6' +7.0 (keys from pass-17 mlp_L28 means: keep_pass17.pt; metric C+within via solveS3) -> E5g_last_delta.pt; neurons_v5.pt (A, B). Pass 18: generate edited(E5g) on S0[:128]+DS[:128], n=2.

## STATUS after pass 17 (code run ~109)
- Best confirmed edit so far: E3 (work/best_E3 = copy of edited dir; E3_delta.pt): dev S0[:128]+DS[:128] n=2: 5/512 matches (all in small cells), parse 60%. Original: 0/512 on S0 (failed pass-1 file) and 0 everywhere.
- E4 (linear lattice keys in last layer) REJECTED (off-position noise). 
- Lattice circuit now: gated neurons in model.layers.26.mlp (15029: fcc@alpha->'6'; 2845: hex@gamma->'1'), act = silu(g.x)*(up.x), g = 60*(wpos - (wpos.muP+0.42)*bP), up=(u-(u.m0+th1)*bP)/(60*0.58), down = slope*unit_vec(token). Linear range only up to implied slope ~24 (measured efficiency 0.2-0.4; slope 36 reverses sign). Constant part of the gap removed by last-layer position-bias keys (need within-position covariance whitening: solveS3).
- E5g = E3 + bias alpha '6' +7.0 + bias gamma '1' +4.0 + rule g60 ('6' +7 at gamma when 60/60 written) [last layer] + neurons v5. Pass 18 = generation dev test of E5g.
- Files: neuron_design.pt, neuron_design2.pt (u_hex2,u_c60), neurons_v5.pt, keep_pass9.pt, keep_pass7.pt, keep_pass17.pt, keep_x26.pt, keep_x26_stats.pt, E5g_last_delta.pt. Raw big arrays of passes 7,9,11,12,15 deleted.
- To restore E3: W=bf16_round_stochastic((W0.double()+torch.load('E3_delta.pt')['D'].double()).float(),seed=0); save_edit(LN,W); drop the three model.layers.26.mlp edits.
- Unused val sets for final confirmation: S2..S9 (VID[512:2560]); DS uses VID[2560:], S0,S1 used for dev.
- Open failure modes in E3 regime: duplicate coordinates (34% of answers), atom count (one formula unit written in 45%; refs Z=1 only 21% overall, 51% of small cells), lattice type (being addressed).

- E5g dev result (pass 18): 7 matches / 512 (E3: 5). Copy saved as work/cand_E5g. Pass 19: TF original on p15_texts.json with hidden 27, 28 + mlp_act 28 (for last-block gated neurons).

- E6 = E3_delta (last-layer down_proj) + gated neurons in model.layers.27.mlp: 15587 (alpha x fcc -> '6', slope 60, th1 .45), 10289 (gamma x hex -> '1', slope 40, th1 .5), 12126 (gamma x 60ctx -> '6', slope 30, th1 .5); gate a=40, th_pos .5; up x30 / down /30 balance; design in x28_design.pt. No block-26 edits. Pass 20: generate edited(E6) on S0[:128]+DS[:128], n=2.

- Pass 20: E6 dev result 11/512 matches, best-of-2 9/256 (E3: 5 / 4). work/best_E6 = copy of edited dir.

- Pass 21: generate edited(E6) on train P21 (p21_ids.json: fcc120/hex120/90x120 small Z=1, hexZ2 40, other_small 50, bcc 22, rest 40), n=1, max_tokens 1000.

- Pass 21 result: E6 on stratified train P21: 80/512 match. Pass 22: TF original on 220 complete pass-21 generations (p22_texts.json) with hidden 27,28 + mlp 28.

- E7 = E6 + neurons 1411 (stop: line-end x complete -> im_end, slope 30 th .45, th_pos .6), 5161 (dup5: sum-gate a=10 K=4 th_pos .9 th1 .35, v=-3*unit('5')), 17824 (dup0 same, '0'); design in x28_design_b.pt. Pass 23: generate edited(E7) dev S0[:128]+DS[:128] n=2.

- E7 (stop + dup neurons) REJECTED (null). E8 = E6 + hex-site neurons in layer 27: 1411 NX (xy-dot x hex ctx; a 40, th_pos .5, th1 .1, v=8*v_hexsite), 5161 NY3 (y-dot x x3 -> '6', slope 40), 17824 NY6 (y-dot x x6 -> '3'); design x28_design_hex.pt. Pass 24: generate edited(E8) on P21 (train), n=1, max_tokens 1000 (E6 baseline = pass 21).

- E8 (hex-site neurons) REJECTED (hex stratum 12 -> 4). Current edit = E6. Pass 25: generate edited(E6) on train p25.json (200 rhombohedral + negatives), n=1, max_tokens 400 (headers).

- Pass 26 (TF original): p26_texts.json = 512 pass-25 own headers + 448 synthetic alpha=beta=33; hidden 27,28 + mlp 28.

- Disk incident (code run 157-159): pass 26 failed (not counted) because the filesystem was full: deleted result arrays were still memory-mapped by the kernel. Fix: delete np.memmap variables in the kernel before/after removing files. Raw arrays of passes 3 (hidden), 17, 19, 22 deleted; kept: fwd_3/mlp_L28.npy, keep_*.pt files.

- E9 = E6 + rhombohedral neurons (x28_design_rh.pt: SPEC9 rows; neurons N27c): R_a (alpha x rh -> '3', slope 90, th1 .45), R_a2, R_b, R_g single-factor context detectors (a=40, th .5; values 20*u3, 20*u3, 15*v3g). Pass 27: generate edited(E9) on p27_ids.json (S0[:128]+DS[:96]+32 val rhombo from VID[2560:]), n=2.

- FINAL CANDIDATE E9b (installed; copy in work/best_E9b after this run): model.layers.27.mlp.down_proj = W0 + E3_delta (stochastic bf16 rounding, seed 0) with columns of hijacked neurons replaced; gate/up rows of neurons N27=[15587,10289,12126] (fcc@alpha, hex@gamma, c60@gamma) and N27c=[1411,5161,17824,8836] (R_a th1 .50, R_a2, R_b, R_g). Raw rows/cols saved in E9b_neuron_weights.pt.
- Confirmation: 'both' passes on S2..S7 (sset(2)..sset(7)), n=1, max_tokens 3072 -> passes 28-33.

- Pass 28 (S2, both, n=1): original 0/256, edited 0/256 matches (edited parse 53%, valid 50%, comp 26%; original parse 6%). S2 had only 4 tiny addressable problems. Requests req_29..req_33 = S3..S7.

- E9b held-out confirmation (passes 28-33, S2..S7, n=1): original 0/1536, edited 10/1536, +0.65% +- 0.21%.
- FINAL EDIT E10 installed (E10_neuron_weights.pt + E3_delta.pt + x28_design*.pt): E9b with u_f2/u_r2 pooled directions, rh th1 .52. Confirmation sets in final_conf_sets.json: req_34..req_41 (n=1 blocks), req_42..req_46 (best-of-10, 25 problems each).

- E10 confirmation in progress: passes 34-36 pooled edited matches 1 / 768 (original 0).

- E10 n=1 confirmation passes 34-38: edited 2 / 1280 matches (original 0). Plan for the rest: passes 39-43 = best-of-10 'both' for E10 (req_42..46 files), then E9b edited-only best-of-10 on the same 125 problems (3 passes), then choose; last passes n=1 'both' on blocks 5-7 for the winner.

- DECISION: final edit = E9b (restored from work/best_E9b into edited/). E10 measured +0.16%+-0.11% (1280 problems, passes 34-38) and 0/50 best-of-10 problems (passes 39-40, no addressable problems among them). Remaining passes: E9b 'both' n=1 on E10's blocks (req_34..38 again, now with E9b), then unused blocks (req_39..41).

- Passes 41,42 = E9b 'both' on S8,S9 (same problems as E10 passes 34,35). Next: 43,44 = blocks 2,3 (req_36,37); 45 = block 4 (req_38); then unused blocks.

- E9b held-out 'both' n=1 so far (S2-S7 + passes 41-46): edited 13 / 3072, original 0. Head-to-head with E10 on 5 shared blocks: E10 2, E9b 2 matches.

## FINAL (after pass 50, code run 189)
- Submitted edit = E9b (./work/edited == ./work/best_E9b).
- E9b held-out paired n=1: 3,584 problems, original 0, edited 15, +0.42% +- 0.11%. Best-of-10 on 50 problems: 4 problems (14/500 answers) vs 0.
- E10 (pooled directions) not submitted: +0.16% +- 0.11% on 1,280 problems; tie with E9b (2 vs 2) on the same problems.

