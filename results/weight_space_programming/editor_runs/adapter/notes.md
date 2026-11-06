# NOTES (weight editor, adapter run)
- Pools: VAL = val shuffled with random.Random(1234) permutation of range(5000); diag = VAL[0:128].
- Pass 1: generate original, diag x4, acts layers 4..28 -> gen_1.jsonl

- Pass 1 result (orig, diag x4): match 18.2%, bo4 29.7%, valid 78.7%, parse 93.9%, comp 92.2%, spg 55.9%. Failures: collisions (<0.5A) 21%; wrong atom count 22% (match 0.9%); spg wrong -> match 4% vs 29%.
- Pass 2: teacher-forced original on 128 diag refs + pass-1 gens (index in tf2_index.json), logprobs+top5 -> fwd_2

- Pass 2 findings: ref NLL 370 nats/text; 69% fresh coord digits, 20% cell; derived-d2 5%; Z/labels/elements deterministic. 43% of gens have contact <1A. Coin experiment on derived digits: no match effect.
- H1: SG-number pathway too weak (failing samples in solvable problems are consistent structures of another SG). Pass 3: attention all layers on refs mp-16456, mp-6873 -> fwd_3

- Pass 3: SG-reading heads (tool layer,head): L15 h16,h27,h21,h19; L14 h16,h17,h19; L20 h27,h22,h24 (model.layers = tool layer-1).
- Pass 4: TF original on TFSET (163 val ids in tfset.json) true prompt + wrong-SG prompt (WSG in tfset.json) -> fwd_4

- Pass 4: wrong-SG prompt raises ref NLL 339.4 -> 374.0 (+34.6+-1.6): len_d0 +5.5, len_lo +7.8, ang_hi +5.2, fresh_d2 +4.0, der_d2 +2.8. Baseline per-class NLL (true prompt) stored in kernel N_true (recompute from fwd_4 rows 0..162; rows 163..325 are wrong-SG).
- Pass 5: prompt-only hidden states L12..20 for 1024 train prompts (p5_ids.json) -> fwd_5

- E3(eps,r,layers): v_proj dW += eps*(W+dW) V^T V, V=top-r right-singular vectors of SG-token inputs (RMSNormed hidden_L{l} at SG positions from fwd_5, common mean of other tokens projected out), refactor to rank 32 by SVD. Pass 6: TF edited E3(eps=1,r=8,layers 13,14,19) on texts4.

- Pass 6: E3 eps=1 worse on ref NLL (+5.2+-0.5), SG gap +1.7+-1.1 -> dropped (edits cleared).
- Calibration (val refs): fresh_d2 conf .40 vs acc .32; len_d0 .73 vs .65 -> overconfident. H3 memorisation. Pass 7: TF original on train seen (S7) vs not seen (U7) ids in p7_ids.json -> fwd_7

- H1 (SG ignored) refuted: 100/117 valid wrong-SG answers are subgroups of request; H3 (memorisation) refuted: seen 51 vs unseen-train 63 vs val 114 leading NLL.
- H5: symmetry-image circuit = model.layers.22 heads [0,1,2,3,5,6,7,12,25], model.layers.23 heads [8,11,13] (tool L23/L24).
- E2(s): o_proj lora_A[:, h*128:(h+1)*128] *= s for those heads. Pass 8: TF edited E2(s=3) on pass-2 texts (texts2 = 128 diag refs + 217 gens, tf2_index.json).

- Pass 8: E2 s=3 (o_proj adapter cols of image heads): no effect on derived digits (dconf -0.001), total dNLL +0.97/text refs. Pass 9: E1 s=3 (q_proj lora_B rows of image heads) on texts2.

- Pass 9: E1 s=3 (q rows of image heads): tiny effect (refs +1.65/text). Head-level adapter scaling is not a lever.
- Rejected hypotheses: H1 SG ignored (wrong-SG answers are subgroups; E3 worsens NLL), H3 memorisation (seen 51 / unseen-train 63 / val 114), M5 near-special coords (no excess), P3 prior shift in special sites (gen 6.1% vs ref 5.8%), K7 volume line inconsistency (written volume within 3% for 90%), cell first-digit coin (paired diff 0.005+-0.031), wrong count not tied to centring.
- Calibration on val: fresh_d2 conf .40 vs acc .32 (overconfident under shift); marginals unbiased.
- Kernel helpers: classify(ids), ck3, CLS, class_sums, cmp, tok_table, feats/F0 (pass-2 baseline), compare_c(k,label), lora_names, get_delta, factor32, build_E3/apply_E3, scale_o_heads/scale_q_heads, clear_edits.

- Luck tests (calibrated sampler model sampler_q10): extra non-top token effect on match: len_d0 -0.10+-0.06, first-row d2 -0.05+-0.03, later rows ~0, fine digits ~0. Earlier strong effects were an artefact (top-5 truncation). Importance re-weighting inconclusive.
- Logit lens: digit margins written in tool layers 24-27 (+3.1,+1.5,+4.7,+6.3), layer 28 removes ~40%.
- Pass 10: TF edited = all 196 adapter deltas truncated to rank 16; texts = TFSET true (163) + S7[:n] + U7[:n].

- Pass 10: rank-16 truncation of all adapters: val NLL +0.59+-0.16 (0.17%), seen +1.25, unseen +0.50 -> free rank but no gain.
- Gated decisiveness circuit judged not worth it (predicted +1.5 pts, knob-like).
- Pass 11: generate both, E3(eps=1,r=8,layers 13,14,19 v_proj) on DEV1=VAL[300:428] n=2.

- Pass 11 (gen both, DEV1 n=2): E3 eps=+1 HARMFUL: match 24.2->16.8% (paired -7.4+-2.4), spg 51.6->38.7, comp 92.6->85.6, bo2 30.5->21.9.
- Pass 12: TF edited E3 eps=-0.5 on texts4.

- Pass 12: E3 eps=-0.5: true NLL +0.74+-0.27, gap -3.3; quadratic optimum eps=-0.08 (gain 0.03 nats): SG pathway strength already optimal.
- Counting check: rows == declared formula in 97.9% of gens; derived-digit accuracy does not degrade with row index. No counting / long-range defect.

- Pass 13: generate both, E3(eps=-0.5,r=8,layers 13,14,19) on DEV2=VAL[428:556] n=2.

- Pass 13 (gen both, DEV2 n=2): E3 eps=-0.5: match 21.1->19.1 (paired -2.0+-1.4), spg 55.9->48.8. SG pathway optimum at trained strength (inverted U). Lever closed.

- Passes 14/15: TF original on own answers of pass 11 / pass 13 (index tf1415_index.json), top5 -> fwd_14, fwd_15 (luck tests).

- Passes 14/15 pooled luck test (812 answers): only first-row first-decimal draws matter: -0.023+-0.012 (within problem), 3.7 draws/answer; cell first digit -0.014+-0.025 (earlier -0.10 was noise).
- Pass 16: gen both, image-head adapter knockout (q rows + o cols of IMH zeroed) on DEV3=VAL[556:684] n=2.

- Pass 16 (gen both, DEV3 n=2): image-head adapter knockout: match 19.1->18.8 (paired -0.4+-1.3), spg 55.1->53.9. No effect: lever closed both ways.

- Occupancy check (val refs, z first decimal at first row of an element, 401 decisions): model prob per fully blocked bin 0.048 vs truth rate 0.009 (free bins 0.109 vs 0.120): model avoids occupied sites only ~2x, reality ~13x. Candidate defect: weak steric avoidance at generator placement.

- Collision creation (1024 orig answers): contact<1.0A in 446; first offending row: first row of element 132, new all-fresh orbit 161, partly fresh 63, derived 90. (<0.5A: 220: 54/60/29/77.) H6: weak occupancy check at generator placement.
- Pass 17: attention all layers + hidden on refs mp-552307, mp-560244 (SG 2) -> fwd_17

- H6: occupancy check heads = model.layers.27 heads 21,23,24 (suppress attended z digit), weak. E4'(eps): o_proj dW += eps*P_dig@W_o_tot[:,cols h] (P_dig=Wd^T (Wd Wd^T)^-1 C Wd), refactor rank 32. Eval set textsE = texts4[:163] + own colliding answers (evalE_index.json: indices into SRC list = pass2 gens(128..344 of TF2), then fwd_14, fwd_15 order). Pass 18: TF E4' eps=6.

- Pass 18: E4' eps=6: blocked-bin prob .048->.050, creating-row logprob -0.09+-0.02, refs NLL +0.67. Too weak.
- E5(kappa): q_proj rows of heads 21,23,24 (layer 27) += kappa*W_q_tot[rows]@P_Q (P_Q top-8 PCA of normed layer-27 inputs at coordinate-digit query positions, from fwd_3+fwd_17). Pass 19: TF E5(kappa=2)+E4'(eps=6) on textsE.

- Pass 19: E5(kappa=2)+E4'(6): blocked-bin log-ratio -0.16+-0.07, creating-row -0.43+-0.09, but refs NLL +9.3 (der_d2 +3.6): dropped.
- Pass 20: gen both E4'(eps=6) alone on DEV4=VAL[700:828] n=2.

- Pass 20 (gen both, DEV4 n=2): E4'(eps=6): match 20.3->21.1 (paired +0.8+-1.4), valid 74.6->77.0, invalid 50->44. Candidate saved in cand_E4p_eps6/.
- Occupancy avoidance attribution: MLPs of tool layers 24, 27 + head tool L27 h26 (model.layers.26 head 26; -0.55 logits; attends to the 2 spaces before the coordinate = plan mover; +0.6 logits on true digit).
- E6(eps): layer-26 o_proj dW += eps*P_dig@W_o_tot[:,26*128:27*128], rank-32 refactor. Pass 21: TF E6 eps=1 on textsE.

- Pass 21: E6 eps=1: blocked-bin log-ratio -0.038+-0.010, conf .404->.406, refs NLL +0.15+-0.08: absorbed downstream. Pass 22: TF E6 eps=8.

- Pass 22: E6 eps=8: blocked-bin .048->.041 (log-ratio -0.19+-0.05), creating-row -0.37+-0.11, refs NLL +4.4, conf .404->.422: not viable. Head levers closed.
- Pass 23: record mlp_act L24, L27 + hidden L28 on 24 low-symmetry val refs (ids in p23_ids.json).

- Pass 23: neuron attribution: L24 MLP -1.57 (top5 -1.16) and L27 MLP -1.10 near-far, but only 2 neurons consistent (268, 10771); no selective neuron lever.
- Pass 24: TF E4' eps=20 on textsE.

- Pass 24: E4' eps=20 harmful (refs NLL +11.2, blocked-bin unchanged). eps=6 is max benign dose.

- FINAL CANDIDATE under replication: E4'(eps=6) = files cand_E4p_eps6/o_proj_27_lora_{A,B}.npy -> save_edit to base_model.model.model.layers.27.self_attn.o_proj.lora_A/B.weight. Held-out pool = holdout_pool.json (VAL[828:] minus used ids); replication pass p uses HO[(p-25)*128:(p-24)*128], n=2, model both. Pooled analysis: paired per-problem diff from gen_<p>_original/edited.jsonl.

- Replication E4'(6): pass 25 +0.4+-1.4, pass 26 -1.2+-1.6 (held-out pooled -0.4+-1.1).
- E7(eps): model.layers.23 down_proj dW[:,J24] += eps*P_dig@W_down_tot[:,J24], J24=[7684,268,9425,16358,3264,3802,12768,5746,991,7470]. Pass 27: TF E7 eps=1 on textsE. NOTE: ./work/edited now holds E7, not the final candidate; restore cand_E4p_eps6 before further replication.

- Pass 27: E7 eps=1 (J24 neurons digit output x2): no effect (blocked log-ratio -0.037, NLL +0.5). Pass 28: TF knockout E6(eps=-1)+E7(eps=-1) on textsE.

- Pass 28: knockout of digit-logit part of head 26.26 and J24 neurons: NLL +0.13+-0.13: P_dig-direction writes before the final block are not causal (absorbed).

- Pass 29: E8(eps=-1) = full knockout of head 26.26 output on pre-coordinate-space values (rank 16, S26 from pass-23 hidden_L26): refs NLL -0.28+-0.17 (improves), fresh_d2 conf .404->.391, blocked-bin log-ratio +0.10. Head adds decisiveness. Pass 30: TF E8 eps=+2.

- Pass 30: E8 eps=+2: conf .404->.430, refs NLL +3.5, blocked-bin log-ratio -0.20+-0.04. Monotone dose-response with pass 29.
- Pass 31: gen both E8(+2) on HO[256:384] n=2; pass 32: E8(-1) same problems.

- Pass 31 (gen both, HO[256:384] n=2): E8(+2): match 25.0->28.1 (paired +3.1+-1.4; 11 gained/3 lost), bo2 31.3->33.6, valid 81.3->80.1, spg 59.0->58.6. Candidate saved cand_E8_p2/. Pass 32: E8(-1) same problems.

- Pass 32 (same problems): E8(-1): match 26.6->27.7 (+1.2+-1.5). Not opposite to +2 result; single-pass noise large (orig 25.0 vs 26.6 on same problems).
- Replicating E8(+2) (cand_E8_p2/o_proj_26_lora_{A,B}.npy -> layers.26.self_attn.o_proj) on HO[384:], passes 33+ (pass p uses HO[(p-30)*128:(p-29)*128]). Decision rule: submit E8(+2) if pooled held-out gain (excluding pass 31) > 0 and > E4'(6); else E4'(6).

- E8(+2) held-out: pass 33 +0.4+-1.4, pass 34 -1.6+-1.2 -> dropped.
- Final protocol: passes 35-40 E8(-1) (cand_E8_m1/o_proj_26_lora_{A,B}.npy), passes 41-44 E4'(6) (cand_E4p_eps6), all HO[(p-30)*128:(p-29)*128] n=2. Choose larger pooled held-out per-answer gain; last passes confirm.

- E8(-1) held-out pooled (passes 32,35,36,37): match -0.49+-0.67, bo2 -1.95+-0.99, valid -2.73+-1.40 -> dropped. New allocation: passes 38-42 E8(+2), 43-47 E4'(6), 48-50 chosen.

- E8(+2) held-out 5 passes (33,34,38,39,40): pooled ~ -0.1+-0.7. Passes 41-45: E4'(6) replication on HO[(p-30)*128:...]. Passes 46-50: chosen candidate.

- DECISION: final candidate E4'(6) (held-out pooled +0.67+-0.49 over passes 25,26,41-45; E8(+2) -0.08+-0.71; E8(-1) -0.49+-0.67). ./work/edited holds E4'(6) (copy in best_final/ and cand_E4p_eps6/). Passes 46-50: further held-out confirmation (HO[(p-30)*128:(p-29)*128]).

- FINAL: all 50 passes used. Submitted edit E4'(6) in ./work/edited (2 tensors of layers.27.self_attn.o_proj). Held-out: confirmation passes 46-50: -0.08+-0.57; all 12 held-out passes: +0.36+-0.37; best-of-2 +0.26+-0.58; valid -0.68+-0.83; SG acc -1.53+-0.76. final_answer.md written.

