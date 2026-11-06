"""Compare the four evaluated conditions on the 100 held-out test materials with every stored run in results/
restricted to the same 100 material ids (per-material best-of-10 from the stored validation CSVs)."""
import json, glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
E = "/workspace/wsp/eval"; REPO = "/workspace/LLM_Crystal_CIF"
ids = pd.read_csv(f"{E}/mpts52_test100_seed42.csv").material_id.tolist()
rows = []
panels = {}
for tag in ("base_unedited", "base_edited", "adapter_unedited", "adapter_edited"):
    p = f"{E}/{tag}/panel.json"
    if os.path.exists(p):
        d = json.load(open(p)); panels[tag] = d
        rows.append({"run": f"[this experiment] {tag}", "best_of_10_on_100": d["best_of_10_pct"], "per_gen_match": d["per_generation_match_pct"],
                     "valid": d["valid_pct"], "comp_ok": d["composition_correct_pct"], "spg_ok": d["space_group_correct_pct"],
                     "strict_rms_med": d["strict_rms_median"], "full_test_best_of_10": None})
def mcnemar(a, b):
    x = sum(1 for i in ids if b[i] and not a[i]); y = sum(1 for i in ids if a[i] and not b[i])
    return x, y, (binomtest(x, x + y, 0.5).pvalue if x + y else 1.0)
for a, b in (("base_unedited", "base_edited"), ("adapter_unedited", "adapter_edited")):
    if a in panels and b in panels:
        x, y, p = mcnemar(panels[a]["per_material_best"], panels[b]["per_material_best"])
        print(f"paired {b} vs {a}: best-of-10 {panels[a]['best_of_10_pct']:.0f}% -> {panels[b]['best_of_10_pct']:.0f}%; "
              f"materials gained {x}, lost {y}; McNemar exact p = {p:.3f}; "
              f"per-generation match {panels[a]['per_generation_match_pct']:.1f}% -> {panels[b]['per_generation_match_pct']:.1f}%")
stored = {}
for f in sorted(glob.glob(f"{REPO}/results/*/*/validation/validation_*_16bit.csv") + glob.glob(f"{REPO}/results/*/*/*/validation/validation_*_16bit.csv")):
    run = f.split("/results/")[1].split("/validation/")[0]
    try:
        v = pd.read_csv(f, usecols=lambda c: c == "material_id" or c.startswith("Match_"))
    except Exception as e:
        continue
    mc = [c for c in v.columns if c.startswith("Match_")]
    if len(mc) != 10 or not set(ids) <= set(v.material_id):
        continue
    best = v.set_index("material_id")[mc].any(axis=1)
    stored[run] = {i: int(best[i]) for i in ids}
    sub = v[v.material_id.isin(ids)]
    rows.append({"run": run, "best_of_10_on_100": 100 * float(best[ids].mean()), "per_gen_match": 100 * float(sub[mc].values.mean()),
                 "full_test_best_of_10": 100 * float(best.mean()) if len(v) >= 8000 else None, "n_full": len(v)})
t = pd.DataFrame(rows)
t.to_csv(f"{E}/comparison_on_test100.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(t.round(2).to_string(index=False))
k = "rank_sweep/rank_r32_s3407"
for tag in ("adapter_unedited", "adapter_edited"):
    if tag in panels and k in stored:
        x, y, p = mcnemar(stored[k], panels[tag]["per_material_best"])
        print(f"{tag} vs stored {k} on the same 100: gained {x}, lost {y}, McNemar p = {p:.3f}")
