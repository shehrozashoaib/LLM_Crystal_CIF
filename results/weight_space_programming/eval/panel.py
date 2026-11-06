"""Metric panel for one eval tag from its generated CSV, with the validator's decision rule (harness/grader.py)."""
import sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, "/workspace/wsp/harness")
from multiprocessing import Pool
import grader
tag = sys.argv[1]; E = "/workspace/wsp/eval"
g = pd.read_csv(glob.glob(f"{E}/{tag}/generated/generated_cifs_*_0_99.csv")[0])
t = pd.read_csv(f"{E}/mpts52_test100_seed42.csv")
assert (g.material_id.values == t.material_id.values).all() and (g.output.values == t.output.values).all()
cols = [c for c in g.columns if c.startswith("generation_")]; assert len(cols) == 10 and len(g) == 100
jobs = [(r[c], r["output"]) for _, r in g.iterrows() for c in cols]
with Pool(48) as p:
    G = p.map(grader.grade_one, jobs, chunksize=4)
M = np.array([x["match"] for x in G]).reshape(100, 10)
f = lambda k: float(np.mean([x[k] for x in G]))
rms = [x["rms"] for x in G if x["match"]]; nm = [x["near_miss_rms"] for x in G if x["near_miss_rms"] is not None]
out = {"tag": tag, "materials": 100, "N": 10, "best_of_10_pct": 100 * float(M.any(1).mean()),
       "per_generation_match_pct": 100 * float(M.mean()), "valid_pct": 100 * f("valid"), "parseable_pct": 100 * f("parseable"),
       "composition_correct_pct": 100 * f("comp_ok"), "space_group_correct_pct": 100 * f("spg_ok"),
       "strict_rms_median": float(np.median(rms)) if rms else None, "strict_rms_mean": float(np.mean(rms)) if rms else None,
       "near_miss_rms_median_valid": float(np.median(nm)) if nm else None,
       "outcomes": pd.Series([x["category"] for x in G]).value_counts().to_dict(),
       "per_material_best": dict(zip(g.material_id, M.any(1).astype(int).tolist())),
       "per_material_matches": dict(zip(g.material_id, M.sum(1).tolist()))}
json.dump(out, open(f"{E}/{tag}/panel.json", "w"), indent=1)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if not k.startswith("per_material")})
