"""Numbers added after the second external review (computed from saved results + physics)."""
import json
import pathlib

import numpy as np
from scipy.stats import wilcoxon

R = pathlib.Path(__file__).resolve().parent / "results"
rows = [json.loads(l) for l in (R / "test2.jsonl").read_text().splitlines()]
ref30 = [r for r in rows if r["seed"] <= 30]
out = {"ref30_mcc": {m: float(np.mean([r[m]["mcc"] for r in ref30])) for m in
                     ["ORACLE", "RAW", "SIGORTH", "TRGC", "GPDC", "PROPOSED", "PROPOSED-DEPLOY", "PROPOSED-POSTINV"]}}
G = {}
for r in (json.loads(l) for l in (R / "conditions2.jsonl").read_text().splitlines()):
    G.setdefault(r["task"]["cond"], []).append(r)
G["REF_1_30"] = ref30
rng = np.random.default_rng(1)
out["deploy_minus_sim"] = {}
for c, rs in G.items():
    d = np.array([r["PROPOSED-DEPLOY"]["mcc"] - r["PROPOSED"]["mcc"] for r in rs])
    b = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    lo, hi = np.percentile(b, [5, 95])
    out["deploy_minus_sim"][c] = {"mean": float(d.mean()), "ci90": [float(lo), float(hi)], "tost": bool(lo > -0.05 and hi < 0.05)}
sc = [r["sim_info"]["coupling_scale"] for r in G["C3_oscillatory"]]
out["c3_coupling_scale"] = [min(sc), max(sc), float(np.median(sc))]
c3 = G["C3_oscillatory"]
out["c3_sim_vs_postinv_p"] = float(wilcoxon([r["PROPOSED"]["mcc"] for r in c3], [r["PROPOSED-POSTINV"]["mcc"] for r in c3]).pvalue)
c6 = G["C6_patch_readout"]
out["c6_sim_vs_deploy_p"] = float(wilcoxon([r["PROPOSED"]["mcc"] for r in c6], [r["PROPOSED-DEPLOY"]["mcc"] for r in c6]).pvalue)
out["overall_fp_rate_absent"] = sum(r["PROPOSED"]["fp"] for r in rows) / sum(78 for r in rows)
import trustnet_harness as th
p = th.load_physics()
M = p["K"] @ p["L"]
Mn = M / np.sqrt(np.outer(np.diag(M), np.diag(M)))
off = ~np.eye(10, dtype=bool)
out["mixing"] = {"asym": float(np.linalg.norm(Mn - Mn.T) / np.linalg.norm(Mn)),
                 "off_median": float(np.median(np.abs(Mn[off]))), "off_max": float(np.abs(Mn[off]).max())}
(R / "round2_numbers.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:1500])
