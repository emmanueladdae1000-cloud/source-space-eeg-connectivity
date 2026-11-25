"""Hub-graph case study numbers (author's run, seeds and CSV round-trips as in the scripts)."""
import json
import pathlib

import numpy as np

import trustnet_harness as th
from benchmark import est_sigorth

ROOT = pathlib.Path(__file__).resolve().parent / "data" / "run_hub"
OUT = pathlib.Path(__file__).resolve().parent / "results"
ld = lambda f, **k: np.loadtxt(ROOT / f"{f}.csv", delimiter=",", **k)
csv6 = lambda x: np.round(x, 6)
fif32 = lambda x: x.astype(np.float32).astype(np.float64)

phys = th.load_physics()
L_csv = ld("L_leadfield")
pc = dict(phys, L=L_csv)
G, Yc = ld("G_true", dtype=int), ld("Y_clean")
Yh = csv6(th.reconstruct(fif32(th.scalp(Yc, phys["L"], np.random.default_rng(42))), phys["K"]))
res = {}


def pooled(est, n=20, pct=99.0, tag=None):
    null = th.null_coefficients(Yc, pc, est, n)
    tau = th.pooled_threshold(null, pct)
    W = est(Yh)
    m = th.metrics(G, (W > tau).astype(int), W)
    m["tau"] = tau
    m["null_mean"] = float(np.mean(np.concatenate([w[th.OFF] for w in null])))
    if tag:
        np.save(OUT / f"hub_null_{tag}.npy", np.concatenate([w[th.OFF] for w in null]))
        np.save(OUT / f"hub_W_{tag}.npy", W)
    return m


res["PROPOSED_beta0.05"] = pooled(lambda Y: th.est_symorth(Y, beta=0.05), tag="prop005")
res["PROPOSED_beta0.10"] = pooled(lambda Y: th.est_symorth(Y, beta=0.10), tag="prop010")
res["RAW"] = pooled(th.est_raw)
res["SIGORTH"] = pooled(est_sigorth)
null_old = th.null_coefficients(Yc, pc, th.est_deblur_signed, 30)
Z, zthr, _, _ = th.edgewise_z(null_old, th.est_deblur_signed(Yh), 99.5)
res["EDGEZ"] = dict(th.metrics(G, (Z > zthr).astype(int), Z), z_threshold=zthr)
(OUT / "hub_case_study.json").write_text(json.dumps(res, indent=2))
for k, m in res.items():
    print(f"{k:18s} {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
          f"MCC={m['mcc']:.4f} AUROC={m['auroc']:.4f} AP={m['ap']:.4f}" + (f" tau={m['tau']:.6f} null_mean={m['null_mean']:.4f}" if 'tau' in m else ""))
