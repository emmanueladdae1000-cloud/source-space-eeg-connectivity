"""(1) Show the harness reproduces the author's saved runs; (2) rerun the Table 2 comparisons on the hub graph."""
import json
import pathlib
import time

import numpy as np

import trustnet_harness as th

ROOT = pathlib.Path(__file__).resolve().parent / "data"
HUB, NETB = ROOT / "run_hub", ROOT / "run_b"
OUT = pathlib.Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
ld = lambda d, f, **k: np.loadtxt(d / f"{f}.csv", delimiter=",", **k)
csv6 = lambda x: np.round(x, 6)                    # CSV round-trip used between the scripts
fif32 = lambda x: x.astype(np.float32).astype(np.float64)   # raw.save() stores single precision
report = {}
t0 = time.time()
phys = th.load_physics()
print(f"physics loaded in {time.time() - t0:.0f}s")
L_csv = ld(HUB, "L_leadfield")
report["leadfield_max_abs_diff_vs_csv"] = float(np.abs(phys["L"] - L_csv).max())

# ---------------------------------------------------------------- (1a) hub run, proposed method
G_h, W_h = ld(HUB, "G_true", dtype=int), ld(HUB, "W_true")
Yc_h = ld(HUB, "Y_clean")
phys_csv = dict(phys, L=L_csv)                     # step5 reads L_leadfield.csv
null_h = th.null_coefficients(Yc_h, phys_csv, th.est_symorth, 20)
tau_h = th.pooled_threshold(null_h, 99.0)
report["hub_tau_reproduced"] = tau_h
report["hub_tau_saved"] = float(np.loadtxt(HUB / "null_threshold.csv"))

Z_h = fif32(th.scalp(Yc_h, phys["L"], np.random.default_rng(42)))       # step3 (seed 42)
Yh_h = th.reconstruct(Z_h, phys["K"])
Yh_saved = ld(HUB, "Y_hat_eloreta")
report["hub_yhat_max_abs_diff"] = float(np.abs(Yh_h - Yh_saved).max())
W_est_h = th.est_symorth(csv6(Yh_h))
report["hub_west_max_abs_diff"] = float(np.abs(W_est_h - ld(HUB, "W_est"))[th.OFF].max())
m_prop = th.metrics(G_h, (W_est_h > tau_h).astype(int), W_est_h)
report["hub_proposed"] = m_prop

# ---------------------------------------------------------------- (1b) Network B run, earlier edge-wise z method
G_b, W_b = ld(NETB, "G_true", dtype=int), ld(NETB, "W_true")
Yc_b = csv6(th.simulate_mvar(W_b, seed=42))        # step2 from W_true.csv, saved to Y_clean.csv
null_b = th.null_coefficients(Yc_b, phys_csv, th.est_deblur_signed, 30)
A_b = th.est_deblur_signed(csv6(th.reconstruct(fif32(th.scalp(Yc_b, phys["L"], np.random.default_rng(42))), phys["K"])))
Zb, zthr_b, mu_b, sd_b = th.edgewise_z(null_b, A_b, 99.5)
report["netb_zthr_reproduced"] = zthr_b
report["netb_zthr_saved"] = float(np.loadtxt(NETB / "null_threshold.csv"))
report["netb_null_mean_max_abs_diff"] = float(np.abs(mu_b - ld(NETB, "A_null_mean"))[th.OFF].max())
report["netb_z_max_abs_diff"] = float(np.abs(Zb - ld(NETB, "Z_est"))[th.OFF].max())
report["netb_edgewise_vs_netb_truth"] = th.metrics(G_b, (Zb > zthr_b).astype(int))

# ---------------------------------------------------------------- (2) Table 2 comparisons on the hub graph
null_old_h = th.null_coefficients(Yc_h, phys_csv, th.est_deblur_signed, 30)
A_h = th.est_deblur_signed(csv6(Yh_h))
Zh, zthr_h, _, _ = th.edgewise_z(null_old_h, A_h, 99.5)
report["hub_edgewise_z_threshold"] = zthr_h
report["hub_edgewise_z"] = th.metrics(G_h, (Zh > zthr_h).astype(int), Zh)

null_raw_h = th.null_coefficients(Yc_h, phys_csv, th.est_raw, 20)
tau_raw = th.pooled_threshold(null_raw_h, 99.0)
W_raw_h = th.est_raw(csv6(Yh_h))
report["hub_raw_ridge_threshold"] = tau_raw
report["hub_raw_ridge_own_null"] = th.metrics(G_h, (W_raw_h > tau_raw).astype(int), W_raw_h)
report["runtime_s"] = round(time.time() - t0)

(OUT / "table2_rerun.json").write_text(json.dumps(report, indent=2))
for k, v in report.items():
    if isinstance(v, dict):
        keys = ["tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1", "mcc", "auroc", "ap"]
        print(f"{k:32s} " + " ".join(f"{x}={v[x]:.4f}" if isinstance(v[x], float) else f"{x}={v[x]}" for x in keys if x in v))
    else:
        print(f"{k:32s} {v}")
