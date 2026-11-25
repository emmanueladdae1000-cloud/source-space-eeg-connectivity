"""Summaries for the benchmark: mean [95% bootstrap CI], paired Wilcoxon (Holm), trap rates."""
import json
import pathlib
import sys

import numpy as np
from scipy.stats import wilcoxon

RES = pathlib.Path(__file__).resolve().parent / "results"
METRICS = ["precision", "recall", "f1", "mcc", "auroc", "ap", "fp"]


def load(name):
    return [json.loads(l) for l in (RES / name).read_text().splitlines() if l.strip()]


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows, methods, ref="PROPOSED"):
    out = {"n_networks": len(rows), "methods": {}}
    for m in methods:
        d = {}
        for k in METRICS:
            d[k] = boot_ci([r[m][k] for r in rows])
        tw = sum(r[m]["fp_two_step"] for r in rows); nw = sum(r[m]["n_two_step"] for r in rows)
        tc = sum(r[m]["fp_common_src"] for r in rows); nc = sum(r[m]["n_common_src"] for r in rows)
        d["two_step_trap_rate"] = [tw, nw, tw / nw if nw else None]
        d["common_source_trap_rate"] = [tc, nc, tc / nc if nc else None]
        d["networks_with_zero_fp"] = sum(r[m]["fp"] == 0 for r in rows)
        out["methods"][m] = d
    # paired tests on MCC vs ref, Holm correction
    others = [m for m in methods if m != ref and m != "ORACLE"]
    pv = []
    for m in others:
        a = np.array([r[ref]["mcc"] for r in rows]); b = np.array([r[m]["mcc"] for r in rows])
        stat = wilcoxon(a, b, zero_method="wilcox") if np.any(a != b) else None
        wins = int((a > b).sum()); ties = int((a == b).sum())
        pv.append([m, float(stat.pvalue) if stat else 1.0, wins, ties, len(a) - wins - ties, float(np.median(a - b))])
    order = np.argsort([p[1] for p in pv])
    k = len(pv)
    adj, running = [None] * k, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * pv[i][1]))
        adj[i] = running
    out["paired_mcc_vs_" + ref] = [{"method": p[0], "p": p[1], "p_holm": adj[i], "wins": p[2], "ties": p[3],
                                   "losses": p[4], "median_diff": p[5]} for i, p in enumerate(pv)]
    return out


def fmt(t, pct=False):
    m, lo, hi = t
    return f"{m*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]" if pct else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "test"
    if which == "test":
        rows = load("test.jsonl")
        methods = ["ORACLE", "RAW", "SIGORTH", "EDGEZ", "PROPOSED", "PROPOSED-DEPLOY"]
        s = summarize(rows, methods)
        (RES / "test_summary.json").write_text(json.dumps(s, indent=2))
        print(f"n = {s['n_networks']}")
        for m in methods:
            d = s["methods"][m]
            print(f"{m:16s} P {fmt(d['precision'], 1)} | R {fmt(d['recall'], 1)} | F1 {fmt(d['f1'])} | MCC {fmt(d['mcc'])} | "
                  f"AUROC {fmt(d['auroc'])} | AP {fmt(d['ap'])} | FP {d['fp'][0]:.1f} | 2step {d['two_step_trap_rate'][2]*100:.1f}% "
                  f"| cs {d['common_source_trap_rate'][2]*100:.1f}% | zeroFP {d['networks_with_zero_fp']}")
        for p in s["paired_mcc_vs_PROPOSED"]:
            print(f"PROPOSED vs {p['method']:16s} p_holm={p['p_holm']:.2e} wins/ties/losses={p['wins']}/{p['ties']}/{p['losses']} median dMCC={p['median_diff']:+.3f}")
    else:
        rows = load("robust.jsonl")
        groups = {}
        for r in rows:
            key = (r["snr_db"], r["n_edges"], r["head"])
            groups.setdefault(key, []).append(r)
        summ = {}
        for key, rs in sorted(groups.items()):
            s = summarize(rs, ["RAW", "PROPOSED"])
            summ[str(key)] = s
            d, e = s["methods"]["RAW"], s["methods"]["PROPOSED"]
            p = s["paired_mcc_vs_PROPOSED"][0]
            print(f"snr={key[0]:>4} edges={key[1]:>2} head={key[2]:9s} n={len(rs)} | RAW MCC {fmt(d['mcc'])} P {d['precision'][0]*100:.0f}% R {d['recall'][0]*100:.0f}% "
                  f"| PROP MCC {fmt(e['mcc'])} P {e['precision'][0]*100:.0f}% R {e['recall'][0]*100:.0f}% AUROC {e['auroc'][0]:.3f} | p={p['p']:.1e} W/T/L={p['wins']}/{p['ties']}/{p['losses']}")
        (RES / "robust_summary.json").write_text(json.dumps(summ, indent=2))
