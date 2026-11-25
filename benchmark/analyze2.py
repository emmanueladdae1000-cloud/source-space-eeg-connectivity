"""Summaries for the extended benchmark (test2 + conditions2)."""
import json
import pathlib
import sys

import numpy as np
from scipy.stats import wilcoxon

RES = pathlib.Path(__file__).resolve().parent / "results"
METHODS = ["ORACLE", "RAW", "SIGORTH", "TRGC", "GPDC", "PROPOSED", "PROPOSED-DEPLOY", "PROPOSED-POSTINV"]
KEYS = ["precision", "recall", "f1", "mcc", "auroc", "ap", "fp"]


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def holm(ps):
    order = np.argsort(ps); k = len(ps); adj = [0.0] * k; run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (k - r) * ps[i])); adj[i] = run
    return adj


def summarize(rows, methods=METHODS, ref="PROPOSED"):
    out = {"n": len(rows), "methods": {}, "paired": []}
    for m in methods:
        d = {k: boot([r[m][k] for r in rows]) for k in KEYS}
        for trap in ("two_step", "common_src"):
            fp_, n_ = sum(r[m][f"fp_{trap}"] for r in rows), sum(r[m][f"n_{trap}"] for r in rows)
            d[f"trap_{trap}"] = [fp_, n_, fp_ / n_ if n_ else None]
        # pooled precision (sum TP / sum detections) and precision excluding empty outputs
        tp, fp = sum(r[m]["tp"] for r in rows), sum(r[m]["fp"] for r in rows)
        d["precision_pooled"] = tp / (tp + fp) if tp + fp else None
        ne = [r[m]["precision"] for r in rows if r[m]["tp"] + r[m]["fp"] > 0]
        d["precision_nonempty"] = [float(np.mean(ne)) if ne else None, len(rows) - len(ne)]
        out["methods"][m] = d
    others = [m for m in methods if m not in (ref, "ORACLE")]
    ps, rec = [], []
    for m in others:
        a = np.array([r[ref]["mcc"] for r in rows]); b = np.array([r[m]["mcc"] for r in rows])
        p = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
        ps.append(p)
        rec.append({"method": m, "p": p, "wins": int((a > b).sum()), "ties": int((a == b).sum()),
                    "losses": int((a < b).sum()), "median_diff": float(np.median(a - b)), "mean_diff": boot(a - b)})
    for r_, a in zip(rec, holm(ps)):
        r_["p_holm"] = a
    out["paired"] = rec
    # equivalence of deployable vs simulation null (TOST via 90% bootstrap CI, margin 0.05)
    d = np.array([r["PROPOSED-DEPLOY"]["mcc"] - r[ref]["mcc"] for r in rows])
    b = d[np.random.default_rng(1).integers(0, len(d), (10000, len(d)))].mean(1)
    lo, hi = np.percentile(b, 5), np.percentile(b, 95)
    out["equivalence_deploy_vs_sim"] = {"mean_diff": float(d.mean()), "ci90": [float(lo), float(hi)],
                                        "margin": 0.05, "equivalent": bool(lo > -0.05 and hi < 0.05),
                                        "min_diff": float(d.min()), "max_diff": float(d.max())}
    return out


def line(m, d):
    f = lambda k, pct=False: (f"{d[k][0]*100:.1f} [{d[k][1]*100:.1f}, {d[k][2]*100:.1f}]" if pct else f"{d[k][0]:.3f} [{d[k][1]:.3f}, {d[k][2]:.3f}]")
    return (f"{m:17s} P {f('precision', 1)} R {f('recall', 1)} MCC {f('mcc')} AUROC {d['auroc'][0]:.3f} AP {d['ap'][0]:.3f} "
            f"FP {d['fp'][0]:.2f} [{d['fp'][1]:.2f}, {d['fp'][2]:.2f}] traps {d['trap_two_step'][2]*100:.1f}%/{d['trap_common_src'][2]*100:.1f}%")


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "test":
        rows = [json.loads(l) for l in (RES / "test2.jsonl").read_text().splitlines()]
        s = summarize(rows)
        (RES / "test2_summary.json").write_text(json.dumps(s, indent=2))
        for m in METHODS:
            print(line(m, s["methods"][m]))
        for p in s["paired"]:
            print(f"PROPOSED vs {p['method']:17s} p_holm={p['p_holm']:.1e} W/T/L={p['wins']}/{p['ties']}/{p['losses']} median dMCC={p['median_diff']:+.3f}")
        print("equivalence deploy vs sim:", s["equivalence_deploy_vs_sim"])
        for m in ["SIGORTH", "PROPOSED"]:
            d = s["methods"][m]; print(m, "precision pooled %.3f, non-empty mean %.3f (empty outputs: %d)" % (d["precision_pooled"], d["precision_nonempty"][0], d["precision_nonempty"][1]))
    else:
        rows = [json.loads(l) for l in (RES / "conditions2.jsonl").read_text().splitlines()]
        groups = {}
        for r in rows:
            groups.setdefault(r["task"]["cond"], []).append(r)
        summ = {c: summarize(rs) for c, rs in groups.items()}
        (RES / "conditions2_summary.json").write_text(json.dumps(summ, indent=2))
        hdr = ["ORACLE", "RAW", "SIGORTH", "TRGC", "GPDC", "PROPOSED", "PROPOSED-DEPLOY", "PROPOSED-POSTINV"]
        print("cond".ljust(20) + " ".join(h[:9].rjust(9) for h in hdr) + "  | best non-oracle")
        for c, s in summ.items():
            mccs = {m: s["methods"][m]["mcc"][0] for m in hdr}
            best = max((m for m in hdr if m != "ORACLE"), key=lambda m: mccs[m])
            print(c.ljust(20) + " ".join(f"{mccs[m]:9.3f}" for m in hdr) + f"  | {best}")
        print("\nFP per network (mean):")
        for c, s in summ.items():
            print(c.ljust(20) + " ".join(f"{s['methods'][m]['fp'][0]:9.2f}" for m in hdr))
        print("\nPROPOSED vs comparators (Holm p, W/T/L):")
        for c, s in summ.items():
            print(c.ljust(20) + "  ".join(f"{p['method'][:8]}:{p['p_holm']:.0e}({p['wins']}/{p['ties']}/{p['losses']})" for p in s["paired"]))
