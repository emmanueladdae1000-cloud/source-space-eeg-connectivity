"""Figures 6-8 for the revised manuscript, drawn only from results/*.json(l) and the hub run files."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)
HUB = HERE / "data" / "run_hub"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
COL = {"ORACLE": "#6B7280", "RAW": "#9CA3AF", "SIGORTH": "#7C3AED", "EDGEZ": "#EA580C",
       "PROPOSED": "#1D4ED8", "PROPOSED-DEPLOY": "#0891B2"}
LAB = {"ORACLE": "Oracle\n(clean sources)", "RAW": "Ridge-MVAR", "SIGORTH": "Signal-level\northog.",
       "EDGEZ": "Edge-wise\nnull z", "PROPOSED": "Proposed", "PROPOSED-DEPLOY": "Proposed,\ndeployable null"}
rows = [json.loads(l) for l in (RES / "test.jsonl").read_text().splitlines()]
summ = json.loads((RES / "test_summary.json").read_text())
hub = json.loads((RES / "hub_case_study.json").read_text())
rob = [json.loads(l) for l in (RES / "robust.jsonl").read_text().splitlines()]

# ---------------------------------------------------------------- Figure 6: hub case study
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.15, 1, 1.25]})
null = np.load(RES / "hub_null_prop005.npy")
tau = hub["PROPOSED_beta0.05"]["tau"]
ax[0].hist(null, bins=40, color="#9CA3AF", edgecolor="white", linewidth=0.3)
ax[0].axvline(tau, color="#EA580C", ls="--", lw=1.2)
ax[0].axvline(null.mean(), color="#374151", ls=":", lw=1.0)
ax[0].text(tau + 0.01, ax[0].get_ylim()[1] * 0.8, f"τ = {tau:.3f}\n(99th pct)", color="#EA580C", fontsize=7.5)
ax[0].text(null.mean() + 0.01, ax[0].get_ylim()[1] * 0.45, f"mean = {null.mean():.3f}", fontsize=7.5)
ax[0].set_xlabel("Null edge weight"); ax[0].set_ylabel("Count")
ax[0].set_title(f"(a) Null distribution ({len(null):,} edges)", fontsize=9)
G = np.loadtxt(HUB / "G_true.csv", delimiter=",", dtype=int)
W = np.load(RES / "hub_W_prop005.npy")
Ge = (W > tau).astype(int)
grid = np.zeros((10, 10)); grid[(G == 1) & (Ge == 1)] = 1; grid[(G == 0) & (Ge == 1)] = 2; grid[(G == 1) & (Ge == 0)] = 3
np.fill_diagonal(grid, 4)
from matplotlib.colors import ListedColormap
ax[1].imshow(grid, cmap=ListedColormap(["#F3F4F6", "#15803D", "#DC2626", "#7C3AED", "#9CA3AF"]), vmin=0, vmax=4)
for i in range(10):
    for j in range(10):
        lab = {1: "TP", 2: "FP", 3: "FN"}.get(int(grid[i, j]))
        if lab:
            ax[1].text(j, i, lab, ha="center", va="center", color="white", fontsize=6, fontweight="bold")
ax[1].set_xticks(range(10)); ax[1].set_yticks(range(10))
ax[1].set_xticklabels(range(1, 11), fontsize=7); ax[1].set_yticklabels(range(1, 11), fontsize=7)
ax[1].set_xlabel("Target node"); ax[1].set_ylabel("Source node")
ax[1].set_title("(b) Recovery (10 TP, 2 FP, 2 FN)", fontsize=9)
methods = [("EDGEZ", "EDGEZ"), ("RAW", "RAW"), ("SIGORTH", "SIGORTH"), ("PROPOSED", "PROPOSED_beta0.05")]
mets = ["precision", "recall", "f1", "mcc"]
x = np.arange(len(mets)); w = 0.2
for k, (name, key) in enumerate(methods):
    vals = [hub[key][m] for m in mets]
    b = ax[2].bar(x + (k - 1.5) * w, vals, w, color=COL[name], label=LAB[name].replace("\n", " "))
    for xi, v in zip(x + (k - 1.5) * w, vals):
        ax[2].text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=5.5, rotation=90)
ax[2].set_xticks(x); ax[2].set_xticklabels(["Precision", "Recall", "F1", "MCC"]); ax[2].set_ylim(0, 1.12)
ax[2].set_title("(c) Methods on this graph (own nulls)", fontsize=9)
fig.legend(*ax[2].get_legend_handles_labels(), fontsize=7, frameon=False, loc="lower right",
           bbox_to_anchor=(0.99, -0.02), ncol=4)
fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(FIG / "fig6_hub_case_study.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 7: benchmark
order = ["ORACLE", "RAW", "SIGORTH", "EDGEZ", "PROPOSED", "PROPOSED-DEPLOY"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
rng = np.random.default_rng(0)
for k, m in enumerate(order):
    v = np.array([r[m]["mcc"] for r in rows])
    ax[0].boxplot(v, positions=[k], widths=0.55, showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor=COL[m], alpha=0.25, edgecolor=COL[m]), medianprops=dict(color=COL[m], lw=1.5),
                  whiskerprops=dict(color=COL[m]), capprops=dict(color=COL[m]))
    ax[0].scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color=COL[m], alpha=0.6, lw=0)
    mu = summ["methods"][m]["mcc"][0]
    ax[0].text(k, 1.04, f"{mu:.2f}", ha="center", fontsize=7.5, color=COL[m], fontweight="bold")
ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels([LAB[m] for m in order], fontsize=7)
ax[0].set_ylabel("MCC per network"); ax[0].set_ylim(-0.1, 1.1)
ax[0].set_title("(a) Edge recovery on 100 held-out random networks (mean above)", fontsize=9)
for m in order:
    d = summ["methods"][m]
    p, r_ = d["precision"], d["recall"]
    ax[1].errorbar(r_[0], p[0], xerr=[[r_[0] - r_[1]], [r_[2] - r_[0]]], yerr=[[p[0] - p[1]], [p[2] - p[0]]],
                   fmt="o", color=COL[m], ms=5, capsize=2, lw=1)
    offs = {"SIGORTH": (-10, 9), "PROPOSED": (9, 1), "ORACLE": (-20, 7), "PROPOSED-DEPLOY": (6, -10),
            "RAW": (6, 4), "EDGEZ": (-30, 8)}[m]
    ax[1].annotate(LAB[m].replace("\n", " "), (r_[0], p[0]), textcoords="offset points", xytext=offs,
                   fontsize=6.5, color=COL[m], ha="left")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_xlim(0.3, 1.05); ax[1].set_ylim(0.1, 1.0)
ax[1].set_title("(b) Mean precision and recall (95% CI)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig7_benchmark.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 8: robustness
def grp(snr=10.0, edges=12, head="template"):
    src = rob if not (snr == 10.0 and edges == 12 and head == "template") else [r for r in rows if r["seed"] <= 30]
    return [r for r in src if r["snr_db"] == snr and r["n_edges"] == edges and r["head"] == head]


def mci(rs, m):
    v = np.array([r[m]["mcc"] for r in rs]); b = np.random.default_rng(0).integers(0, len(v), (5000, len(v)))
    mm = v[b].mean(1); return v.mean(), np.percentile(mm, 2.5), np.percentile(mm, 97.5)


fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
panels = [("SNR (dB)", [0.0, 5.0, 10.0, 20.0], lambda v: grp(snr=v)),
          ("True edges (of 90)", [9, 12, 15, 18], lambda v: grp(edges=v)),
          ("Head model used for analysis", ["template", "mismatch", "tilt30"], lambda v: grp(head=v))]
for a, (xl, vals, getter) in zip(ax, panels):
    for m in ["RAW", "PROPOSED"]:
        stats = [mci(getter(v), m) for v in vals]
        xs = np.arange(len(vals))
        mu = [s[0] for s in stats]; lo = [s[0] - s[1] for s in stats]; hi = [s[2] - s[0] for s in stats]
        a.errorbar(xs, mu, yerr=[lo, hi], marker="o", ms=4, capsize=2, color=COL[m], label=LAB[m].replace("\n", " "))
    a.set_xticks(np.arange(len(vals)))
    a.set_xticklabels(["exact", "skull σ", "30° tilt"] if xl.startswith("Head") else [f"{v:g}" for v in vals], fontsize=7.5)
    a.set_xlabel(xl); a.set_ylim(0, 1)
ax[0].set_ylabel("MCC (mean, 95% CI, n = 30)")
ax[0].legend(fontsize=7, frameon=False, loc="lower right")
for a, t in zip(ax, ["(a) Sensor noise", "(b) Network density", "(c) Forward-model error"]):
    a.set_title(t, fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig8_robustness.png"); plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
