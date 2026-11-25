"""Figures 6-8 for manuscript v3 (extended benchmark)."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
COL = {"ORACLE": "#6B7280", "RAW": "#9CA3AF", "SIGORTH": "#7C3AED", "TRGC": "#B45309", "GPDC": "#059669",
       "EDGEZ": "#EA580C", "PROPOSED": "#1D4ED8", "PROPOSED-DEPLOY": "#0891B2", "PROPOSED-POSTINV": "#DC2626"}
LAB = {"ORACLE": "Oracle", "RAW": "Ridge-MVAR", "SIGORTH": "Signal-level\northog.", "TRGC": "TRGC", "GPDC": "gPDC",
       "EDGEZ": "Edge-wise\nnull z", "PROPOSED": "Proposed\n(sim. null)", "PROPOSED-DEPLOY": "Proposed\n(deploy. null)",
       "PROPOSED-POSTINV": "Proposed\n(post-inv. null)"}
ORDER = ["ORACLE", "RAW", "SIGORTH", "TRGC", "GPDC", "PROPOSED", "PROPOSED-DEPLOY", "PROPOSED-POSTINV"]
rows = [json.loads(l) for l in (RES / "test2.jsonl").read_text().splitlines()]
summ = json.loads((RES / "test2_summary.json").read_text())
cond = json.loads((RES / "conditions2_summary.json").read_text())
hub = json.loads((RES / "hub_case_study.json").read_text())

# ---------------------------------------------------------------- Figure 6 (case study, 2 panels)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1.3]})
null = np.load(RES / "hub_null_prop005.npy"); tau = hub["PROPOSED_beta0.05"]["tau"]
ax[0].hist(null, bins=40, color="#9CA3AF", edgecolor="white", linewidth=0.3)
ax[0].axvline(tau, color="#EA580C", ls="--", lw=1.2); ax[0].axvline(null.mean(), color="#374151", ls=":", lw=1.0)
ax[0].text(tau + 0.008, ax[0].get_ylim()[1] * 0.8, f"τ = {tau:.3f}\n(99th pct)", color="#EA580C", fontsize=7.5)
ax[0].text(null.mean() + 0.01, ax[0].get_ylim()[1] * 0.45, f"mean = {null.mean():.3f}", fontsize=7.5)
ax[0].set_xlabel("Null edge weight"); ax[0].set_ylabel("Count")
ax[0].set_title(f"(a) Pooled null ({len(null):,} weights)", fontsize=9)
meth = [("EDGEZ", "EDGEZ"), ("RAW", "RAW"), ("SIGORTH", "SIGORTH"), ("PROPOSED", "PROPOSED_beta0.05")]
mets = ["precision", "recall", "f1", "mcc"]; x = np.arange(4); w = 0.2
for k, (name, key) in enumerate(meth):
    vals = [hub[key][m] for m in mets]
    ax[1].bar(x + (k - 1.5) * w, vals, w, color=COL[name], label=LAB[name].replace("\n", " "))
    for xi, v in zip(x + (k - 1.5) * w, vals):
        ax[1].text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=5.5, rotation=90)
ax[1].set_xticks(x); ax[1].set_xticklabels(["Precision", "Recall", "F1", "MCC"]); ax[1].set_ylim(0, 1.12)
ax[1].set_title("(b) Methods on the case-study network (own nulls)", fontsize=9)
fig.legend(*ax[1].get_legend_handles_labels(), fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(0.99, -0.02), ncol=4)
fig.tight_layout(rect=[0, 0.08, 1, 1]); fig.savefig(FIG / "fig6_case_study_v3.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 7 (benchmark, 3 panels)
fig = plt.figure(figsize=(7.2, 5.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.25, 1])
a0 = fig.add_subplot(gs[0, :]); a1 = fig.add_subplot(gs[1, 0]); a2 = fig.add_subplot(gs[1, 1])
rng = np.random.default_rng(0)
for k, m in enumerate(ORDER):
    v = np.array([r[m]["mcc"] for r in rows])
    a0.boxplot(v, positions=[k], widths=0.55, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=COL[m], alpha=0.25, edgecolor=COL[m]), medianprops=dict(color=COL[m], lw=1.5),
               whiskerprops=dict(color=COL[m]), capprops=dict(color=COL[m]))
    a0.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=3.5, color=COL[m], alpha=0.55, lw=0)
    a0.text(k, 1.05, f"{summ['methods'][m]['mcc'][0]:.2f}", ha="center", fontsize=7.5, color=COL[m], fontweight="bold")
a0.set_xticks(range(len(ORDER))); a0.set_xticklabels([LAB[m] for m in ORDER], fontsize=7)
a0.set_ylabel("MCC per network"); a0.set_ylim(-0.12, 1.12)
a0.set_title("(a) Edge recovery on 100 held-out random networks (each method tuned on 20 separate networks; mean above)", fontsize=9)
offs = {"ORACLE": (-12, -13), "RAW": (6, 3), "SIGORTH": (-34, 9), "TRGC": (6, -3), "GPDC": (6, 3),
        "PROPOSED": (7, 2), "PROPOSED-DEPLOY": (7, -9), "PROPOSED-POSTINV": (-46, 8)}
for m in ORDER:
    d = summ["methods"][m]; p, r_ = d["precision"], d["recall"]
    a1.errorbar(r_[0], p[0], xerr=[[r_[0] - r_[1]], [r_[2] - r_[0]]], yerr=[[p[0] - p[1]], [p[2] - p[0]]],
                fmt="o", color=COL[m], ms=4.5, capsize=2, lw=1)
    a1.annotate(LAB[m].replace("\n", " "), (r_[0], p[0]), textcoords="offset points", xytext=offs[m], fontsize=6.3, color=COL[m])
a1.set_xlabel("Recall"); a1.set_ylabel("Precision"); a1.set_xlim(0.25, 1.05); a1.set_ylim(0.1, 0.95)
a1.set_title("(b) Mean precision and recall (95% CI)", fontsize=9)
nulls = ["PROPOSED", "PROPOSED-DEPLOY", "PROPOSED-POSTINV"]
fp = [summ["methods"][m]["fp"] for m in nulls]
bars = a2.bar(range(3), [f[0] for f in fp], yerr=[[f[0] - f[1] for f in fp], [f[2] - f[0] for f in fp]],
              color=[COL[m] for m in nulls], capsize=3)
for i, (f, m) in enumerate(zip(fp, nulls)):
    a2.text(i, f[2] + 1.5, f"{f[0]:.1f}\nMCC {summ['methods'][m]['mcc'][0]:.2f}", ha="center", fontsize=7)
a2.axhline(0.78, color="#374151", ls=":", lw=1); a2.text(2.45, 3.0, "nominal\n0.78", fontsize=6.5, ha="right")
a2.set_xticks(range(3)); a2.set_xticklabels(["Simulation\nnull", "Deployable\nnull", "Post-inversion\nnull"], fontsize=7.5)
a2.set_ylabel("False positives / network"); a2.set_ylim(0, 72)
a2.set_title("(c) Null ablation (same estimator)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig7_benchmark_v3.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 8 (conditions heatmap)
names = {"C1_unequal": "C1 unequal innovation variance", "C2_correlated": "C2 correlated innovations (zero-lag)",
         "C3_oscillatory": "C3 oscillatory dynamics (8–12 Hz)", "C4_background": "C4 background sources (−3 dB)",
         "C5_random_locations": "C5 random source locations", "C6_patch_readout": "C6 patch (PCA) read-out",
         "C7_short_20s": "C7 short recordings (20 s)", "C8_combined": "C8 combined (C1+C4+C5+C6)",
         "R_snr0": "SNR 0 dB", "R_snr5": "SNR 5 dB", "R_snr20": "SNR 20 dB", "R_edges9": "9 edges",
         "R_edges15": "15 edges", "R_edges18": "18 edges", "R_skull": "Skull-conductivity error", "R_tilt30": "30° orientation error"}
keys = list(names)
M = np.array([[cond[c]["methods"][m]["mcc"][0] for m in ORDER] for c in keys])
fig, ax = plt.subplots(figsize=(7.2, 5.0))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(len(keys)):
    best = max(range(1, len(ORDER)), key=lambda j: M[i, j])
    for j in range(len(ORDER)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if M[i, j] > 0.55 else "#111827", fontweight="bold" if j == best else "normal")
    ax.add_patch(plt.Rectangle((best - 0.5, i - 0.5), 1, 1, fill=False, ec="#EA580C", lw=1.6))
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LAB[m] for m in ORDER], fontsize=7)
ax.set_yticks(range(len(keys))); ax.set_yticklabels([names[k] for k in keys], fontsize=7.5)
ax.axhline(7.5, color="#111827", lw=0.8)
ax.set_title("Mean MCC over 30 networks per condition (orange box: best non-oracle method)", fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("MCC", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig8_conditions_v3.png"); plt.close(fig)
print("ok")
