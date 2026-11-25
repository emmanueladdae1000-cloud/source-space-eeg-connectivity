import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef


def plot_results_matrix(
    g_true_csv="G_true.csv",
    g_est_csv="G_est.csv",
    thresh_csv="null_threshold.csv",
    out_png="figure_results_matrix_12edge.png",
):
    G_true = np.loadtxt(g_true_csv, delimiter=",", dtype=int)
    G_est = np.loadtxt(g_est_csv, delimiter=",", dtype=int)
    thresh = float(np.loadtxt(thresh_csv, delimiter=","))

    n = G_true.shape[0]
    mask = ~np.eye(n, dtype=bool)
    y_true = G_true[mask]
    y_pred = G_est[mask]

    tp = int(np.sum((G_true == 1) & (G_est == 1) & mask))
    fp = int(np.sum((G_true == 0) & (G_est == 1) & mask))
    fn = int(np.sum((G_true == 1) & (G_est == 0) & mask))
    tn = int(np.sum((G_true == 0) & (G_est == 0) & mask))

    density_pct = 100.0 * np.sum(y_true) / len(y_true)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    palette = {
        "gt_edge": "#1A237E",   # Deep Royal Indigo
        "est_edge": "#0B4F6C",  # Deep Ocean Teal
        "tp": "#116530",        # Deep Forest Green
        "fp": "#B91C1C",        # Deep Crimson Red
        "fn": "#6D28D9",        # Deep Royal Purple
        "diag": "#64748B",      # Deep Slate Gray
        "tn_bg": "#EEF2F6",     # Crisp Light Cool Gray
        "zero_txt": "#7C8A99",  # Medium Cool Gray for 0 labels
    }

    def prep_binary(M):
        grid = M.copy()
        np.fill_diagonal(grid, 2)
        return grid

    diag_grid = np.zeros((n, n), dtype=int)
    diag_grid[(G_true == 1) & (G_est == 1)] = 1  # TP
    diag_grid[(G_true == 0) & (G_est == 1)] = 2  # FP
    diag_grid[(G_true == 1) & (G_est == 0)] = 3  # FN
    np.fill_diagonal(diag_grid, 4)               # Excluded self-loop

    cmap_gt = ListedColormap([palette["tn_bg"], palette["gt_edge"], palette["diag"]])
    cmap_est = ListedColormap([palette["tn_bg"], palette["est_edge"], palette["diag"]])
    cmap_diag = ListedColormap([
        palette["tn_bg"],
        palette["tp"],
        palette["fp"],
        palette["fn"],
        palette["diag"],
    ])

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.6), dpi=300, facecolor="white")
    labels = [f"N{i}" for i in range(1, n + 1)]

    titles = [
        f"(a) Ground-Truth Graph ($G_{{true}}$)\n[{int(np.sum(y_true))} Directed Edges, {density_pct:.1f}% Density]",
        f"(b) Recovered Estimate ($\\hat{{G}}$)\n[Ridge-MVAR $\\alpha=0.5$, Sym. Orthog., $\\tau={thresh:.3f}$]",
        f"(c) Diagnostic Overlay\n[F1 = {f1:.3f} | Prec = {prec*100:.1f}% | Rec = {rec*100:.1f}% | MCC = {mcc:.3f}]",
    ]

    grids = [prep_binary(G_true), prep_binary(G_est), diag_grid]
    cmaps = [cmap_gt, cmap_est, cmap_diag]
    vmaxs = [2, 2, 4]

    for ax_idx, (ax, grid, cmap, vmax, title) in enumerate(zip(axes, grids, cmaps, vmaxs, titles)):
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")

        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_edgecolor("#1E293B")
            spine.set_linewidth(1.1)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, fontsize=8.8, fontweight="bold", color="#1E293B")
        ax.set_yticklabels(labels, fontsize=8.8, fontweight="bold", color="#1E293B")

        ax.set_xlabel("Target Node (Receiver $j$)", fontsize=9.8, fontweight="bold", labelpad=7, color="#111827")
        if ax_idx == 0:
            ax.set_ylabel("Source Node (Sender $i$)", fontsize=9.8, fontweight="bold", labelpad=7, color="#111827")

        ax.set_title(title, fontsize=10.2, fontweight="bold", pad=11, color="#111827")

        for r in range(n):
            for c in range(n):
                val = grid[r, c]
                if ax_idx < 2:
                    if val == 2:
                        ax.text(c, r, "-", ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
                    elif val == 1:
                        ax.text(c, r, "1", ha="center", va="center", fontsize=9.2, fontweight="bold", color="white")
                    else:
                        ax.text(c, r, "0", ha="center", va="center", fontsize=8.0, color=palette["zero_txt"])
                else:
                    if val == 4:
                        ax.text(c, r, "-", ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
                    elif val == 1:
                        ax.text(c, r, "TP", ha="center", va="center", fontsize=8.0, fontweight="bold", color="white")
                    elif val == 2:
                        ax.text(c, r, "FP", ha="center", va="center", fontsize=8.0, fontweight="bold", color="white")
                    elif val == 3:
                        ax.text(c, r, "FN", ha="center", va="center", fontsize=8.0, fontweight="bold", color="white")
                    else:
                        ax.text(c, r, "0", ha="center", va="center", fontsize=8.0, color=palette["zero_txt"])

    legend_handles = [
        Patch(facecolor=palette["gt_edge"], edgecolor="white", label="Ground Truth (a)"),
        Patch(facecolor=palette["est_edge"], edgecolor="white", label="Recovered (b)"),
        Patch(facecolor=palette["tp"], edgecolor="white", label=f"True Positive - TP ({tp})"),
        Patch(facecolor=palette["fp"], edgecolor="white", label=f"False Positive - FP ({fp})"),
        Patch(facecolor=palette["fn"], edgecolor="white", label=f"False Negative - FN ({fn})"),
        Patch(facecolor=palette["tn_bg"], edgecolor="#94A3B8", label=f"True Negative - TN ({tn})"),
        Patch(facecolor=palette["diag"], edgecolor="white", label="Excluded Self-Loop"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#94A3B8",
        fontsize=8.6,
        columnspacing=1.6,
        bbox_to_anchor=(0.5, 0.005),
    )

    plt.subplots_adjust(left=0.05, right=0.98, top=0.84, bottom=0.20, wspace=0.18)
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    print(f"Saved deep-color 3-panel matrix figure to: {out_png}")


if __name__ == "__main__":
    plot_results_matrix()