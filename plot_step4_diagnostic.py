import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_step4_diagnostic(
    y_clean_csv="Y_clean.csv",
    y_hat_csv="Y_hat_eloreta.csv",
    out_png="figure_step4_leakage_diagnostic.png",
):
    Y_clean = np.loadtxt(y_clean_csv, delimiter=",")
    Y_hat = np.loadtxt(y_hat_csv, delimiter=",")

    n = Y_clean.shape[0]
    labels = [f"N{i}" for i in range(1, n + 1)]
    mask = ~np.eye(n, dtype=bool)

    # Panel (a): Per-node waveform correlation |r|
    node_r = np.array([abs(np.corrcoef(Y_clean[i], Y_hat[i])[0, 1]) for i in range(n)])
    mean_node_r = float(np.mean(node_r))

    # Panel (b) & (c): Pairwise zero-lag |r| matrices
    R_true = np.abs(np.corrcoef(Y_clean))
    R_hat = np.abs(np.corrcoef(Y_hat))
    mean_r_true = float(R_true[mask].mean())
    mean_r_hat = float(R_hat[mask].mean())

    R_true_plot = np.ma.masked_where(np.eye(n, dtype=bool), R_true)
    R_hat_plot = np.ma.masked_where(np.eye(n, dtype=bool), R_hat)

    vmax = max(float(R_true[mask].max()), float(R_hat[mask].max()), 0.5)
    vmax = min(1.0, np.ceil(vmax * 20.0) / 20.0)

    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#CBD5E1")

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    fig = plt.figure(figsize=(15.6, 4.6), dpi=300, facecolor="white")
    gs = GridSpec(1, 4, width_ratios=[1.18, 1.0, 1.0, 0.055], wspace=0.22, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])

    # Panel (a): Bar chart of waveform recovery
    x = np.arange(n)
    bars = ax0.bar(x, node_r, width=0.62, color="#1D6FD8", edgecolor="#0F4C9A", linewidth=0.6, zorder=3)
    ax0.axhline(
        mean_node_r, color="#475569", linestyle="--", linewidth=1.25,
        label=f"Mean = {mean_node_r:.2f}", zorder=4,
    )
    for bar, val in zip(bars, node_r):
        ax0.text(
            bar.get_x() + bar.get_width() / 2.0, val + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#1E293B",
        )
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=8.5, fontweight="bold", color="#1E293B")
    ax0.set_ylim(0.0, 1.15)
    ax0.set_ylabel("Waveform Correlation |r| ($Y_{clean}$ vs. $\\hat{Y}$)", fontsize=9.2, fontweight="bold")
    ax0.set_title("(a) Waveform Recovery per Node", fontsize=10.4, fontweight="bold", pad=10, color="#111827")
    ax0.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.4)
    ax0.grid(axis="y", linestyle=":", color="#CBD5E1", zorder=0)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # Panel (b) & (c): Zero-lag |r| heatmaps with shared scale
    for ax, mat, title in [
        (ax1, R_true_plot, f"(b) True Sources Zero-Lag |r| (Mean = {mean_r_true:.2f})"),
        (ax2, R_hat_plot, f"(c) eLORETA Estimates Zero-Lag |r| (Mean = {mean_r_hat:.2f})"),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, fontsize=8.2, fontweight="bold", color="#1E293B")
        ax.set_yticklabels(labels, fontsize=8.2, fontweight="bold", color="#1E293B")
        ax.set_title(title, fontsize=10.1, fontweight="bold", pad=10, color="#111827")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
            spine.set_linewidth(0.9)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Zero-Lag |r|", fontsize=9.2, fontweight="bold", color="#1E293B")
    cbar.ax.tick_params(labelsize=8.2)

    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")

    print("=== Verified Step 4 Diagnostic Numbers ===")
    for i, r in enumerate(node_r, 1):
        print(f"Node {i:2d} |r| : {r:.4f}")
    print(f"Mean Waveform |r|        : {mean_node_r:.4f}")
    print(f"True Zero-Lag Mean |r|   : {mean_r_true:.4f} (max = {R_true[mask].max():.4f})")
    print(f"eLORETA Zero-Lag Mean |r|: {mean_r_hat:.4f} (max = {R_hat[mask].max():.4f})")
    print(f"Saved real figure to     : {out_png}")


if __name__ == "__main__":
    plot_step4_diagnostic()