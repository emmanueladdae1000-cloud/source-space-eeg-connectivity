import numpy as np
import matplotlib.pyplot as plt


def zscore_rows(Y):
    return (Y - np.mean(Y, axis=1, keepdims=True)) / np.std(Y, axis=1, keepdims=True)


def fit_ridge_mvar_raw(Y_z, alpha=0.5, order=1):
    """Standard Ridge-MVAR without zero-lag residual orthogonalization."""
    X = Y_z.T
    n_samples, n_nodes = X.shape
    Y_curr = X[order:]
    X_lag = np.hstack([X[order - p : n_samples - p] for p in range(1, order + 1)])
    n_features = X_lag.shape[1]
    A_full = np.linalg.solve(X_lag.T @ X_lag + alpha * np.eye(n_features), X_lag.T @ Y_curr)
    A_lags = A_full.reshape(order, n_nodes, n_nodes)
    W = np.linalg.norm(A_lags, axis=0)
    np.fill_diagonal(W, 0.0)
    return W


def plot_per_edge_recovery(
    w_true_csv="W_true.csv",
    g_true_csv="G_true.csv",
    y_clean_csv="Y_clean.csv",
    y_hat_csv="Y_hat_eloreta.csv",
    w_est_csv="W_est.csv",
    thresh_csv="null_threshold.csv",
    alpha=0.5,
    out_png="figure_per_edge_recovery_final.png",
):
    W_true = np.loadtxt(w_true_csv, delimiter=",")
    G_true = np.loadtxt(g_true_csv, delimiter=",", dtype=int)
    Y_clean = np.loadtxt(y_clean_csv, delimiter=",")
    Y_hat = np.loadtxt(y_hat_csv, delimiter=",")
    W_sym = np.loadtxt(w_est_csv, delimiter=",")
    thresh = float(np.loadtxt(thresh_csv, delimiter=","))

    W_clean = fit_ridge_mvar_raw(zscore_rows(Y_clean), alpha=alpha)
    W_raw = fit_ridge_mvar_raw(zscore_rows(Y_hat), alpha=alpha)

    in_deg = G_true.sum(axis=0)
    edges = [(int(i), int(j)) for i, j in zip(*np.where(G_true == 1))]
    edges.sort(key=lambda e: abs(W_true[e[0], e[1]]), reverse=True)

    x = np.arange(len(edges))
    offsets = [-0.27, -0.09, 0.09, 0.27]

    w_t_vals = [abs(W_true[i, j]) for i, j in edges]
    w_c_vals = [W_clean[i, j] for i, j in edges]
    w_r_vals = [W_raw[i, j] for i, j in edges]
    w_s_vals = [W_sym[i, j] for i, j in edges]

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    fig, ax = plt.subplots(figsize=(13.8, 4.9), dpi=300, facecolor="white")

    for idx in range(len(edges)):
        ax.plot(
            [x[idx] + offsets[2], x[idx] + offsets[3]],
            [w_r_vals[idx], w_s_vals[idx]],
            color="#CBD5E1", linewidth=1.2, zorder=2,
        )

    ax.scatter(x + offsets[0], w_t_vals, marker="D", s=46, color="#475569", label="W_true", zorder=4)
    ax.scatter(x + offsets[1], w_c_vals, marker="s", s=42, color="#0F9D58", label="Clean sources (oracle)", zorder=4)
    ax.scatter(x + offsets[2], w_r_vals, marker="^", s=44, color="#7C3AED", label="eLORETA, raw Ridge-MVAR", zorder=4)
    ax.scatter(x + offsets[3], w_s_vals, marker="o", s=48, color="#1D6FD8", label="eLORETA + Sym. Orthog. (Proposed)", zorder=5)

    ax.axhline(
        thresh, color="#D9480F", linestyle="--", linewidth=1.6,
        label=f"Locked cutoff = {thresh:.3f}", zorder=3,
    )

    for idx, val in enumerate(w_s_vals):
        if val > thresh:
            ax.text(x[idx], -0.055, "hit", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#1D6FD8")
        else:
            ax.text(x[idx], -0.055, "miss", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#B91C1C")

    xlabels = [f"{i+1}->{j+1}\n(in {in_deg[j]})" for i, j in edges]
    ymax = max(max(w_t_vals), max(w_c_vals), max(w_r_vals), max(w_s_vals), thresh) * 1.18
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.4, fontweight="bold", color="#1E293B")
    ax.set_xlim(-0.6, len(edges) - 0.4)
    ax.set_ylim(-0.11, ymax)
    ax.set_ylabel("Coupling / Score", fontsize=9.8, fontweight="bold", color="#111827")
    ax.set_xlabel("True Directed Edge (Target In-Degree)", fontsize=9.8, fontweight="bold", color="#111827", labelpad=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=5, frameon=False, fontsize=8.4)
    ax.grid(axis="y", linestyle=":", color="#CBD5E1", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")

    hits = sum(v > thresh for v in w_s_vals)
    print(f"Saved {out_png} | True hits above {thresh:.4f}: {hits} / {len(edges)}")


if __name__ == "__main__":
    plot_per_edge_recovery()