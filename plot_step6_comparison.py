import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_phase1_comparison():
    G_true = np.loadtxt("G_true.csv", delimiter=",", dtype=int)
    G_est = np.loadtxt("G_est.csv", delimiter=",", dtype=int)
    W_est = np.loadtxt("W_est.csv", delimiter=",")

    # Build an overlay matrix: 0 = True Negative, 1 = False Positive (Leakage), 2 = True Positive
    overlay = np.zeros_like(G_true)
    overlay[(G_true == 0) & (G_est == 1)] = 1  # False Positive (Orange)
    overlay[(G_true == 1) & (G_est == 1)] = 2  # True Positive (Green)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150)
    nodes = [f"N{i}" for i in range(1, 11)]

    # Panel 1: Ground Truth (G_true)
    axes[0].imshow(G_true, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("1. Ground Truth Blueprint (G_true)\n[5 True Causal Pathways]", fontsize=12, fontweight="bold")

    # Panel 2: Recovered Binary Network (G_est)
    axes[1].imshow(G_est, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("2. Recovered eLORETA + Ridge-MVAR (G_est)\n[Alpha = 0.5, 99.9th Pct Threshold]", fontsize=12, fontweight="bold")

    # Panel 3: Diagnostic Overlay
    cmap_overlay = ListedColormap(["#f7f7f7", "#ff9800", "#2ca02c"])
    axes[2].imshow(overlay, cmap=cmap_overlay, vmin=0, vmax=2)
    axes[2].set_title("3. Diagnostic Overlay (94.4% Accuracy)\n[Green = True Hit (5/5) | Orange = Spatial Leakage (5)]", fontsize=12, fontweight="bold")

    for idx, ax in enumerate(axes):
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(nodes, fontweight="bold")
        ax.set_yticklabels(nodes, fontweight="bold")
        ax.set_xlabel("Target Node (Receiver j)", fontweight="bold")
        ax.set_ylabel("Source Node (Sender i)", fontweight="bold")
        # Draw gridlines and numbers inside cells
        for i in range(10):
            for j in range(10):
                val = G_true[i, j] if idx == 0 else G_est[i, j]
                text_color = "white" if (val == 1 and idx < 2) or (idx == 2 and overlay[i, j] > 0) else "gray"
                ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig("step6_phase1_validation.png")
    plt.close()
    print("Saved visual comparison to: 'step6_phase1_validation.png'")

if __name__ == "__main__":
    plot_phase1_comparison()