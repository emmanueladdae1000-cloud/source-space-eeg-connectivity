import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, ConnectionStyle
from matplotlib.lines import Line2D
import mne


EDGE_STYLES = {
    "convergent": {
        "color": "#D62728",  # Crimson Red
        "label": "Convergent Hub Inputs (into N5)",
    },
    "relay": {
        "color": "#2CA02C",  # Emerald Green
        "label": "Relay Hub Pathways (via N3)",
    },
    "cascade": {
        "color": "#1F77B4",  # Royal Blue
        "label": "Primary Source Cascade (from N9)",
    },
    "divergent": {
        "color": "#8E44AD",  # Rich Purple
        "label": "Divergent Branch (from N8)",
    },
}


def classify_edge(i, j, in_deg, out_deg):
    """Assigns each directed edge (i -> j) to a structural motif color group."""
    if in_deg[j] >= 3:
        return "convergent"
    if (in_deg[i] >= 2 and out_deg[i] >= 2) or (in_deg[j] >= 2 and out_deg[j] >= 2):
        return "relay"
    if in_deg[i] >= 1 and out_deg[i] >= 2:
        return "divergent"
    return "cascade"


def get_arc_samples(p1, p2, rad, n_pts=25):
    """Samples interior points along matplotlib's exact Arc3 Bezier curve."""
    path = ConnectionStyle("Arc3", rad=rad)(p1, p2)
    v = path.vertices  # [start, control, end]
    t = np.linspace(0.12, 0.88, n_pts)[:, None]
    return ((1 - t) ** 2) * v[0] + 2 * (1 - t) * t * v[1] + (t ** 2) * v[2]


def choose_clean_rad(i, j, xy, used_curves):
    """Selects arc curvature that avoids neighboring nodes and previous edges."""
    candidates = (-0.18, 0.18, -0.28, 0.28, -0.38, 0.38, -0.48, 0.48)
    p1, p2 = xy[i], xy[j]
    other_idx = [k for k in range(len(xy)) if k not in (i, j)]
    others = xy[other_idx]

    best_rad, best_pts, best_score = 0.18, None, -1e9
    for rad in candidates:
        pts = get_arc_samples(p1, p2, rad)
        min_node_d = float(np.linalg.norm(pts[:, None, :] - others[None, :, :], axis=2).min())
        min_curve_d = 25.0
        for prev in used_curves:
            d_c = float(np.linalg.norm(pts[:, None, :] - prev[None, :, :], axis=2).min())
            min_curve_d = min(min_curve_d, d_c)
        score = min(min_node_d, 16.0) * 2.5 + min(min_curve_d, 12.0) * 0.9 - abs(rad) * 3.5
        if score > best_score:
            best_score = score
            best_rad = rad
            best_pts = pts
    return best_rad, best_pts


def draw_numbered_nodes_2d(ax, xy, node_colors, node_size=260, fontsize=8.5):
    """Draws clean circular nodes with centered white node IDs (1-10)."""
    ax.scatter(
        xy[:, 0], xy[:, 1],
        s=node_size, c=node_colors, edgecolors="white", linewidths=1.8, zorder=5,
    )
    ax.scatter(
        xy[:, 0], xy[:, 1],
        s=node_size, facecolors="none", edgecolors="#1E293B", linewidths=1.0, zorder=6,
    )
    for i in range(len(xy)):
        ax.text(
            xy[i, 0], xy[i, 1], str(i + 1),
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white",
            zorder=7,
        )


def plot_figure2_connected(
    g_true_csv="G_true.csv",
    vertices_csv="selected_vertices.csv",
    out_png="figure2_3d_connected_graph.png",
):
    G_true = np.loadtxt(g_true_csv, delimiter=",", dtype=int)
    selected_idx = np.loadtxt(vertices_csv, delimiter=",", dtype=int)

    # Load MNE sample cortical surface coordinates (meters -> mm)
    data_path = mne.datasets.sample.data_path()
    fwd_fname = data_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=False)
    fwd = mne.pick_types_forward(fwd, meg=False, eeg=True, exclude=[])

    rr_lh = fwd["src"][0]["rr"][fwd["src"][0]["vertno"]]
    rr_rh = fwd["src"][1]["rr"][fwd["src"][1]["vertno"]]
    rr_all = np.vstack([rr_lh, rr_rh]) * 1000.0
    coords = rr_all[selected_idx]

    edges = [(int(i), int(j)) for i, j in zip(*np.where(G_true == 1))]
    in_deg = G_true.sum(axis=0)
    out_deg = G_true.sum(axis=1)

    # Node colors matching hub roles
    node_colors = []
    for i in range(10):
        if in_deg[i] >= 3:
            node_colors.append("#D62728")  # Convergent hub (Red)
        elif in_deg[i] >= 2 and out_deg[i] >= 2:
            node_colors.append("#2CA02C")  # Relay hub (Green)
        elif out_deg[i] >= 2 and in_deg[i] >= 1:
            node_colors.append("#8E44AD")  # Divergent hub (Purple)
        else:
            node_colors.append("#1F77B4")  # Connected cortical node (Blue)

    arrow_halo = [pe.withStroke(linewidth=2.3, foreground="white", alpha=0.88)]
    text_halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    small_arrow = "-|>,head_length=4.0,head_width=2.2"

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    fig = plt.figure(figsize=(13.5, 11.2), dpi=300, facecolor="white")
    fig.suptitle(
        f"3D MNE Cortical Surface & Connected Ground-Truth Effective Connectivity ({len(edges)} Directed Edges)",
        fontsize=13.5,
        fontweight="bold",
        color="#1A1A1A",
        y=0.975,
    )

    # ==========================================================
    # Panel (a): 3D Isometric Brain Model
    # ==========================================================
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.set_facecolor("white")
    for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
        axis.pane.set_facecolor((0.97, 0.98, 0.99, 0.65))
        axis.pane.set_edgecolor("#E2E8F0")

    ax1.scatter(
        rr_all[::3, 0], rr_all[::3, 1], rr_all[::3, 2],
        s=1.0, c="#CBD5E1", alpha=0.20, linewidths=0,
    )

    for i, j in edges:
        cat = classify_edge(i, j, in_deg, out_deg)
        color = EDGE_STYLES[cat]["color"]
        vec = coords[j] - coords[i]
        # Shorten start/end slightly so arrow tips stop cleanly at node edges
        start = coords[i] + 0.08 * vec
        span = 0.82 * vec
        ax1.quiver(
            start[0], start[1], start[2],
            span[0], span[1], span[2],
            color=color,
            arrow_length_ratio=0.085,
            linewidth=1.35,
            alpha=0.90,
        )

    ax1.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        s=115, c=node_colors, edgecolors="white", linewidths=1.2, depthshade=False,
    )
    ax1.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        s=115, facecolors="none", edgecolors="#1E293B", linewidths=0.9, depthshade=False,
    )
    for i in range(10):
        ax1.text(
            coords[i, 0] + 3.5, coords[i, 1] + 2.5, coords[i, 2] + 4.0,
            f"N{i+1}", fontsize=8.0, fontweight="bold", color="#1A1A1A",
            path_effects=text_halo,
        )

    ax1.view_init(elev=24, azim=-52)
    ax1.set_title("(a) 3D Isometric Brain Model", fontsize=11.5, fontweight="bold", pad=8, color="#1E293B")
    ax1.set_xlabel("X: Left (-) to Right (+) [mm]", fontsize=8.5, labelpad=5)
    ax1.set_ylabel("Y: Back (-) to Front (+) [mm]", fontsize=8.5, labelpad=5)
    ax1.set_zlabel("Z: Inf (-) to Sup (+) [mm]", fontsize=8.5, labelpad=5)
    ax1.tick_params(labelsize=7.5)

    # ==========================================================
    # Panel (b): Top-Down (Axial) View [Nose at Top]
    # ==========================================================
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(rr_all[:, 0], rr_all[:, 1], s=2.2, c="#CBD5E1", alpha=0.30, linewidths=0, zorder=1)
    ax2.axvline(0, color="#94A3B8", linestyle="--", linewidth=0.9, alpha=0.75, zorder=2)

    xy_axial = coords[:, [0, 1]]
    used_axial = []
    for i, j in edges:
        cat = classify_edge(i, j, in_deg, out_deg)
        color = EDGE_STYLES[cat]["color"]
        rad, pts = choose_clean_rad(i, j, xy_axial, used_axial)
        used_axial.append(pts)
        arr = FancyArrowPatch(
            tuple(xy_axial[i]), tuple(xy_axial[j]),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=small_arrow,
            color=color,
            linewidth=1.45,
            shrinkA=10.5,
            shrinkB=11.0,
            alpha=0.93,
            path_effects=arrow_halo,
            zorder=3,
        )
        ax2.add_patch(arr)

    draw_numbered_nodes_2d(ax2, xy_axial, node_colors, node_size=240, fontsize=8.0)

    ax2.text(-64, 105, "Left Hemisphere", color="#64748B", fontsize=8.8, fontweight="bold")
    ax2.text(14, 105, "Right Hemisphere", color="#64748B", fontsize=8.8, fontweight="bold")
    ax2.set_title("(b) Top-Down (Axial) View [Nose at Top]", fontsize=11.5, fontweight="bold", pad=8, color="#1E293B")
    ax2.set_xlabel("X: Left (-) to Right (+) [mm]", fontsize=9)
    ax2.set_ylabel("Y: Posterior (-) to Anterior (+) [mm]", fontsize=9)
    ax2.set_xlim(-70, 70)
    ax2.set_ylim(-68, 115)
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, linestyle=":", color="#E2E8F0", zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ==========================================================
    # Panel (c): Side (Sagittal) View [Facing Right]
    # ==========================================================
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(rr_all[:, 1], rr_all[:, 2], s=2.2, c="#CBD5E1", alpha=0.30, linewidths=0, zorder=1)

    xy_sag = coords[:, [1, 2]]
    used_sag = []
    for i, j in edges:
        cat = classify_edge(i, j, in_deg, out_deg)
        color = EDGE_STYLES[cat]["color"]
        rad, pts = choose_clean_rad(i, j, xy_sag, used_sag)
        used_sag.append(pts)
        arr = FancyArrowPatch(
            tuple(xy_sag[i]), tuple(xy_sag[j]),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=small_arrow,
            color=color,
            linewidth=1.45,
            shrinkA=10.5,
            shrinkB=11.0,
            alpha=0.93,
            path_effects=arrow_halo,
            zorder=3,
        )
        ax3.add_patch(arr)

    draw_numbered_nodes_2d(ax3, xy_sag, node_colors, node_size=240, fontsize=8.0)

    ax3.set_title("(c) Side (Sagittal) View [Facing Right]", fontsize=11.5, fontweight="bold", pad=8, color="#1E293B")
    ax3.set_xlabel("Y: Posterior (-) to Anterior (+) [mm]", fontsize=9)
    ax3.set_ylabel("Z: Inferior (-) to Superior (+) [mm]", fontsize=9)
    ax3.set_xlim(-68, 115)
    ax3.set_ylim(10, 140)
    ax3.set_aspect("equal", adjustable="box")
    ax3.grid(True, linestyle=":", color="#E2E8F0", zorder=0)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ==========================================================
    # Panel (d): Uncluttered Zero-Crossing Topology Schematic
    # ==========================================================
    ax4 = fig.add_subplot(2, 2, 4)
    # Exact zero-crossing circular ordering for this 12-edge graph:
    # 1-indexed: [1, 3, 4, 2, 6, 9, 8, 7, 5, 10]
    ring_order = [0, 2, 3, 1, 5, 8, 7, 6, 4, 9]
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 10, endpoint=False)
    radius = 1.0
    xy_ring = np.zeros((10, 2))
    for slot, node_idx in enumerate(ring_order):
        xy_ring[node_idx] = [radius * np.cos(angles[slot]), radius * np.sin(angles[slot])]

    # Draw subtle guide ring
    circle_theta = np.linspace(0, 2 * np.pi, 200)
    ax4.plot(
        radius * np.cos(circle_theta), radius * np.sin(circle_theta),
        color="#E2E8F0", linestyle="--", linewidth=0.9, zorder=1,
    )

    slot_map = {node_idx: slot for slot, node_idx in enumerate(ring_order)}
    for i, j in edges:
        cat = classify_edge(i, j, in_deg, out_deg)
        color = EDGE_STYLES[cat]["color"]
        # Adjacent nodes along the ring curve slightly outward; chords stay crisp and gentle
        step = abs(slot_map[i] - slot_map[j])
        step = min(step, 10 - step)
        rad = 0.14 if step == 1 else 0.08
        arr = FancyArrowPatch(
            tuple(xy_ring[i]), tuple(xy_ring[j]),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=4.5,head_width=2.5",
            color=color,
            linewidth=1.65,
            shrinkA=14.5,
            shrinkB=15.0,
            alpha=0.95,
            path_effects=arrow_halo,
            zorder=3,
        )
        ax4.add_patch(arr)

    draw_numbered_nodes_2d(ax4, xy_ring, node_colors, node_size=380, fontsize=9.5)

    # Subtle outer role tags for key hubs
    hub_tags = {
        4: "Convergent Hub\n(4 inputs)",
        2: "Relay Hub\n(2-in, 2-out)",
        7: "Divergent Hub\n(2 outputs)",
    }
    for idx_node, tag in hub_tags.items():
        pos = xy_ring[idx_node]
        norm = np.linalg.norm(pos)
        label_pos = pos * (1.26 / norm)
        ax4.text(
            label_pos[0], label_pos[1], tag,
            ha="center", va="center", fontsize=7.8, fontweight="bold",
            color="#475569",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F8FAFC", ec="#CBD5E1", lw=0.7),
        )

    ax4.set_title("(d) Uncluttered Directed Topology (Zero Crossings)", fontsize=11.5, fontweight="bold", pad=8, color="#1E293B")
    ax4.set_xlim(-1.45, 1.45)
    ax4.set_ylim(-1.42, 1.42)
    ax4.set_aspect("equal", adjustable="box")
    ax4.axis("off")

    # ==========================================================
    # Unified Color Legend Beneath Panels
    # ==========================================================
    legend_handles = [
        Line2D([0], [0], color=info["color"], lw=2.2, marker=">", markersize=5.0, label=info["label"])
        for info in EDGE_STYLES.values()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=9.3,
        columnspacing=2.2,
        bbox_to_anchor=(0.5, 0.012),
    )

    plt.tight_layout(rect=[0.01, 0.075, 0.99, 0.945], h_pad=2.2, w_pad=2.2)
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved clean 2x2 Figure 2 to: {out_png}")


if __name__ == "__main__":
    plot_figure2_connected()