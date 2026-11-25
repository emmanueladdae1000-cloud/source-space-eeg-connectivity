import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def plot_3d_brain_connectivity():
    """
    Visualizes the 10 selected cortical vertices and their directed Ground Truth
    connectivity pathways inside MNE's 3D Brain Model (Left & Right Hemispheres).
    """
    # 1. Load your Step 1 Answer Key and Step 3 Selected Vertices
    G_true = np.loadtxt("G_true.csv", delimiter=",", dtype=int)
    selected_idx = np.loadtxt("selected_vertices.csv", delimiter=",", dtype=int)
    n_nodes = len(selected_idx)

    # 2. Load MNE's 3D Forward Model to get exact 3D (X, Y, Z) brain coordinates
    print("Loading MNE 3D cortical surface coordinates...")
    data_path = mne.datasets.sample.data_path()
    fwd_fname = data_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=False)
    fwd = mne.pick_types_forward(fwd, meg=False, eeg=True, exclude=[])
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)

    # Convert 3D coordinates from meters to millimeters (mm)
    all_coords_mm = fwd["source_rr"] * 1000.0
    node_coords = all_coords_mm[selected_idx]

    # Separate Left Hemisphere (LH) and Right Hemisphere (RH) vertices
    n_lh = len(fwd["src"][0]["vertno"])
    print("\n=== 3D Cortical Coordinates of Your 10 Nodes ===")
    for i, v_idx in enumerate(selected_idx):
        hemi = "Left Hemisphere (LH)" if v_idx < n_lh else "Right Hemisphere (RH)"
        x, y, z = node_coords[i]
        print(f"Node {i+1:2d} (Vertex {v_idx:4d}) | {hemi} | X={x:6.1f} mm, Y={y:6.1f} mm, Z={z:6.1f} mm")

    # Define colors for each motif
    def get_node_color(node_num):
        if node_num in [1, 2, 3]:
            return "#1f77b4"  # Blue: Cascade (1 -> 2 -> 3)
        elif node_num in [4, 5, 6]:
            return "#d62728"  # Red: Common Source (4 -> 5, 6)
        elif node_num in [7, 8]:
            return "#2ca02c"  # Green: Direct (7 -> 8)
        else:
            return "#7f7f7f"  # Gray: Isolated (9, 10)

    node_colors = [get_node_color(i + 1) for i in range(n_nodes)]

    # 3. Create a 3-Panel Neuroimaging Glass-Brain Figure
    fig = plt.figure(figsize=(18, 6.5), dpi=150)

    # --- PANEL 1: 3D Isometric Brain View ---
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    # Plot the 7,498 cortical vertices as a translucent 3D brain cloud
    ax3d.scatter(
        all_coords_mm[:, 0], all_coords_mm[:, 1], all_coords_mm[:, 2],
        c="lightgray", s=2, alpha=0.18, label="MNE 3D Cortex (7,498 vertices)"
    )
    # Plot the 10 Cortical Nodes
    for i in range(n_nodes):
        x, y, z = node_coords[i]
        ax3d.scatter(x, y, z, c=node_colors[i], s=140, edgecolors="black", linewidth=1.2, depthshade=False)
        ax3d.text(x + 3, y + 3, z + 4, f"N{i+1}", fontsize=9, fontweight="bold", color="black")

    # Plot 3D Directed Connectivity Lines
    for src in range(n_nodes):
        for dst in range(n_nodes):
            if G_true[src, dst] == 1:
                p1 = node_coords[src]
                p2 = node_coords[dst]
                c = get_node_color(src + 1)
                ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=c, linewidth=2.5)
                # Add 3D directional arrowhead near target node
                vec = p2 - p1
                ax3d.quiver(
                    p1[0], p1[1], p1[2], vec[0], vec[1], vec[2],
                    color=c, arrow_length_ratio=0.18, linewidth=2.2
                )

    ax3d.view_init(elev=25, azim=-60)
    ax3d.set_title("1. 3D Isometric Brain Model", fontsize=12, fontweight="bold")
    ax3d.set_xlabel("X: Left (-) / Right (+) [mm]")
    ax3d.set_ylabel("Y: Back (-) / Front (+) [mm]")
    ax3d.set_zlabel("Z: Down (-) / Up (+) [mm]")

    # --- PANEL 2: Top-Down Axial View (Looking Down at Top of Head) ---
    ax_top = fig.add_subplot(1, 3, 2)
    ax_top.scatter(all_coords_mm[:, 0], all_coords_mm[:, 1], c="gainsboro", s=4, alpha=0.35)
    for i in range(n_nodes):
        x, y, _ = node_coords[i]
        ax_top.scatter(x, y, c=node_colors[i], s=180, edgecolors="black", linewidth=1.5, zorder=4)
        ax_top.text(x + 3, y + 3, f"Node {i+1}", fontsize=9, fontweight="bold", zorder=5)

    for src in range(n_nodes):
        for dst in range(n_nodes):
            if G_true[src, dst] == 1:
                p1 = node_coords[src]
                p2 = node_coords[dst]
                arrow = FancyArrowPatch(
                    (p1[0], p1[1]), (p2[0], p2[1]),
                    connectionstyle="arc3,rad=0.15",
                    arrowstyle="-|>", mutation_scale=18,
                    color=get_node_color(src + 1), linewidth=2.5, zorder=3
                )
                ax_top.add_patch(arrow)

    ax_top.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_top.text(-65, 95, "Left Hemisphere", fontsize=10, fontweight="bold", color="dimgray")
    ax_top.text(15, 95, "Right Hemisphere", fontsize=10, fontweight="bold", color="dimgray")
    ax_top.set_title("2. Top-Down (Axial) View [Nose at Top]", fontsize=12, fontweight="bold")
    ax_top.set_xlabel("X: Left <---> Right (mm)")
    ax_top.set_ylabel("Y: Back <---> Front (mm)")
    ax_top.set_aspect("equal")

    # --- PANEL 3: Side Sagittal View (Looking from Right Side of Head) ---
    ax_side = fig.add_subplot(1, 3, 3)
    ax_side.scatter(all_coords_mm[:, 1], all_coords_mm[:, 2], c="gainsboro", s=4, alpha=0.35)
    for i in range(n_nodes):
        _, y, z = node_coords[i]
        ax_side.scatter(y, z, c=node_colors[i], s=180, edgecolors="black", linewidth=1.5, zorder=4)
        ax_side.text(y + 3, z + 3, f"Node {i+1}", fontsize=9, fontweight="bold", zorder=5)

    for src in range(n_nodes):
        for dst in range(n_nodes):
            if G_true[src, dst] == 1:
                p1 = node_coords[src]
                p2 = node_coords[dst]
                arrow = FancyArrowPatch(
                    (p1[1], p1[2]), (p2[1], p2[2]),
                    connectionstyle="arc3,rad=0.15",
                    arrowstyle="-|>", mutation_scale=18,
                    color=get_node_color(src + 1), linewidth=2.5, zorder=3
                )
                ax_side.add_patch(arrow)

    ax_side.set_title("3. Side (Sagittal) View [Facing Right]", fontsize=12, fontweight="bold")
    ax_side.set_xlabel("Y: Back <---> Front (mm)")
    ax_side.set_ylabel("Z: Inferior <---> Superior (mm)")
    ax_side.set_aspect("equal")

    plt.suptitle(
        "Phase 1: 3D MNE Cortical Surface & Ground-Truth Directed Connectivity\n"
        "Blue = Cascade (1->2->3) | Red = Common Source (4->5, 4->6) | Green = Direct (7->8) | Gray = Isolated (9, 10)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("step3_3d_brain_connectivity.png")
    plt.close()
    print("\nSaved 3D brain visualization to: 'step3_3d_brain_connectivity.png'")

if __name__ == "__main__":
    plot_3d_brain_connectivity()