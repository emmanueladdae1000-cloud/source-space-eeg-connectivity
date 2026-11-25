import sys
import numpy as np


def create_random_ground_truth(
    n_nodes=10,
    n_edges=12,
    weight_range=(0.40, 0.60),
    max_spectral_radius=0.85,
    allow_negative=False,
    seed=None
):
    """
    Step 1: Construct a random, weakly connected, stationary 10x10 MVAR ground truth.
    Convention: G_true[i, j] = 1 means Source i -> Target j.
    """
    if n_edges < n_nodes - 1:
        raise ValueError(f"Need at least {n_nodes - 1} edges to connect all {n_nodes} nodes.")
    if n_edges > n_nodes * (n_nodes - 1):
        raise ValueError("n_edges exceeds maximum possible directed off-diagonal edges.")

    rng = np.random.default_rng(seed)

    for _ in range(1000):
        G_true = np.zeros((n_nodes, n_nodes), dtype=int)
        W_true = np.zeros((n_nodes, n_nodes), dtype=float)

        # 1. Build a random spanning tree (n_nodes - 1 edges) so every node is connected
        perm = rng.permutation(n_nodes)
        for k in range(1, n_nodes):
            u = perm[rng.integers(0, k)]
            v = perm[k]
            if rng.random() < 0.5:
                G_true[u, v] = 1
            else:
                G_true[v, u] = 1

        # 2. Add remaining random directed edges up to n_edges
        extra_needed = n_edges - (n_nodes - 1)
        if extra_needed > 0:
            off_diag = ~np.eye(n_nodes, dtype=bool)
            candidates = np.argwhere((G_true == 0) & off_diag)
            chosen = candidates[rng.choice(len(candidates), size=extra_needed, replace=False)]
            for i, j in chosen:
                G_true[i, j] = 1

        # 3. Assign random coupling weights to all active edges
        edge_coords = np.argwhere(G_true == 1)
        weights = rng.uniform(weight_range[0], weight_range[1], size=len(edge_coords))
        if allow_negative:
            weights *= rng.choice([-1.0, 1.0], size=len(edge_coords))

        for (i, j), w in zip(edge_coords, weights):
            W_true[i, j] = w

        # 4. Enforce MVAR stationarity via spectral radius check: max(|eigenvalues|) < 0.85
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(W_true))))
        if spectral_radius < max_spectral_radius:
            return G_true, W_true, spectral_radius

    raise RuntimeError("Could not sample a stationary MVAR matrix; try lowering n_edges or weight_range.")


def summarize_motifs(G_true):
    """Counts naturally occurring Common Sources, Common Targets, and Indirect Cascades."""
    out_deg = np.sum(G_true, axis=1)
    in_deg = np.sum(G_true, axis=0)

    common_sources = int(np.sum(out_deg >= 2))
    common_targets = int(np.sum(in_deg >= 2))

    two_hop = (G_true @ G_true) > 0
    np.fill_diagonal(two_hop, False)
    indirect_cascades = int(np.sum(two_hop & (G_true == 0)))
    return common_sources, common_targets, indirect_cascades


if __name__ == "__main__":
    # Leave blank for a new random ground truth each run, or pass an integer seed (e.g., python3 step1_ground_truth.py 42)
    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else int(np.random.default_rng().integers(0, 100000))
    N_EDGES = 12  # 12/90 = 13.3% density

    G_true, W_true, rho = create_random_ground_truth(
        n_nodes=10,
        n_edges=N_EDGES,
        weight_range=(0.40, 0.60),
        max_spectral_radius=0.85,
        seed=SEED
    )

    np.savetxt("G_true.csv", G_true, delimiter=",", fmt="%d")
    np.savetxt("W_true.csv", W_true, delimiter=",", fmt="%.4f")

    cs, ct, casc = summarize_motifs(G_true)
    active_nodes = int(np.sum((np.sum(G_true, axis=0) + np.sum(G_true, axis=1)) > 0))

    print(f"=== Step 1: Random Ground Truth Complete (Seed = {SEED}) ===")
    print("10x10 Binary Adjacency Matrix (G_true):")
    print(G_true)
    print(f"\nConnected Nodes    : {active_nodes} / {G_true.shape[0]}")
    print(f"Total Edges        : {np.sum(G_true)} ({np.sum(G_true)/90*100:.1f}% density)")
    print(f"Spectral Radius    : {rho:.4f} (Stationary: < 0.85)")
    print(f"Natural Motifs     : {cs} Common Sources | {ct} Common Targets | {casc} Indirect Cascades")
    print("Saved files: 'G_true.csv' and 'W_true.csv'")