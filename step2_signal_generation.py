import numpy as np

def generate_mvar_signals(W_true, n_samples=15000, burn_in=1000, seed=42):
    """
    Step 2: Generate clean 10-channel continuous MVAR time series (Y).
    Input: W_true (10x10 causal weights matrix from Step 1)
    Output: Y_clean of shape (10, n_samples)
    """
    rng = np.random.default_rng(seed)
    n_nodes = W_true.shape[0]
    total_steps = n_samples + burn_in

    # Copy W_true and add self-memory (0.40) on the diagonal
    A1 = W_true.copy()
    np.fill_diagonal(A1, 0.40)

    # Check mathematical stability: largest absolute eigenvalue must be < 1.0
    eigenvalues = np.linalg.eigvals(A1)
    max_eig = np.max(np.abs(eigenvalues))
    print(f"MVAR Stability Check -> Max Eigenvalue: {max_eig:.4f} (Must be < 1.0)")
    assert max_eig < 1.0, "Error: MVAR system is unstable!"

    # Initialize signal array and driving white noise
    Y = np.zeros((total_steps, n_nodes))
    noise = rng.standard_normal((total_steps, n_nodes))

    # Generate continuous time series step-by-step
    for t in range(1, total_steps):
        Y[t] = Y[t - 1] @ A1 + noise[t]

    # Remove the burn-in warmup steps and transpose to (10 channels, n_samples)
    Y_clean = Y[burn_in:].T
    return Y_clean

if __name__ == "__main__":
    print("=== Step 2: Dynamic Signal Generation (Y) ===")
    
    # 1. Load W_true.csv generated in Step 1
    W_true = np.loadtxt("W_true.csv", delimiter=",")
    print(f"Loaded 'W_true.csv' with shape: {W_true.shape}")

    # 2. Generate 10 channels x 15,000 time points of clean MVAR data
    Y_clean = generate_mvar_signals(W_true, n_samples=15000, burn_in=1000, seed=42)

    # 3. Save clean signals as a CSV file (10 rows = 10 nodes, 15000 columns = time points)
    np.savetxt("Y_clean.csv", Y_clean, delimiter=",", fmt="%.6f")

    print(f"Generated clean MVAR signals (Y_clean) with shape: {Y_clean.shape}")
    print(f"Signal Mean: {np.mean(Y_clean):.4f} | Signal Std Dev: {np.std(Y_clean):.4f}")
    print("Saved file: 'Y_clean.csv'")