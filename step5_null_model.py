import numpy as np
import mne


def phase_randomize(Y, seed=None):
    rng = np.random.default_rng(seed)
    _, n_samples = Y.shape

    fft_vals = np.fft.rfft(Y, axis=1)
    amplitudes = np.abs(fft_vals)

    random_phases = rng.uniform(0, 2 * np.pi, size=fft_vals.shape)
    random_phases[:, 0] = 0.0
    if n_samples % 2 == 0:
        random_phases[:, -1] = 0.0

    fft_scrambled = amplitudes * np.exp(1j * random_phases)
    return np.fft.irfft(fft_scrambled, n=n_samples, axis=1)


def fit_ridge_mvar(Y_z, alpha=0.5, order=1, beta=0.05):
    """
    Fits Ridge-MVAR (alpha=0.5) and applies symmetric innovation-residual
    orthogonalization to suppress zero-lag spatial leakage without hub cannibalization.
    """
    X = Y_z.T
    n_samples, n_nodes = X.shape

    Y_curr = X[order:]
    X_lag = np.hstack([X[order - p : n_samples - p] for p in range(1, order + 1)])

    n_features = X_lag.shape[1]
    XtX = X_lag.T @ X_lag
    XtY = X_lag.T @ Y_curr
    A_full = np.linalg.solve(XtX + alpha * np.eye(n_features), XtY)

    E = Y_curr - (X_lag @ A_full)
    s_e = np.maximum(np.std(E, axis=0), 1e-8)
    C_res = np.corrcoef(E, rowvar=False)

    C_reg = (1.0 - beta) * C_res + beta * np.eye(n_nodes)
    eigvals, eigvecs = np.linalg.eigh(C_reg)
    eigvals = np.maximum(eigvals, 1e-6)
    C_half = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    C_inv_half = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T

    A_lags = A_full.reshape(order, n_nodes, n_nodes)
    A_unmixed = np.zeros_like(A_lags)
    for p in range(order):
        A_scaled = (s_e[:, None] * A_lags[p]) / s_e[None, :]
        A_unmixed[p] = C_half @ A_scaled @ C_inv_half

    W_est = np.linalg.norm(A_unmixed, axis=0)
    np.fill_diagonal(W_est, 0.0)
    return W_est


def run_step5_null_model(
    y_clean_csv="Y_clean.csv",
    leadfield_csv="L_leadfield.csv",
    vertices_csv="selected_vertices.csv",
    raw_fif="Z_scalp_raw.fif",
    alpha=0.5,
    percentile=99.0,
    n_surrogates=20,
    snr_db=10.0,
    lambda2=1.0 / 9.0
):
    print("=== Step 5: Synthetic Null Model Generation (Symmetric Un-Mixing) ===")
    Y_clean = np.loadtxt(y_clean_csv, delimiter=",")
    L = np.loadtxt(leadfield_csv, delimiter=",")
    selected_idx = np.loadtxt(vertices_csv, delimiter=",", dtype=int)
    raw_template = mne.io.read_raw_fif(raw_fif, preload=False, verbose=False)
    info = raw_template.info.copy()

    data_path = mne.datasets.sample.data_path()
    fwd_fname = data_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=False)
    fwd = mne.pick_types_forward(fwd, meg=False, eeg=True, exclude=[])
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)

    noise_cov = mne.make_ad_hoc_cov(info, verbose=False)
    inv_op = mne.minimum_norm.make_inverse_operator(
        info, fwd, noise_cov, loose=0.0, depth=None, fixed=True, verbose=False
    )

    n_nodes = Y_clean.shape[0]
    off_diag_mask = ~np.eye(n_nodes, dtype=bool)
    all_null_scores = []
    W_null_max = np.zeros((n_nodes, n_nodes))

    for s in range(n_surrogates):
        rng = np.random.default_rng(1000 + s)
        Y_scrambled = phase_randomize(Y_clean, seed=100 + s)

        Z_clean_null = L @ Y_scrambled
        sig_power = np.mean(Z_clean_null ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        Z_noisy_null = Z_clean_null + rng.normal(0, np.sqrt(noise_power), size=Z_clean_null.shape)

        raw_null = mne.io.RawArray(Z_noisy_null, info, verbose=False)
        raw_null.set_eeg_reference("average", projection=True, verbose=False)
        raw_null.apply_proj(verbose=False)

        stc_null = mne.minimum_norm.apply_inverse_raw(
            raw_null, inv_op, lambda2=lambda2, method="eLORETA", verbose=False
        )
        Y_hat_null = stc_null.data[selected_idx, :]
        Y_hat_null_z = (
            (Y_hat_null - np.mean(Y_hat_null, axis=1, keepdims=True))
            / np.std(Y_hat_null, axis=1, keepdims=True)
        )

        W_null = fit_ridge_mvar(Y_hat_null_z, alpha=alpha)
        all_null_scores.append(W_null[off_diag_mask])
        W_null_max = np.maximum(W_null_max, W_null)

    null_distribution = np.concatenate(all_null_scores)
    thresh_locked = float(np.percentile(null_distribution, percentile))

    np.savetxt("W_null_artifacts.csv", W_null_max, delimiter=",", fmt="%.6f")
    np.savetxt("null_threshold.csv", np.array([thresh_locked]), delimiter=",", fmt="%.6f")

    print(f"Null Edges Evaluated : {len(null_distribution)} ({n_surrogates} runs x 90 edges)")
    print(f"Mean Null Score      : {np.mean(null_distribution):.5f}")
    print(f"Locked Cutoff (p={percentile}) : {thresh_locked:.6f}")


if __name__ == "__main__":
    run_step5_null_model()