"""TrustNet experiment harness.

Re-implements the author's step1-step6 pipeline (scripts in
step*.py in the repository root) as in-memory functions so that
many networks can be evaluated. Faithfulness notes:

* Signal generation, forward model, sensor noise, CAR, phase randomization and
  both estimators are copied from the author's scripts (same seeds by default).
* eLORETA is a linear operator. Instead of reconstructing all 7,498 sources and
  reading out 10 rows (step4 / step5), we compute the 10 x 60 effective kernel
  once by applying the identical inverse to an identity "recording" that went
  through the identical CAR projection. Y_hat = K_sel @ Z_noisy is then
  mathematically identical to the scripts' output, at a fraction of the memory.
"""
from __future__ import annotations

import numpy as np
import mne

SFREQ, SNR_DB, LAMBDA2, N_NODES = 250.0, 10.0, 1.0 / 9.0, 10
OFF = ~np.eye(N_NODES, dtype=bool)


# ----------------------------------------------------------------------------- physics
def load_physics(verbose=False):
    """Forward model, sensor info, lead field L (60 x 10) and the 10 x 60 eLORETA kernel."""
    data_path = mne.datasets.sample.data_path(download=False)
    meg = data_path / "MEG" / "sample"
    raw_sample = mne.io.read_raw_fif(meg / "sample_audvis_raw.fif", preload=False, verbose=False)
    info = mne.pick_info(raw_sample.info, mne.pick_types(raw_sample.info, meg=False, eeg=True, exclude=[]))
    with info._unlock():
        info["sfreq"] = SFREQ
        info["lowpass"] = SFREQ / 2.0
        info["highpass"] = 0.0
    info["bads"] = []
    fwd = mne.read_forward_solution(meg / "sample_audvis-meg-eeg-oct-6-fwd.fif", verbose=False)
    fwd = mne.pick_types_forward(fwd, meg=False, eeg=True, exclude=[])
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)
    n_src = fwd["sol"]["data"].shape[1]
    sel = np.linspace(100, n_src - 100, N_NODES, dtype=int)
    L = fwd["sol"]["data"][:, sel]

    # Info carrying the average-reference projector, exactly as the scripts' raw objects do
    dummy = mne.io.RawArray(np.zeros((len(info["ch_names"]), 10)), info, verbose=False)
    dummy.set_eeg_reference("average", projection=True, verbose=False)
    dummy.apply_proj(verbose=False)
    inv_info = dummy.info
    noise_cov = mne.make_ad_hoc_cov(inv_info, verbose=False)
    inv = mne.minimum_norm.make_inverse_operator(inv_info, fwd, noise_cov, loose=0.0, depth=None,
                                                 fixed=True, verbose=False)
    K_sel = effective_kernel(inv, info, sel)
    return {"info": inv_info, "fwd": fwd, "inv": inv, "sel": sel, "L": L, "K": K_sel}


def effective_kernel(inv, base_info, sel):
    """10 x 60 matrix M with M @ Z_noisy == (eLORETA after CAR applied to Z_noisy)[sel]."""
    n_ch = len(base_info["ch_names"])
    raw_eye = mne.io.RawArray(np.eye(n_ch), base_info.copy(), verbose=False)
    # CAR projection, applied like set_eeg_reference(projection=True) + apply_proj() in the scripts
    raw_eye.set_eeg_reference("average", projection=True, verbose=False)
    raw_eye.apply_proj(verbose=False)
    stc = mne.minimum_norm.apply_inverse_raw(raw_eye, inv, lambda2=LAMBDA2, method="eLORETA", verbose=False)
    return stc.data[sel, :]           # columns = response to each unit electrode input


# ----------------------------------------------------------------------------- simulation
def random_ground_truth(seed, n_edges=12, weight_range=(0.40, 0.60), max_rho=0.85):
    """Copy of step1_ground_truth.create_random_ground_truth (weights rounded to 4 dp like W_true.csv)."""
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        G = np.zeros((N_NODES, N_NODES), dtype=int)
        W = np.zeros((N_NODES, N_NODES))
        perm = rng.permutation(N_NODES)
        for k in range(1, N_NODES):
            u, v = perm[rng.integers(0, k)], perm[k]
            if rng.random() < 0.5:
                G[u, v] = 1
            else:
                G[v, u] = 1
        extra = n_edges - (N_NODES - 1)
        if extra > 0:
            cand = np.argwhere((G == 0) & OFF)
            for i, j in cand[rng.choice(len(cand), size=extra, replace=False)]:
                G[i, j] = 1
        coords = np.argwhere(G == 1)
        for (i, j), w in zip(coords, rng.uniform(*weight_range, size=len(coords))):
            W[i, j] = w
        W4 = np.round(W, 4)
        A_full = W4 + 0.40 * np.eye(N_NODES)          # the matrix actually simulated in step 2
        if (np.max(np.abs(np.linalg.eigvals(W4))) < max_rho
                and np.max(np.abs(np.linalg.eigvals(A_full))) < 1.0):
            return G, W4
    raise RuntimeError("no stationary network found")


def simulate_mvar(W_true, n_samples=15000, burn_in=1000, seed=42):
    """Copy of step2_signal_generation.generate_mvar_signals."""
    rng = np.random.default_rng(seed)
    A1 = W_true.copy()
    np.fill_diagonal(A1, 0.40)
    assert np.max(np.abs(np.linalg.eigvals(A1))) < 1.0
    Y = np.zeros((n_samples + burn_in, N_NODES))
    noise = rng.standard_normal((n_samples + burn_in, N_NODES))
    for t in range(1, n_samples + burn_in):
        Y[t] = Y[t - 1] @ A1 + noise[t]
    return Y[burn_in:].T


def scalp(Y, L, rng):
    """Volume conduction + 10 dB white sensor noise (steps 3 and 5). CAR is inside the kernel."""
    Z = L @ Y
    noise_power = np.mean(Z ** 2) / (10 ** (SNR_DB / 10.0))
    return Z + rng.normal(0, np.sqrt(noise_power), size=Z.shape)


def reconstruct(Z_noisy, K):
    Yh = K @ Z_noisy
    return (Yh - Yh.mean(1, keepdims=True)) / Yh.std(1, keepdims=True)


def phase_randomize(Y, seed=None):
    """Copy of step5_null_model.phase_randomize."""
    rng = np.random.default_rng(seed)
    n = Y.shape[1]
    F = np.fft.rfft(Y, axis=1)
    ph = rng.uniform(0, 2 * np.pi, size=F.shape)
    ph[:, 0] = 0.0
    if n % 2 == 0:
        ph[:, -1] = 0.0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=n, axis=1)


# ----------------------------------------------------------------------------- estimators
def _ridge(Y_z, alpha):
    X = Y_z.T
    Yc, Xl = X[1:], X[:-1]
    A = np.linalg.solve(Xl.T @ Xl + alpha * np.eye(N_NODES), Xl.T @ Yc)
    return A, Yc - Xl @ A


def est_raw(Y_z, alpha=0.5):
    """Plain Ridge-MVAR(1): |A|, diagonal zeroed."""
    A, _ = _ridge(Y_z, alpha)
    W = np.abs(A)
    np.fill_diagonal(W, 0.0)
    return W


def est_symorth(Y_z, alpha=0.5, beta=0.05):
    """Proposed: Ridge-MVAR + symmetric innovation-residual orthogonalization (current step5)."""
    A, E = _ridge(Y_z, alpha)
    s_e = np.maximum(E.std(0), 1e-8)
    C = (1.0 - beta) * np.corrcoef(E, rowvar=False) + beta * np.eye(N_NODES)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-6)
    Ch, Cih = (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T
    W = np.abs(Ch @ ((s_e[:, None] * A) / s_e[None, :]) @ Cih)
    np.fill_diagonal(W, 0.0)
    return W


def est_deblur_signed(Y_z, alpha=0.5, beta=0.10):
    """Earlier step5 estimator (used with edge-wise z): receiver-side deblurring, signed output."""
    A, E = _ridge(Y_z, alpha)
    C_inv = np.linalg.inv(np.corrcoef(E, rowvar=False) + beta * np.eye(N_NODES))
    A_est = (1.0 + beta) * (A @ C_inv)
    np.fill_diagonal(A_est, 0.0)
    return A_est


# ----------------------------------------------------------------------------- null calibration
def null_coefficients(Y_clean, phys, estimator, n_surrogates, seed_offset=0):
    """Replay phase-randomized clean signals through the identical chain; seeds as in step5."""
    out = []
    for s in range(n_surrogates):
        rng = np.random.default_rng(1000 + s + seed_offset)
        Ys = phase_randomize(Y_clean, seed=100 + s + seed_offset)
        out.append(estimator(reconstruct(scalp(Ys, phys["L"], rng), phys["K"])))
    return np.stack(out)


def pooled_threshold(null_stack, percentile=99.0):
    return float(np.percentile(np.concatenate([w[OFF] for w in null_stack]), percentile))


def edgewise_z(null_stack, A_est, percentile=99.5):
    """Earlier step5/step6 edge-wise z calibration (leave-one-out, SD floored at 0.25 x median)."""
    mu = null_stack.mean(0)
    sd = null_stack.std(0, ddof=1)
    floor = 0.25 * np.median(sd[OFF])
    sd_reg = np.maximum(sd, floor)
    np.fill_diagonal(mu, 0.0)
    np.fill_diagonal(sd_reg, 1.0)
    zs = []
    for s in range(len(null_stack)):
        others = np.delete(null_stack, s, axis=0)
        z = np.abs(null_stack[s] - others.mean(0)) / np.maximum(others.std(0, ddof=1), floor)
        zs.append(z[OFF])
    z_thr = float(np.percentile(np.concatenate(zs), percentile))
    Z = np.abs(A_est - mu) / sd_reg
    np.fill_diagonal(Z, 0.0)
    return Z, z_thr, mu, sd_reg


# ----------------------------------------------------------------------------- metrics
def metrics(G_true, G_est, scores=None):
    t, p = G_true[OFF], G_est[OFF]
    tp = int(((t == 1) & (p == 1)).sum()); fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum()); tn = int(((t == 0) & (p == 0)).sum())
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    m = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / 90,
         "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
         "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "mcc": (tp * tn - fp * fn) / den if den else 0.0}
    if scores is not None:
        s = scores[OFF]
        pos, neg = s[t == 1], s[t == 0]
        m["auroc"] = float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))
        order = np.argsort(-s, kind="mergesort")
        yy = t[order]
        cum = np.cumsum(yy)
        m["ap"] = float(np.sum(np.diff(np.concatenate([[0], cum / yy.sum()])) * cum / np.arange(1, len(yy) + 1)))
    return m
