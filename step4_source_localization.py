import numpy as np
import mne

def run_step4_source_localization(
    raw_fif="Z_scalp_raw.fif",
    vertices_csv="selected_vertices.csv",
    y_clean_csv="Y_clean.csv",
    lambda2=1.0 / 9.0
):
    """
    Step 4: Source Localization and Dimensional Alignment (eLORETA)
    Takes the confounded 60-channel scalp EEG (Z) and uses eLORETA to reconstruct
    the 10 cortical source signals (Y_hat) at their original 3D coordinates.
    """
    print("=== Step 4: Source Localization via eLORETA ===")

    # 1. Load the confounded 60-channel scalp EEG and the 10 target vertex IDs
    raw_eeg = mne.io.read_raw_fif(raw_fif, preload=True, verbose=False)
    selected_idx = np.loadtxt(vertices_csv, delimiter=",", dtype=int)
    print(f"Loaded confounded scalp EEG '{raw_fif}' with shape: {raw_eeg.get_data().shape}")

    # 2. Load the MNE 3D Forward Model (Lead-Field physics)
    data_path = mne.datasets.sample.data_path()
    fwd_fname = data_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=False)
    fwd = mne.pick_types_forward(fwd, meg=False, eeg=True, exclude=[])
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)

    # 3. Build the eLORETA Inverse Operator
    print("Building eLORETA inverse operator to reverse skull volume conduction...")
    noise_cov = mne.make_ad_hoc_cov(raw_eeg.info, verbose=False)
    inv_op = mne.minimum_norm.make_inverse_operator(
        raw_eeg.info, fwd, noise_cov, loose=0.0, depth=None, fixed=True, verbose=False
    )

    # 4. Apply eLORETA to reconstruct cortical source signals across the 3D brain
    stc = mne.minimum_norm.apply_inverse_raw(
        raw_eeg, inv_op, lambda2=lambda2, method="eLORETA", verbose=False
    )

    # 5. Dimensional Alignment: Extract the 10 original cortical coordinates (10 x 15000)
    Y_hat_raw = stc.data[selected_idx, :]

    # Standardize (Z-score) each reconstructed node so mean = 0 and std = 1
    Y_hat = (Y_hat_raw - np.mean(Y_hat_raw, axis=1, keepdims=True)) / np.std(Y_hat_raw, axis=1, keepdims=True)

    # 6. Save the reconstructed 10-node signals as CSV
    np.savetxt("Y_hat_eloreta.csv", Y_hat, delimiter=",", fmt="%.6f")
    print(f"Reconstructed cortical matrix (Y_hat_eloreta) shape: {Y_hat.shape}")
    print("Saved file: 'Y_hat_eloreta.csv'")

    # 7. Check signal recovery quality against Step 2 Y_clean
    Y_clean = np.loadtxt(y_clean_csv, delimiter=",")
    print("\n--- Signal Recovery Check (Correlation between Y_clean and Y_hat_eloreta) ---")
    corrs = []
    for i in range(len(selected_idx)):
        r = np.corrcoef(Y_clean[i], Y_hat[i])[0, 1]
        corrs.append(abs(r))
        print(f"  Node {i+1:2d} (Vertex {selected_idx[i]:4d}): |r| = {abs(r):.4f}")
    print(f"Average Waveform Correlation across 10 Nodes: {np.mean(corrs):.4f}")

if __name__ == "__main__":
    run_step4_source_localization()