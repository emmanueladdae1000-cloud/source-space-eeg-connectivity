import numpy as np
import mne

def run_step3_forward_modeling(y_csv="Y_clean.csv", sfreq=250.0, snr_db=10.0, seed=42):
    """
    Step 3: Forward Modeling & Confounder Injection (Z)
    Projects 10 clean cortical signals (Y) through an MNE 3D BEM head model (L)
    to create confounded scalp EEG recordings (Z).
    """
    rng = np.random.default_rng(seed)

    # 1. Load the 10 x 15000 clean MVAR signals from Step 2
    Y_clean = np.loadtxt(y_csv, delimiter=",")
    n_nodes, n_samples = Y_clean.shape
    print(f"Loaded '{y_csv}' with shape: {Y_clean.shape} (10 cortical nodes, {n_samples} time points)")

    # 2. Load MNE's standard 3D Boundary Element Model (BEM) & Forward Solution
    print("Loading MNE 3D head model...")
    data_path = mne.datasets.sample.data_path()
    meg_path = data_path / "MEG" / "sample"
    raw_fname = meg_path / "sample_audvis_raw.fif"
    fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"

    # Extract standard scalp EEG electrode information (60 EEG channels in MNE sample cap)
    raw_sample = mne.io.read_raw_fif(raw_fname, preload=False, verbose=False)
    eeg_picks = mne.pick_types(raw_sample.info, meg=False, eeg=True, exclude=[])
    info = mne.pick_info(raw_sample.info, eeg_picks)
    
    # Unlock the MNE info header to set our 250 Hz sampling rate
    with info._unlock():
        info["sfreq"] = sfreq
        info["lowpass"] = sfreq / 2.0
        info["highpass"] = 0.0
    info["bads"] = []  # Keep all 60 scalp EEG channels active

    # Load forward operator and lock dipole orientation perpendicular to the cortical surface
    fwd = mne.read_forward_solution(fwd_fname, verbose=False)
    fwd = mne.pick_types_forward(fwd, meg=False, eeg=True, exclude=[])
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)

    # 3. Assign the 10 nodes to distinct, well-separated 3D cortical coordinates
    n_total_vertices = fwd["sol"]["data"].shape[1]
    selected_idx = np.linspace(100, n_total_vertices - 100, n_nodes, dtype=int)

    # Extract the Lead-Field Matrix (L) of shape (n_eeg_channels, 10)
    L = fwd["sol"]["data"][:, selected_idx]
    print(f"Extracted Lead-Field Matrix (L) with shape: {L.shape} (Scalp Electrodes x Cortical Nodes)")

    # 4. Confounder 1: Physical Volume Conduction (Z_clean = L @ Y_clean)
    Z_clean = L @ Y_clean

    # 5. Confounder 2: Scalp Sensor Noise (10 dB SNR)
    signal_power = np.mean(Z_clean ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    sensor_noise = rng.normal(0, np.sqrt(noise_power), size=Z_clean.shape)
    Z_noisy = Z_clean + sensor_noise

    # 6. Confounder 3: Reference Smearing (Common Average Reference across the scalp)
    raw_eeg = mne.io.RawArray(Z_noisy, info, verbose=False)
    raw_eeg.set_eeg_reference("average", projection=True, verbose=False)
    raw_eeg.apply_proj(verbose=False)

    # Extract final confounded scalp matrix Z
    Z_scalp = raw_eeg.get_data()

    # 7. Save outputs as CSVs (for inspection) and FIF (for Step 4 eLORETA)
    np.savetxt("L_leadfield.csv", L, delimiter=",", fmt="%.6f")
    np.savetxt("selected_vertices.csv", selected_idx, delimiter=",", fmt="%d")
    np.savetxt("Z_scalp.csv", Z_scalp, delimiter=",", fmt="%.6f")
    raw_eeg.save("Z_scalp_raw.fif", overwrite=True, verbose=False)

    print("\n=== Step 3: Forward Modeling & Confounder Injection Complete ===")
    print(f"Final Confounded Scalp Matrix (Z_scalp) shape: {Z_scalp.shape}")
    print("Saved files:")
    print("  - 'Z_scalp.csv'           (Confounded scalp EEG time series)")
    print("  - 'L_leadfield.csv'       (Lead-field volume conduction weights)")
    print("  - 'selected_vertices.csv' (3D cortical vertex indices for the 10 nodes)")
    print("  - 'Z_scalp_raw.fif'       (MNE EEG object ready for Step 4 eLORETA)")

if __name__ == "__main__":
    run_step3_forward_modeling()