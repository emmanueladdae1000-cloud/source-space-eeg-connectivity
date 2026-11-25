# Source-space EEG directed connectivity: physics-matched null calibration

Code, data and per-network results for the manuscript

> **Physics-Matched Surrogate Calibration with Innovation-Residual Orthogonalization for Directed Source-Space EEG Connectivity: A Pre-Specified Biophysical Benchmark**
> Emmanuel Addo Addae, University of South Carolina (manuscript under review)

The pipeline simulates a random directed network of 10 cortical sources, projects it through a realistic head model to 60 EEG electrodes, and reconstructs the sources with eLORETA. It then estimates directed connections with a multivariate autoregressive (MVAR) model and decides which connections to report using a threshold calibrated on surrogate data, which are replayed through the same head model. Everything runs on simulated data. The only external input is the free MNE-Python *sample* head model.

The repository has two parts:

| Part | Where | What it does |
|---|---|---|
| Step-by-step pipeline | repository root (`step1_*.py` … `step6_*.py`, `plot_*.py`) | Runs one network through the six stages, writing every intermediate result to CSV. Produces the case study (Figures 1–5 of the paper). |
| Benchmark | `benchmark/` | Re-implements the same pipeline as functions and evaluates many networks. Produces the 100-network benchmark, the 16 stress conditions, Tables 2–4, S1 and Figures 6–8. |

---

## 1. Installation

Requires Python 3.10 or newer and about 2 GB of free disk space for the head model.

```bash
git clone https://github.com/emmanueladdae1000-cloud/source-space-eeg-connectivity.git
cd source-space-eeg-connectivity
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download the MNE sample dataset once (about 1.6 GB, about 5 minutes):

```bash
python -c "import mne; print(mne.datasets.sample.data_path())"
```

The step-by-step scripts download it automatically if it is missing. The benchmark scripts do not, so run this command first.

---

## 2. Quick check (about 1 minute)

To confirm that your installation reproduces the published numbers, run:

```bash
cd benchmark
python validate_and_rerun_table2.py   # case-study network: reproduces the saved run of the original scripts
python benchmark2.py check 2          # re-runs test networks 1 and 2 with all 8 methods
```

`benchmark2.py check` compares every method's MCC with the saved `results/test2.jsonl` and prints `OK: matches the saved results` if they agree. `validate_and_rerun_table2.py` writes `results/table2_rerun.json`. In it, `hub_tau_reproduced` should equal `hub_tau_saved` (0.196174), and `hub_yhat_max_abs_diff` and `hub_west_max_abs_diff` should be below 10⁻⁶.

---

## 3. Step-by-step pipeline (one network)

Run the scripts in order from the repository root. Each one reads the CSV files written by the previous step.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 Ground truth | `step1_ground_truth.py [seed]` | none | `G_true.csv` (adjacency), `W_true.csv` (weights) |
| 2 Source signals | `step2_signal_generation.py` | `W_true.csv` | `Y_clean.csv` (10 × 15,000; MVAR(1), seed 42) |
| 3 Forward model | `step3_forward_modeling.py` | `Y_clean.csv` | `L_leadfield.csv`, `selected_vertices.csv`, `Z_scalp.csv`, `Z_scalp_raw.fif` (60 electrodes, 10 dB noise, average reference) |
| 4 Source reconstruction | `step4_source_localization.py` | `Z_scalp_raw.fif`, `selected_vertices.csv` | `Y_hat_eloreta.csv` (eLORETA, λ² = 1/9) |
| 5 Null calibration | `step5_null_model.py` | `Y_clean.csv`, `L_leadfield.csv`, … | `null_threshold.csv` (τ, 99th percentile of 20 replayed surrogates), `W_null_artifacts.csv` |
| 6 Estimation and scoring | `step6_metric_evaluation.py` | `Y_hat_eloreta.csv`, `G_true.csv`, `null_threshold.csv` | `W_est.csv`, `G_est.csv`, metrics printed to screen |

```bash
python step1_ground_truth.py 7      # any integer seed; omit for a random network
python step2_signal_generation.py
python step3_forward_modeling.py
python step4_source_localization.py
python step5_null_model.py
python step6_metric_evaluation.py
```

The `plot_*.py` scripts draw the figures from these CSV files: `plot_step2_brainwaves.py`, `plot_step3_3d_brain.py`, `plot_figure2_connected.py` (network placement), `plot_step4_diagnostic.py` (reconstruction fidelity and leakage), `plot_per_edge_recovery.py`, `plot_results_matrix.py` and `plot_step6_comparison.py`.

**Reproducing the paper's case-study network exactly.** Its random seed was not recorded, so start from the saved ground truth instead of step 1:

```bash
cp benchmark/data/run_hub/G_true.csv benchmark/data/run_hub/W_true.csv .
python step2_signal_generation.py   # then steps 3-6 as above
```

This gives τ = 0.196174 and 10 TP / 2 FP / 2 FN / 76 TN (MCC 0.808). The saved outputs of the original run are in `benchmark/data/run_hub/` for comparison.

> **Note on step 1.** `step1_ground_truth.py` checks stability on the coupling matrix *W* only. The simulated process also keeps 0.40 of each node's previous value, so the full transition matrix is *W* + 0.40 *I*. The benchmark generator (`trustnet_harness.random_ground_truth`) checks both, which rejects the rare unstable networks (paper, Section 3.2).

---

## 4. Benchmark (`benchmark/`)

The design was written down before any test run. It is in [`benchmark/BENCHMARK_DESIGN.md`](benchmark/BENCHMARK_DESIGN.md), together with the amendments made before testing.

### Files

| File | Purpose |
|---|---|
| `trustnet_harness.py` | The six pipeline stages as functions (physics, ground truth, simulation, reconstruction, surrogates, estimators, metrics). eLORETA is linear, so the 10 × 60 read-out matrix is computed once; this equals full reconstruction to within 10⁻⁶. |
| `benchmark2.py` | **Main benchmark of the paper**: tuning, 100 test networks, 16 stress conditions; 8 methods. |
| `analyze2.py` | Bootstrap CIs, paired Wilcoxon tests (Holm-corrected), equivalence (TOST) → `*_summary.json`. |
| `make_figures2.py` | Figures 6–8 of the paper. |
| `hub_case_study.py` | Table 2 (all methods on the case-study network). |
| `validate_and_rerun_table2.py` | Checks the harness against the saved outputs of the original scripts. |
| `round2_numbers.py` | Additional numbers reported in the paper (per-condition equivalence, mixing asymmetry, …). |
| `benchmark.py`, `analyze.py`, `make_figures.py` | First benchmark run (6 methods, robustness checks); some functions are reused by `benchmark2.py`. |
| `data/run_hub/`, `data/run_b/` | Saved outputs of the original scripts for the case-study network and a second network. |
| `results/` | Per-network results (`*.jsonl`, one line per network) and summaries (`*.json`) used in the paper. |
| `figures/` | The generated figures. |

### Methods compared

| Name in code | Method in the paper |
|---|---|
| `PROPOSED` | Ridge-MVAR + innovation-residual orthogonalization, simulation null |
| `PROPOSED-DEPLOY` | same estimator, deployable null (surrogates of the reconstructed signals, replayed through the head model) |
| `PROPOSED-POSTINV` | same estimator, conventional post-inversion null (no replay) |
| `RAW` | Ridge-MVAR |
| `SIGORTH` | signal-level symmetric (Löwdin) orthogonalization + Ridge-MVAR |
| `TRGC` | time-reversed conditional Granger causality |
| `GPDC` | generalized partial directed coherence, 1–40 Hz |
| `ORACLE` | Ridge-MVAR on the clean simulated sources (reference ceiling) |

### Full replication

Run from inside `benchmark/`. The times below were measured with 8 parallel processes on a 16-core laptop.

```bash
cd benchmark
python benchmark2.py tune          # 20 tuning networks -> results/tune2.jsonl, tune2_choice.json   (~4 min)
python benchmark2.py test          # 100 test networks  -> results/test2.jsonl                     (~5 min)
python benchmark2.py conditions    # 16 conditions x 30 -> results/conditions2.jsonl               (~25 min)
python analyze2.py test            # -> results/test2_summary.json        (Table 3)
python analyze2.py conditions      # -> results/conditions2_summary.json  (Table 4, Table S1)
python hub_case_study.py           # -> results/hub_case_study.json       (Table 2)
python round2_numbers.py           # -> results/round2_numbers.json
python make_figures2.py            # -> figures/fig6-8_*_v3.png
```

First benchmark run (optional; cited in the paper as the "earlier run"):

```bash
python benchmark.py tune
python benchmark.py test 0.05
python benchmark.py robust 0.05
python analyze.py test
python analyze.py robust
python make_figures.py
```

Set the environment variable `TRUSTNET_WORKERS` to change the number of parallel processes. The default is min(8, number of CPUs). This does not change the results, because every network has its own fixed random seeds:

| Quantity | Seed |
|---|---|
| Ground truth and source signals | network seed *s* (tuning 10001–10020, test 1–100, conditions 1–30) |
| Sensor noise | *s* + 500,000 |
| Surrogates | offset by 1,000 × *s* |
| Dipole tilt (orientation-error condition) | *s* + 777,000 |

The skull-conductivity condition computes a second forward model on first use and caches it in `benchmark/cache/` (about 25 MB, a few minutes).

### Where each result in the paper comes from

| Paper | File |
|---|---|
| Table 2, Figure 6 | `results/hub_case_study.json`, `results/hub_null_prop005.npy` |
| Section 4.2 (tuning) | `results/tune2.jsonl`, `results/tune2_choice.json` |
| Table 3, Figure 7, Sections 4.3–4.5 | `results/test2.jsonl`, `results/test2_summary.json` |
| Table 4, Table S1, Figure 8, Section 4.6 | `results/conditions2.jsonl`, `results/conditions2_summary.json` |
| Per-condition equivalence, mixing asymmetry, C3 coupling scale | `results/round2_numbers.json` |
| "Earlier run" (oracle with α = 0.5) | `results/test_summary.json` |

---

## 5. Exact reproducibility

With the versions in `requirements.txt` (tested: Python 3.14.3, numpy 2.4.2, scipy 1.17.0, mne 1.12.1, matplotlib 3.10.8, scikit-learn 1.8.0, Windows 11), rerunning the scripts reproduces every file in `results/` byte for byte. The only exception is the `runtime_s` field in `results/table2_rerun.json`. With other versions of numpy, scipy, MNE or the BLAS library, floating-point results can differ in the last digits. Rarely, an edge weight lying almost exactly on the threshold can then flip, which changes that network's metrics slightly. The summary statistics should agree to within their confidence intervals.

---

## 6. Citation

If you use this code, please cite the manuscript above. Full citation details will be added on publication.
