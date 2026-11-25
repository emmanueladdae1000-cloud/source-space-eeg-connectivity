# TrustNet benchmark design (fixed before any test-set run, 2026-11-24)

## Networks
- Generator: the author's `step1_ground_truth.py` (random spanning tree + extra random edges, weights U[0.40, 0.60], coupling spectral radius < 0.85), 10 nodes, 12 edges unless stated.
- **Tuning set:** seeds 10001–10020 (20 networks). Used only to choose β.
- **Test set:** seeds 1–100 (100 networks). Never used for any choice.
- Per network: MVAR(1) signals with seed = network seed; sensor-noise seed = network seed + 500000; surrogate seeds offset by 1000 × network seed. All other physics as in the author's scripts (MNE sample BEM, 60 electrodes, 10 dB, CAR, eLORETA λ² = 1/9, z-scored read-out at the 10 fixed vertices).

## Methods (each thresholded with its own null replayed through the identical chain)
| ID | Method | Null |
|---|---|---|
| RAW | Ridge-MVAR(1), α = 0.5 | pooled, 20 surrogates, 99th percentile |
| SIGORTH | symmetric (Löwdin) orthogonalization of the reconstructed signals before Ridge-MVAR (signal-level leakage correction in the spirit of Colclough et al. 2015) | pooled, 20 surrogates, 99th percentile |
| EDGEZ | earlier pipeline: Ridge-MVAR + receiver-side deblurring (β = 0.10), edge-wise leave-one-out z | 30 surrogates, 99.5th percentile |
| PROPOSED | Ridge-MVAR + symmetric innovation-residual orthogonalization, β chosen on the tuning set | pooled, 20 surrogates, 99th percentile |
| PROPOSED-DEPLOY | same estimator; surrogates built from the *reconstructed* signals (phase-randomized Ŷ re-projected through forward model, noise, CAR and inverse), i.e. without access to the true sources | pooled, 20 surrogates, 99th percentile |
| ORACLE | Ridge-MVAR on the clean simulated sources (no measurement chain); reference ceiling | pooled on clean surrogates, 20, 99th percentile |

## β selection (tuning set only)
Grid β ∈ {0.01, 0.02, 0.05, 0.10, 0.20, 0.50}; choose the β with the highest median MCC over the 20 tuning networks (ties → smaller β). The chosen β is then frozen for all test runs.

## Outcomes (per network, over the 90 off-diagonal directed pairs)
Precision, recall, F1, MCC (thresholded); AUROC and average precision (threshold-free, from the continuous scores; for EDGEZ the z-scores); number of false positives; false-detection rate on two-step cascade traps (ordered pairs linked by a two-edge path without a direct edge) and on common-source cross-pairs (two targets of one source, no edge between them).
Summaries: mean with 95% percentile-bootstrap CI over networks (10,000 resamples); paired two-sided Wilcoxon signed-rank test of PROPOSED vs each other method on MCC, Holm-corrected.

## Robustness (test seeds 1–30, RAW vs PROPOSED)
1. SNR ∈ {0, 5, 10, 20} dB.
2. Edge count ∈ {9, 12, 15, 18}.
3. Head-model mismatch: data simulated with the sample forward model; inverse operator **and** null built with a forward model recomputed from the same BEM surfaces with a different skull conductivity (0.3 / 0.0042 / 0.3 S/m instead of the template's 3-layer model).

## Amendment before any results (smoke test, 2026-11-24)
The author's generator checks stationarity on the coupling matrix W only (spectral radius < 0.85), but step 2 simulates A = W + 0.40 I. Some accepted networks therefore have spectral radius of A ≥ 1 and step 2 aborts (first seen on tuning seed 10001). The generator now keeps the author's constraint and additionally requires spectral radius of A < 1, the correct MVAR stability condition. No outcome had been computed when this change was made.

Second amendment (same smoke test, before any tuning or test results): the skull-conductivity mismatch changes the lead field by 16% (Frobenius) but leaves each node's topography almost unchanged in shape (cosine > 0.998), so after z-scoring it is a weak test. It is kept and reported as such, and a stronger mismatch is added: data simulated with each dipole tilted 30° from the cortical normal (random azimuth per node, seed = network seed + 777000) while the inverse and the null assume surface-normal dipoles (condition `tilt30`).

## Extension after external review (specified 2026-11-24, before any extension run)
An independent review of the first manuscript draft asked for experiments that the first benchmark did not contain. They are added here, before running, and are reported in full regardless of outcome. The first-benchmark results stay on file (`results/test.jsonl`, `results/robust.jsonl`).

**Equal tuning (tuning seeds 10001–10020 only).** Every method gets the same tuning budget, choosing by highest median MCC (ties → first in grid order):
- Ridge-MVAR and signal-level orthogonalization: α ∈ {0.5, 50, 500, 5000}.
- Proposed: α ∈ {0.5, 50, 500, 5000} × β ∈ {0.01, 0.02, 0.05, 0.10, 0.20, 0.50}.
- TRGC and gPDC: use the α chosen for Ridge-MVAR.

**Added comparators** (each with its own pooled 99th-percentile simulation null, 20 surrogates):
- TRGC: time-reversed conditional Granger causality, GC(data) − GC(time-reversed data), from ridge-regularized MVAR (Winkler et al. 2016).
- gPDC: generalized partial directed coherence, averaged over 1–40 Hz.

**Null ablation for the proposed estimator:**
- **(a) simulation null:** clean sources scrambled, then replayed through the chain.
- **(b) deployable null:** reconstructed sources scrambled, then replayed.
- **(c) post-inversion null:** reconstructed sources scrambled, no replay; the standard approach.

**Equivalence.** Deployable vs simulation null: two one-sided tests on the paired MCC difference with margin ±0.05, using a 90% bootstrap CI.

**Test set.** The 100 test networks are rerun with all methods at their tuned settings.

**Conditions that violate the estimator's assumptions or add realism** (test seeds 1–30, all methods):
- **C1 unequal innovations:** innovation variances U(0.5, 2.0).
- **C2 correlated innovations:** genuine zero-lag interactions; three random node pairs with innovation correlation 0.3.
- **C3 oscillatory dynamics:** each node a damped 8–12 Hz resonator (MVAR(2) diagonal, r = 0.9) with lag-1 coupling; estimators use order 2 (coupling down-scaled until the companion matrix is stable; the scale is recorded).
- **C4 background activity:** 200 random background sources with independent AR(1) activity (coefficient 0.9) at −3 dB relative to the network's sensor power; the simulation null does not contain the background (it is unknown to the analyst), whereas the deployable null does.
- **C5 random source locations:** 10 vertices drawn per network with pairwise distance ≥ 20 mm.
- **C6 patch read-out:** each node read out as the first principal component of the 20 vertices nearest to it, instead of the true vertex.
- **C7 short recordings:** T = 5,000 samples (20 s).
- **C8 combined realism:** C1 + C4 + C5 + C6.

*Amendment to C3 (smoke test on seeds 3–5, before any extension run):* with stability as the only constraint, cascades of resonators amplified some node variances to about 10^6 times others, which is non-physiological and made even the oracle fail. The coupling is therefore shrunk by factors of 0.9 until the companion matrix is stable **and** the largest stationary node variance (discrete Lyapunov solution) is at most 20 times the smallest; the resulting scale is recorded per network.

**First-benchmark robustness conditions** (SNR 0/5/20 dB, 9/15/18 edges, skull-conductivity error, 30° tilt) are rerun with all methods. False positives per network are reported with CIs for every condition, and the robustness p-values are Holm-corrected within each comparison.

## Reporting rule
All results are reported as obtained, including where PROPOSED is not better. No re-running with different seeds or settings after seeing test results.
