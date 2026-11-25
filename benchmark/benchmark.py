"""TrustNet benchmark (design fixed in BENCHMARK_DESIGN.md).

Usage:
  python benchmark.py tune            # beta grid on tuning seeds 10001-10020
  python benchmark.py test BETA       # 100 test networks, all methods
  python benchmark.py robust BETA     # SNR / density / head-model mismatch on seeds 1-30
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np

import trustnet_harness as th

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
CACHE = HERE / "cache"
RES.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
N_WORKERS = int(os.environ.get("TRUSTNET_WORKERS", min(8, os.cpu_count() or 1)))
_PHYS = {}


# ----------------------------------------------------------------------------- physics (per worker)
def physics(kind="template"):
    if kind not in _PHYS:
        base = _PHYS.get("template") or th.load_physics()
        _PHYS["template"] = base
        if kind == "mismatch":
            _PHYS["mismatch"] = mismatch_physics(base)
    return _PHYS[kind]


def mismatch_physics(base):
    """Forward model recomputed from the same BEM surfaces with a different skull conductivity."""
    import mne
    f = CACHE / "fwd_mismatch_skull0042-fwd.fif"
    sp = mne.datasets.sample.data_path(download=False)
    subjects_dir = sp / "subjects"
    if not f.exists():
        # Same 3-layer surfaces as the template BEM; only the skull conductivity changes
        model = mne.read_bem_surfaces(subjects_dir / "sample" / "bem" / "sample-5120-5120-5120-bem.fif", verbose=False)
        sig_before = [float(s["sigma"]) for s in model]
        for s in model:
            if s["id"] == mne.io.constants.FIFF.FIFFV_BEM_SURF_ID_SKULL:
                s["sigma"] = 0.0042
        (CACHE / "mismatch_conductivities.json").write_text(json.dumps(
            {"template": sig_before, "mismatch": [float(s["sigma"]) for s in model]}))
        bem = mne.make_bem_solution(model, verbose=False)
        src = mne.read_source_spaces(subjects_dir / "sample" / "bem" / "sample-oct-6-src.fif", verbose=False)
        trans = sp / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
        fwd = mne.make_forward_solution(base["info"], trans, src, bem, meg=False, eeg=True, mindist=5.0, verbose=False)
        mne.write_forward_solution(f, fwd, overwrite=True, verbose=False)
    fwd = mne.read_forward_solution(f, verbose=False)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)
    # map the template's 10 vertices into this forward model by vertex number
    vt = [s["vertno"] for s in base["fwd"]["src"]]
    vm = [s["vertno"] for s in fwd["src"]]
    n_lh_t = len(vt[0])
    sel_m = []
    for idx in base["sel"]:
        hemi, k = (0, idx) if idx < n_lh_t else (1, idx - n_lh_t)
        vertex = vt[hemi][k]
        pos = int(np.where(vm[hemi] == vertex)[0][0])
        sel_m.append(pos if hemi == 0 else len(vm[0]) + pos)
    sel_m = np.array(sel_m)
    noise_cov = mne.make_ad_hoc_cov(base["info"], verbose=False)
    inv = mne.minimum_norm.make_inverse_operator(base["info"], fwd, noise_cov, loose=0.0, depth=None, fixed=True, verbose=False)
    import mne as _m  # noqa: F401
    base_info = _m.pick_info(base["info"], range(len(base["info"]["ch_names"])))
    with base_info._unlock():
        base_info["projs"] = []
    K = th.effective_kernel(inv, base_info, sel_m)
    return {"L": fwd["sol"]["data"][:, sel_m], "K": K, "sel": sel_m}


# ----------------------------------------------------------------------------- estimators
def est_sigorth(Y_z, alpha=0.5):
    """Symmetric (Loewdin) orthogonalization of the signals, then plain Ridge-MVAR."""
    C = np.corrcoef(Y_z)
    w, V = np.linalg.eigh(C)
    Yo = (V / np.sqrt(np.maximum(w, 1e-6))) @ V.T @ Y_z
    Yo = (Yo - Yo.mean(1, keepdims=True)) / Yo.std(1, keepdims=True)
    return th.est_raw(Yo, alpha)


# ----------------------------------------------------------------------------- traps
def trap_masks(G):
    two = ((G @ G) > 0) & (G == 0) & th.OFF
    cs = np.zeros_like(G, dtype=bool)
    for k in range(10):
        tg = np.where(G[k] == 1)[0]
        for i in tg:
            for j in tg:
                if i != j and G[i, j] == 0:
                    cs[i, j] = True
    return two, cs


def add_traps(m, G, Gest):
    two, cs = trap_masks(G)
    m["n_two_step"], m["fp_two_step"] = int(two.sum()), int((Gest.astype(bool) & two).sum())
    m["n_common_src"], m["fp_common_src"] = int(cs.sum()), int((Gest.astype(bool) & cs).sum())
    return m


# ----------------------------------------------------------------------------- one network
def surrogates(source, sim_L, model, estimator, n, off, snr_db, measure=True):
    out = []
    for s in range(n):
        Ys = th.phase_randomize(source, seed=off + 100 + s)
        if measure:
            rng = np.random.default_rng(off + 1000 + s)
            Ys = th.reconstruct(scalp(Ys, model["L"], rng, snr_db), model["K"])
        else:
            Ys = (Ys - Ys.mean(1, keepdims=True)) / Ys.std(1, keepdims=True)
        out.append(estimator(Ys))
    return np.stack(out)


def scalp(Y, L, rng, snr_db):
    Z = L @ Y
    return Z + rng.normal(0, np.sqrt(np.mean(Z ** 2) / (10 ** (snr_db / 10.0))), size=Z.shape)


def tilted_leadfield(seed, angle_deg=30.0):
    """True lead field when each source dipole is tilted angle_deg from the cortical normal (random azimuth)."""
    import mne
    if "G3" not in _PHYS:
        sp = mne.datasets.sample.data_path(download=False)
        fwd = mne.read_forward_solution(sp / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif", verbose=False)
        fwd = mne.pick_types_forward(fwd, meg=False, eeg=True, exclude=[])
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False, verbose=False)
        sel = physics("template")["sel"]
        _PHYS["G3"] = np.stack([fwd["sol"]["data"][:, 3 * k:3 * k + 3] for k in sel])   # (10, 60, 3); col 2 = normal
    rng = np.random.default_rng(seed + 777000)
    th_ = np.deg2rad(angle_deg)
    cols = []
    for G3 in _PHYS["G3"]:
        phi = rng.uniform(0, 2 * np.pi)
        u = np.array([np.sin(th_) * np.cos(phi), np.sin(th_) * np.sin(phi), np.cos(th_)])
        cols.append(G3 @ u)
    return np.stack(cols, axis=1)


def run_network(task):
    seed, beta, methods, snr_db, n_edges, head = (task[k] for k in ("seed", "beta", "methods", "snr_db", "n_edges", "head"))
    sim = dict(physics("template"))
    if head == "tilt30":                  # data generated with tilted dipoles; analyst assumes normal orientation
        sim["L"] = tilted_leadfield(seed, 30.0)
        model = physics("template")
    else:
        model = physics(head)             # what the analyst believes (inverse + null)
    model = {"L": model["L"], "K": model["K"]}
    G, W = th.random_ground_truth(seed, n_edges=n_edges)
    Yc = th.simulate_mvar(W, seed=seed)
    Yh = th.reconstruct(scalp(Yc, sim["L"], np.random.default_rng(seed + 500000), snr_db), model["K"])
    off = 1000 * seed
    prop = lambda Y: th.est_symorth(Y, beta=beta)
    out = {"seed": seed, "beta": beta, "snr_db": snr_db, "n_edges": n_edges, "head": head,
           "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(W))))}
    for mth in methods:
        if mth == "EDGEZ":
            null = surrogates(Yc, sim["L"], model, th.est_deblur_signed, 30, off, snr_db)
            Z, zthr, _, _ = th.edgewise_z(null, th.est_deblur_signed(Yh), 99.5)
            Gest, score = (Z > zthr).astype(int), Z
        elif mth == "ORACLE":
            Yz = (Yc - Yc.mean(1, keepdims=True)) / Yc.std(1, keepdims=True)
            null = surrogates(Yc, None, None, th.est_raw, 20, off, snr_db, measure=False)
            score = th.est_raw(Yz)
            Gest = (score > th.pooled_threshold(null, 99.0)).astype(int)
        else:
            est = {"RAW": th.est_raw, "SIGORTH": est_sigorth, "PROPOSED": prop, "PROPOSED-DEPLOY": prop}[mth]
            src = Yh if mth == "PROPOSED-DEPLOY" else Yc
            null = surrogates(src, sim["L"], model, est, 20, off, snr_db)
            score = est(Yh)
            Gest = (score > th.pooled_threshold(null, 99.0)).astype(int)
        np.fill_diagonal(Gest, 0)
        out[mth] = add_traps(th.metrics(G, Gest, score), G, Gest)
    return out


def run_many(tasks, out_name):
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        rows = pool.map(run_network, tasks, chunksize=1)
    path = RES / out_name
    path.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"{len(rows)} networks -> {path.name} in {time.time() - t0:.0f}s")
    return rows


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "tune":
        grid = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
        tasks = [{"seed": s, "beta": b, "methods": ["PROPOSED"], "snr_db": 10.0, "n_edges": 12, "head": "template"}
                 for b in grid for s in range(10001, 10021)]
        rows = run_many(tasks, "tune.jsonl")
        for b in grid:
            mcc = [r["PROPOSED"]["mcc"] for r in rows if r["beta"] == b]
            print(f"beta={b:<5} median MCC={np.median(mcc):.4f} mean MCC={np.mean(mcc):.4f}")
    elif stage == "test":
        beta = float(sys.argv[2])
        methods = ["ORACLE", "RAW", "SIGORTH", "EDGEZ", "PROPOSED", "PROPOSED-DEPLOY"]
        run_many([{"seed": s, "beta": beta, "methods": methods, "snr_db": 10.0, "n_edges": 12, "head": "template"}
                  for s in range(1, 101)], "test.jsonl")
    elif stage == "robust":
        beta = float(sys.argv[2])
        m = ["RAW", "PROPOSED"]
        tasks = [{"seed": s, "beta": beta, "methods": m, "snr_db": snr, "n_edges": 12, "head": "template"}
                 for snr in (0.0, 5.0, 20.0) for s in range(1, 31)]
        tasks += [{"seed": s, "beta": beta, "methods": m, "snr_db": 10.0, "n_edges": e, "head": "template"}
                  for e in (9, 15, 18) for s in range(1, 31)]
        tasks += [{"seed": s, "beta": beta, "methods": m, "snr_db": 10.0, "n_edges": 12, "head": h}
                  for h in ("mismatch", "tilt30") for s in range(1, 31)]
        run_many(tasks, "robust.jsonl")
