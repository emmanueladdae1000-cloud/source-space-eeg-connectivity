"""Extended TrustNet benchmark (design: BENCHMARK_DESIGN.md, 'Extension after external review').

Usage:
  python benchmark2.py tune                 # equal tuning on seeds 10001-10020 -> results/tune2.json
  python benchmark2.py test                 # 100 test networks, all methods, tuned settings
  python benchmark2.py conditions           # C1-C8 + first-benchmark robustness conditions, seeds 1-30
  python benchmark2.py check [n]            # quick check: rerun test networks 1..n (default 2) and compare
                                            # with the saved results/test2.jsonl

Set TRUSTNET_WORKERS to change the number of parallel processes (default: min(8, CPU count)).
The worker count does not affect the results: every network has its own fixed seeds.
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
RES, CACHE = HERE / "results", HERE / "cache"
OFF = th.OFF
N = 10
FS = 250.0
_P = {}
N_WORKERS = int(os.environ.get("TRUSTNET_WORKERS", min(8, os.cpu_count() or 1)))
ALL = ["ORACLE", "RAW", "SIGORTH", "TRGC", "GPDC", "PROPOSED", "PROPOSED-DEPLOY", "PROPOSED-POSTINV"]


# ============================================================================ physics
def phys():
    if "t" not in _P:
        import mne
        base = th.load_physics()
        info = mne.pick_info(base["info"], range(len(base["info"]["ch_names"])))
        with info._unlock():
            info["projs"] = []
        n_src = base["fwd"]["sol"]["data"].shape[1]
        base["K_full"] = th.effective_kernel(base["inv"], info, np.arange(n_src))      # (7498, 60)
        base["L_full"] = base["fwd"]["sol"]["data"]                                     # (60, 7498)
        base["rr"] = base["fwd"]["source_rr"] * 1000.0                                  # mm
        _P["t"] = base
    return _P["t"]


def model_for(head):
    if head == "template":
        p = phys()
        return {"L": p["L"], "K": p["K"]}
    import benchmark as b1
    b1._PHYS["template"] = phys()
    m = b1.physics("mismatch")
    return {"L": m["L"], "K": m["K"]}


# ============================================================================ estimators (order p)
def ridge(Yz, alpha, p=1):
    X = Yz.T
    T = X.shape[0]
    Yc = X[p:]
    Xl = np.hstack([X[p - k:T - k] for k in range(1, p + 1)])
    A = np.linalg.solve(Xl.T @ Xl + alpha * np.eye(N * p), Xl.T @ Yc)
    return A.reshape(p, N, N), Yc - Xl @ A, Xl, Yc


def est_raw(Yz, alpha, p=1, **_):
    A, *_ = ridge(Yz, alpha, p)
    W = np.linalg.norm(A, axis=0)
    np.fill_diagonal(W, 0)
    return W


def est_sigorth(Yz, alpha, p=1, **_):
    C = np.corrcoef(Yz)
    w, V = np.linalg.eigh(C)
    Yo = (V / np.sqrt(np.maximum(w, 1e-6))) @ V.T @ Yz
    Yo = (Yo - Yo.mean(1, keepdims=True)) / Yo.std(1, keepdims=True)
    return est_raw(Yo, alpha, p)


def est_prop(Yz, alpha, p=1, beta=0.10, **_):
    A, E, *_ = ridge(Yz, alpha, p)
    s = np.maximum(E.std(0), 1e-8)
    C = (1 - beta) * np.corrcoef(E, rowvar=False) + beta * np.eye(N)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-6)
    Ch, Cih = (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T
    W = np.linalg.norm(np.stack([Ch @ ((s[:, None] * Ak) / s[None, :]) @ Cih for Ak in A]), axis=0)
    np.fill_diagonal(W, 0)
    return W


def _cond_gc(Yz, alpha, p):
    """Conditional GC_{i->j} = log(var_j | without lags of i) - log(var_j | all lags)."""
    _, E, Xl, Yc = ridge(Yz, alpha, p)
    v_full = E.var(0)
    G = np.zeros((N, N))
    for i in range(N):
        keep = [c for c in range(N * p) if c % N != i]
        Xr = Xl[:, keep]
        Ar = np.linalg.solve(Xr.T @ Xr + alpha * np.eye(len(keep)), Xr.T @ Yc)
        v_red = (Yc - Xr @ Ar).var(0)
        G[i] = np.log(np.maximum(v_red, 1e-12) / np.maximum(v_full, 1e-12))
    np.fill_diagonal(G, 0)
    return G


def est_trgc(Yz, alpha, p=1, **_):
    """Time-reversed Granger causality score (forward GC minus time-reversed GC), floored at 0."""
    D = _cond_gc(Yz, alpha, p) - _cond_gc(Yz[:, ::-1].copy(), alpha, p)
    W = np.maximum(D, 0)
    np.fill_diagonal(W, 0)
    return W


def est_gpdc(Yz, alpha, p=1, **_):
    """Generalized PDC averaged over 1-40 Hz; W[i, j] = influence i -> j."""
    A, E, *_ = ridge(Yz, alpha, p)
    sig = np.maximum(E.std(0), 1e-8)
    B = np.stack([Ak.T for Ak in A])                       # B[k][target, source]
    acc = np.zeros((N, N))
    freqs = np.arange(1, 41)
    for f in freqs:
        Af = np.eye(N, dtype=complex) - sum(B[k] * np.exp(-2j * np.pi * f * (k + 1) / FS) for k in range(p))
        num = np.abs(Af) / sig[:, None]                    # |A_ts| / sigma_t
        den = np.sqrt(np.sum((np.abs(Af) ** 2) / (sig[:, None] ** 2), axis=0))
        acc += (num / den[None, :]).T                       # transpose -> [source, target]
    W = acc / len(freqs)
    np.fill_diagonal(W, 0)
    return W


EST = {"RAW": est_raw, "SIGORTH": est_sigorth, "PROPOSED": est_prop, "PROPOSED-DEPLOY": est_prop,
       "PROPOSED-POSTINV": est_prop, "TRGC": est_trgc, "GPDC": est_gpdc, "ORACLE": est_raw}


# ============================================================================ simulation variants
def simulate(W, seed, T=15000, innov="equal", dyn="mvar1", burn=1000):
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((T + burn, N))
    r2 = np.random.default_rng(seed + 333000)
    info = {}
    if innov == "unequal":
        var = r2.uniform(0.5, 2.0, N)
        E = E * np.sqrt(var)
        info["innov_var"] = var.round(3).tolist()
    elif innov == "correlated":
        C = np.eye(N)
        pairs = []
        while len(pairs) < 3:
            i, j = r2.choice(N, 2, replace=False)
            if (min(i, j), max(i, j)) not in pairs:
                pairs.append((min(i, j), max(i, j)))
                C[i, j] = C[j, i] = 0.3
        E = E @ np.linalg.cholesky(C).T
        info["zero_lag_pairs"] = [[int(a) + 1, int(b) + 1] for a, b in pairs]
    Y = np.zeros((T + burn, N))
    if dyn == "mvar1":
        A1 = W + 0.40 * np.eye(N)
        for t in range(1, T + burn):
            Y[t] = Y[t - 1] @ A1 + E[t]
    else:                                                   # damped 8-12 Hz resonators + lag-1 coupling
        f0 = r2.uniform(8, 12, N)
        rr = 0.9
        a1, a2 = np.diag(2 * rr * np.cos(2 * np.pi * f0 / FS)), np.diag(np.full(N, -rr ** 2))
        from scipy.linalg import solve_discrete_lyapunov
        scale = 1.0
        while True:                                     # stable AND node variances within a factor of 20
            A1 = a1 + scale * W
            comp = np.block([[A1.T, a2.T], [np.eye(N), np.zeros((N, N))]])
            if np.max(np.abs(np.linalg.eigvals(comp))) < 0.99:
                Q = np.zeros((2 * N, 2 * N)); Q[:N, :N] = np.eye(N)
                var = np.diag(solve_discrete_lyapunov(comp, Q))[:N]
                if var.max() / var.min() <= 20.0:
                    break
            scale *= 0.9
        info["coupling_scale"] = round(scale, 4)
        for t in range(2, T + burn):
            Y[t] = Y[t - 1] @ A1 + Y[t - 2] @ a2 + E[t]
    return Y[burn:].T, info


def random_locations(seed, min_mm=20.0):
    rr = phys()["rr"]
    rng = np.random.default_rng(seed + 444000)
    while True:
        sel = []
        for cand in rng.permutation(len(rr)):
            if all(np.linalg.norm(rr[cand] - rr[s]) >= min_mm for s in sel):
                sel.append(cand)
            if len(sel) == N:
                return np.array(sel)


def patch_index(sel, k=20):
    rr = phys()["rr"]
    return [np.argsort(np.linalg.norm(rr - rr[s], axis=1))[:k] for s in sel]


def background(sel, seed, n_bg=200, min_mm=20.0):
    rr = phys()["rr"]
    rng = np.random.default_rng(seed + 555000)
    far = np.where(np.min(np.linalg.norm(rr[:, None, :] - rr[sel][None, :, :], axis=2), axis=1) >= min_mm)[0]
    return rng.choice(far, n_bg, replace=False)


def ar1_sources(n, T, seed, coef=0.9):
    rng = np.random.default_rng(seed + 666000)
    E = rng.standard_normal((T + 500, n))
    X = np.zeros_like(E)
    for t in range(1, T + 500):
        X[t] = coef * X[t - 1] + E[t]
    return X[500:].T


# ============================================================================ measurement + read-out
class Chain:
    """Forward model + noise + CAR-eLORETA + read-out for one network/condition."""

    def __init__(self, sel, patches=None, bg=None, model_override=None, snr_db=10.0):
        p = phys()
        self.L_true = p["L_full"][:, sel]
        if model_override is not None:                 # analyst's (possibly wrong) model at the template vertices
            self.L_model, self.K_rows = model_override["L"], [model_override["K"][i:i + 1] for i in range(N)]
        else:
            self.L_model = self.L_true
            self.K_rows = [p["K_full"][idx] for idx in patches] if patches is not None else [p["K_full"][[s]] for s in sel]
        self.bg_L = p["L_full"][:, bg] if bg is not None else None
        self.snr_db = snr_db

    def scalp(self, Y, rng, L=None, bg_Z=None):
        Z = (self.L_true if L is None else L) @ Y
        if bg_Z is not None:
            Z = Z + bg_Z
        return Z + rng.normal(0, np.sqrt(np.mean(Z ** 2) / 10 ** (self.snr_db / 10)), size=Z.shape)

    def readout(self, Z):
        out = []
        for Kr in self.K_rows:
            P = Kr @ Z
            if P.shape[0] == 1:
                x = P[0]
            else:                                          # first principal component of the patch
                Pc = P - P.mean(1, keepdims=True)
                u, _, _ = np.linalg.svd(Pc @ Pc.T)
                x = u[:, 0] @ Pc
            out.append((x - x.mean()) / x.std())
        return np.stack(out)


# ============================================================================ one network
def run_network(task):
    seed, methods = task["seed"], task["methods"]
    snr, ne, head = task.get("snr_db", 10.0), task.get("n_edges", 12), task.get("head", "template")
    T, innov, dyn = task.get("T", 15000), task.get("innov", "equal"), task.get("dyn", "mvar1")
    cfg = task["cfg"]                                      # tuned params per method: {"RAW": {"alpha":..}, ...}
    p_order = 2 if dyn == "osc2" else 1
    p = phys()
    G, W = th.random_ground_truth(seed, n_edges=ne)
    Yc, sim_info = simulate(W, seed, T=T, innov=innov, dyn=dyn)
    sel = random_locations(seed) if task.get("locations") == "random" else p["sel"]
    patches = patch_index(sel) if task.get("readout") == "patch" else None
    bg = background(sel, seed) if task.get("background") else None
    override = None
    if head == "mismatch":
        override = model_for("mismatch")
    chain = Chain(sel, patches, bg, override, snr)
    if head == "tilt30":
        import benchmark as b1
        b1._PHYS["template"] = p
        chain.L_true = b1.tilted_leadfield(seed, 30.0)
    bg_Z = None
    if bg is not None:
        Zs = chain.L_true @ Yc
        Zb = chain.bg_L @ ar1_sources(len(bg), T, seed)
        bg_Z = Zb * np.sqrt(0.5 * np.mean(Zs ** 2) / np.mean(Zb ** 2))     # -3 dB relative to network
    Yh = chain.readout(chain.scalp(Yc, np.random.default_rng(seed + 500000), bg_Z=bg_Z))
    off = 1000 * seed
    out = {"seed": seed, "task": {k: v for k, v in task.items() if k != "cfg"}, "sim_info": sim_info}
    for m in methods:
        prm = dict(cfg.get(m, cfg.get("RAW", {"alpha": 0.5})))
        f = lambda Y, _m=m, _p=prm: EST[_m](Y, p=p_order, **_p)
        null = []
        for s in range(20):
            rng = np.random.default_rng(off + 1000 + s)
            if m == "ORACLE":
                Ys = th.phase_randomize(Yc, seed=off + 100 + s)
                null.append(f((Ys - Ys.mean(1, keepdims=True)) / Ys.std(1, keepdims=True)))
                continue
            if m == "PROPOSED-POSTINV":
                Ys = th.phase_randomize(Yh, seed=off + 100 + s)
                null.append(f((Ys - Ys.mean(1, keepdims=True)) / Ys.std(1, keepdims=True)))
                continue
            src = Yh if m == "PROPOSED-DEPLOY" else Yc
            Ys = th.phase_randomize(src, seed=off + 100 + s)
            null.append(f(chain.readout(chain.scalp(Ys, rng, L=chain.L_model))))
        null = np.stack(null)
        tau = float(np.percentile(np.concatenate([w[OFF] for w in null]), 99.0))
        score = f((Yc - Yc.mean(1, keepdims=True)) / Yc.std(1, keepdims=True)) if m == "ORACLE" else f(Yh)
        Gest = (score > tau).astype(int)
        np.fill_diagonal(Gest, 0)
        mm = th.metrics(G, Gest, score)
        import benchmark as b1
        out[m] = b1.add_traps(mm, G, Gest)
    return out


def pool_run(tasks, name):
    t0 = time.time()
    with Pool(N_WORKERS) as pl:
        rows = pl.map(run_network, tasks, chunksize=1)
    (RES / name).write_text("\n".join(json.dumps(r) for r in rows))
    print(f"{len(rows)} tasks -> {name} in {time.time() - t0:.0f}s")
    return rows


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "tune":
        tasks, A, B = [], [0.5, 50.0, 500.0, 5000.0], [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
        for a in A:
            for s in range(10001, 10021):
                tasks.append({"seed": s, "methods": ["RAW", "SIGORTH"], "cfg": {"RAW": {"alpha": a}, "SIGORTH": {"alpha": a}}, "grid": ["alpha", a]})
                for b in B:
                    tasks.append({"seed": s, "methods": ["PROPOSED"], "cfg": {"PROPOSED": {"alpha": a, "beta": b}}, "grid": ["alpha_beta", a, b]})
        rows = pool_run(tasks, "tune2.jsonl")
        choice = {}
        for m in ["RAW", "SIGORTH"]:
            med = {a: np.median([r[m]["mcc"] for r in rows if m in r and r["task"]["grid"] == ["alpha", a]]) for a in A}
            choice[m] = {"alpha": max(A, key=lambda a: (med[a], -A.index(a)))}
            print(m, {a: round(v, 4) for a, v in med.items()}, "->", choice[m])
        grid = [(a, b) for a in A for b in B]
        med = {g: np.median([r["PROPOSED"]["mcc"] for r in rows if "PROPOSED" in r and r["task"]["grid"] == ["alpha_beta", *g]]) for g in grid}
        best = max(grid, key=lambda g: (med[g], -grid.index(g)))
        choice["PROPOSED"] = {"alpha": best[0], "beta": best[1]}
        print("PROPOSED top:", sorted(((round(v, 4), g) for g, v in med.items()), reverse=True)[:6], "->", choice["PROPOSED"])
        for m in ["PROPOSED-DEPLOY", "PROPOSED-POSTINV"]:
            choice[m] = dict(choice["PROPOSED"])
        for m in ["TRGC", "GPDC", "ORACLE"]:
            choice[m] = dict(choice["RAW"])
        (RES / "tune2_choice.json").write_text(json.dumps(choice, indent=2))
    elif stage == "check":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        cfg = json.loads((RES / "tune2_choice.json").read_text())
        saved = {r["seed"]: r for r in map(json.loads, (RES / "test2.jsonl").read_text().splitlines())}
        t0, worst = time.time(), 0.0
        for s in range(1, n + 1):
            r = run_network({"seed": s, "methods": ALL, "cfg": cfg})
            for m in ALL:
                d = abs(r[m]["mcc"] - saved[s][m]["mcc"])
                worst = max(worst, d)
                print(f"network {s:3d} {m:17s} MCC {r[m]['mcc']:.4f}  saved {saved[s][m]['mcc']:.4f}  diff {d:.1e}")
        print(f"largest MCC difference {worst:.1e} ({time.time() - t0:.0f}s)")
        print("OK: matches the saved results" if worst < 1e-6 else "MISMATCH: see README, 'Exact reproducibility'")
    else:
        cfg = json.loads((RES / "tune2_choice.json").read_text())
        if stage == "test":
            pool_run([{"seed": s, "methods": ALL, "cfg": cfg} for s in range(1, 101)], "test2.jsonl")
        elif stage == "conditions":
            conds = {"C1_unequal": {"innov": "unequal"}, "C2_correlated": {"innov": "correlated"},
                     "C3_oscillatory": {"dyn": "osc2"}, "C4_background": {"background": True},
                     "C5_random_locations": {"locations": "random"}, "C6_patch_readout": {"readout": "patch"},
                     "C7_short_20s": {"T": 5000},
                     "C8_combined": {"innov": "unequal", "background": True, "locations": "random", "readout": "patch"},
                     "R_snr0": {"snr_db": 0.0}, "R_snr5": {"snr_db": 5.0}, "R_snr20": {"snr_db": 20.0},
                     "R_edges9": {"n_edges": 9}, "R_edges15": {"n_edges": 15}, "R_edges18": {"n_edges": 18},
                     "R_skull": {"head": "mismatch"}, "R_tilt30": {"head": "tilt30"}}
            tasks = [dict({"seed": s, "methods": ALL, "cfg": cfg, "cond": c}, **kw) for c, kw in conds.items() for s in range(1, 31)]
            pool_run(tasks, "conditions2.jsonl")
