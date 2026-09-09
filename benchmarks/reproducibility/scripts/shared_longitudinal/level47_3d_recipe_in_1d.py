"""
level47_3d_recipe_in_1d.py
==========================
2026-08-10: run the 1D theory model with the 3D-solver field recipe
for ALL THREE intensities (f_sc = 0.5 / 0.7 / 1.0), full 240 turns,
and re-compare against the extended 3D data.

Field recipe (= 3D solver, plain words): slice the bunch into 128
slices over the min/max arc, sum the disk-kernel field per slice
(exact O(N^2)), and give every particle the field of ITS slice
(step assignment, no interpolation).

Init: each run's own 3D turn-1 particles, N=20000, seed 42;
1000 steps/turn, field update every 10 steps (baseline recipe).

Caches (curves + snapshots + per-snapshot slice field):
  outputs/level47_1d_cache_recipe3d_f{0p5,0p7,1p0}.npz
Figures:
  level47_sigma_t_all.png   (3 intensities x sigma_s/sigma_E,
                             3D vs 1D baseline vs 1D 3D-recipe)
  level47_phasespace_all.png (3 intensities x 4 turns,
                              3D top, 1D 3D-recipe bottom)
  level47_Es_all.png        (Es(s) 3D binned vs 1D 3D-recipe)
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

LEVEL0_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEVEL0_DIR.parent))
from bundle_paths import REPO_ROOT
sys.path.insert(0, str(LEVEL0_DIR))

from level5_bucket_sampled_tracking import (
    load_seo_freq_table, rk4_step_numba, E0_MEV, BETA_C, Q_TOTAL,
)
from level33_cache_3d_and_bare1D import load_turn_particles

# ================================================================
# CONFIG — edit here
# ================================================================

RUNS_3D = {
    0.5: REPO_ROOT / "HPC_dir" / "Longitudinal_0p5",
    0.7: REPO_ROOT / "HPC_dir" / "Longitudinal_0p7",
    1.0: REPO_ROOT / "HPC_dir" / "Longitudinal_1p0",
}
V_TURN = 200e3
N_TURNS = 240
N_PART = 20000
N_STEPS_PER_TURN = 1000
ES_UPDATE_INTERVAL = 10
NS_3D = 128
SEED = 42
SNAP_EVERY = 10

FIG_TURNS_3D = (51, 101, 151, 221)
S_LIM, E_LIM = 20.0, 10.0
OUT_DIR = LEVEL0_DIR / "outputs"
SHOW_FIGS = False
SAVE_FIGS = True
A_DISK = 0.04
EPS0 = 8.854187817e-12

# ================================================================


def fname_f(f):
    return f"f{f:.1f}".replace(".", "p")


def disk_sum_1d(z, q, a=A_DISK):
    """O(N^2) exact disk-kernel field (numpy, ns=128 -> trivial)."""
    r = z[:, None] - z[None, :]
    ar = np.abs(r)
    K = np.sign(r) * (1.0 - ar / np.sqrt(r * r + a * a))
    K[np.isnan(K)] = 0.0
    return (1.0 / (2.0 * np.pi * EPS0 * a * a)) * (K @ q)


def track_recipe3d(zd0, dEk0, P, t_lookup, n_turns, n_steps,
                   snap_turns, es_update=10):
    """1D tracker with the 3D-solver field recipe."""
    zd = zd0.copy()
    dEk = dEk0.copy()
    dt = P["Tc"] / n_steps
    f_sc = P["f_sc"]
    E_min, dE_table, T_table = t_lookup
    Tc, w0, fis, V_turn = P["Tc"], P["w0"], P["fis"], P["V_turn"]
    # 2026-08-11 fix: build the field at FULL intensity; the f_sc
    # scaling is applied once in the RHS (level5 rk4_step_numba).
    # Previously this read `Q_TOTAL * f_sc / len(zd0)`, which with
    # the RHS factor gave an effective f_sc**2 (invisible at f=1,
    # caused the 12-18% undershoot at f=0.5/0.7).
    q_macro = Q_TOTAL / len(zd0)

    hist_turn, hist_sig_s, hist_sig_E = [], [], []
    snapshots, es_snaps = {}, {}
    Es_part = np.zeros(len(zd), dtype=np.float64)

    def build_es(s_part):
        lo, hi = float(s_part.min()), float(s_part.max())
        s_grid = np.linspace(lo, hi, NS_3D)
        ds = s_grid[1] - s_grid[0]
        idx = np.clip(((s_part - lo) / ds).astype(int), 0, NS_3D - 1)
        counts = np.bincount(idx, minlength=NS_3D).astype(np.float64)
        Ef = disk_sum_1d(s_grid, counts * q_macro)
        return Ef[idx], s_grid, Ef

    for turn in range(n_turns + 1):
        if turn % 10 == 0:
            print(f"  turn {turn:>4d}/{n_turns} ...", flush=True)
        s_part = -BETA_C * zd
        hist_turn.append(turn)
        hist_sig_s.append(np.std(s_part))
        hist_sig_E.append(np.std(dEk))
        if turn in snap_turns:
            snapshots[turn] = (s_part.copy(), dEk.copy())
            _, sg, Ef = build_es(s_part)
            # store the EFFECTIVE field (as applied to particles)
            es_snaps[turn] = (sg.copy(), Ef.copy() * f_sc)
        if turn == n_turns:
            break
        for step_i in range(n_steps):
            if step_i % es_update == 0:
                Es_part, _, _ = build_es(-BETA_C * zd)
            zd, dEk = rk4_step_numba(zd, dEk, dt, Tc, w0, fis, V_turn,
                                     E_min, dE_table, T_table,
                                     f_sc, BETA_C, Es_part)

    return {"turns": np.array(hist_turn), "sig_s": np.array(hist_sig_s),
            "sig_E": np.array(hist_sig_E), "snapshots": snapshots,
            "es_snaps": es_snaps}


def run_all():
    res = {}
    for f, run_dir in sorted(RUNS_3D.items()):
        cache = OUT_DIR / f"level47_1d_cache_recipe3d_{fname_f(f)}.npz"
        if cache.exists():
            d = np.load(cache)
            snap_t = list(d["snap_turns"])
            res[f] = {"turns": d["turns"], "sig_s": d["sig_s"],
                      "sig_E": d["sig_E"],
                      "snapshots": {int(t): (s, e) for t, s, e in
                                    zip(snap_t, d["s_snap"],
                                        d["dek_snap"])},
                      "es_snaps": {int(t): (d["es_s_list"][i],
                                            d["es_v_list"][i])
                                   for i, t in enumerate(snap_t)}}
            print(f"[f={f}] loaded cache")
            continue
        s1, dek1, _ = load_turn_particles(run_dir, 1)
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(s1), min(N_PART, len(s1)), replace=False)
        zd0 = (-s1[idx] / BETA_C).astype(np.float64)
        dEk0 = dek1[idx].astype(np.float64)

        seo = (REPO_ROOT / "Bmap_FD16" / "resultsSEO_noError"
               / "SEO_ini.txt")
        Ek_grid, Freq_grid = load_seo_freq_table(str(seo))
        T_grid = 1.0 / Freq_grid
        Tc = float(np.interp(E0_MEV, Ek_grid, T_grid))
        E_tab = np.linspace(E0_MEV - 30.0, E0_MEV + 30.0, 20001)
        T_tab = np.interp(E_tab, Ek_grid, T_grid).astype(np.float64)
        t_lookup = (E0_MEV - 30.0, E_tab[1] - E_tab[0], T_tab)
        P = {"Tc": Tc, "w0": 2.0 * np.pi / Tc, "fis": 0.0,
             "V_turn": V_TURN, "f_sc": f,
             "Ek_grid": Ek_grid.astype(np.float64),
             "T_grid": T_grid.astype(np.float64)}

        print(f"[f={f}] tracking {N_TURNS} turns with the "
              f"3D-recipe field ...")
        t0 = time.perf_counter()
        r = track_recipe3d(zd0, dEk0, P, t_lookup, N_TURNS,
                           N_STEPS_PER_TURN,
                           set(range(0, N_TURNS + 1, SNAP_EVERY)),
                           ES_UPDATE_INTERVAL)
        print(f"  done in {time.perf_counter()-t0:.0f} s")
        snap_t = sorted(r["snapshots"].keys())
        np.savez_compressed(
            cache, turns=r["turns"], sig_s=r["sig_s"],
            sig_E=r["sig_E"],
            snap_turns=np.array(snap_t, dtype=np.int32),
            s_snap=np.stack([r["snapshots"][t][0] for t in snap_t]
                            ).astype(np.float32),
            dek_snap=np.stack([r["snapshots"][t][1] for t in snap_t]
                              ).astype(np.float32),
            es_s_list=np.stack([r["es_snaps"][t][0] for t in snap_t]
                               ).astype(np.float32),
            es_v_list=np.stack([r["es_snaps"][t][1] for t in snap_t]
                               ).astype(np.float32),
        )
        print(f"  -> {cache.name}")
        res[f] = r
    return res


def fig_sigma_t(res):
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.5), sharex=True)
    for i, f in enumerate(sorted(res)):
        d3 = np.load(OUT_DIR / f"level33_3d_cache_{fname_f(f)[1:]}.npz")
        d1 = np.load(OUT_DIR / f"level35_1d_cache_{fname_f(f)}.npz")
        for j, (k3, k1, ylab) in enumerate([
                ("sigma_s", "sig_s", r"$\sigma_s$ [m]"),
                ("sigma_E", "sig_E", r"$\sigma_E$ [MeV]")]):
            ax = axes[i, j]
            ax.plot(d3["turn"], d3[k3], "o-", ms=3.5, lw=1,
                    color="C1", label="3D")
            ax.plot(d1["turns"] + 1, d1[k1], "-", lw=1.2, color="C2",
                    alpha=0.65, label="1D baseline (smooth field)")
            ax.plot(res[f]["turns"] + 1, res[f][k1], "-", lw=1.6,
                    color="C3", label="1D, 3D-slice field")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=9)
            if i == 2:
                ax.set_xlabel("turn (3D turn = 1D turn + 1)")
        axes[i, 0].set_title(f"f_sc = {f}", fontsize=10, loc="left")
    fig.suptitle("1D with the 3D-solver slice field vs 3D tracking "
                 "(V=200 kV, extended)", y=0.995)
    fig.tight_layout()
    if SAVE_FIGS:
        fn = OUT_DIR / "level47_sigma_t_all.png"
        fig.savefig(fn, dpi=200)
        print(f"[saved] {fn}")
    plt.close(fig)


def fig_phasespace(res):
    ncol = len(FIG_TURNS_3D)
    fig, axes = plt.subplots(2 * len(res), ncol,
                             figsize=(4.2 * ncol, 2.1 * 2 * len(res)),
                             squeeze=False)
    for i, f in enumerate(sorted(res)):
        d3 = np.load(OUT_DIR / f"level33_3d_cache_{fname_f(f)[1:]}.npz")
        snap_t = sorted(res[f]["snapshots"].keys())
        for j, t3 in enumerate(FIG_TURNS_3D):
            k3 = int(np.argmin(np.abs(d3["turn"] - t3)))
            t3r = int(d3["turn"][k3])
            t1 = t3r - 1
            if t1 not in snap_t:
                t1 = min(snap_t, key=lambda x: abs(x - t1))
            rows = [("3D", d3["s_sub"][k3], d3["dek_sub"][k3], t3r),
                    ("1D-3Dfield", *res[f]["snapshots"][t1], t1)]
            for ii, (lab, sd, ed, tt) in enumerate(rows):
                ax = axes[2 * i + ii][j]
                h = ax.hist2d(sd, ed, bins=200,
                              range=[[-S_LIM, S_LIM], [-E_LIM, E_LIM]],
                              norm=LogNorm(vmin=1), cmap="viridis")
                fig.colorbar(h[3], ax=ax, label="counts")
                ax.set_title(rf"{lab} turn {tt} | $\sigma_s$="
                             rf"{np.std(sd):.2f}, $\sigma_E$="
                             rf"{np.std(ed):.2f}", fontsize=8)
                ax.set_xlabel(r"$s$ [m]", fontsize=8)
                ax.set_ylabel(r"$\Delta E_k$ [MeV]", fontsize=8)
                ax.tick_params(labelsize=7)
        axes[2 * i][0].annotate(f"f_sc = {f}", xy=(-0.55, 1.05),
                                xycoords="axes fraction", fontsize=11,
                                fontweight="bold")
    fig.suptitle("phase space: 3D vs 1D with 3D-slice field",
                 y=0.998)
    fig.tight_layout(rect=(0.02, 0, 1, 0.995))
    if SAVE_FIGS:
        fn = OUT_DIR / "level47_phasespace_all.png"
        fig.savefig(fn, dpi=180)
        print(f"[saved] {fn}")
    plt.close(fig)


def fig_es(res):
    ncol = len(FIG_TURNS_3D)
    fig, axes = plt.subplots(len(res), ncol,
                             figsize=(4.2 * ncol, 3.0 * len(res)),
                             sharex=True, squeeze=False)
    for i, f in enumerate(sorted(res)):
        es3 = np.load(OUT_DIR
                      / f"level35_3d_es_cache_{fname_f(f)}.npz")
        snap_t = sorted(res[f]["es_snaps"].keys())
        for j, t3 in enumerate(FIG_TURNS_3D):
            k3 = int(np.argmin(np.abs(es3["turn"] - t3)))
            t3r = int(es3["turn"][k3])
            t1 = t3r - 1
            if t1 not in snap_t:
                t1 = min(snap_t, key=lambda x: abs(x - t1))
            sg, Ef = res[f]["es_snaps"][t1]
            ax = axes[i][j]
            ax.plot(es3["s_centers"], es3["es_binned"][k3] / 1e3,
                    "-", color="C1", lw=1.5, label=f"3D turn {t3r}")
            ax.plot(sg, Ef / 1e3, "-", color="C3", lw=1.5,
                    label=f"1D-3Dfield turn {t1}")
            ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
            ax.set_title(f"f={f}, turn {t3r}/{t1}", fontsize=9)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(r"$E_s$ [kV/m]")
            if i == len(res) - 1:
                ax.set_xlabel(r"$s$ [m]")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("longitudinal field: 3D binned Efi vs 1D "
                 "3D-slice field", y=0.995)
    fig.tight_layout()
    if SAVE_FIGS:
        fn = OUT_DIR / "level47_Es_all.png"
        fig.savefig(fn, dpi=200)
        print(f"[saved] {fn}")
    plt.close(fig)


def summary(res):
    print("\n=== sigma_s/sigma_E: 3D / 1D baseline / 1D 3D-slice field ===")
    for f in sorted(res):
        d3 = np.load(OUT_DIR / f"level33_3d_cache_{fname_f(f)[1:]}.npz")
        d1 = np.load(OUT_DIR / f"level35_1d_cache_{fname_f(f)}.npz")
        print(f"--- f_sc={f} ---")
        for t3 in FIG_TURNS_3D:
            k3 = int(np.argmin(np.abs(d3["turn"] - t3)))
            t3r = int(d3["turn"][k3])
            k1 = int(np.argmin(np.abs(d1["turns"] + 1 - t3r)))
            kr = int(np.argmin(np.abs(res[f]["turns"] + 1 - t3r)))
            print(f"t={t3r:3d}: {d3['sigma_s'][k3]:6.2f}"
                  f"/{d3['sigma_E'][k3]:5.2f}"
                  f" | {d1['sig_s'][k1]:6.2f}/{d1['sig_E'][k1]:5.2f}"
                  f" | {res[f]['sig_s'][kr]:6.2f}"
                  f"/{res[f]['sig_E'][kr]:5.2f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = run_all()
    fig_sigma_t(res)
    fig_phasespace(res)
    fig_es(res)
    summary(res)
    if SHOW_FIGS:
        plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
