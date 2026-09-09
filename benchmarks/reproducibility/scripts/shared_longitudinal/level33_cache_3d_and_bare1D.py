"""
level33_cache_3d_and_bare1D.py
==============================
2026-08-10: new HPC sync extended the 200 kV runs beyond turn 151
(0p0 -> 291, 0p5/0p7 -> 221, 1p0 -> 231).  This script

  Stage 1: caches 3D per-turn observables + subsampled particles for
           ALL four intensities into outputs/level33_3d_cache_*.npz
           (avoids re-reading ~5000 per-rank npz dumps later)
  Stage 2: runs the missing 1D BARE (f_sc=0.0) cache, 300 turns,
           init from Longitudinal_0p0 turn-1 particles
           -> outputs/level33_1d_cache_bare.npz
  Stage 3: 1D vs 3D bare comparison figure (f_sc=0p0) over the full
           291-turn range: sigma_s/sigma_E (+ 5%-trimmed core sigma),
           envelope f_s via FFT, phase-space snapshots.

Stages are idempotent: existing caches are not recomputed.
Style: no CLI args; edit CONFIG below.
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
    track_bunch, load_seo_freq_table, disk_kernel_numba,
    E0_MEV, BETA_C,
)

# ================================================================
# CONFIG — edit here
# ================================================================

DO_3D_CACHE = True
DO_1D_BARE = True
DO_FIG = True

RUNS = {                       # run name -> (dir, f_sc)
    "0p0": (REPO_ROOT / "HPC_dir" / "Longitudinal_0p0", 0.0),
    "0p5": (REPO_ROOT / "HPC_dir" / "Longitudinal_0p5", 0.5),
    "0p7": (REPO_ROOT / "HPC_dir" / "Longitudinal_0p7", 0.7),
    "1p0": (REPO_ROOT / "HPC_dir" / "Longitudinal_1p0", 1.0),
}

C_PER = 71.53                  # m, SEO perimeter @300 MeV (level21/22 value)
HALO_S_CUT = 22.0              # m
CORE_FRAC = 0.90               # central fraction for trimmed core sigma
N_SUB = 40000                  # subsampled particles stored per turn
SEED = 42

# 1D bare run
N_TURNS_1D = 300               # covers 3D turn 291 (= 1D turn 290)
N_PART_1D = 20000
N_STEPS_PER_TURN = 1000
V_TURN = 200e3
SC_NS = 1024                   # grid unused in bare mode, kept for API
SNAP_EVERY = 10                # 1D snapshot cadence (turns)

# comparison figure
TURNS_3D_FIG = (51, 151, 291)  # 3D turns for phase space
SNAP_1D_FIG = (50, 150, 290)   # matching 1D turns (3D turn = 1D turn + 1)

OUT_DIR = LEVEL0_DIR / "outputs"
SHOW_FIGS = False
SAVE_FIGS = True

# ================================================================


def list_turns(run_dir):
    turns = []
    for d in run_dir.glob("TBT_Dump_*"):
        try:
            turns.append(int(d.name.split("_")[-1]))
        except ValueError:
            pass
    return sorted(turns)


def load_turn_particles(run_dir, turn):
    """Return (s, dEk) full arrays from per-rank npz dumps."""
    d = run_dir / f"TBT_Dump_{turn}"
    files = sorted(d.glob("crossing_angle_*_rank_*.npz"))
    if not files:
        return None
    data = np.vstack([np.load(f)["particles"] for f in files])
    ph = data[:, 9]
    ph_mean = np.arctan2(np.mean(np.sin(ph)), np.mean(np.cos(ph)))
    dph = np.arctan2(np.sin(ph - ph_mean), np.cos(ph - ph_mean))
    s = -dph / (2.0 * np.pi) * C_PER
    return s, data[:, 5] - E0_MEV, float(np.degrees(ph_mean))


def core_sigmas(s, dEk, frac=CORE_FRAC):
    """std of s / dEk within the central `frac` of the s distribution."""
    lo, hi = np.percentile(s, [50 * (1 - frac), 50 * (1 + frac)])
    m = (s >= lo) & (s <= hi)
    return float(np.std(s[m])), float(np.std(dEk[m]))


# ----------------------------------------------------------------
# Stage 1: 3D cache
# ----------------------------------------------------------------

def cache_3d():
    for name, (run_dir, f_sc) in RUNS.items():
        out = OUT_DIR / f"level33_3d_cache_{name}.npz"
        if out.exists():
            print(f"[3D cache] {name}: exists, skipped ({out.name})")
            continue
        print(f"[3D cache] {name}: {run_dir.name} ...")
        t0 = time.perf_counter()
        rows, s_sub, dek_sub = [], [], []
        rng = np.random.default_rng(SEED)
        for t in list_turns(run_dir):
            res = load_turn_particles(run_dir, t)
            if res is None:
                continue
            s, dEk, ph_mean_deg = res
            cs, ce = core_sigmas(s, dEk)
            rows.append((t, np.std(s), np.std(dEk), cs, ce,
                         np.mean(np.abs(s) > HALO_S_CUT),
                         np.max(np.abs(s)), ph_mean_deg, np.mean(dEk),
                         len(s)))
            idx = rng.choice(len(s), min(N_SUB, len(s)), replace=False)
            s_sub.append(s[idx].astype(np.float32))
            dek_sub.append(dEk[idx].astype(np.float32))
        rows = np.array(rows)
        np.savez_compressed(
            out,
            turn=rows[:, 0].astype(np.int32),
            sigma_s=rows[:, 1], sigma_E=rows[:, 2],
            sigma_s_core=rows[:, 3], sigma_E_core=rows[:, 4],
            halo_frac=rows[:, 5], s_max=rows[:, 6],
            phase_mean_deg=rows[:, 7], dEk_mean=rows[:, 8],
            N=rows[:, 9].astype(np.int32),
            f_sc=f_sc, s_sub=np.vstack(s_sub), dek_sub=np.vstack(dek_sub),
        )
        print(f"  {len(rows)} turns ({int(rows[0,0])}..{int(rows[-1,0])}), "
              f"{time.perf_counter()-t0:.0f} s -> {out.name}")


# ----------------------------------------------------------------
# Stage 2: 1D bare cache
# ----------------------------------------------------------------

def run_1d_bare():
    out = OUT_DIR / "level33_1d_cache_bare.npz"
    if out.exists():
        print(f"[1D bare] cache exists, skipped ({out.name})")
        return
    run_dir = RUNS["0p0"][0]
    s1, dek1, _ = load_turn_particles(run_dir, 1)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(s1), min(N_PART_1D, len(s1)), replace=False)
    zd0 = (-s1[idx] / BETA_C).astype(np.float64)
    dEk0 = dek1[idx].astype(np.float64)
    print(f"[1D bare] init from 0p0 turn 1: N={len(zd0)}, "
          f"sigma_s0={np.std(s1[idx]):.3f} m, "
          f"sigma_E0={np.std(dEk0):.3f} MeV")

    seo = REPO_ROOT / "Bmap_FD16" / "resultsSEO_noError" / "SEO_ini.txt"
    Ek_grid, Freq_grid = load_seo_freq_table(str(seo))
    T_grid = 1.0 / Freq_grid
    Tc = float(np.interp(E0_MEV, Ek_grid, T_grid))
    w0 = 2.0 * np.pi / Tc
    E_min_tab = E0_MEV - 30.0
    E_tab = np.linspace(E_min_tab, E0_MEV + 30.0, 20001)
    T_tab = np.interp(E_tab, Ek_grid, T_grid).astype(np.float64)
    t_lookup = (E_min_tab, E_tab[1] - E_tab[0], T_tab)

    s_edges = np.linspace(-14.0, 14.0, SC_NS + 1)
    s_c = 0.5 * (s_edges[:-1] + s_edges[1:])
    precomp_kernel = (s_c[1] - s_c[0]) * disk_kernel_numba(s_c)
    counts_buf = np.zeros(SC_NS, dtype=np.float64)

    P = {"Tc": Tc, "w0": w0, "fis": 0.0, "V_turn": V_TURN,
         "f_sc": 0.0, "Ek_grid": Ek_grid.astype(np.float64),
         "T_grid": T_grid.astype(np.float64)}

    snap_turns = set(range(0, N_TURNS_1D + 1, SNAP_EVERY))
    print(f"[1D bare] tracking {N_TURNS_1D} turns ...")
    t0 = time.perf_counter()
    res = track_bunch(zd0, dEk0, "bare", N_TURNS_1D, N_STEPS_PER_TURN,
                      P, s_edges, snap_turns, 0.08,
                      precomp_kernel, counts_buf, 10, t_lookup)
    print(f"[1D bare] done in {time.perf_counter()-t0:.0f} s")

    snap_t = sorted(res["snapshots"].keys())
    s_snap = np.stack([res["snapshots"][t][2] for t in snap_t]
                      ).astype(np.float32)
    dek_snap = np.stack([res["snapshots"][t][1] for t in snap_t]
                        ).astype(np.float32)
    np.savez_compressed(
        out, turns=res["turns"], sig_s=res["sig_s"], sig_E=res["sig_E"],
        snap_turns=np.array(snap_t, dtype=np.int32),
        s_snap=s_snap, dek_snap=dek_snap, V_turn=V_TURN, f_sc=0.0,
    )
    print(f"[1D bare] -> {out.name}")


# ----------------------------------------------------------------
# Stage 3: 1D vs 3D bare comparison (f_sc = 0.0)
# ----------------------------------------------------------------

def fit_period(turns, values):
    """FFT peak + parabolic refine -> dominant period [turns]."""
    t = np.asarray(turns, float)
    v = np.asarray(values, float) - np.mean(values)
    dt = np.median(np.diff(t))
    V = np.fft.rfft(v)
    freqs = np.fft.rfftfreq(len(v), d=dt)
    V[0] = 0.0
    k = int(np.argmax(np.abs(V)))
    dk = 0.0
    if 0 < k < len(V) - 1:
        y0, y1, y2 = abs(V[k - 1]), abs(V[k]), abs(V[k + 1])
        dk = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    f = (k + dk) * (freqs[1] - freqs[0])
    return (1.0 / f) if f > 0 else np.nan


def make_figure():
    d3 = np.load(OUT_DIR / "level33_3d_cache_0p0.npz")
    d1 = np.load(OUT_DIR / "level33_1d_cache_bare.npz")

    t3, t1 = d3["turn"], d1["turns"] + 1     # 3D turn = 1D turn + 1
    f_rev = 2.7349e6                          # Hz

    # envelope (2 f_s) periods
    T3 = fit_period(t3, d3["sigma_s"])
    T1 = fit_period(t1, d1["sig_s"])
    fs3, fs1 = f_rev / (2 * T3), f_rev / (2 * T1)
    print(f"\n[bare f_s] 3D: T_env={T3:.1f} turns -> f_s={fs3/1e3:.2f} kHz")
    print(f"[bare f_s] 1D: T_env={T1:.1f} turns -> f_s={fs1/1e3:.2f} kHz "
          f"(diff {100*(fs1-fs3)/fs3:+.1f}%)")

    # ---- figure 1: sigma(t) ----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].plot(t3, d3["sigma_s"], "o-", ms=3, lw=1, color="C0",
                 label="3D bare")
    axes[0].plot(t1, d1["sig_s"], "-", lw=1.4, color="C0", alpha=0.55,
                 label="1D bare")
    axes[0].set_ylabel(r"$\sigma_s$ [m]")
    axes[1].plot(t3, d3["sigma_E"], "o-", ms=3, lw=1, color="C0",
                 label="3D bare")
    axes[1].plot(t1, d1["sig_E"], "-", lw=1.4, color="C0", alpha=0.55,
                 label="1D bare")
    axes[1].set_ylabel(r"$\sigma_E$ [MeV]")
    axes[2].plot(t3, d3["sigma_s_core"], "s-", ms=3, lw=1, color="C2",
                 label="3D core (90%)")
    axes[2].plot(t3, d3["sigma_s"], "o-", ms=3, lw=1, color="C0",
                 label="3D rms")
    axes[2].plot(t1, d1["sig_s"], "-", lw=1.4, color="C0", alpha=0.55,
                 label="1D rms")
    axes[2].set_ylabel(r"$\sigma_s$ [m]")
    axes[2].set_title("rms vs trimmed core", fontsize=10)
    for a in axes:
        a.set_xlabel("turn (3D turn = 1D turn + 1)")
        a.legend(fontsize=9)
        a.grid(alpha=0.3)
    fig.suptitle(f"1D vs 3D BARE (f_sc=0, V=200 kV): "
                 f"f_s 3D={fs3/1e3:.2f} kHz, 1D={fs1/1e3:.2f} kHz", y=1.00)
    fig.tight_layout()
    if SAVE_FIGS:
        f1 = OUT_DIR / "level33_sigma_t_bare.png"
        fig.savefig(f1, dpi=200)
        print(f"[saved] {f1}")

    # ---- figure 2: phase space ----
    snap_t = list(d1["snap_turns"])
    ncol = len(TURNS_3D_FIG)
    fig, axes = plt.subplots(2, ncol, figsize=(4.6 * ncol, 8.6),
                             squeeze=False)
    s_lim, e_lim = 18.0, 9.0
    for j, (tt3, tt1) in enumerate(zip(TURNS_3D_FIG, SNAP_1D_FIG)):
        k3 = int(np.argmin(np.abs(t3 - tt3)))
        k1 = snap_t.index(tt1)
        s3, e3 = d3["s_sub"][k3], d3["dek_sub"][k3]
        s1d, e1d = d1["s_snap"][k1], d1["dek_snap"][k1]
        for i, (sd, ed, lab, tt) in enumerate([
                (s3, e3, "3D", tt3), (s1d, e1d, "1D", tt1)]):
            ax = axes[i][j]
            h = ax.hist2d(sd, ed, bins=220,
                          range=[[-s_lim, s_lim], [-e_lim, e_lim]],
                          norm=LogNorm(vmin=1), cmap="viridis")
            fig.colorbar(h[3], ax=ax, label="counts/bin")
            ax.set_title(rf"{lab} turn {tt} | $\sigma_s$={np.std(sd):.2f} m"
                         rf", $\sigma_E$={np.std(ed):.2f} MeV", fontsize=10)
            ax.set_xlabel(r"$s$ [m]")
            if j == 0:
                ax.set_ylabel(r"$\Delta E_k$ [MeV]")
    fig.suptitle("phase space, BARE (f_sc=0, V=200 kV): 3D (top) vs 1D",
                 y=1.00)
    fig.tight_layout()
    if SAVE_FIGS:
        f2 = OUT_DIR / "level33_phasespace_bare.png"
        fig.savefig(f2, dpi=200)
        print(f"[saved] {f2}")

    # ---- summary print ----
    print("\n" + "=" * 64)
    print(f"{'3D turn':>8} {'sig_s 3D':>9} {'sig_s 1D':>9} "
          f"{'core 3D':>9} {'sig_E 3D':>9} {'sig_E 1D':>9}")
    print("-" * 64)
    for tt3 in range(1, int(t3[-1]) + 1, 50):
        k3 = int(np.argmin(np.abs(t3 - tt3)))
        k1 = int(np.argmin(np.abs(t1 - tt3)))
        print(f"{int(t3[k3]):>8} {d3['sigma_s'][k3]:>9.3f} "
              f"{d1['sig_s'][k1]:>9.3f} {d3['sigma_s_core'][k3]:>9.3f} "
              f"{d3['sigma_E'][k3]:>9.3f} {d1['sig_E'][k1]:>9.3f}")
    print("=" * 64)

    if SHOW_FIGS:
        plt.show()
    plt.close("all")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DO_3D_CACHE:
        cache_3d()
    if DO_1D_BARE:
        run_1d_bare()
    if DO_FIG:
        make_figure()
    print("Done.")


if __name__ == "__main__":
    main()
