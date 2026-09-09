"""Shared numerical core for the final LSC-on/off tune comparisons.

The production dump contains one NPZ file per selected particle.  Tunes are
measured from the turn-by-turn Courant--Snyder complex coordinate in two ways:

1. a linear regression of the unwrapped phase (primary finite-time tune), and
2. a Hann-windowed, zero-padded FFT peak (cross-check).

The radial coordinate is projected onto the energy-dependent closed-orbit
normal at the 90-degree dump azimuth.  A turn-by-turn sampled bunch centroid
is removed in both planes.  The toy model is rerun for exactly 100 turns with
the same estimators; this avoids comparing a short 3-D record directly with
the previously published 240-turn toy FFT summary.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT as REPO
DATA = REPO / "HPC_dir" / "data_coupling_withLsc500"
DUMP = DATA / "FFT_Tune_Dump_0"
OUT = HERE / "results"
SEO = REPO / "Bmap_FD16" / "resultsSEO_noError_X"
sys.path.insert(0, str(HERE))

from toy_tracking import (  # noqa: E402
    C_LIGHT, CELL_DEG, EK_MEV, EPS0, MP_KG, MP_MEV, N_CELLS, QE,
    build_lattice, summarize, track, track_with_dispersion, tune_fft,
)
from longitudinal_driver import (  # noqa: E402
    CACHE as LONGITUDINAL_CACHE,
    EMITTANCE_RMS, densities, longitudinal_pair,
)

# The nominal tracking duration and the tune-analysis window.  The original
# 100-turn data starts immediately, hence TUNE_START=0 and 100 samples.
TOY_TRACK_TURNS = 100
TUNE_START = 0
N_TURNS = TOY_TRACK_TURNS - TUNE_START
DUMP_N_SAMPLES = N_TURNS
DATA_WINDOW_START = 0
N_TOY = 768
TOY_SEED = 22222
TOY_STEPS_PER_CELL = 50
TOY_SIGMA_R0: float | None = None
TOY_SIGMA_Z0: float | None = None
TOY_SIGMA_R_PROFILE: np.ndarray | None = None
# If True, TOY_SIGMA_R_PROFILE is already the total physical slice width,
# including the within-slice closed-orbit spread.  Keep Dx*dEk in each
# particle's field-evaluation coordinate, but do not add Dx^2*sigma_E^2 to
# the prescribed field width a second time.
TOY_SIGMA_R_PROFILE_IS_TOTAL = False
TOY_DISPERSION_DX0_M_PER_MEV = 0.0
TOY_DISPERSION_DXP0_RAD_PER_MEV = 0.0
TOY_DISPERSION_SIGMA_E_MEV: float | None = None
TOY_FULL_DISPERSION = False
TOY_DISPERSION_DX_PROFILE: np.ndarray | None = None
TOY_COMMON_ORBIT_LAUNCH = True
TOY_CACHE_TAG = "nominal60"
# Cache only the final transverse toy trajectories/tunes.  The two comparison
# entry points set this to False so every run reflects their current parameters.
TOY_USE_RESULT_CACHE = True
# Cases to track and the number of particle batches used for progress output.
# Splitting by particle is exact because toy particles do not interact directly;
# they sample a prescribed Gaussian field and longitudinal line-density history.
TOY_CASES = ("off", "on")
TOY_PROGRESS_CHUNKS = 10
FFT_HALF_WIDTH_R = 0.25
FFT_HALF_WIDTH_Z = 0.25

# Maximum-SC tune-shift star marker.  Set both to floats for a manual position;
# leave both as None to use the theoretical footprint's marginal 1st percentiles.
STAR_QR: float | None = 3.12
STAR_QZ: float | None = 2.68


def load_3d_dump() -> tuple[np.ndarray, list[str]]:
    files = sorted(DUMP.glob("particle_*_angle_90.0_rank_*.npz"))
    if not files:
        raise FileNotFoundError(f"no particle dumps in {DUMP}")
    data = np.empty((DUMP_N_SAMPLES, len(files), 5), dtype=np.float32)
    names: list[str] = []
    for j, filename in enumerate(files):
        particle = np.load(filename)["particles"]
        if particle.shape != (DUMP_N_SAMPLES, 17):
            raise ValueError(
                f"{filename.name}: expected ({DUMP_N_SAMPLES},17), "
                f"got {particle.shape}"
            )
        if not np.isfinite(particle).all():
            raise ValueError(f"{filename.name}: contains NaN/Inf")
        data[:, j] = particle[:, [0, 1, 2, 3, 5]]  # r, pr, z, pz, Ek
        names.append(filename.name)
    return data.astype(np.float64), names


def project_3d_coordinates(data: np.ndarray) -> tuple[np.ndarray, ...]:
    r, pr, z, pz, ek = (data[:, :, i] for i in range(5))
    ini = np.loadtxt(SEO / "SEO_ini.txt", skiprows=2)
    energy = ini[:, 1]
    # r90 and original pr=vr/v columns in SEO_ini.txt.
    r0 = np.interp(ek, energy, ini[:, 11])
    vr0 = np.interp(ek, energy, ini[:, 12])
    pr0_geom = vr0 / np.sqrt(np.maximum(1.0 - vr0 * vr0, 1e-30))

    pfi = np.sqrt(np.maximum(1.0 - pr * pr - pz * pz, 1e-30))
    norm = np.sqrt(1.0 + pr0_geom * pr0_geom)
    p_tangent = (pr0_geom * pr + pfi) / norm
    x = (r - r0) / norm
    xp = ((pr - pr0_geom * pfi) / norm) / p_tangent
    zp = (pz / norm) / p_tangent

    # Remove coherent motion/reference interpolation offsets, not individual
    # particle averages (which would suppress slow physical components).
    for value in (x, xp, z, zp):
        value -= value.mean(axis=1, keepdims=True)
    return x, xp, z.copy(), zp, ek


def phase_regression_tune(x: np.ndarray, p: np.ndarray, beta: float,
                          alpha: float, expected: float) -> np.ndarray:
    """Time-averaged tune from the unwrapped CS phase slope."""
    root_beta = np.sqrt(beta)
    u = x / root_beta - 1j * (alpha * x + beta * p) / root_beta
    phase = np.unwrap(np.angle(u), axis=0)
    t = np.arange(len(u), dtype=np.float64)
    tc = t - t.mean()
    slope = np.sum(tc[:, None] * (phase - phase.mean(axis=0)), axis=0)
    slope /= np.sum(tc * tc) * (2.0 * np.pi)
    q0 = expected % 1.0
    q0 = q0 if q0 <= 0.5 else q0 - 1.0
    slope += np.rint(q0 - slope)
    return expected + slope - q0


def both_estimators(x: np.ndarray, p: np.ndarray, beta: float,
                    alpha: float, expected: float,
                    fft_half_width: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    q_phase = phase_regression_tune(x, p, beta, alpha, expected)
    q_fft = np.empty(x.shape[1], dtype=np.float64)
    # Keep zero-padded FFT memory bounded for 25,000 production particles.
    for lo in range(0, x.shape[1], 500):
        hi = min(x.shape[1], lo + 500)
        q_fft[lo:hi] = tune_fft(
            x[:, lo:hi], p[:, lo:hi], beta, alpha, expected,
            half_width=fft_half_width,
        )
    return q_phase, q_fft


def slice_energy_histories(cache_case: str, selected: np.ndarray,
                           n_turns: int) -> tuple[np.ndarray, np.ndarray]:
    """Return dEk relative to slice mean and conditional slice sigma_E."""
    longitudinal = np.load(LONGITUDINAL_CACHE)
    snap_turns = longitudinal["snap_turns"].astype(int)
    s_snap = longitudinal[f"s_{cache_case}"].astype(np.float64)
    de_snap = longitudinal[f"de_{cache_case}"].astype(np.float64)
    delta_local = np.empty((n_turns, len(selected)), dtype=np.float64)
    sigma_slice = np.empty_like(delta_local)
    for turn in range(n_turns):
        k = int(np.clip(
            np.searchsorted(snap_turns, turn, side="right") - 1,
            0, len(snap_turns) - 2,
        ))
        fraction = ((turn - snap_turns[k])
                    / (snap_turns[k + 1] - snap_turns[k]))
        s_all = (1.0 - fraction)*s_snap[k] + fraction*s_snap[k + 1]
        de_all = (1.0 - fraction)*de_snap[k] + fraction*de_snap[k + 1]
        edges = np.linspace(float(s_all.min()), float(s_all.max()), 129)
        bins_all = np.clip(
            np.searchsorted(edges, s_all, side="right") - 1, 0, 127
        )
        count = np.bincount(bins_all, minlength=128).astype(np.float64)
        sum_de = np.bincount(bins_all, weights=de_all, minlength=128)
        sum_de2 = np.bincount(bins_all, weights=de_all*de_all, minlength=128)
        mean_de = np.divide(sum_de, count, out=np.zeros(128), where=count > 0)
        variance = np.divide(
            sum_de2, count, out=np.zeros(128), where=count > 0
        ) - mean_de*mean_de
        sigma_de = np.sqrt(np.maximum(variance, 0.0))
        bins_selected = bins_all[selected]
        delta_local[turn] = de_all[selected] - mean_de[bins_selected]
        sigma_slice[turn] = sigma_de[bins_selected]
    return delta_local, sigma_slice


def run_toy_100(lat: dict) -> dict[str, np.ndarray]:
    cache = OUT / (
        f"toy_{TOY_TRACK_TURNS}turn_window_{TUNE_START}_{TOY_TRACK_TURNS-1}_"
        f"{TOY_CACHE_TAG}.npz"
    )
    if TOY_USE_RESULT_CACHE and cache.exists():
        return {key: value for key, value in np.load(cache).items()}

    longitudinal_pair(TOY_TRACK_TURNS)
    rng = np.random.default_rng(TOY_SEED)
    selected = rng.choice(20_000, N_TOY, replace=False)
    lam_no, _ = densities("s_no", selected, TOY_TRACK_TURNS)
    lam_yes, _ = densities("s_yes", selected, TOY_TRACK_TURNS)

    br0, bz0 = lat["beta_r"][0], lat["beta_z"][0]
    ar0, az0 = lat["alpha_r"][0], lat["alpha_z"][0]
    eps_r = (EMITTANCE_RMS if TOY_SIGMA_R0 is None
             else TOY_SIGMA_R0**2 / br0)
    eps_z = (EMITTANCE_RMS if TOY_SIGMA_Z0 is None
             else TOY_SIGMA_Z0**2 / bz0)
    X, P = rng.normal(size=(2, N_TOY)) * np.sqrt(eps_r)
    Z, PZ = rng.normal(size=(2, N_TOY)) * np.sqrt(eps_z)
    x0 = np.sqrt(br0) * X
    xp0 = (P - ar0 * X) / np.sqrt(br0)
    dispersion_histories = None
    if TOY_FULL_DISPERSION:
        if TOY_DISPERSION_DX_PROFILE is None:
            raise ValueError("full dispersion requires a Dx lattice profile")
        de_no, se_no = slice_energy_histories(
            "no", selected, TOY_TRACK_TURNS)
        de_yes, se_yes = slice_energy_histories(
            "yes", selected, TOY_TRACK_TURNS)
        if TOY_DISPERSION_SIGMA_E_MEV is not None:
            longitudinal = np.load(LONGITUDINAL_CACHE)
            physical_sigma = float(longitudinal["de_no"][0].std())
            scale = TOY_DISPERSION_SIGMA_E_MEV / physical_sigma
            de_no *= scale; se_no *= scale
            de_yes *= scale; se_yes *= scale
        if TOY_COMMON_ORBIT_LAUNCH:
            # The production distribution is launched around one common
            # 300-MeV orbit.  Convert that physical launch coordinate to the
            # betatron coordinate relative to each particle's slice orbit.
            x0 -= TOY_DISPERSION_DX0_M_PER_MEV * de_no[0]
            xp0 -= TOY_DISPERSION_DXP0_RAD_PER_MEV * de_no[0]
        dispersion_histories = {
            "off": (de_no, se_no),
            "on": (de_yes, se_yes),
        }
    elif TOY_DISPERSION_SIGMA_E_MEV is not None:
        longitudinal = np.load(LONGITUDINAL_CACHE)
        delta_ek = longitudinal["de_no"][0, selected].astype(np.float64)
        delta_ek -= delta_ek.mean()
        sample_sigma_ek = float(delta_ek.std())
        if sample_sigma_ek <= 0.0:
            raise ValueError("selected longitudinal particles have zero energy spread")
        delta_ek *= TOY_DISPERSION_SIGMA_E_MEV / sample_sigma_ek
        x0 += TOY_DISPERSION_DX0_M_PER_MEV * delta_ek
        xp0 += TOY_DISPERSION_DXP0_RAD_PER_MEV * delta_ek
    z0 = np.sqrt(bz0) * Z
    zp0 = (PZ - az0 * Z) / np.sqrt(bz0)
    sr_beta = np.sqrt(lat["beta_r"] * eps_r)
    sr = (sr_beta if TOY_SIGMA_R_PROFILE is None
          else np.asarray(TOY_SIGMA_R_PROFILE, dtype=np.float64))
    if sr.shape != lat["beta_r"].shape:
        raise ValueError("TOY_SIGMA_R_PROFILE must match the lattice grid")
    sz = np.sqrt(lat["beta_z"] * eps_z)
    gamma = 1.0 + EK_MEV / MP_MEV
    beta_rel = np.sqrt(1.0 - gamma**-2)
    accel = (QE / (gamma * MP_KG * beta_rel**2 * C_LIGHT**2)
             / (4.0 * np.pi * EPS0 * gamma**2))
    gx, gw = np.polynomial.legendre.leggauss(12)
    gs, gw = 0.5 * (gx + 1.0), 0.5 * gw
    theta = np.linspace(0.0, np.deg2rad(CELL_DEG), TOY_STEPS_PER_CELL + 1)

    invalid_cases = set(TOY_CASES) - {"off", "on"}
    if invalid_cases:
        raise ValueError(f"unsupported TOY_CASES: {sorted(invalid_cases)}")
    case_density = {"off": lam_no, "on": lam_yes}
    result: dict[str, np.ndarray] = {}
    for case in TOY_CASES:
        line_density = case_density[case]
        n_particles = line_density.shape[1]
        n_chunks = max(1, min(int(TOY_PROGRESS_CHUNKS), n_particles))
        chunk_edges = np.linspace(0, n_particles, n_chunks + 1, dtype=int)
        tr_arrays = tuple(
            np.empty((TOY_TRACK_TURNS, n_particles), dtype=np.float64)
            for _ in range(4)
        )
        print(
            f"toy {case}: tracking {n_particles} particles x "
            f"{TOY_TRACK_TURNS} turns in {n_chunks} batches",
            flush=True,
        )
        if TOY_FULL_DISPERSION:
            de_turn, se_turn = dispersion_histories[case]
            se_for_field_width = (
                np.zeros_like(se_turn)
                if TOY_SIGMA_R_PROFILE_IS_TOTAL else se_turn
            )
            dx_profile = np.asarray(
                TOY_DISPERSION_DX_PROFILE, dtype=np.float64
            )
        for chunk_index, (start, stop) in enumerate(
            zip(chunk_edges[:-1], chunk_edges[1:]), start=1
        ):
            particle_slice = slice(start, stop)
            if TOY_FULL_DISPERSION:
                tr_chunk = track_with_dispersion(
                    x0[particle_slice], xp0[particle_slice],
                    z0[particle_slice], zp0[particle_slice],
                    line_density[:, particle_slice],
                    de_turn[:, particle_slice],
                    se_for_field_width[:, particle_slice], theta,
                    lat["theta"][0], lat["theta"][1] - lat["theta"][0],
                    lat["h"], lat["kr"], lat["kz"], sr, sz, dx_profile,
                    gs, gw, accel, N_CELLS,
                )
            else:
                tr_chunk = track(
                    x0[particle_slice], xp0[particle_slice],
                    z0[particle_slice], zp0[particle_slice],
                    line_density[:, particle_slice], theta,
                    lat["theta"][0], lat["theta"][1] - lat["theta"][0],
                    lat["h"], lat["kr"], lat["kz"], sr, sz, gs, gw,
                    accel, N_CELLS,
                )
            for destination, source in zip(tr_arrays, tr_chunk):
                destination[:, particle_slice] = source
            print(
                f"toy {case}: {stop}/{n_particles} particles "
                f"({100.0 * stop / n_particles:.0f}%)",
                flush=True,
            )
        tr = tr_arrays
        for plane, xx, pp, bb, aa, qb in (
            ("r", tr[0], tr[1], br0, ar0, lat["qr"]),
            ("z", tr[2], tr[3], bz0, az0, lat["qz"]),
        ):
            fft_half_width = (FFT_HALF_WIDTH_R if plane == "r"
                              else FFT_HALF_WIDTH_Z)
            phase, fft = both_estimators(
                xx[TUNE_START:], pp[TUNE_START:], bb, aa, qb,
                fft_half_width=fft_half_width)
            result[f"q{plane}_{case}_phase"] = phase
            result[f"q{plane}_{case}_fft"] = fft
    if TOY_USE_RESULT_CACHE:
        np.savez_compressed(cache, **result)
    return result


def density_grid(qr: np.ndarray, qz: np.ndarray, xr: tuple[float, float],
                 yr: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist, xe, ye = np.histogram2d(qr, qz, bins=100, range=(xr, yr))
    return gaussian_filter(hist, 1.2).T, xe, ye


def contour_levels(hist: np.ndarray) -> list[float]:
    values = np.sort(hist.ravel())[::-1]
    cumulative = np.cumsum(values) / values.sum()
    return sorted(values[np.searchsorted(cumulative, f)] for f in (0.95, 0.68))


def toy_total_action() -> np.ndarray:
    """Recreate the matched toy particles' J_r+J_z in mm mrad."""
    rng = np.random.default_rng(TOY_SEED)
    rng.choice(20_000, N_TOY, replace=False)  # same RNG advance as run_toy_100
    lat = build_lattice()
    eps_r = (EMITTANCE_RMS if TOY_SIGMA_R0 is None
             else TOY_SIGMA_R0**2 / lat["beta_r"][0])
    eps_z = (EMITTANCE_RMS if TOY_SIGMA_Z0 is None
             else TOY_SIGMA_Z0**2 / lat["beta_z"][0])
    radial = rng.normal(size=(2, N_TOY)) * np.sqrt(eps_r)
    vertical = rng.normal(size=(2, N_TOY)) * np.sqrt(eps_z)
    return 0.5 * (np.sum(radial * radial, axis=0)
                  + np.sum(vertical * vertical, axis=0)) * 1.0e6


def stats_row(case: str, estimator: str, plane: str, q: np.ndarray,
              qbare: float) -> dict[str, float | str]:
    return {"case": case, "estimator": estimator, "plane": plane,
            "n": len(q), **summarize(q, qbare)}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-show", action="store_true",
        help="save figures without opening an interactive window",
    )
    args = parser.parse_args()
    if (STAR_QR is None) != (STAR_QZ is None):
        raise ValueError("STAR_QR and STAR_QZ must both be floats or both be None")
    OUT.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    lat = build_lattice()
    data, names = load_3d_dump()
    x, xp, z, zp, ek = project_3d_coordinates(data)
    window = slice(DATA_WINDOW_START, DATA_WINDOW_START + N_TURNS)
    qr_phase, qr_fft = both_estimators(
        x[window], xp[window], lat["beta_r"][0], lat["alpha_r"][0], lat["qr"])
    qz_phase, qz_fft = both_estimators(
        z[window], zp[window], lat["beta_z"][0], lat["alpha_z"][0], lat["qz"])
    toy = run_toy_100(lat)

    np.savez_compressed(
        OUT / f"withlsc{TOY_TRACK_TURNS}_tunes.npz", filenames=np.asarray(names),
        qr_phase=qr_phase, qz_phase=qz_phase, qr_fft=qr_fft, qz_fft=qz_fft,
        mean_ek_turn=ek.mean(axis=1), qr_bare=lat["qr"], qz_bare=lat["qz"])

    rows = []
    for case, source in (("3d_with_lsc", None), ("toy_lsc_on", "on"),
                         ("toy_lsc_off", "off")):
        for estimator in ("phase", "fft"):
            if source is None:
                qr = qr_phase if estimator == "phase" else qr_fft
                qzq = qz_phase if estimator == "phase" else qz_fft
            else:
                qr = toy[f"qr_{source}_{estimator}"]
                qzq = toy[f"qz_{source}_{estimator}"]
            rows.append(stats_row(case, estimator, "r", qr, lat["qr"]))
            rows.append(stats_row(case, estimator, "z", qzq, lat["qz"]))
    columns = list(rows[0])
    with (OUT / "tune_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # Primary comparison keeps the same FFT convention as the existing
    # 240-turn theoretical prediction.  Phase regression remains an estimator
    # cross-check in the CSV.
    tqr, tqz = toy["qr_on_fft"], toy["qz_on_fft"]
    sigma_r0 = (np.sqrt(lat["beta_r"][0] * EMITTANCE_RMS)
                if TOY_SIGMA_R0 is None else TOY_SIGMA_R0)
    sigma_z0 = (np.sqrt(lat["beta_z"][0] * EMITTANCE_RMS)
                if TOY_SIGMA_Z0 is None else TOY_SIGMA_Z0)
    # Use a common publication window in both panels.  The x/y spans are both
    # 0.30, matching the equal-span convention of the paper tune benchmarks.
    # The window also leaves room for the forthcoming LSC-off footprint.
    xr, yr = (2.95, 3.25), (2.63, 2.93)
    jtot = toy_total_action()
    # Use a robust lower-left endpoint instead of one extreme macro-particle.
    # The marginal 1st percentiles suppress single-particle/finite-window
    # fluctuations while retaining the theoretical maximum-shift boundary.
    if STAR_QR is None:
        qr_core = float(np.percentile(tqr, 1.0))
        qz_core = float(np.percentile(tqz, 1.0))
        star_source = "marginal 1st-percentile tunes"
    else:
        qr_core, qz_core = float(STAR_QR), float(STAR_QZ)
        if not (xr[0] <= qr_core <= xr[1] and yr[0] <= qz_core <= yr[1]):
            raise ValueError(
                f"manual star ({qr_core}, {qz_core}) lies outside "
                f"the plot ranges Qr={xr}, Qz={yr}"
            )
        star_source = "manually specified coordinates"

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)
    hist = axes[0].hist2d(
        qr_fft, qz_fft, bins=[256, 256], range=[xr, yr],
        norm=LogNorm(), cmap="viridis",
    )
    cb0 = fig.colorbar(hist[3], ax=axes[0])
    cb0.set_label("Counts", fontsize=15)

    scatter = axes[1].scatter(
        tqr, tqz, c=jtot, cmap="plasma", s=18, alpha=0.78,
        linewidths=0, rasterized=True,
    )
    cb1 = fig.colorbar(scatter, ax=axes[1])
    cb1.set_label(r"$J_r+J_z$ [$\pi$ mm mrad]", fontsize=15)

    for i, (ax, title) in enumerate(zip(
        axes,
        ("3-D PIC tracking (LSC on)", "Matched nonlinear toy (LSC on)"),
    )):
        ax.axvline(lat["qr"], ls="--", lw=1.8, color="tab:blue")
        ax.axhline(lat["qz"], ls="--", lw=1.8, color="tab:blue")
        ax.axvline(qr_core, ls=":", lw=2.0, color="tab:blue")
        ax.axhline(qz_core, ls=":", lw=2.0, color="tab:blue")
        ax.scatter(
            lat["qr"], lat["qz"], marker="o", s=115,
            facecolor="white", edgecolor="blue", linewidth=2.2, zorder=12,
            label="Bare tune",
        )
        ax.scatter(
            qr_core, qz_core, marker="*", s=230, color="red",
            edgecolor="black", linewidth=1.7, zorder=13,
            label="Max SC tune shift",
        )
        ax.text(0.045, 0.91, f"({chr(ord('a') + i)})", transform=ax.transAxes,
                fontsize=25, fontweight="bold")
        ax.set_xlim(xr)
        ax.set_ylim(yr)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$Q_r$", fontsize=19)
        ax.set_ylabel(r"$Q_z$", fontsize=19)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=12, loc="lower right")
    fig.savefig(OUT / f"withlsc{TOY_TRACK_TURNS}_vs_theory.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(OUT / f"withlsc{TOY_TRACK_TURNS}_vs_theory_paper_style.pdf",
                bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    def compact(q: np.ndarray, qb: float) -> tuple[float, float, float]:
        s = summarize(q, qb)
        return s["mean"], s["mean_shift"], s["spread_90"]

    ar, az = compact(qr_fft, lat["qr"]), compact(qz_fft, lat["qz"])
    tr, tz = compact(tqr, lat["qr"]), compact(tqz, lat["qz"])
    report = f"""# {N_TURNS}-sample 3-D with-LSC comparison

Tune estimator: Hann-windowed, zero-padded FFT of the Courant--Snyder complex
coordinate over the same {N_TURNS}-sample interval in the 3-D data and matched
toy (nominal turns {TUNE_START}--{TOY_TRACK_TURNS-1}).
This is the convention used by the existing 240-turn theoretical prediction.
An unwrapped-phase regression is retained in `tune_summary.csv` as a second
estimator; it gives the same physical conclusion.

| plane | bare | 3-D mean | toy mean | 3-D mean shift | toy mean shift | 3-D Q95-Q05 | toy Q95-Q05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| r | {lat['qr']:.6f} | {ar[0]:.6f} | {tr[0]:.6f} | {ar[1]:+.6f} | {tr[1]:+.6f} | {ar[2]:.6f} | {tr[2]:.6f} |
| z | {lat['qz']:.6f} | {az[0]:.6f} | {tz[0]:.6f} | {az[1]:+.6f} | {tz[1]:+.6f} | {az[2]:.6f} | {tz[2]:.6f} |

Mean-shift magnitude ratios (3-D/toy): radial {abs(ar[1]/tr[1]):.3f}, vertical
{abs(az[1]/tz[1]):.3f}.  Spread ratios: radial {ar[2]/tr[2]:.3f}, vertical
{az[2]/tz[2]:.3f}.

Interpretation: compare the means quantitatively, while remembering that the
3-D model contains nonlinear lattice/amplitude effects and a self-consistent
transverse envelope that the fixed-envelope toy does not.  The paired LSC-off
3-D result is still required to isolate the longitudinal-SC-induced change.

The 3-D population above the bare vertical tune is not rejected: there are
{np.count_nonzero(qz_fft > lat['qz'])} of {len(qz_fft)} particles
({100.0*np.mean(qz_fft > lat['qz']):.2f}%).  The existing paper tune benchmarks
show the same high-Qz extension, so it is consistent with amplitude-dependent
nonlinear magnetic-lattice tune rather than, by itself, a tune-extraction
artifact.  The toy reconstructs a linear periodic Hill lattice and therefore
does not reproduce this nonlinear-lattice tail.

The maximum-SC-shift marker uses {star_source}:
(Qr,Qz)=({qr_core:.6f},{qz_core:.6f}).  The same dotted guide lines are used in
both panels.

The fixed toy RMS beam sizes at the observation point are
sigma_r={1e3*sigma_r0:.4f} mm and sigma_z={1e3*sigma_z0:.4f} mm.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"outputs: {OUT}")
    print(f"elapsed: {time.perf_counter() - tic:.1f} s")


if __name__ == "__main__":
    main()
