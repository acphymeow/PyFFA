"""Compare turn-resolved LSC-off 3-D and dispersion-mismatch sigma_x.

The selected-particle FFT dump begins at nominal turn 10.  Its first 100
records therefore cover turns 10--109.  The analytical curve begins at turn 1
and describes a beam launched around one common-energy closed orbit instead
of around each particle's energy-dependent closed orbit.

All generated files stay below ``results/lsc_off/sigma_x_turns``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import comparison_core as analysis
from dispersion import centered_dispersion_curves


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT as REPO
FFT_DUMP = REPO / "HPC_dir" / "simulation1" / "FFT_Tune_Dump_0"
SEO = REPO / "Bmap_FD16" / "resultsSEO_noError_X"
LONG_CACHE = HERE / "outputs" / "gauss_waterbag_longitudinal_pair.npz"
OUT = HERE / "results" / "lsc_off" / "sigma_x_turns"

N_3D_TURNS = 100
FIRST_3D_TURN = 10
N_THEORY_TURNS = 100
N_SLICES = 128
EMITTANCE_RMS = 60.0e-6


def largest_gap_bins(periodic_coordinate: np.ndarray, n_slices: int) -> np.ndarray:
    phase = np.mod(np.asarray(periodic_coordinate, dtype=np.float64), 2.0 * np.pi)
    ordered = np.sort(phase)
    gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
    cut = np.mod(ordered[int(np.argmax(gaps))] + 0.5 * float(gaps.max()),
                 2.0 * np.pi)
    continuous = np.mod(phase - cut, 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, n_slices + 1)
    return np.clip(np.searchsorted(edges, continuous, side="right") - 1,
                   0, n_slices - 1)


def measure_3d_turns(cache_path: Path) -> dict[str, np.ndarray]:
    if cache_path.exists():
        return {key: value for key, value in np.load(cache_path).items()}

    files = sorted(FFT_DUMP.glob("particle_*_angle_90.0_rank_*.npz"))
    if not files:
        raise FileNotFoundError(f"no particle tune dumps in {FFT_DUMP}")

    # Only r, Ek and RF phase are needed.  Keeping three dense arrays is much
    # faster than reopening 25,000 small NPZ files once for every turn.
    radius = np.empty((N_3D_TURNS, len(files)), dtype=np.float64)
    energy = np.empty_like(radius)
    phase = np.empty_like(radius)
    for index, filename in enumerate(files):
        particles = np.load(filename)["particles"]
        if particles.shape[0] < N_3D_TURNS or particles.shape[1] != 17:
            raise ValueError(f"{filename.name}: incomplete shape {particles.shape}")
        radius[:, index] = particles[:N_3D_TURNS, 0]
        energy[:, index] = particles[:N_3D_TURNS, 5]
        phase[:, index] = particles[:N_3D_TURNS, 9]
        if (index + 1) % 5000 == 0:
            print(f"loaded {index + 1}/{len(files)} particle files")

    seo_ini = np.loadtxt(SEO / "SEO_ini.txt", skiprows=2)
    energy_axis = seo_ini[:, 1]
    sigma_x = np.empty(N_3D_TURNS)
    sigma_ek = np.empty(N_3D_TURNS)
    mean_ek = np.empty(N_3D_TURNS)
    slice_count = np.empty(N_3D_TURNS, dtype=np.int64)
    slice_number = np.empty(N_3D_TURNS, dtype=np.int64)

    for sample in range(N_3D_TURNS):
        bins = largest_gap_bins(phase[sample], N_SLICES)
        counts = np.bincount(bins, minlength=N_SLICES)
        densest = int(np.argmax(counts))
        selected = bins == densest
        ek_mean = float(energy[sample, selected].mean())
        r_ref = float(np.interp(ek_mean, energy_axis, seo_ini[:, 11]))
        vr_ref = float(np.interp(ek_mean, energy_axis, seo_ini[:, 12]))
        pr_ref = vr_ref / np.sqrt(max(1.0 - vr_ref * vr_ref, 1.0e-30))
        x = (radius[sample, selected] - r_ref) / np.sqrt(1.0 + pr_ref * pr_ref)
        sigma_x[sample] = x.std(ddof=0)
        sigma_ek[sample] = energy[sample, selected].std(ddof=0)
        mean_ek[sample] = ek_mean
        slice_count[sample] = int(selected.sum())
        slice_number[sample] = densest

    result = {
        "turn": np.arange(FIRST_3D_TURN,
                          FIRST_3D_TURN + N_3D_TURNS, dtype=np.int64),
        "sigma_x_m": sigma_x,
        "sigma_ek_mev": sigma_ek,
        "mean_ek_mev": mean_ek,
        "slice_count": slice_count,
        "slice_number": slice_number,
        "n_particles": np.asarray(len(files), dtype=np.int64),
    }
    np.savez_compressed(cache_path, **result)
    return result


def interpolate_longitudinal_snapshot(values: np.ndarray,
                                      snapshot_turns: np.ndarray,
                                      turn: int) -> np.ndarray:
    exact = np.flatnonzero(snapshot_turns == turn)
    if len(exact):
        return values[int(exact[0])].astype(np.float64)
    upper = int(np.searchsorted(snapshot_turns, turn, side="right"))
    lower = upper - 1
    fraction = ((turn - snapshot_turns[lower])
                / (snapshot_turns[upper] - snapshot_turns[lower]))
    return ((1.0 - fraction) * values[lower]
            + fraction * values[upper]).astype(np.float64)


def theoretical_slice_sigma_ek(turns: np.ndarray) -> np.ndarray:
    cache = np.load(LONG_CACHE)
    snapshot_turns = cache["snap_turns"].astype(np.int64)
    s_snap = cache["s_no"]
    de_snap = cache["de_no"]
    result = np.empty(len(turns))
    for index, turn in enumerate(turns):
        s = interpolate_longitudinal_snapshot(s_snap, snapshot_turns, int(turn))
        de = interpolate_longitudinal_snapshot(de_snap, snapshot_turns, int(turn))
        edges = np.linspace(float(s.min()), float(s.max()), N_SLICES + 1)
        bins = np.clip(np.searchsorted(edges, s, side="right") - 1,
                       0, N_SLICES - 1)
        counts = np.bincount(bins, minlength=N_SLICES)
        selected = bins == int(np.argmax(counts))
        result[index] = de[selected].std(ddof=0)
    return result


def dispersion_mismatch_theory() -> dict[str, np.ndarray]:
    lattice = analysis.build_lattice()
    phi, dx, dxp = centered_dispersion_curves()
    dispersion = float(np.interp(0.0, phi, dx, period=2.0 * np.pi))
    dispersion_prime = float(np.interp(0.0, phi, dxp, period=2.0 * np.pi))
    beta = float(lattice["beta_r"][0])
    alpha = float(lattice["alpha_r"][0])
    tune = float(lattice["qr"])
    sigma_beta = float(np.sqrt(beta * EMITTANCE_RMS))

    turns = np.arange(1, N_THEORY_TURNS + 1, dtype=np.int64)
    sigma_ek = theoretical_slice_sigma_ek(turns)
    phase_advance = 2.0 * np.pi * tune * turns
    m11 = np.cos(phase_advance) + alpha * np.sin(phase_advance)
    m12 = beta * np.sin(phase_advance)

    # At injection all energies share the central-energy orbit.  Relative to
    # each off-energy closed orbit the initial mismatch is (-D*dE,-D'*dE).
    # At the same observation azimuth after n turns, the energy-dependent
    # physical displacement is [D - (M11 D + M12 D')] dE.
    energy_to_x = dispersion - (m11 * dispersion + m12 * dispersion_prime)
    sigma_x = np.sqrt(sigma_beta * sigma_beta
                      + (energy_to_x * sigma_ek) ** 2)
    return {
        "turn": turns,
        "sigma_x_m": sigma_x,
        "sigma_ek_mev": sigma_ek,
        "energy_to_x_m_per_mev": energy_to_x,
        "sigma_beta_m": np.asarray(sigma_beta),
        "dispersion_m_per_mev": np.asarray(dispersion),
        "dispersion_prime_rad_per_mev": np.asarray(dispersion_prime),
        "beta_m": np.asarray(beta),
        "alpha": np.asarray(alpha),
        "tune": np.asarray(tune),
    }


def dominant_period(values: np.ndarray) -> float:
    detrended = values - np.polyval(np.polyfit(np.arange(len(values)), values, 1),
                                    np.arange(len(values)))
    amplitude = np.abs(np.fft.rfft(detrended))
    amplitude[0] = 0.0
    index = int(np.argmax(amplitude))
    return float(len(values) / index) if index else float("inf")


def fitted_harmonic_amplitude(turns: np.ndarray, values: np.ndarray,
                              frequency: float) -> float:
    """Return sinusoidal amplitude after removing offset and linear drift."""
    omega_t = 2.0 * np.pi * frequency * turns
    design = np.column_stack((np.ones(len(turns)), turns,
                              np.cos(omega_t), np.sin(omega_t)))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(np.hypot(coefficients[2], coefficients[3]))


def write_csv(path: Path, data_3d: dict[str, np.ndarray],
              theory: dict[str, np.ndarray], fixed_sigma_m: float) -> None:
    by_3d = {int(turn): i for i, turn in enumerate(data_3d["turn"])}
    by_theory = {int(turn): i for i, turn in enumerate(theory["turn"])}
    turns = sorted(set(by_3d) | set(by_theory))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "turn", "sigma_x_3d_mm", "sigma_ek_3d_mev", "slice_count_3d",
            "sigma_x_mismatch_theory_mm", "sigma_ek_theory_mev",
            "toy_prescribed_field_sigma_x_mm",
        ])
        for turn in turns:
            i3 = by_3d.get(turn)
            it = by_theory.get(turn)
            writer.writerow([
                turn,
                "" if i3 is None else 1.0e3 * data_3d["sigma_x_m"][i3],
                "" if i3 is None else data_3d["sigma_ek_mev"][i3],
                "" if i3 is None else data_3d["slice_count"][i3],
                "" if it is None else 1.0e3 * theory["sigma_x_m"][it],
                "" if it is None else theory["sigma_ek_mev"][it],
                "" if it is None else 1.0e3 * fixed_sigma_m,
            ])


def make_figure(data_3d: dict[str, np.ndarray], theory: dict[str, np.ndarray],
                fixed_sigma_m: float) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.4), sharex=True)
    axes[0].plot(data_3d["turn"], 1.0e3 * data_3d["sigma_x_m"],
                 color="tab:blue", lw=1.4, label="3-D densest slice (turns 10--109)")
    axes[0].plot(theory["turn"], 1.0e3 * theory["sigma_x_m"],
                 color="tab:orange", lw=1.2,
                 label="linear common-orbit mismatch theory (turns 1--100)")
    axes[0].axhline(1.0e3 * fixed_sigma_m, color="tab:red", lw=1.1, ls="--",
                    label="current toy prescribed field width")
    axes[0].set_ylabel(r"densest-slice $\sigma_x$ [mm]")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, ncol=1)

    axes[1].plot(data_3d["turn"], data_3d["sigma_ek_mev"],
                 color="tab:blue", lw=1.4, label="3-D")
    axes[1].plot(theory["turn"], theory["sigma_ek_mev"],
                 color="tab:orange", lw=1.2, label="longitudinal toy")
    axes[1].set_xlabel("turn number")
    axes[1].set_ylabel(r"densest-slice $\sigma_E$ [MeV]")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle("LSC-off turn-resolved horizontal size at the 90-degree dump")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--rebuild-3d", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / "simulation1_turn10_109_slice_sigma.npz"
    if args.rebuild_3d and cache_path.exists():
        cache_path.unlink()

    data_3d = measure_3d_turns(cache_path)
    theory = dispersion_mismatch_theory()
    fixed_sigma_m = float(data_3d["sigma_x_m"][0])
    np.savez_compressed(OUT / "dispersion_mismatch_theory_turn1_100.npz", **theory)
    write_csv(OUT / "sigma_x_turn_by_turn.csv", data_3d, theory, fixed_sigma_m)
    figure = make_figure(data_3d, theory, fixed_sigma_m)
    figure.savefig(OUT / "sigma_x_turn_by_turn.png", dpi=240)
    figure.savefig(OUT / "sigma_x_turn_by_turn.pdf")

    overlap = np.intersect1d(data_3d["turn"], theory["turn"])
    x3 = np.asarray([data_3d["sigma_x_m"][np.flatnonzero(data_3d["turn"] == t)[0]]
                     for t in overlap])
    xt = np.asarray([theory["sigma_x_m"][np.flatnonzero(theory["turn"] == t)[0]]
                     for t in overlap])
    rmse = float(np.sqrt(np.mean((x3 - xt) ** 2)))
    correlation = float(np.corrcoef(x3, xt)[0, 1])
    fractional_tune = float(theory["tune"]) % 1.0
    amplitude_3d = fitted_harmonic_amplitude(
        data_3d["turn"], 1.0e3 * data_3d["sigma_x_m"], fractional_tune)
    amplitude_theory = fitted_harmonic_amplitude(
        theory["turn"], 1.0e3 * theory["sigma_x_m"], fractional_tune)
    summary = (
        f"3-D turns {data_3d['turn'][0]}--{data_3d['turn'][-1]}: "
        f"sigma_x mean/min/max = {1e3*data_3d['sigma_x_m'].mean():.6f} / "
        f"{1e3*data_3d['sigma_x_m'].min():.6f} / "
        f"{1e3*data_3d['sigma_x_m'].max():.6f} mm\n"
        f"theory turns {theory['turn'][0]}--{theory['turn'][-1]}: "
        f"sigma_x mean/min/max = {1e3*theory['sigma_x_m'].mean():.6f} / "
        f"{1e3*theory['sigma_x_m'].min():.6f} / "
        f"{1e3*theory['sigma_x_m'].max():.6f} mm\n"
        f"turn 10: 3-D={1e3*data_3d['sigma_x_m'][0]:.6f} mm, "
        f"theory={1e3*theory['sigma_x_m'][9]:.6f} mm\n"
        f"overlap turns {overlap[0]}--{overlap[-1]}: "
        f"RMSE={1e3*rmse:.6f} mm, correlation={correlation:.6f}\n"
        f"dominant sigma_x period: 3-D={dominant_period(data_3d['sigma_x_m']):.6f} "
        f"turns, theory={dominant_period(theory['sigma_x_m']):.6f} turns\n"
        f"amplitude at bare fractional Qr={fractional_tune:.9f}: "
        f"3-D={amplitude_3d:.6f} mm, theory={amplitude_theory:.6f} mm\n"
        f"theory beta/alpha/Qr = {float(theory['beta_m']):.9f} m / "
        f"{float(theory['alpha']):.9f} / {float(theory['tune']):.9f}\n"
        f"theory Dx/Dx' = {1e3*float(theory['dispersion_m_per_mev']):.9f} "
        f"mm/MeV / {1e3*float(theory['dispersion_prime_rad_per_mev']):.9f} "
        f"mrad/MeV\n"
    )
    (OUT / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"saved: {OUT / 'sigma_x_turn_by_turn.png'}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
