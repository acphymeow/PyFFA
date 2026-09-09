#!/usr/bin/env python3
"""Run the self-contained KV tune benchmark with third-order resonances."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.signal import savgol_filter


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bundle_paths import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FFAG_MathTools import Bilinear_interp_2D_vect_uniform
from paper_resonance_overlay import _label_position, _rotation
from plot_resonance_lines_order3 import (
    generate_resonances,
    is_reviewer_highlight,
    line_segment_in_window,
)


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_OUTPUT_STEM = "fig_tune_kv90_with_resonances"

# ---------------------------------------------------------------------------
# Physics and data configuration.  Emittances are geometric RMS values in
# m rad; 70e-6 corresponds to the 70 pi mm mrad convention used in the paper.
# Keep the emittance, input directory, panel label, and output stem consistent
# when switching between the 50, 70, and 90 pi mm mrad benchmark cases.
# ---------------------------------------------------------------------------
E_CHARGE = 1.602176634e-19
M_PROTON = 1.67262192369e-27
C_LIGHT = 2.99792458e8
EPS0 = 8.8541878128e-12

CIRCUMFERENCE_M = 70.0
KINETIC_ENERGY_MEV = 300.0
QX_BARE = 3.23
QZ_BARE = 2.80
N_TOTAL_PROTONS = 2.08e13
BUNCH_FULL_LENGTH_M = 12.0
EPS_X_RMS = 90.0e-6
EPS_Z_RMS = 90.0e-6
N_SIGMA = 4.0
FIGURE_LABEL = "b"
DATA_DIR = REPO_ROOT.parent / "output" / "NewBenchmark" / "FFT_Tune_kv90"

SEO_DIR = REPO_ROOT / "Bmap_FD16" / "resultsSEO_noError_16000"
BETA_R_PATH = SEO_DIR / "BetaFuncR.txt"
BETA_Z_PATH = SEO_DIR / "BetaFuncZ.txt"
SEO_R_PATH = SEO_DIR / "SEO_r.txt"
SEO_PR_PATH = SEO_DIR / "SEO_pr.txt"


COL_X, COL_XP = 0, 1
COL_Z, COL_ZP = 2, 3
COL_FI, COL_EK = 4, 5
PAD_FACTOR = 8
USE_SAVGOL = False
SG_WINDOW, SG_ORDER = 18, 4
QX_CELL_SHIFT = 3.0
QZ_CELL_SHIFT = 2.0

QX_LIMITS = (2.3, 3.3)
QZ_LIMITS = (1.9, 2.9)
RESONANCE_LINE_COLOR = "#9f9f9f"
RESONANCE_TEXT_COLOR = "#858585"
RESONANCE_LINEWIDTH = 0.58
RESONANCE_LINE_ALPHA = 0.95
RESONANCE_LABEL_FONTSIZE = 7.8
RESONANCE_TEXT_ALPHA = 0.82
RESONANCE_LABEL_BOX_ALPHA = 0.36


def beta_gamma_from_kinetic_energy(
    kinetic_energy_mev: float,
    rest_energy_mev: float = 938.2720813,
) -> tuple[float, float]:
    total_energy = kinetic_energy_mev + rest_energy_mev
    gamma = total_energy / rest_energy_mev
    beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
    return beta, gamma


def truncated_gaussian_peak_line_density(
    n_total: float,
    sigma_s: float,
    n_sigma: float,
) -> tuple[float, float]:
    cut = n_sigma * sigma_s
    normalization = math.erf(cut / (math.sqrt(2.0) * sigma_s))
    peak = n_total / (math.sqrt(2.0 * math.pi) * sigma_s * normalization)
    return peak, normalization


def kv_field_slopes(
    line_charge: float,
    beta_x: np.ndarray,
    beta_z: np.ndarray,
    eps_x_rms: float,
    eps_z_rms: float,
) -> tuple[np.ndarray, np.ndarray]:
    sigma_x = np.maximum(np.sqrt(np.asarray(beta_x) * eps_x_rms), 1e-12)
    sigma_z = np.maximum(np.sqrt(np.asarray(beta_z) * eps_z_rms), 1e-12)
    ex_over_x = line_charge / (
        4.0 * math.pi * EPS0 * sigma_x * (sigma_x + sigma_z)
    )
    ez_over_z = line_charge / (
        4.0 * math.pi * EPS0 * sigma_z * (sigma_x + sigma_z)
    )
    return ex_over_x, ez_over_z


def space_charge_focusing_arrays(
    line_charge: float,
    beta_x: np.ndarray,
    beta_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    beta_rel, gamma = beta_gamma_from_kinetic_energy(KINETIC_ENERGY_MEV)
    ex_over_x, ez_over_z = kv_field_slopes(
        line_charge,
        beta_x,
        beta_z,
        EPS_X_RMS,
        EPS_Z_RMS,
    )
    factor = -E_CHARGE / (
        gamma**3 * M_PROTON * beta_rel**2 * C_LIGHT**2
    )
    return factor * ex_over_x, factor * ez_over_z


def tune_shift_from_one_cell(
    azimuth_deg: np.ndarray,
    beta_function: np.ndarray,
    focusing: np.ndarray,
    circumference_m: float,
    n_cells: int,
) -> float:
    cell_span_deg = azimuth_deg[-1] - azimuth_deg[0]
    cell_length = circumference_m * cell_span_deg / 360.0
    s_cell = (azimuth_deg - azimuth_deg[0]) / cell_span_deg * cell_length
    cell_integral = np.trapz(beta_function * focusing, s_cell)
    return n_cells * cell_integral / (4.0 * math.pi)


def compute_theory_kv_gaussian_peak() -> dict:
    beta_r_data = np.loadtxt(BETA_R_PATH, skiprows=1)
    beta_z_data = np.loadtxt(BETA_Z_PATH, skiprows=1)
    azimuth_deg = beta_r_data[0, 1:]
    beta_x = beta_r_data[2, 1:]
    beta_z = beta_z_data[2, 1:]
    n_cells = int(round(360.0 / (azimuth_deg[-1] - azimuth_deg[0])))

    sigma_s = BUNCH_FULL_LENGTH_M / (2.0 * N_SIGMA)
    peak_particles_per_m, truncation_norm = truncated_gaussian_peak_line_density(
        N_TOTAL_PROTONS,
        sigma_s,
        N_SIGMA,
    )
    peak_charge_per_m = peak_particles_per_m * E_CHARGE
    k_sc_x, k_sc_z = space_charge_focusing_arrays(
        peak_charge_per_m,
        beta_x,
        beta_z,
    )
    dqx_max = tune_shift_from_one_cell(
        azimuth_deg,
        beta_x,
        k_sc_x,
        CIRCUMFERENCE_M,
        n_cells,
    )
    dqz_max = tune_shift_from_one_cell(
        azimuth_deg,
        beta_z,
        k_sc_z,
        CIRCUMFERENCE_M,
        n_cells,
    )
    qx_max = QX_BARE + dqx_max
    qz_max = QZ_BARE + dqz_max

    print("=== Theory: transverse KV + longitudinal Gaussian peak tune shift ===")
    print(f"sigma_s = {sigma_s:.6f} m")
    print(f"lambda_p_peak = {peak_particles_per_m:.6e} particles/m")
    print(f"lambda_q_peak = {peak_charge_per_m:.6e} C/m")
    print(f"dQx_max = {dqx_max:.8f}")
    print(f"dQz_max = {dqz_max:.8f}")
    print(f"Bare tune = ({QX_BARE:.8f}, {QZ_BARE:.8f})")
    print(f"Max SC tune shift point = ({qx_max:.8f}, {qz_max:.8f})")
    return {
        "bare": (QX_BARE, QZ_BARE),
        "gaussian_peak": (qx_max, qz_max),
        "dQ": (dqx_max, dqz_max),
        "sigma_s": sigma_s,
        "norm_trunc": truncation_norm,
        "lambda_p_peak": peak_particles_per_m,
        "lambda_q_peak": peak_charge_per_m,
    }


def parabolic_interpolate(amplitude: np.ndarray, index: int) -> float:
    y0, y1, y2 = amplitude[index - 1], amplitude[index], amplitude[index + 1]
    denominator = y0 - 2.0 * y1 + y2
    if denominator == 0:
        return float(index)
    return index + 0.5 * (y0 - y2) / denominator


def tune_from_phase_space(
    coordinate: np.ndarray,
    momentum: np.ndarray,
    pad_factor: int = PAD_FACTOR,
) -> float:
    coordinate = np.asarray(coordinate)
    momentum = np.asarray(momentum)
    if coordinate.size != momentum.size:
        raise ValueError("coordinate and momentum must have the same length")
    if coordinate.size < 8:
        raise ValueError("too few turn-by-turn points for FFT tune analysis")

    sigma_coordinate = coordinate.std(ddof=0)
    sigma_momentum = momentum.std(ddof=0)
    momentum_scale = (
        sigma_momentum / sigma_coordinate
        if sigma_coordinate > 0 and sigma_momentum > 0
        else 1.0
    )
    signal = coordinate + 1j * momentum / momentum_scale
    signal -= signal.mean()
    n_fft = int(pad_factor * coordinate.size)
    spectrum = np.fft.fft(signal, n=n_fft)
    frequencies = np.fft.fftfreq(n_fft, d=1.0)
    amplitude = np.abs(spectrum)
    amplitude[0] = 0.0
    peak_index = int(np.argmax(amplitude))
    usable_index = min(max(peak_index, 1), n_fft - 2)
    interpolated_index = parabolic_interpolate(amplitude, usable_index)
    return (interpolated_index % n_fft) * (frequencies[1] - frequencies[0])


def fractional_tunes(vx_raw: float, vz_raw: float) -> tuple[float, float]:
    qx = 1.0 - vx_raw
    qz = 1.0 - vz_raw
    if qz > 0.25:
        qz -= 1.0
    return qx, qz


def compute_fft_tunes() -> tuple[np.ndarray, np.ndarray]:
    seo_r_info = np.loadtxt(SEO_R_PATH, skiprows=1)
    seo_pr_info = np.loadtxt(SEO_PR_PATH, skiprows=1)
    seo_azimuth_axis = seo_r_info[0, 1:]
    seo_energy_axis = seo_r_info[1:, 0]
    seo_r_matrix = seo_r_info[1:, 1:]
    seo_pr_matrix = seo_pr_info[1:, 1:]
    particle_paths = sorted(DATA_DIR.glob("particle_*.npz"))
    print(f"Particle files found: {len(particle_paths)}")

    tune_pairs: list[tuple[float, float]] = []
    for path in particle_paths:
        try:
            particles = np.load(path)["particles"]
            particles = particles[particles[:, -1].argsort()]
            if len(particles) < max(16, SG_WINDOW):
                continue

            x = particles[:, COL_X].copy()
            px = particles[:, COL_XP].copy()
            z = particles[:, COL_Z].copy()
            pz = particles[:, COL_ZP].copy()
            azimuth = np.mod(particles[:, COL_FI], 2.0 * np.pi)
            kinetic_energy = particles[:, COL_EK]
            r_seo = np.zeros_like(x)
            pr_seo = np.zeros_like(x)
            r_flag = np.zeros_like(x)
            pr_flag = np.zeros_like(x)

            Bilinear_interp_2D_vect_uniform(
                seo_energy_axis[0],
                seo_energy_axis[1] - seo_energy_axis[0],
                len(seo_energy_axis),
                seo_azimuth_axis[0],
                seo_azimuth_axis[1] - seo_azimuth_axis[0],
                len(seo_azimuth_axis),
                seo_r_matrix,
                kinetic_energy,
                azimuth,
                r_seo,
                r_flag,
            )
            Bilinear_interp_2D_vect_uniform(
                seo_energy_axis[0],
                seo_energy_axis[1] - seo_energy_axis[0],
                len(seo_energy_axis),
                seo_azimuth_axis[0],
                seo_azimuth_axis[1] - seo_azimuth_axis[0],
                len(seo_azimuth_axis),
                seo_pr_matrix,
                kinetic_energy,
                azimuth,
                pr_seo,
                pr_flag,
            )

            if USE_SAVGOL:
                window = SG_WINDOW if SG_WINDOW % 2 else SG_WINDOW + 1
                x -= savgol_filter(x, window, SG_ORDER)
                px -= savgol_filter(px, window, SG_ORDER)
                z -= savgol_filter(z, window, SG_ORDER)
                pz -= savgol_filter(pz, window, SG_ORDER)

            x -= r_seo
            px -= pr_seo
            vx_raw = tune_from_phase_space(x, px)
            vz_raw = tune_from_phase_space(z, pz)
            qx_fractional, qz_fractional = fractional_tunes(vx_raw, vz_raw)
            tune_pairs.append(
                (
                    qx_fractional + QX_CELL_SHIFT,
                    qz_fractional + QZ_CELL_SHIFT,
                )
            )
        except Exception as exc:
            print(f"FFT tune failed for {path.name}: {exc}")

    if not tune_pairs:
        return np.array([]), np.array([])
    qx_all, qz_all = np.asarray(tune_pairs).T
    print(f"FFT tunes extracted successfully: {len(qx_all)}")
    return qx_all, qz_all


def apply_original_tune_branches(
    qx_all: np.ndarray,
    qz_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the absolute-tune branches used by the benchmark."""
    qx_plot = np.asarray(qx_all, dtype=float).copy()
    qz_plot = np.asarray(qz_all, dtype=float).copy()
    qx_plot[qx_plot > 3.5] -= 1.0
    qz_plot += 1.0
    return qx_plot, qz_plot


def draw_gaussian_style_resonances(
    axis: plt.Axes,
    qx_data: np.ndarray,
    qz_data: np.ndarray,
    *,
    max_order: int,
    label_mode: str,
) -> None:
    """Match the resonance-line style used by the Gaussian tune script."""
    qx_data = np.asarray(qx_data, dtype=float)
    qz_data = np.asarray(qz_data, dtype=float)
    good = (
        np.isfinite(qx_data)
        & np.isfinite(qz_data)
        & (qx_data >= QX_LIMITS[0])
        & (qx_data <= QX_LIMITS[1])
        & (qz_data >= QZ_LIMITS[0])
        & (qz_data <= QZ_LIMITS[1])
    )
    qx_good = qx_data[good]
    qz_good = qz_data[good]
    if qx_good.size:
        data_center = (float(np.median(qx_good)), float(np.median(qz_good)))
        qx_low, qx_high = np.percentile(qx_good, [3.0, 97.0])
        qz_low, qz_high = np.percentile(qz_good, [3.0, 97.0])
    else:
        data_center = (0.5 * sum(QX_LIMITS), 0.5 * sum(QZ_LIMITS))
        qx_low = qx_high = data_center[0]
        qz_low = qz_high = data_center[1]

    x_span = QX_LIMITS[1] - QX_LIMITS[0]
    z_span = QZ_LIMITS[1] - QZ_LIMITS[0]
    axis.set_xlim(QX_LIMITS)
    axis.set_ylim(QZ_LIMITS)
    axis.figure.canvas.draw()

    avoid_boxes = [
        (
            qx_low - 0.025 * x_span,
            qx_high + 0.025 * x_span,
            qz_low - 0.025 * z_span,
            qz_high + 0.025 * z_span,
        ),
        (
            QX_LIMITS[0],
            QX_LIMITS[0] + 0.15 * x_span,
            QZ_LIMITS[0] + 0.60 * z_span,
            QZ_LIMITS[0] + 0.76 * z_span,
        ),
    ]
    legend = axis.get_legend()
    if legend is not None:
        renderer = axis.figure.canvas.get_renderer()
        bbox = legend.get_window_extent(renderer=renderer)
        data_bbox = axis.transData.inverted().transform(
            np.asarray([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
        )
        avoid_boxes.append(
            (
                data_bbox[0, 0] - 0.015 * x_span,
                data_bbox[1, 0] + 0.015 * x_span,
                data_bbox[0, 1] - 0.015 * z_span,
                data_bbox[1, 1] + 0.015 * z_span,
            )
        )

    occupied: list[tuple[float, float]] = []
    for line in generate_resonances(QX_LIMITS, QZ_LIMITS, max_order=max_order):
        segment = line_segment_in_window(line, QX_LIMITS, QZ_LIMITS)
        if segment is None:
            continue
        (x0, z0), (x1, z1) = segment
        axis.plot(
            [x0, x1],
            [z0, z1],
            color=RESONANCE_LINE_COLOR,
            linewidth=RESONANCE_LINEWIDTH,
            alpha=RESONANCE_LINE_ALPHA,
            linestyle="-",
            zorder=3,
        )

        should_label = label_mode == "all" or (
            label_mode == "highlighted" and is_reviewer_highlight(line)
        )
        if not should_label:
            continue
        position = _label_position(
            segment,
            data_center,
            avoid_boxes,
            occupied,
            QX_LIMITS,
            QZ_LIMITS,
        )
        occupied.append(position)
        axis.text(
            position[0],
            position[1],
            line.label,
            color=RESONANCE_TEXT_COLOR,
            alpha=RESONANCE_TEXT_ALPHA,
            fontsize=RESONANCE_LABEL_FONTSIZE,
            fontweight="normal",
            rotation=_rotation(axis, segment),
            rotation_mode="anchor",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": RESONANCE_LABEL_BOX_ALPHA,
            },
            clip_on=True,
            zorder=-1,
        )


def plot_kv_tunes_with_resonances(
    theory_dict: dict,
    qx_all: np.ndarray,
    qz_all: np.ndarray,
    output_dir: Path,
    output_stem: str,
    *,
    max_order: int,
    label_mode: str,
) -> tuple[plt.Figure, Path, Path]:
    """Reproduce the original KV tune plot with a resonance overlay."""
    qx_bare, qz_bare = theory_dict["bare"]
    qx_max, qz_max = theory_dict["gaussian_peak"]
    dqx_max, dqz_max = theory_dict["dQ"]

    fig, axis = plt.subplots(
        1,
        1,
        figsize=(7.2, 6.0),
        constrained_layout=True,
    )

    histogram = None
    if qx_all.size:
        histogram = axis.hist2d(
            qx_all,
            qz_all,
            bins=[256, 256],
            range=[QX_LIMITS, QZ_LIMITS],
            norm=LogNorm(),
            cmap="viridis",
            zorder=1,
        )

    u_min = np.exp(-0.5 * N_SIGMA * N_SIGMA)
    u_line = np.linspace(u_min, 1.0, 500)
    qx_line = qx_bare + dqx_max * u_line
    qz_line = qz_bare + dqz_max * u_line
    axis.plot(
        qx_line,
        qz_line,
        "--",
        color="red",
        linewidth=3.0,
        label="_nolegend_",
        zorder=8,
    )

    axis.scatter(
        qx_bare,
        qz_bare,
        marker="o",
        s=160,
        facecolor="white",
        edgecolor="blue",
        linewidth=2.8,
        label="Bare tune",
        zorder=12,
    )
    axis.scatter(
        qx_max,
        qz_max,
        marker="*",
        s=300,
        color="red",
        edgecolor="black",
        linewidth=2.4,
        label="Max SC tune shift",
        zorder=13,
    )
    axis.text(
        0.04,
        0.70,
        rf"({FIGURE_LABEL})",
        transform=axis.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
        zorder=20,
    )

    axis.set_xlabel(r"$Q_x$", fontsize=24)
    axis.set_ylabel(r"$Q_z$", fontsize=24)
    axis.set_xlim(QX_LIMITS)
    axis.set_ylim(QZ_LIMITS)
    axis.tick_params(axis="both", labelsize=20)
    legend = axis.legend(fontsize=15, loc="upper left")
    legend.set_zorder(30)

    draw_gaussian_style_resonances(
        axis,
        qx_all,
        qz_all,
        max_order=max_order,
        label_mode=label_mode,
    )

    if histogram is not None:
        colorbar = fig.colorbar(histogram[3], ax=axis)
        colorbar.set_label("Counts", fontsize=19)
        colorbar.ax.tick_params(labelsize=12)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return fig, png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emittance",
        type=float,
        default=EPS_X_RMS * 1.0e6,
        help="Geometric rms emittance in pi mm mrad.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figure-label", default=FIGURE_LABEL)
    parser.add_argument("--max-order", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--label-mode",
        choices=("all", "highlighted", "none"),
        default="all",
        help="Label all lines, only Qx=3 and 3Qz=8, or no lines.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    global EPS_X_RMS, EPS_Z_RMS, DATA_DIR, FIGURE_LABEL
    args = parse_args()
    EPS_X_RMS = args.emittance * 1.0e-6
    EPS_Z_RMS = args.emittance * 1.0e-6
    DATA_DIR = args.data_dir
    FIGURE_LABEL = args.figure_label
    theory_dict = compute_theory_kv_gaussian_peak()
    qx_all, qz_all = compute_fft_tunes()
    if not qx_all.size:
        raise RuntimeError(f"No valid tune data found in {DATA_DIR}")
    qx_all, qz_all = apply_original_tune_branches(qx_all, qz_all)

    print(
        "Qx FFT min / mean / max = "
        f"{qx_all.min():.8f}, {qx_all.mean():.8f}, {qx_all.max():.8f}"
    )
    print(
        "Qz FFT min / mean / max = "
        f"{qz_all.min():.8f}, {qz_all.mean():.8f}, {qz_all.max():.8f}"
    )

    figure, png_path, pdf_path = plot_kv_tunes_with_resonances(
        theory_dict,
        qx_all,
        qz_all,
        args.output_dir,
        args.output_stem,
        max_order=args.max_order,
        label_mode=args.label_mode,
    )
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
