#!/usr/bin/env python3
"""RMS-envelope benchmark in both transverse directions.

This is the reviewer-response version of the root-level
``Benchmark_SC_envlope_stable.py``.  The original file is kept unchanged.
The numerical envelope model is retained, while the data reduction and plots
are generalized from the vertical-only ``sigma_y`` result to the horizontal
``sigma_x`` and vertical ``sigma_z`` results requested by Referee A, Comment 4.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.signal import savgol_filter

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    path
    for path in SCRIPT_DIR.parents
    if (path / "src" / "FFAG_Bunch.py").is_file() and (path / "benchmarks").is_dir()
)

ANGLE_PATTERN = re.compile(r"crossing_angle_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_start_")

# ============================================================
# 用户参数区：从仓库根目录直接运行脚本时，只需要修改这里
# ============================================================

CASES = (
    {
        "key": "with_sc",
        "label": "With SC",
        "tbt_dir": REPO_ROOT / "output" / "Benchmark_env_2e14" / "merged_files",
        "theory_particles": 2.0e14,
    },
    {
        "key": "without_sc",
        "label": "Without SC",
        "tbt_dir": REPO_ROOT / "output" / "Benchmark_env_2e13" / "merged_files",
        "theory_particles": 0.0,
    },
)

BETA_R_PATH = REPO_ROOT / "Bmap" / "resultsSEO" / "BetaFuncR.txt"
BETA_Z_PATH = REPO_ROOT / "Bmap" / "resultsSEO" / "BetaFuncZ.txt"
BETA_ROW_INDEX = 2

CIRCUMFERENCE_M = 72.0
EMITTANCE_X_M_RAD = 60e-6
EMITTANCE_Z_M_RAD = 60e-6
BUNCH_CUT_RATIO = 0.2
GAMMA_REL = 1.32

TURNS_TO_COMPARE = (0,)
SMOOTH_BETA = True
SG_WINDOW = 11
SG_POLYNOMIAL = 3

# 设为正整数可只读前几个 crossing 文件做快速检查；正式出图使用 None。
MAX_FILES = None
PHI_LIMITS_DEG = (0.0, 200.0)

OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_FIGURES = {
    ("with_sc", "x"): "fig_rms_env_sc_x.png",
    ("with_sc", "z"): "fig_rms_env_sc_z.png",
    ("without_sc", "x"): "fig_rms_env_nosc_x.png",
    ("without_sc", "z"): "fig_rms_env_nosc_z.png",
}

# 水平两图共用纵轴范围，竖直两图共用纵轴范围。
YLIMITS_CM = {
    "x": (1.2, 2.8),
    "z": (1.5, 3.5),
}


def load_beta_from_txt(path: Path, row_index_for_energy: int):
    data = np.loadtxt(path, skiprows=1)
    return data[0, 1:], data[row_index_for_energy, 1:]


def phi_to_s_cell(phi_deg, circumference):
    phi_deg = np.asarray(phi_deg, dtype=float)
    phi0 = phi_deg[0]
    dphi = phi_deg[-1] - phi0
    if dphi <= 0:
        raise ValueError("The azimuth grid must have a positive span")
    cell_length = circumference * dphi / 360.0
    s_cell = (phi_deg - phi0) / dphi * cell_length
    return s_cell, cell_length, dphi


def maybe_savgol(values, window, polynomial):
    if not HAS_SCIPY:
        return np.asarray(values, dtype=float)
    max_odd = len(values) if len(values) % 2 == 1 else len(values) - 1
    window = min(window, max_odd)
    if window < 5:
        return np.asarray(values, dtype=float)
    return savgol_filter(values, window, polynomial)


def focusing_from_beta(s_cell, beta, smooth, window, polynomial):
    beta_used = np.asarray(beta, dtype=float).copy()
    if smooth:
        beta_used = maybe_savgol(beta_used, window, polynomial)
    beta_prime = np.gradient(beta_used, s_cell)
    beta_second = np.gradient(beta_prime, s_cell)
    focusing = (
        1.0 / beta_used**2
        + beta_prime**2 / (4.0 * beta_used**2)
        - beta_second / (2.0 * beta_used)
    )
    return focusing, beta_used


def matched_initial_from_beta(s_cell, beta, emittance, smooth, window, polynomial):
    beta_for_derivative = np.asarray(beta, dtype=float)
    if smooth:
        beta_for_derivative = maybe_savgol(beta_for_derivative, window, polynomial)
    beta_prime = np.gradient(beta_for_derivative, s_cell)
    beta0 = beta_for_derivative[0]
    radius0 = 2.0 * np.sqrt(beta0 * emittance)
    slope0 = np.sqrt(emittance) * beta_prime[0] / np.sqrt(beta0)
    return radius0, slope0


def tile_cell(s_cell, values, number_of_cells):
    cell_length = s_cell[-1] - s_cell[0]
    s_parts = []
    value_parts = []
    for index in range(number_of_cells):
        keep = slice(None) if index == 0 else slice(1, None)
        s_parts.append(s_cell[keep] + index * cell_length)
        value_parts.append(values[keep])
    return np.concatenate(s_parts), np.concatenate(value_parts)


def integrate_envelopes(
    s_long,
    kx_long,
    kz_long,
    emittance_x,
    emittance_z,
    line_charge,
    beta_rel,
    gamma_rel,
    initial_x,
    initial_z,
):
    """Retain the original Euler-Cromer envelope integration and conventions."""
    epsilon0 = 8.8541878128e-12
    speed_of_light = 299792458.0
    elementary_charge = 1.602176634e-19
    proton_mass = 1.67262192369e-27

    # The integrated radii are a_x=2 sigma_x and a_z=2 sigma_z.
    emittance_kv_x = 4.0 * emittance_x
    emittance_kv_z = 4.0 * emittance_z
    sc_coefficient = (
        elementary_charge
        / (proton_mass * speed_of_light**2)
        * line_charge
        / (4.0 * np.pi * epsilon0)
        / (beta_rel**2 * gamma_rel**3)
    )

    state = np.array([initial_x[0], initial_x[1], initial_z[0], initial_z[1]])
    radius_x = np.empty_like(s_long)
    radius_z = np.empty_like(s_long)

    for index in range(len(s_long)):
        ax, ax_prime, az, az_prime = state
        radius_x[index] = ax
        radius_z[index] = az
        denominator = ax + az
        if denominator <= 0:
            raise RuntimeError(f"Envelope collapsed at s={s_long[index]:.6f} m")
        ax_second = (
            -kx_long[index] * ax
            + emittance_kv_x**2 / ax**3
            + 4.0 * sc_coefficient / denominator
        )
        az_second = (
            -kz_long[index] * az
            + emittance_kv_z**2 / az**3
            + 4.0 * sc_coefficient / denominator
        )
        if index < len(s_long) - 1:
            step = s_long[index + 1] - s_long[index]
            state[1] += step * ax_second
            state[0] += step * state[1]
            state[3] += step * az_second
            state[2] += step * state[3]

    return 0.5 * radius_x, 0.5 * radius_z


def parse_angle(path: Path):
    match = ANGLE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse crossing angle from {path.name}")
    return float(match.group(1))


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean((values - np.mean(values)) ** 2)))


def tracking_sigma_maps(tbt_dir: Path, selected_turns, max_files=None):
    """Read each crossing file once and return sigma_x and sigma_z.

    Column 0 is the radial coordinate and column 2 is the vertical z
    coordinate.  At a fixed azimuth, subtracting the radial mean makes the rms
    invariant to the constant reference-orbit radius, so std(r)=sigma_x.
    """
    files = sorted(tbt_dir.glob("*.csv"), key=parse_angle)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No crossing-angle CSV files found in {tbt_dir}")

    selected_turns = tuple(sorted(set(selected_turns)))
    angles = np.array([parse_angle(path) for path in files])
    sigma_x = np.full((len(selected_turns), len(files)), np.nan)
    sigma_z = np.full_like(sigma_x, np.nan)

    for file_index, path in enumerate(files):
        # Reading only r, z and phi substantially reduces the memory footprint.
        radius, vertical, azimuth = np.loadtxt(
            path,
            delimiter=",",
            usecols=(0, 2, 4),
            unpack=True,
        )
        turn_index = np.floor(azimuth / (2.0 * np.pi)).astype(np.int64)
        for row_index, turn in enumerate(selected_turns):
            mask = turn_index == turn
            if np.any(mask):
                sigma_x[row_index, file_index] = rms(radius[mask])
                sigma_z[row_index, file_index] = rms(vertical[mask])

    return angles, np.array(selected_turns), sigma_x, sigma_z, files


def theoretical_envelopes(number_of_particles):
    phi_cell, beta_x_raw = load_beta_from_txt(BETA_R_PATH, BETA_ROW_INDEX)
    phi_z, beta_z_raw = load_beta_from_txt(BETA_Z_PATH, BETA_ROW_INDEX)
    if len(phi_cell) != len(phi_z) or not np.allclose(phi_cell, phi_z, atol=1e-9, rtol=0):
        raise ValueError("BetaFuncR and BetaFuncZ do not use the same azimuth grid")

    s_cell, cell_length, cell_span_deg = phi_to_s_cell(phi_cell, CIRCUMFERENCE_M)
    kx_cell, beta_x = focusing_from_beta(
        s_cell, beta_x_raw, SMOOTH_BETA, SG_WINDOW, SG_POLYNOMIAL
    )
    kz_cell, beta_z = focusing_from_beta(
        s_cell, beta_z_raw, SMOOTH_BETA, SG_WINDOW, SG_POLYNOMIAL
    )
    cells_per_turn = int(round(360.0 / cell_span_deg))
    number_of_turns = max(TURNS_TO_COMPARE) + 1
    s_long, kx_long = tile_cell(s_cell, kx_cell, cells_per_turn * number_of_turns)
    _, kz_long = tile_cell(s_cell, kz_cell, cells_per_turn * number_of_turns)

    initial_x = matched_initial_from_beta(
        s_cell,
        beta_x,
        EMITTANCE_X_M_RAD,
        SMOOTH_BETA,
        SG_WINDOW,
        SG_POLYNOMIAL,
    )
    initial_z = matched_initial_from_beta(
        s_cell,
        beta_z,
        EMITTANCE_Z_M_RAD,
        SMOOTH_BETA,
        SG_WINDOW,
        SG_POLYNOMIAL,
    )

    elementary_charge = 1.602176634e-19
    bunch_length = CIRCUMFERENCE_M * BUNCH_CUT_RATIO
    line_charge = number_of_particles * elementary_charge / bunch_length
    beta_rel = np.sqrt(1.0 - 1.0 / GAMMA_REL**2)
    sigma_x, sigma_z = integrate_envelopes(
        s_long,
        kx_long,
        kz_long,
        EMITTANCE_X_M_RAD,
        EMITTANCE_Z_M_RAD,
        line_charge,
        beta_rel,
        GAMMA_REL,
        initial_x,
        initial_z,
    )
    phi_global = 360.0 * s_long / CIRCUMFERENCE_M
    print(
        f"Theory grid: {cells_per_turn} cells/turn, "
        f"cell length={cell_length:.6f} m, cell span={cell_span_deg:.6f} deg"
    )
    return phi_global, sigma_x, sigma_z


def save_tracking_summary(path, case_key, angles, turns, sigma_x, sigma_z, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("case", "turn", "azimuth_deg", "sigma_x_m", "sigma_z_m", "source_file")
        )
        for turn_index, turn in enumerate(turns):
            for angle_index, angle in enumerate(angles):
                writer.writerow(
                    (
                        case_key,
                        int(turn),
                        angle,
                        sigma_x[turn_index, angle_index],
                        sigma_z[turn_index, angle_index],
                        files[angle_index].name,
                    )
                )


def make_figures(results):
    # 与根目录最初脚本一致的论文绘图风格。
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 26,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
            "figure.titlesize": 28,
            "lines.linewidth": 2.5,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for result in results:
        directions = (
            ("x", result["theory_x"], result["track_x"], r"$\sigma_x$ [cm]"),
            ("z", result["theory_z"], result["track_z"], r"$\sigma_z$ [cm]"),
        )
        for direction, theory, tracking, ylabel in directions:
            fig, axis = plt.subplots(figsize=(10, 4))
            sc_label = "with SC" if result["key"] == "with_sc" else "without SC"
            axis.plot(
                result["phi_theory"],
                theory * 100.0,
                lw=2.4,
                color="blue",
                label=f"Theory ({sc_label})",
            )
            for turn_index, turn in enumerate(result["turns"]):
                label = f"PyFFAG Tracking ({sc_label})" if turn_index == 0 else None
                axis.scatter(
                    result["angles"] + 360.0 * turn,
                    tracking[turn_index] * 100.0,
                    marker="*",
                    s=280,
                    color="red",
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=10,
                    linestyle="None",
                    label=label,
                )
            axis.set_xlabel(r"Azimuth $\phi$ [deg]")
            axis.set_ylabel(ylabel)
            axis.set_xlim(*PHI_LIMITS_DEG)
            axis.set_ylim(*YLIMITS_CM[direction])
            axis.grid(True)
            # 原图未显示 legend 和 title，这里保持相同风格。
            fig.tight_layout()
            output = OUTPUT_DIR / OUTPUT_FIGURES[(result["key"], direction)]
            fig.savefig(output, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {output}")


def main():
    if min(TURNS_TO_COMPARE) < 0:
        raise ValueError("Turn numbers must be non-negative")
    if not HAS_SCIPY and SMOOTH_BETA:
        print("Warning: scipy is unavailable; beta smoothing is skipped")

    results = []
    for case in CASES:
        print(f"Processing {case['label']}: {case['tbt_dir']}")
        angles, turns, track_x, track_z, files = tracking_sigma_maps(
            case["tbt_dir"], TURNS_TO_COMPARE, MAX_FILES
        )
        phi_theory, theory_x, theory_z = theoretical_envelopes(case["theory_particles"])
        save_tracking_summary(
            OUTPUT_DIR / f"rms_envelope_{case['key']}_tracking.csv",
            case["key"],
            angles,
            turns,
            track_x,
            track_z,
            files,
        )
        results.append(
            {
                **case,
                "angles": angles,
                "turns": turns,
                "track_x": track_x,
                "track_z": track_z,
                "phi_theory": phi_theory,
                "theory_x": theory_x,
                "theory_z": theory_z,
            }
        )
    make_figures(results)


if __name__ == "__main__":
    main()
