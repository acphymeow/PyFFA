"""Standard transverse dynamic-aperture scan for PyFFA analytical lattices.

The scan follows the usual polar-grid tracking protocol in normalized
transverse amplitude space.  At a fixed energy and azimuth, each launch point
is described by

    epsilon_r = rho**2 * cos(theta)**2
    epsilon_z = rho**2 * sin(theta)**2

where epsilon is the geometric Courant--Snyder invariant in PyFFA's
``pi mm mrad`` convention.  The physical launch coordinates are reconstructed
from the local closed orbit and Twiss functions.  Several transverse phase
pairs may be tracked for each polar point; a point is conservatively called
stable only when every requested phase pair survives the requested number of
turns.

RF, space charge, acceleration and magnetic-error harmonics are intentionally
absent.  This is a fixed-energy single-particle DA calculation.

Typical screening run (run from the repository root)::

    python project_reverse_injection/scripts/tool_scan_lattice_da.py \
      --bmap ./Bmaps/Bmap_FD16_k7p63_B1p65_R265_Fp0p75_s49p97 \
      --bmap ./Bmaps/Bmap_FD16_k7p63_B1p65_R365_Fp0p75_s49p97 \
      --energy-mev 300 --turns 200 --phi-step-rad 0.001 \
      --max-emit-pi-mm-mrad 2000 --n-radius 41 --n-angle 19

For a final result, repeat the boundary calculation with more turns, a finer
radial grid and at least twice the azimuthal step density to demonstrate
numerical convergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FFAG_Distribution import (  # noqa: E402
    LoadSEOParams,
)
from FFAG_Field import FFAG_Bfield_analytical  # noqa: E402
from FFAG_ParasAndConversion import FFAG_ConversionTools, FFAG_GlobalParameters  # noqa: E402
from FFAG_track import FFAG_RungeKutta, FunctionForSEOUnified  # noqa: E402


DEFAULT_PHASE_PAIRS_DEG = "0,0;90,0;0,90;90,90"


@dataclass(frozen=True)
class ScanSettings:
    energy_mev: float
    azimuth_deg: float
    turns: int
    steps_per_turn: int
    max_emit_pi_mm_mrad: float
    n_radius: int
    n_angle: int
    phase_pairs_deg: tuple[tuple[float, float], ...]
    aperture_r_min_m: float
    aperture_r_max_m: float
    aperture_z_abs_m: float


def parse_phase_pairs(text: str) -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        fields = [value.strip() for value in item.split(",")]
        if len(fields) != 2:
            raise ValueError(
                "Each phase pair must be 'phi_r_deg,phi_z_deg'; "
                f"got {item!r}."
            )
        pairs.append((float(fields[0]), float(fields[1])))
    if not pairs:
        raise ValueError("At least one transverse phase pair is required.")
    return tuple(pairs)


def load_numeric_table(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path)
    except ValueError:
        return np.loadtxt(path, skiprows=1)


def interpolate_periodic_table(
    table: np.ndarray,
    energy_mev: float,
    coordinate: float,
) -> float:
    """Linearly interpolate an SEO/Twiss table in energy and periodic phase.

    The production helper uses a four-point Lagrange stencil.  Several scan
    lattices contain only two energy rows, so that stencil repeats endpoints
    and can return NaN.  DA launch generation needs a well-defined local
    linear interpolation instead.
    """
    energy_axis = np.asarray(table[1:, 0], dtype=np.float64)
    phase_axis = np.asarray(table[0, 1:], dtype=np.float64)
    values = np.asarray(table[1:, 1:], dtype=np.float64)
    if energy_axis.size < 1 or phase_axis.size < 2:
        raise ValueError("SEO/Twiss table does not contain enough grid points.")
    if not energy_axis[0] <= energy_mev <= energy_axis[-1]:
        raise ValueError(
            f"Energy {energy_mev:g} MeV is outside table range "
            f"[{energy_axis[0]:g}, {energy_axis[-1]:g}] MeV."
        )

    energy_line = np.array(
        [np.interp(energy_mev, energy_axis, values[:, j]) for j in range(values.shape[1])],
        dtype=np.float64,
    )
    period = float(phase_axis[-1] - phase_axis[0])
    if period <= 0.0:
        raise ValueError("SEO/Twiss phase axis is not strictly increasing.")
    phase = float((coordinate - phase_axis[0]) % period + phase_axis[0])
    return float(np.interp(phase, phase_axis, energy_line))


def detect_max_order(bmap_dir: Path) -> int:
    config_path = bmap_dir / "config_Bmap.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    value = config.get("config", {}).get("expand_order")
    if value is not None:
        return int(value)

    orders = []
    for path in bmap_dir.glob("Bz_order_*.npz"):
        try:
            orders.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(orders, default=0)


def validate_lattice_inputs(bmap_dir: Path, max_order: int) -> Path:
    if not bmap_dir.is_dir():
        raise FileNotFoundError(f"Bmap directory does not exist: {bmap_dir}")
    if not (bmap_dir / "config_Bmap.json").is_file():
        raise FileNotFoundError(f"Missing config_Bmap.json in {bmap_dir}")

    seo_dir = bmap_dir / "resultsSEO"
    required_seo = (
        "SEO_ini.txt",
        "SEO_r.txt",
        "SEO_pr.txt",
        "BetaFuncR.txt",
        "BetaFuncZ.txt",
        "AlphaFuncR.txt",
        "AlphaFuncZ.txt",
    )
    missing = [name for name in required_seo if not (seo_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete SEO/Twiss data in {seo_dir}; missing: {', '.join(missing)}"
        )

    for component in ("Br", "Bz", "Bfi"):
        for order in range(max_order + 1):
            path = bmap_dir / f"{component}_order_{order}.npz"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Requested max_order={max_order}, but map file is missing: {path}"
                )
    return seo_dir


def lattice_optics_at_launch(
    bmap_dir: Path,
    seo_dir: Path,
    energy_mev: float,
    azimuth_deg: float,
) -> dict[str, float | np.ndarray]:
    seo_ini, _, seo_info = LoadSEOParams(str(seo_dir / "SEO_ini.txt"), energy_mev)
    seo_r = load_numeric_table(seo_dir / "SEO_r.txt")
    seo_pr = load_numeric_table(seo_dir / "SEO_pr.txt")
    fi0 = float(np.deg2rad(azimuth_deg))
    r0 = interpolate_periodic_table(seo_r, energy_mev, fi0)
    pr0 = interpolate_periodic_table(seo_pr, energy_mev, fi0)
    beta_z = interpolate_periodic_table(
        load_numeric_table(seo_dir / "BetaFuncZ.txt"), energy_mev, azimuth_deg
    )
    beta_r = interpolate_periodic_table(
        load_numeric_table(seo_dir / "BetaFuncR.txt"), energy_mev, azimuth_deg
    )
    alpha_z = interpolate_periodic_table(
        load_numeric_table(seo_dir / "AlphaFuncZ.txt"), energy_mev, azimuth_deg
    )
    alpha_r = interpolate_periodic_table(
        load_numeric_table(seo_dir / "AlphaFuncR.txt"), energy_mev, azimuth_deg
    )
    values = {
        "fi0_rad": float(fi0),
        "r0_m": float(r0),
        "pr0": float(pr0),
        "beta_r_m": float(beta_r),
        "alpha_r": float(alpha_r),
        "beta_z_m": float(beta_z),
        "alpha_z": float(alpha_z),
        "period_s": float(seo_ini[3]),
        "seo_info": seo_info,
    }
    for name in ("beta_r_m", "beta_z_m"):
        if not np.isfinite(values[name]) or values[name] <= 0.0:
            raise ValueError(f"Invalid Twiss parameter {name}={values[name]} in {bmap_dir}")
    return values


def make_polar_launches(
    optics: dict[str, float | np.ndarray],
    settings: ScanSettings,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # rho is sampled linearly because DA is an amplitude, not an action.
    rho_axis = np.linspace(
        0.0,
        math.sqrt(settings.max_emit_pi_mm_mrad),
        settings.n_radius,
    )
    theta_axis = np.linspace(0.0, 0.5 * np.pi, settings.n_angle)

    states: list[list[float]] = []
    meta: dict[str, list[float | int]] = {
        "base_id": [],
        "angle_index": [],
        "radius_index": [],
        "phase_index": [],
        "theta_deg": [],
        "rho_sqrt_pi_mm_mrad": [],
        "emit_total_pi_mm_mrad": [],
        "emit_r_pi_mm_mrad": [],
        "emit_z_pi_mm_mrad": [],
        "phase_r_deg": [],
        "phase_z_deg": [],
        "x_r_m": [],
        "xp_r": [],
        "z_m": [],
        "zp_z": [],
    }

    beta_r = float(optics["beta_r_m"])
    alpha_r = float(optics["alpha_r"])
    beta_z = float(optics["beta_z_m"])
    alpha_z = float(optics["alpha_z"])
    r0 = float(optics["r0_m"])
    pr0 = float(optics["pr0"])
    momentum = float(FFAG_ConversionTools().Ek2P(settings.energy_mev))

    for angle_index, theta in enumerate(theta_axis):
        cos_theta = math.cos(float(theta))
        sin_theta = math.sin(float(theta))
        for radius_index, rho in enumerate(rho_axis):
            base_id = angle_index * settings.n_radius + radius_index
            emit_total = float(rho * rho)
            emit_r = emit_total * cos_theta * cos_theta
            emit_z = emit_total * sin_theta * sin_theta
            emit_r_si = emit_r * 1.0e-6
            emit_z_si = emit_z * 1.0e-6

            for phase_index, (phase_r_deg, phase_z_deg) in enumerate(
                settings.phase_pairs_deg
            ):
                phase_r = math.radians(phase_r_deg)
                phase_z = math.radians(phase_z_deg)

                sqrt_beta_emit_r = math.sqrt(beta_r * emit_r_si)
                sqrt_emit_over_beta_r = math.sqrt(emit_r_si / beta_r)
                x_r = sqrt_beta_emit_r * math.cos(phase_r)
                xp_r = -sqrt_emit_over_beta_r * (
                    alpha_r * math.cos(phase_r) + math.sin(phase_r)
                )

                sqrt_beta_emit_z = math.sqrt(beta_z * emit_z_si)
                sqrt_emit_over_beta_z = math.sqrt(emit_z_si / beta_z)
                z = sqrt_beta_emit_z * math.cos(phase_z)
                zp_z = -sqrt_emit_over_beta_z * (
                    alpha_z * math.cos(phase_z) + math.sin(phase_z)
                )

                states.append([r0 + x_r, pr0 + xp_r, z, zp_z, momentum, 0.0])
                meta["base_id"].append(base_id)
                meta["angle_index"].append(angle_index)
                meta["radius_index"].append(radius_index)
                meta["phase_index"].append(phase_index)
                meta["theta_deg"].append(math.degrees(float(theta)))
                meta["rho_sqrt_pi_mm_mrad"].append(float(rho))
                meta["emit_total_pi_mm_mrad"].append(emit_total)
                meta["emit_r_pi_mm_mrad"].append(emit_r)
                meta["emit_z_pi_mm_mrad"].append(emit_z)
                meta["phase_r_deg"].append(phase_r_deg)
                meta["phase_z_deg"].append(phase_z_deg)
                meta["x_r_m"].append(x_r)
                meta["xp_r"].append(xp_r)
                meta["z_m"].append(z)
                meta["zp_z"].append(zp_z)

    meta_arrays = {name: np.asarray(values) for name, values in meta.items()}
    return np.asarray(states, dtype=np.float64), meta_arrays


def track_fixed_energy_da(
    states: np.ndarray,
    bmap: FFAG_Bfield_analytical,
    optics: dict[str, float | np.ndarray],
    settings: ScanSettings,
    progress_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_parameters = FFAG_GlobalParameters()
    global_parameters.AddBMap(bmap)
    global_parameters.BHarmonics = None

    rk = FFAG_RungeKutta()
    n_particles = states.shape[0]
    alive = np.ones(n_particles, dtype=bool)
    loss_turn = np.full(n_particles, float(settings.turns), dtype=np.float64)
    loss_reason = np.full(n_particles, "survived", dtype="<U32")

    initial_finite = np.all(np.isfinite(states), axis=1)
    initial_pfi2 = 1.0 - states[:, 1] ** 2 - states[:, 3] ** 2
    initial_inside = (
        initial_finite
        & (initial_pfi2 > 1.0e-12)
        & (states[:, 0] >= settings.aperture_r_min_m)
        & (states[:, 0] <= settings.aperture_r_max_m)
        & (np.abs(states[:, 2]) <= settings.aperture_z_abs_m)
    )
    alive &= initial_inside
    loss_turn[~initial_inside] = 0.0
    loss_reason[~initial_finite] = "nonfinite_initial"
    loss_reason[initial_finite & (initial_pfi2 <= 1.0e-12)] = "invalid_momentum"
    outside_initial = initial_finite & (initial_pfi2 > 1.0e-12) & (~initial_inside)
    loss_reason[outside_initial] = "outside_aperture_initial"

    step_phi = 2.0 * np.pi / settings.steps_per_turn
    total_steps = settings.turns * settings.steps_per_turn
    fi0 = float(optics["fi0_rad"])

    for step_index in range(total_steps):
        active_index = np.flatnonzero(alive)
        if active_index.size == 0:
            break

        phi = fi0 + step_index * step_phi
        active_states = states[active_index]
        delta = rk.rk4_step(
            FunctionForSEOUnified,
            phi,
            active_states,
            step_phi,
            global_parameters,
        )
        active_states = active_states + delta
        states[active_index] = active_states

        finite = np.all(np.isfinite(active_states), axis=1)
        pfi2 = 1.0 - active_states[:, 1] ** 2 - active_states[:, 3] ** 2
        inside_r = (
            (active_states[:, 0] >= settings.aperture_r_min_m)
            & (active_states[:, 0] <= settings.aperture_r_max_m)
        )
        inside_z = np.abs(active_states[:, 2]) <= settings.aperture_z_abs_m
        still_alive = finite & (pfi2 > 1.0e-12) & inside_r & inside_z

        if not np.all(still_alive):
            lost_global = active_index[~still_alive]
            lost_state = active_states[~still_alive]
            lost_finite = finite[~still_alive]
            lost_pfi2 = pfi2[~still_alive]
            lost_inside_r = inside_r[~still_alive]
            lost_inside_z = inside_z[~still_alive]
            alive[lost_global] = False
            loss_turn[lost_global] = (step_index + 1) / settings.steps_per_turn
            for local_i, global_i in enumerate(lost_global):
                if not lost_finite[local_i]:
                    loss_reason[global_i] = "nonfinite"
                elif lost_pfi2[local_i] <= 1.0e-12:
                    loss_reason[global_i] = "invalid_momentum"
                elif not lost_inside_r[local_i]:
                    loss_reason[global_i] = "radial_aperture"
                elif not lost_inside_z[local_i]:
                    loss_reason[global_i] = "vertical_aperture"
                else:
                    loss_reason[global_i] = "unknown"

        completed_steps = step_index + 1
        if completed_steps % settings.steps_per_turn == 0:
            completed_turn = completed_steps // settings.steps_per_turn
            if completed_turn == 1 or completed_turn % 10 == 0 or completed_turn == settings.turns:
                print(
                    f"[{progress_label}] turn {completed_turn}/{settings.turns}, "
                    f"survived {np.count_nonzero(alive)}/{n_particles}",
                    flush=True,
                )

    return alive, loss_turn, loss_reason


def aggregate_polar_results(
    alive: np.ndarray,
    loss_turn: np.ndarray,
    meta: dict[str, np.ndarray],
    settings: ScanSettings,
) -> dict[str, np.ndarray | float]:
    n_phase = len(settings.phase_pairs_deg)
    n_base = settings.n_angle * settings.n_radius
    alive_by_base = alive.reshape(n_base, n_phase)
    turn_by_base = loss_turn.reshape(n_base, n_phase)

    stable_all = np.all(alive_by_base, axis=1)
    survival_fraction = np.mean(alive_by_base, axis=1)
    min_turn = np.min(turn_by_base, axis=1)

    first_phase = np.arange(n_base) * n_phase
    theta = meta["theta_deg"][first_phase].reshape(
        settings.n_angle, settings.n_radius
    )
    rho = meta["rho_sqrt_pi_mm_mrad"][first_phase].reshape(
        settings.n_angle, settings.n_radius
    )
    emit_total = meta["emit_total_pi_mm_mrad"][first_phase].reshape(
        settings.n_angle, settings.n_radius
    )
    emit_r = meta["emit_r_pi_mm_mrad"][first_phase].reshape(
        settings.n_angle, settings.n_radius
    )
    emit_z = meta["emit_z_pi_mm_mrad"][first_phase].reshape(
        settings.n_angle, settings.n_radius
    )
    stable_grid = stable_all.reshape(settings.n_angle, settings.n_radius)
    survival_fraction_grid = survival_fraction.reshape(
        settings.n_angle, settings.n_radius
    )
    min_turn_grid = min_turn.reshape(settings.n_angle, settings.n_radius)

    boundary_radius_index = np.zeros(settings.n_angle, dtype=np.int64)
    boundary_rho = np.zeros(settings.n_angle, dtype=np.float64)
    boundary_censored = np.zeros(settings.n_angle, dtype=bool)
    for angle_index in range(settings.n_angle):
        line = stable_grid[angle_index]
        first_unstable = np.flatnonzero(~line)
        if first_unstable.size == 0:
            last_stable_index = settings.n_radius - 1
            boundary_censored[angle_index] = True
        else:
            last_stable_index = max(0, int(first_unstable[0]) - 1)
        if not line[0]:
            last_stable_index = 0
            boundary_rho[angle_index] = 0.0
        else:
            boundary_rho[angle_index] = rho[angle_index, last_stable_index]
        boundary_radius_index[angle_index] = last_stable_index

    theta_rad = np.deg2rad(theta[:, 0])
    if theta_rad.size > 1:
        # np.trapz is retained for compatibility with the NumPy versions used
        # on the existing PyFFA/HPC environments.
        da_mean_rho = float((2.0 / np.pi) * np.trapz(boundary_rho, theta_rad))
        da_mean_emit = float(
            (2.0 / np.pi) * np.trapz(boundary_rho**2, theta_rad)
        )
    else:
        da_mean_rho = float(boundary_rho[0])
        da_mean_emit = float(boundary_rho[0] ** 2)

    return {
        "theta_deg": theta,
        "rho": rho,
        "emit_total": emit_total,
        "emit_r": emit_r,
        "emit_z": emit_z,
        "stable_grid": stable_grid,
        "survival_fraction_grid": survival_fraction_grid,
        "min_turn_grid": min_turn_grid,
        "boundary_radius_index": boundary_radius_index,
        "boundary_rho": boundary_rho,
        "boundary_censored": boundary_censored,
        "boundary_emit_total": boundary_rho**2,
        "da_mean_rho": da_mean_rho,
        "da_mean_emit": da_mean_emit,
        "da_min_rho": float(np.min(boundary_rho)),
        "da_max_rho": float(np.max(boundary_rho)),
        "horizontal_emit": float(boundary_rho[0] ** 2),
        "vertical_emit": float(boundary_rho[-1] ** 2),
        "diagonal_emit": float(
            boundary_rho[int(np.argmin(np.abs(theta[:, 0] - 45.0)))] ** 2
        ),
        "stable_base_fraction": float(np.mean(stable_grid)),
    }


def ensure_output_target(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --overwrite to replace result files."
        )
    path.mkdir(parents=True, exist_ok=True)


def write_particle_results(
    path: Path,
    states_initial: np.ndarray,
    states_final: np.ndarray,
    alive: np.ndarray,
    loss_turn: np.ndarray,
    loss_reason: np.ndarray,
    meta: dict[str, np.ndarray],
) -> None:
    fields = [
        "sample_id",
        *meta.keys(),
        "r_initial_m",
        "pr_initial",
        "z_initial_m",
        "pz_initial",
        "r_final_m",
        "pr_final",
        "z_final_m",
        "pz_final",
        "stable",
        "turns_survived",
        "loss_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(states_initial.shape[0]):
            row = {name: meta[name][index] for name in meta}
            row.update(
                {
                    "sample_id": index,
                    "r_initial_m": states_initial[index, 0],
                    "pr_initial": states_initial[index, 1],
                    "z_initial_m": states_initial[index, 2],
                    "pz_initial": states_initial[index, 3],
                    "r_final_m": states_final[index, 0],
                    "pr_final": states_final[index, 1],
                    "z_final_m": states_final[index, 2],
                    "pz_final": states_final[index, 3],
                    "stable": int(alive[index]),
                    "turns_survived": loss_turn[index],
                    "loss_reason": loss_reason[index],
                }
            )
            writer.writerow(row)


def write_grid_results(
    path: Path,
    aggregate: dict[str, np.ndarray | float],
) -> None:
    theta = np.asarray(aggregate["theta_deg"])
    rho = np.asarray(aggregate["rho"])
    emit_total = np.asarray(aggregate["emit_total"])
    emit_r = np.asarray(aggregate["emit_r"])
    emit_z = np.asarray(aggregate["emit_z"])
    stable = np.asarray(aggregate["stable_grid"])
    fraction = np.asarray(aggregate["survival_fraction_grid"])
    min_turn = np.asarray(aggregate["min_turn_grid"])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "theta_deg",
                "rho_sqrt_pi_mm_mrad",
                "emit_total_pi_mm_mrad",
                "emit_r_pi_mm_mrad",
                "emit_z_pi_mm_mrad",
                "stable_all_phases",
                "phase_survival_fraction",
                "min_turns_survived",
            ]
        )
        for angle_index in range(theta.shape[0]):
            for radius_index in range(theta.shape[1]):
                writer.writerow(
                    [
                        theta[angle_index, radius_index],
                        rho[angle_index, radius_index],
                        emit_total[angle_index, radius_index],
                        emit_r[angle_index, radius_index],
                        emit_z[angle_index, radius_index],
                        int(stable[angle_index, radius_index]),
                        fraction[angle_index, radius_index],
                        min_turn[angle_index, radius_index],
                    ]
                )


def write_boundary_results(
    path: Path,
    aggregate: dict[str, np.ndarray | float],
    optics: dict[str, float | np.ndarray],
) -> None:
    theta_deg = np.asarray(aggregate["theta_deg"])[:, 0]
    theta_rad = np.deg2rad(theta_deg)
    boundary_rho = np.asarray(aggregate["boundary_rho"])
    boundary_censored = np.asarray(aggregate["boundary_censored"], dtype=np.int8)
    boundary_emit = boundary_rho**2
    emit_r = boundary_emit * np.cos(theta_rad) ** 2
    emit_z = boundary_emit * np.sin(theta_rad) ** 2
    x_mm = np.sqrt(float(optics["beta_r_m"]) * emit_r * 1.0e-6) * 1.0e3
    z_mm = np.sqrt(float(optics["beta_z_m"]) * emit_z * 1.0e-6) * 1.0e3
    data = np.column_stack(
        (
            theta_deg,
            boundary_rho,
            boundary_emit,
            emit_r,
            emit_z,
            x_mm,
            z_mm,
            boundary_censored,
        )
    )
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=(
            "theta_deg,boundary_rho_sqrt_pi_mm_mrad,"
            "boundary_emit_total_pi_mm_mrad,boundary_emit_r_pi_mm_mrad,"
            "boundary_emit_z_pi_mm_mrad,x_phase0_mm,z_phase0_mm,"
            "scan_limit_reached"
        ),
        comments="",
        fmt="%.12e",
    )


def plot_lattice_result(
    path: Path,
    lattice_key: str,
    aggregate: dict[str, np.ndarray | float],
    settings: ScanSettings,
) -> None:
    theta_deg = np.asarray(aggregate["theta_deg"])[:, 0]
    theta_rad = np.deg2rad(theta_deg)
    rho = np.asarray(aggregate["rho"])
    min_turn = np.asarray(aggregate["min_turn_grid"])
    boundary_rho = np.asarray(aggregate["boundary_rho"])
    boundary_emit = boundary_rho**2
    boundary_censored = np.asarray(
        aggregate.get("boundary_censored", np.zeros_like(boundary_rho)),
        dtype=bool,
    )

    theta_grid = np.deg2rad(np.asarray(aggregate["theta_deg"]))
    amp_r = rho * np.cos(theta_grid)
    amp_z = rho * np.sin(theta_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    scatter = axes[0].scatter(
        amp_r,
        amp_z,
        c=min_turn,
        s=20,
        cmap="viridis",
        vmin=0.0,
        vmax=float(settings.turns),
    )
    axes[0].plot(
        boundary_rho * np.cos(theta_rad),
        boundary_rho * np.sin(theta_rad),
        "o-",
        color="red",
        ms=3.5,
        lw=2.0,
        label="origin-connected all-phase boundary",
    )
    boundary_x = boundary_rho * np.cos(theta_rad)
    boundary_y = boundary_rho * np.sin(theta_rad)
    for angle, x_value, y_value, emit_value, censored in zip(
        theta_deg,
        boundary_x,
        boundary_y,
        boundary_emit,
        boundary_censored,
    ):
        limit_prefix = "≥" if censored else ""
        label = f"{limit_prefix}{emit_value:.0f}"
        if angle >= 75.0:
            offset = (-5, 5)
            horizontal_alignment = "right"
        else:
            offset = (5, 5)
            horizontal_alignment = "left"
        axes[0].annotate(
            label,
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal_alignment,
            va="bottom",
            fontsize=6.5,
            color="darkred",
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72},
        )
    axes[0].set_xlabel(r"$\sqrt{\epsilon_r}$ [$\sqrt{\pi\,\mathrm{mm\,mrad}}$]")
    axes[0].set_ylabel(r"$\sqrt{\epsilon_z}$ [$\sqrt{\pi\,\mathrm{mm\,mrad}}$]")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(
        r"Boundary labels: $\epsilon_{\rm DA}=\epsilon_r+\epsilon_z=x^2+y^2$"
        "\n"
        + rf"min $\epsilon_{{\rm DA}}$={np.min(boundary_emit):.0f}, "
        + rf"$\langle\epsilon_{{\rm DA}}\rangle_\theta$="
        + rf"{float(aggregate.get('da_mean_emit', np.mean(boundary_emit))):.0f} "
        + r"$\pi$ mm mrad",
        fontsize=9,
    )
    colorbar = fig.colorbar(scatter, ax=axes[0])
    colorbar.set_label("minimum turns survived across phases")

    axes[1].plot(theta_deg, boundary_emit, "o-", lw=1.8)
    axes[1].axhline(
        float(aggregate.get("da_mean_emit", np.mean(boundary_emit))),
        color="0.3",
        ls="--",
        lw=1.0,
        label=r"directional mean $\langle\epsilon_{\rm DA}\rangle_\theta$",
    )
    axes[1].axhline(
        float(aggregate["da_mean_rho"]) ** 2,
        color="0.5",
        ls=":",
        lw=1.0,
        label=r"equivalent $(\langle\sqrt{\epsilon_{\rm DA}}\rangle_\theta)^2$",
    )
    for index, (angle, emit_value, censored) in enumerate(
        zip(theta_deg, boundary_emit, boundary_censored)
    ):
        limit_prefix = "≥" if censored else ""
        label = f"{limit_prefix}{emit_value:.0f}"
        axes[1].annotate(
            label,
            (angle, emit_value),
            xytext=(0, 7 if index % 2 == 0 else -11),
            textcoords="offset points",
            ha="center",
            va="bottom" if index % 2 == 0 else "top",
            fontsize=6.5,
            color="darkred",
        )
    axes[1].set_xlabel(r"polar direction $\theta$ [deg]")
    axes[1].set_ylabel(r"boundary $\epsilon_r+\epsilon_z$ [$\pi$ mm mrad]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[1].set_title(
        r"$\epsilon_r=\rho^2\cos^2\theta$, "
        r"$\epsilon_z=\rho^2\sin^2\theta$; "
        r"$\geq$ = scan limit",
        fontsize=9,
    )

    fig.suptitle(
        f"{lattice_key}: {settings.energy_mev:g} MeV, {settings.turns} turns",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_one_lattice(
    bmap_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    bmap_dir = bmap_dir.resolve()
    max_order = detect_max_order(bmap_dir) if args.max_order is None else args.max_order
    seo_dir = validate_lattice_inputs(bmap_dir, max_order)
    lattice_key = bmap_dir.name
    output_dir = output_root / lattice_key
    ensure_output_target(output_dir, args.overwrite)

    print(f"Loading lattice {lattice_key} (max_order={max_order})", flush=True)
    bmap = FFAG_Bfield_analytical(str(bmap_dir / "config_Bmap.json"), max_order)
    optics = lattice_optics_at_launch(
        bmap_dir,
        seo_dir,
        args.energy_mev,
        args.azimuth_deg,
    )

    aperture_r_min = (
        float(bmap.r_min) if args.aperture_r_min_m is None else args.aperture_r_min_m
    )
    aperture_r_max = (
        float(bmap.r_max)
        if args.aperture_r_max_m is None
        else args.aperture_r_max_m
    )
    if aperture_r_min >= aperture_r_max:
        raise ValueError(
            f"Invalid radial aperture [{aperture_r_min}, {aperture_r_max}] for {lattice_key}"
        )

    steps_per_turn = (
        int(args.steps_per_turn)
        if args.steps_per_turn is not None
        else int(math.ceil(2.0 * math.pi / args.phi_step_rad))
    )

    settings = ScanSettings(
        energy_mev=args.energy_mev,
        azimuth_deg=args.azimuth_deg,
        turns=args.turns,
        steps_per_turn=steps_per_turn,
        max_emit_pi_mm_mrad=args.max_emit_pi_mm_mrad,
        n_radius=args.n_radius,
        n_angle=args.n_angle,
        phase_pairs_deg=parse_phase_pairs(args.phase_pairs_deg),
        aperture_r_min_m=aperture_r_min,
        aperture_r_max_m=aperture_r_max,
        aperture_z_abs_m=args.aperture_z_abs_m,
    )
    states, meta = make_polar_launches(optics, settings)
    states_initial = states.copy()
    alive, loss_turn, loss_reason = track_fixed_energy_da(
        states,
        bmap,
        optics,
        settings,
        lattice_key,
    )
    aggregate = aggregate_polar_results(alive, loss_turn, meta, settings)

    particle_csv = output_dir / "da_particles.csv"
    grid_csv = output_dir / "da_polar_grid.csv"
    boundary_csv = output_dir / "da_boundary.csv"
    plot_png = output_dir / "da_survival_and_boundary.png"
    summary_json = output_dir / "da_summary.json"
    write_particle_results(
        particle_csv,
        states_initial,
        states,
        alive,
        loss_turn,
        loss_reason,
        meta,
    )
    write_grid_results(grid_csv, aggregate)
    write_boundary_results(boundary_csv, aggregate, optics)
    plot_lattice_result(plot_png, lattice_key, aggregate, settings)

    summary: dict[str, object] = {
        "lattice": lattice_key,
        "bmap_dir": str(bmap_dir),
        "max_order": max_order,
        "energy_mev": settings.energy_mev,
        "azimuth_deg": settings.azimuth_deg,
        "turns": settings.turns,
        "steps_per_turn": settings.steps_per_turn,
        "phi_step_rad": 2.0 * np.pi / settings.steps_per_turn,
        "max_emit_pi_mm_mrad": settings.max_emit_pi_mm_mrad,
        "n_radius": settings.n_radius,
        "n_angle": settings.n_angle,
        "phase_pairs_deg": settings.phase_pairs_deg,
        "particle_count": int(states.shape[0]),
        "aperture_r_min_m": settings.aperture_r_min_m,
        "aperture_r_max_m": settings.aperture_r_max_m,
        "aperture_z_abs_m": settings.aperture_z_abs_m,
        "r0_m": float(optics["r0_m"]),
        "pr0": float(optics["pr0"]),
        "beta_r_m": float(optics["beta_r_m"]),
        "alpha_r": float(optics["alpha_r"]),
        "beta_z_m": float(optics["beta_z_m"]),
        "alpha_z": float(optics["alpha_z"]),
        "da_mean_rho_sqrt_pi_mm_mrad": float(aggregate["da_mean_rho"]),
        "da_mean_equivalent_emit_pi_mm_mrad": float(aggregate["da_mean_rho"]) ** 2,
        "da_directional_mean_emit_pi_mm_mrad": float(aggregate["da_mean_emit"]),
        "da_min_rho_sqrt_pi_mm_mrad": float(aggregate["da_min_rho"]),
        "da_min_equivalent_emit_pi_mm_mrad": float(aggregate["da_min_rho"]) ** 2,
        "horizontal_boundary_emit_pi_mm_mrad": float(aggregate["horizontal_emit"]),
        "vertical_boundary_emit_pi_mm_mrad": float(aggregate["vertical_emit"]),
        "diagonal_boundary_emit_pi_mm_mrad": float(aggregate["diagonal_emit"]),
        "stable_base_fraction": float(aggregate["stable_base_fraction"]),
        "boundary_scan_limit_fraction": float(
            np.mean(np.asarray(aggregate["boundary_censored"], dtype=np.float64))
        ),
        "particle_csv": str(particle_csv),
        "grid_csv": str(grid_csv),
        "boundary_csv": str(boundary_csv),
        "plot_png": str(plot_png),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_json)
    summary["boundary_theta_deg"] = np.asarray(aggregate["theta_deg"])[:, 0]
    summary["boundary_emit_total"] = np.asarray(aggregate["boundary_emit_total"])

    print(
        f"[{lattice_key}] mean DA amplitude={float(aggregate['da_mean_rho']):.6g} "
        f"sqrt(pi mm mrad), min={float(aggregate['da_min_rho']):.6g}, "
        f"scan-limit directions="
        f"{100.0 * float(summary['boundary_scan_limit_fraction']):.1f}%",
        flush=True,
    )
    return summary


def write_comparison_outputs(output_root: Path, summaries: Iterable[dict[str, object]]) -> None:
    summaries = list(summaries)
    csv_path = output_root / "lattice_da_summary.csv"
    fields = [
        "lattice",
        "bmap_dir",
        "max_order",
        "energy_mev",
        "turns",
        "steps_per_turn",
        "phi_step_rad",
        "particle_count",
        "da_mean_rho_sqrt_pi_mm_mrad",
        "da_mean_equivalent_emit_pi_mm_mrad",
        "da_directional_mean_emit_pi_mm_mrad",
        "da_min_rho_sqrt_pi_mm_mrad",
        "da_min_equivalent_emit_pi_mm_mrad",
        "horizontal_boundary_emit_pi_mm_mrad",
        "vertical_boundary_emit_pi_mm_mrad",
        "diagonal_boundary_emit_pi_mm_mrad",
        "stable_base_fraction",
        "boundary_scan_limit_fraction",
        "summary_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})

    if not summaries:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for summary in summaries:
        axes[0].plot(
            summary["boundary_theta_deg"],
            summary["boundary_emit_total"],
            marker="o",
            ms=3,
            label=str(summary["lattice"]),
        )
    axes[0].set_xlabel(r"polar direction $\theta$ [deg]")
    axes[0].set_ylabel(r"boundary $\epsilon_r+\epsilon_z$ [$\pi$ mm mrad]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)

    labels = [str(summary["lattice"]) for summary in summaries]
    x = np.arange(len(labels))
    mean_emit = [
        float(summary["da_directional_mean_emit_pi_mm_mrad"])
        for summary in summaries
    ]
    min_emit = [float(summary["da_min_equivalent_emit_pi_mm_mrad"]) for summary in summaries]
    width = 0.38
    mean_bars = axes[1].bar(
        x - width / 2,
        mean_emit,
        width,
        label=r"directional mean $\langle\epsilon_{\rm DA}\rangle_\theta$",
    )
    min_bars = axes[1].bar(
        x + width / 2,
        min_emit,
        width,
        label=r"minimum-direction $\epsilon_{\rm DA}$",
    )
    axes[1].bar_label(mean_bars, fmt="%.0f", padding=2, fontsize=7)
    axes[1].bar_label(min_bars, fmt="%.0f", padding=2, fontsize=7)
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel(r"[$\pi$ mm mrad]")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(output_root / "lattice_da_comparison.png", dpi=180)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track a standard polar grid in normalized transverse amplitude space "
            "and compare fixed-energy dynamic aperture across analytical PyFFA lattices."
        )
    )
    parser.add_argument(
        "--bmap",
        action="append",
        required=True,
        help="Analytical Bmap directory. Repeat this option to compare lattices.",
    )
    parser.add_argument("--output-dir", default="./lattice_da_output")
    parser.add_argument("--energy-mev", type=float, default=300.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--turns", type=int, default=200)
    parser.add_argument(
        "--phi-step-rad",
        type=float,
        default=0.001,
        help=(
            "Requested RK4 azimuth step in rad. The code uses ceil(2*pi/h) "
            "steps so that every turn ends exactly at 2*pi. Default: 0.001."
        ),
    )
    parser.add_argument(
        "--steps-per-turn",
        type=int,
        default=None,
        help="Explicitly override --phi-step-rad with an integer step count.",
    )
    parser.add_argument("--max-emit-pi-mm-mrad", type=float, default=2000.0)
    parser.add_argument("--n-radius", type=int, default=41)
    parser.add_argument("--n-angle", type=int, default=19)
    parser.add_argument(
        "--phase-pairs-deg",
        default=DEFAULT_PHASE_PAIRS_DEG,
        help=(
            "Semicolon-separated (phi_r,phi_z) pairs in degrees. "
            f"Default: {DEFAULT_PHASE_PAIRS_DEG!r}."
        ),
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=None,
        help="Taylor-map order; default is config_Bmap.json expand_order.",
    )
    parser.add_argument("--aperture-r-min-m", type=float, default=None)
    parser.add_argument("--aperture-r-max-m", type=float, default=None)
    parser.add_argument("--aperture-z-abs-m", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.turns < 1:
        raise ValueError("--turns must be at least 1.")
    if args.phi_step_rad <= 0.0:
        raise ValueError("--phi-step-rad must be positive.")
    if args.steps_per_turn is not None and args.steps_per_turn < 16:
        raise ValueError("--steps-per-turn must be at least 16.")
    if args.max_emit_pi_mm_mrad <= 0.0:
        raise ValueError("--max-emit-pi-mm-mrad must be positive.")
    if args.n_radius < 2:
        raise ValueError("--n-radius must be at least 2.")
    if args.n_angle < 2:
        raise ValueError("--n-angle must be at least 2.")
    if args.aperture_z_abs_m <= 0.0:
        raise ValueError("--aperture-z-abs-m must be positive.")
    parse_phase_pairs(args.phase_pairs_deg)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_arguments(args)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for bmap_text in args.bmap:
        summaries.append(run_one_lattice(Path(bmap_text), output_root, args))
    write_comparison_outputs(output_root, summaries)
    print(f"DA comparison summary: {output_root / 'lattice_da_summary.csv'}")
    print(f"DA comparison plot   : {output_root / 'lattice_da_comparison.png'}")


if __name__ == "__main__":
    main()
