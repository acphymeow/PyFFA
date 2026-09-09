"""Regenerate completed DA figures with boundary-emittance annotations.

This is a plotting-only utility: it reads existing CSV/JSON outputs and never
reruns particle tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import tool_scan_lattice_da as da


REQUIRED_FILES = ("da_polar_grid.csv", "da_boundary.csv", "da_summary.json")


def load_completed_case(case_dir: Path) -> tuple[dict[str, object], dict[str, object], da.ScanSettings]:
    summary_path = case_dir / "da_summary.json"
    summary: dict[str, object] = json.loads(summary_path.read_text(encoding="utf-8"))
    grid = np.atleast_1d(
        np.genfromtxt(case_dir / "da_polar_grid.csv", delimiter=",", names=True)
    )
    boundary = np.atleast_1d(
        np.genfromtxt(case_dir / "da_boundary.csv", delimiter=",", names=True)
    )

    n_angle = int(summary["n_angle"])
    n_radius = int(summary["n_radius"])
    expected_size = n_angle * n_radius
    if grid.size != expected_size:
        raise ValueError(
            f"{case_dir}: polar grid has {grid.size} rows; expected {expected_size}."
        )

    theta_grid = np.asarray(grid["theta_deg"], dtype=float).reshape(n_angle, n_radius)
    rho_grid = np.asarray(
        grid["rho_sqrt_pi_mm_mrad"], dtype=float
    ).reshape(n_angle, n_radius)
    boundary_rho = np.asarray(
        boundary["boundary_rho_sqrt_pi_mm_mrad"], dtype=float
    )
    boundary_emit = np.asarray(
        boundary["boundary_emit_total_pi_mm_mrad"], dtype=float
    )
    boundary_censored = np.asarray(boundary["scan_limit_reached"], dtype=bool)
    theta_rad = np.deg2rad(np.asarray(boundary["theta_deg"], dtype=float))
    if theta_rad.size > 1:
        mean_emit = float((2.0 / np.pi) * np.trapz(boundary_emit, theta_rad))
    else:
        mean_emit = float(boundary_emit[0])

    aggregate: dict[str, object] = {
        "theta_deg": theta_grid,
        "rho": rho_grid,
        "min_turn_grid": np.asarray(
            grid["min_turns_survived"], dtype=float
        ).reshape(n_angle, n_radius),
        "boundary_rho": boundary_rho,
        "boundary_censored": boundary_censored,
        "boundary_emit_total": boundary_emit,
        "da_mean_rho": float(summary["da_mean_rho_sqrt_pi_mm_mrad"]),
        "da_mean_emit": mean_emit,
    }
    phase_pairs = tuple(
        (float(pair[0]), float(pair[1]))
        for pair in summary["phase_pairs_deg"]
    )
    settings = da.ScanSettings(
        energy_mev=float(summary["energy_mev"]),
        azimuth_deg=float(summary["azimuth_deg"]),
        turns=int(summary["turns"]),
        steps_per_turn=int(summary["steps_per_turn"]),
        max_emit_pi_mm_mrad=float(summary["max_emit_pi_mm_mrad"]),
        n_radius=n_radius,
        n_angle=n_angle,
        phase_pairs_deg=phase_pairs,
        aperture_r_min_m=float(summary["aperture_r_min_m"]),
        aperture_r_max_m=float(summary["aperture_r_max_m"]),
        aperture_z_abs_m=float(summary["aperture_z_abs_m"]),
    )

    summary["summary_json"] = str(summary_path.resolve())
    summary["boundary_theta_deg"] = np.asarray(boundary["theta_deg"], dtype=float)
    summary["boundary_emit_total"] = boundary_emit
    summary["da_directional_mean_emit_pi_mm_mrad"] = mean_emit
    return summary, aggregate, settings


def find_cases(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and all((path / filename).is_file() for filename in REQUIRED_FILES)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot completed DA outputs with emittance labels; no tracking."
    )
    parser.add_argument("root", type=Path, help="Directory containing completed DA case folders.")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help="Optional directory in which to regenerate the comparison plot/CSV.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    cases = find_cases(root)
    if not cases:
        raise FileNotFoundError(f"No completed DA case folders found under {root}")

    summaries: list[dict[str, object]] = []
    for case_dir in cases:
        summary, aggregate, settings = load_completed_case(case_dir)
        output = case_dir / "da_survival_and_boundary.png"
        da.plot_lattice_result(output, str(summary["lattice"]), aggregate, settings)
        summaries.append(summary)
        print(f"Replotted: {output}")

    if args.comparison_dir is not None:
        comparison_dir = args.comparison_dir.resolve()
        comparison_dir.mkdir(parents=True, exist_ok=True)
        da.write_comparison_outputs(comparison_dir, summaries)
        print(f"Replotted: {comparison_dir / 'lattice_da_comparison.png'}")


if __name__ == "__main__":
    main()
