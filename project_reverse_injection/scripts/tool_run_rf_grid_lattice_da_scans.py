"""Run the 4 x 3 DF-ratio/F-angle lattice DA grid strictly in series.

The twelve cases are the two-parameter grid already used in the PPT:

    R = 0.265074, 0.365074, 0.465074, 0.565074
    F = 0.0, 0.5, 0.75 deg

The run order is row-major in the intended 3 x 4 PPT layout: one F value per
row and four R values per row.  Every case uses ``max_order=2`` (field terms
through fifth order).  Missing order-1/order-2 map files are generated with a
strictly serial generator before tracking.  No worker pool, multiprocessing,
or concurrent subprocess is used; only one lattice is prepared/tracked at a
time.

Run from the repository root::

    python -u project_reverse_injection/scripts/tool_run_rf_grid_lattice_da_scans.py

The earlier max_order=0 results are not reused.  Use ``--overwrite`` to
recompute completed max_order=2 DA results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tool_scan_lattice_da as da  # noqa: E402
from FFAG_HighOrder import (  # noqa: E402
    BmapParams_to_ConvexParams,
    Generate_Bmaps_Serial,
    calculate_s,
)


MAX_ORDER = 2
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "project_reverse_injection"
    / "lattice_da_200turn_RF_grid_12_order2_300MeV"
)

R_CASES = (
    (0.265074, "265"),
    (0.365074, "365"),
    (0.465074, "465"),
    (0.565074, "565"),
)
F_CASES = (
    (0.0, "F0"),
    (0.5, "Fp0p5"),
    (0.75, "Fp0p75"),
)

RESULT_FILES = (
    "da_particles.csv",
    "da_polar_grid.csv",
    "da_boundary.csv",
    "da_survival_and_boundary.png",
    "da_summary.json",
)


@dataclass(frozen=True)
class GridCase:
    row: int
    column: int
    r_ratio: float
    f_deg: float
    lattice: str


def build_grid_cases() -> tuple[GridCase, ...]:
    cases = []
    for row, (f_deg, f_tag) in enumerate(F_CASES, start=1):
        for column, (r_ratio, r_tag) in enumerate(R_CASES, start=1):
            cases.append(
                GridCase(
                    row=row,
                    column=column,
                    r_ratio=r_ratio,
                    f_deg=f_deg,
                    lattice=(
                        f"Bmap_FD16_k7p63_B1p65_R{r_tag}_{f_tag}_s49p97"
                    ),
                )
            )
    return tuple(cases)


GRID_CASES = build_grid_cases()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 12-point R/DF-ratio and F-sector-angle DA grid one "
            "lattice at a time."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--row",
        type=int,
        choices=(1, 2, 3),
        help=(
            "Run one PPT row only: 1=F0, 2=F0.5 deg, 3=F0.75 deg. "
            "Default: run all three rows."
        ),
    )
    parser.add_argument("--energy-mev", type=float, default=300.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--turns", type=int, default=200)
    parser.add_argument("--phi-step-rad", type=float, default=0.001)
    parser.add_argument("--max-emit-pi-mm-mrad", type=float, default=4000.0)
    parser.add_argument("--n-radius", type=int, default=13)
    parser.add_argument("--n-angle", type=int, default=7)
    parser.add_argument(
        "--phase-pairs-deg",
        default=da.DEFAULT_PHASE_PAIRS_DEG,
        help="Same format as tool_scan_lattice_da.py.",
    )
    parser.add_argument("--aperture-r-min-m", type=float, default=10.8)
    parser.add_argument("--aperture-r-max-m", type=float, default=12.0)
    parser.add_argument("--aperture-z-abs-m", type=float, default=0.10)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute completed max_order=2 DA results.",
    )
    parser.add_argument(
        "--regenerate-maps",
        action="store_true",
        help=(
            "Regenerate order 0/1/2 field-map files even when all are present."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate/validate all max_order=2 maps serially, then stop before DA.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs and print the serial 3 x 4 run plan only.",
    )
    return parser


def make_da_args(cli: argparse.Namespace, overwrite: bool) -> argparse.Namespace:
    return argparse.Namespace(
        energy_mev=cli.energy_mev,
        azimuth_deg=cli.azimuth_deg,
        turns=cli.turns,
        phi_step_rad=cli.phi_step_rad,
        steps_per_turn=None,
        max_emit_pi_mm_mrad=cli.max_emit_pi_mm_mrad,
        n_radius=cli.n_radius,
        n_angle=cli.n_angle,
        phase_pairs_deg=cli.phase_pairs_deg,
        max_order=MAX_ORDER,
        aperture_r_min_m=cli.aperture_r_min_m,
        aperture_r_max_m=cli.aperture_r_max_m,
        aperture_z_abs_m=cli.aperture_z_abs_m,
        overwrite=overwrite,
    )


def expected_settings(cli: argparse.Namespace) -> dict[str, object]:
    steps_per_turn = int(math.ceil(2.0 * math.pi / cli.phi_step_rad))
    return {
        "max_order": MAX_ORDER,
        "energy_mev": float(cli.energy_mev),
        "azimuth_deg": float(cli.azimuth_deg),
        "turns": int(cli.turns),
        "steps_per_turn": steps_per_turn,
        "max_emit_pi_mm_mrad": float(cli.max_emit_pi_mm_mrad),
        "n_radius": int(cli.n_radius),
        "n_angle": int(cli.n_angle),
        "phase_pairs_deg": [
            list(pair) for pair in da.parse_phase_pairs(cli.phase_pairs_deg)
        ],
        "aperture_r_min_m": float(cli.aperture_r_min_m),
        "aperture_r_max_m": float(cli.aperture_r_max_m),
        "aperture_z_abs_m": float(cli.aperture_z_abs_m),
    }


def values_match(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def load_completed_summary(
    case_dir: Path,
    cli: argparse.Namespace,
    *,
    strict: bool,
) -> dict[str, object] | None:
    if not all((case_dir / name).is_file() for name in RESULT_FILES):
        return None

    summary_path = case_dir / "da_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mismatches = []
    for key, expected in expected_settings(cli).items():
        if not values_match(summary.get(key), expected):
            mismatches.append(
                f"{key}: file={summary.get(key)!r}, requested={expected!r}"
            )
    if mismatches:
        if not strict:
            return None
        raise ValueError(
            f"Existing result settings do not match {case_dir.name}:\n  "
            + "\n  ".join(mismatches)
            + "\nUse --overwrite to recalculate the 12-point grid."
        )

    boundary_path = case_dir / "da_boundary.csv"
    boundary = np.genfromtxt(boundary_path, delimiter=",", names=True)
    summary["summary_json"] = str(summary_path.resolve())
    summary["boundary_theta_deg"] = np.atleast_1d(boundary["theta_deg"])
    summary["boundary_emit_total"] = np.atleast_1d(
        boundary["boundary_emit_total_pi_mm_mrad"]
    )
    if "da_directional_mean_emit_pi_mm_mrad" not in summary:
        theta_rad = np.deg2rad(summary["boundary_theta_deg"])
        boundary_emit = summary["boundary_emit_total"]
        if theta_rad.size > 1:
            summary["da_directional_mean_emit_pi_mm_mrad"] = float(
                (2.0 / np.pi) * np.trapz(boundary_emit, theta_rad)
            )
        else:
            summary["da_directional_mean_emit_pi_mm_mrad"] = float(
                boundary_emit[0]
            )
    summary["particle_csv"] = str((case_dir / "da_particles.csv").resolve())
    summary["grid_csv"] = str((case_dir / "da_polar_grid.csv").resolve())
    summary["boundary_csv"] = str(boundary_path.resolve())
    summary["plot_png"] = str(
        (case_dir / "da_survival_and_boundary.png").resolve()
    )
    return summary


def validate_lattice(case: GridCase) -> Path:
    bmap_dir = REPO_ROOT / "Bmaps" / case.lattice
    da.validate_lattice_inputs(bmap_dir, 0)
    return bmap_dir


def high_order_maps_complete(bmap_dir: Path) -> bool:
    return all(
        (bmap_dir / f"{component}_order_{order}.npz").is_file()
        for component in ("Br", "Bz", "Bfi")
        for order in range(MAX_ORDER + 1)
    )


def update_config_order(bmap_dir: Path) -> None:
    config_path = bmap_dir / "config_Bmap.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("config", {}).get("expand_order", -1)) == MAX_ORDER:
        return
    config.setdefault("config", {})["expand_order"] = MAX_ORDER
    temp_path = config_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(config, indent=4), encoding="utf-8")
    temp_path.replace(config_path)


def save_npz_atomic(path: Path, key: str, values: np.ndarray) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as stream:
        np.savez(stream, **{key: values})
    temp_path.replace(path)


def generate_order2_maps_serial(bmap_dir: Path) -> None:
    config_path = bmap_dir / "config_Bmap.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bmap = config["Bmap"]
    rmin, rmax, rstep = map(float, bmap["rmin_max_step_m"])
    sector_period = 2.0 * np.pi / int(bmap["NSector"])
    theta_step = float(bmap["theta_step_rad"])
    r_axis = np.linspace(
        rmin,
        rmax,
        int(round((rmax - rmin) / rstep)) + 1,
    )
    fi_axis = np.linspace(
        0.0,
        sector_period,
        int(round(sector_period / theta_step)) + 1,
    )
    s_values = calculate_s(
        r_axis,
        0.0,
        float(bmap["SpiralAngle_deg"]),
    )
    s_fit = Polynomial.fit(r_axis, s_values, 5)
    convex = BmapParams_to_ConvexParams(bmap)
    coefficients = bmap.get("coefficients_tupel")
    if coefficients is None:
        coefficients = (float(bmap["B0_T"]), float(bmap["k_value"]))

    print(
        f"Generating max_order={MAX_ORDER} maps serially: {bmap_dir.name} "
        f"(Nr={r_axis.size}, Nphi={fi_axis.size})",
        flush=True,
    )
    br_maps, bz_maps, bfi_maps = Generate_Bmaps_Serial(
        MAX_ORDER,
        r_axis,
        fi_axis,
        convex,
        s_fit,
        coefficients,
    )
    for order in range(MAX_ORDER + 1):
        save_npz_atomic(
            bmap_dir / f"Br_order_{order}.npz",
            "Br_Taylor_coeff",
            br_maps[order],
        )
        save_npz_atomic(
            bmap_dir / f"Bz_order_{order}.npz",
            "Bz_Taylor_coeff",
            bz_maps[order],
        )
        save_npz_atomic(
            bmap_dir / f"Bfi_order_{order}.npz",
            "Bfi_Taylor_coeff",
            bfi_maps[order],
        )
    update_config_order(bmap_dir)
    da.validate_lattice_inputs(bmap_dir, MAX_ORDER)


def ensure_order2_maps(bmap_dir: Path, regenerate: bool) -> None:
    if regenerate or not high_order_maps_complete(bmap_dir):
        generate_order2_maps_serial(bmap_dir)
    else:
        update_config_order(bmap_dir)
        da.validate_lattice_inputs(bmap_dir, MAX_ORDER)


def write_grid_summary(
    output_path: Path,
    case_summaries: list[tuple[GridCase, dict[str, object]]],
) -> None:
    fields = [
        "slot",
        "row",
        "column",
        "R_ratio",
        "F_deg",
        "lattice",
        "max_order",
        "energy_mev",
        "turns",
        "steps_per_turn",
        "phi_step_rad",
        "particle_count",
        "da_mean_equivalent_emit_pi_mm_mrad",
        "da_min_equivalent_emit_pi_mm_mrad",
        "horizontal_boundary_emit_pi_mm_mrad",
        "vertical_boundary_emit_pi_mm_mrad",
        "diagonal_boundary_emit_pi_mm_mrad",
        "stable_base_fraction",
        "boundary_scan_limit_fraction",
        "plot_png",
        "summary_json",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for slot, (case, summary) in enumerate(case_summaries, start=1):
            record = {field: summary.get(field, "") for field in fields}
            record.update(
                {
                    "slot": slot,
                    "row": case.row,
                    "column": case.column,
                    "R_ratio": case.r_ratio,
                    "F_deg": case.f_deg,
                    "lattice": case.lattice,
                }
            )
            writer.writerow(record)


def planned_action(
    case: GridCase,
    bmap_dir: Path,
    output_root: Path,
    cli: argparse.Namespace,
) -> str:
    map_action = (
        "regenerate order2 maps"
        if cli.regenerate_maps
        else "prepare order2 maps"
        if not high_order_maps_complete(bmap_dir)
        else "order2 maps ready"
    )
    if not cli.overwrite:
        current = load_completed_summary(
            output_root / case.lattice,
            cli,
            strict=True,
        )
        if current is not None:
            return f"{map_action}; resume DA"
    return f"{map_action}; run DA"


def main() -> None:
    parser = build_parser()
    cli = parser.parse_args()
    da.validate_arguments(make_da_args(cli, overwrite=cli.overwrite))
    output_root = cli.output_dir.resolve()
    selected_cases = tuple(
        case for case in GRID_CASES if cli.row is None or case.row == cli.row
    )

    validated = {case.lattice: validate_lattice(case) for case in selected_cases}
    run_label = "all rows" if cli.row is None else f"row {cli.row}"
    print(f"Strictly serial R/F DA grid ({run_label}):")
    for slot, case in enumerate(selected_cases, start=1):
        action = planned_action(case, validated[case.lattice], output_root, cli)
        print(
            f"  {slot:02d}/{len(selected_cases):02d}  "
            f"row={case.row} col={case.column}  "
            f"R={case.r_ratio:.6f}  F={case.f_deg:.2f} deg  "
            f"[{action}]  {case.lattice}"
        )

    if cli.dry_run:
        print("Dry run finished; no tracking output was written.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    case_summaries: list[tuple[GridCase, dict[str, object]]] = []

    for slot, case in enumerate(selected_cases, start=1):
        bmap_dir = validated[case.lattice]
        print(
            f"[{slot:02d}/{len(selected_cases):02d}] ensure "
            f"max_order={MAX_ORDER}: {case.lattice}"
        )
        ensure_order2_maps(bmap_dir, regenerate=cli.regenerate_maps)
        if cli.prepare_only:
            continue

        case_dir = output_root / case.lattice
        summary = None
        if not cli.overwrite:
            summary = load_completed_summary(case_dir, cli, strict=True)
        if summary is not None:
            print(
                f"[{slot:02d}/{len(selected_cases):02d}] "
                f"resume completed: {case.lattice}"
            )
        else:
            print(
                f"[{slot:02d}/{len(selected_cases):02d}] start serial case: "
                f"R={case.r_ratio:.6f}, "
                f"F={case.f_deg:.2f} deg, max_order={MAX_ORDER}"
            )
            partial_output = case_dir.exists() and any(case_dir.iterdir())
            run_args = make_da_args(
                cli,
                overwrite=bool(cli.overwrite or partial_output),
            )
            summary = da.run_one_lattice(
                bmap_dir,
                output_root,
                run_args,
            )
        if summary is None:
            raise RuntimeError(f"No DA summary produced for {case.lattice}")
        case_summaries.append((case, summary))

    if cli.prepare_only:
        print(
            f"All {len(selected_cases)} selected max_order={MAX_ORDER} maps "
            f"are ready; DA was not started."
        )
        return

    summaries = [summary for _case, summary in case_summaries]
    if cli.row is None:
        comparison_dir = output_root
        grid_summary_path = output_root / "rf_grid_da_summary.csv"
    else:
        f_deg = selected_cases[0].f_deg
        comparison_dir = output_root / f"row_{cli.row}_F{f_deg:g}"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        grid_summary_path = output_root / f"rf_grid_da_summary_row{cli.row}.csv"
    da.write_comparison_outputs(comparison_dir, summaries)
    write_grid_summary(grid_summary_path, case_summaries)
    manifest = {
        "serial": True,
        "max_order": MAX_ORDER,
        "selected_row": cli.row,
        "layout": {"rows": 3, "columns": 4},
        "R_values": [value for value, _tag in R_CASES],
        "F_values_deg": [value for value, _tag in F_CASES],
        "case_count": len(selected_cases),
        "settings": expected_settings(cli),
        "output_dir": str(output_root),
    }
    manifest_name = "run_manifest.json" if cli.row is None else f"run_manifest_row{cli.row}.json"
    (output_root / manifest_name).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"All {len(selected_cases)} selected serial R/F DA cases finished: {output_root}")
    print(f"Grid summary: {grid_summary_path}")
    print(f"Comparison: {comparison_dir / 'lattice_da_comparison.png'}")


if __name__ == "__main__":
    main()
