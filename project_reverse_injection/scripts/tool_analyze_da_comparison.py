"""Aggregate completed PyFFA DA scans into comparison tables and plots."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STRIP_RATES_PERCENT = {265: 0.142, 365: 0.184, 465: 0.235}


def lattice_r_code(name: str) -> int:
    match = re.search(r"_R(\d+)_", name)
    if not match:
        raise ValueError(f"Cannot extract R code from lattice name: {name}")
    return int(match.group(1))


def load_tunes(summary: dict[str, object]) -> tuple[float, float]:
    seo_path = Path(str(summary["bmap_dir"])) / "resultsSEO" / "SEO_ini.txt"
    data = np.loadtxt(seo_path, skiprows=2)
    energy = float(summary["energy_mev"])
    index = int(np.argmin(np.abs(data[:, 1] - energy)))
    return float(data[index, 2]), float(data[index, 3])


def load_result(lattice_dir: Path) -> dict[str, object]:
    summary = json.loads((lattice_dir / "da_summary.json").read_text(encoding="utf-8"))
    boundary = np.genfromtxt(
        lattice_dir / "da_boundary.csv", delimiter=",", names=True, dtype=float
    )
    particles = np.genfromtxt(
        lattice_dir / "da_particles.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    r_code = lattice_r_code(str(summary["lattice"]))
    qr, qz = load_tunes(summary)
    return {
        "summary": summary,
        "boundary": boundary,
        "particles": particles,
        "r_code": r_code,
        "label": f"R={r_code / 1000:.3f}",
        "strip_rate": STRIP_RATES_PERCENT.get(r_code, np.nan),
        "qr": qr,
        "qz": qz,
    }


def loss_bins(particles: np.ndarray, turns: int) -> dict[str, int]:
    stable = particles["stable"].astype(int) == 1
    loss_turn = particles["turns_survived"].astype(float)
    return {
        "initial": int(np.sum((~stable) & (loss_turn == 0.0))),
        "turn_0_to_1": int(np.sum((~stable) & (loss_turn > 0.0) & (loss_turn <= 1.0))),
        "turn_1_to_10": int(np.sum((~stable) & (loss_turn > 1.0) & (loss_turn <= 10.0))),
        "turn_10_to_50": int(np.sum((~stable) & (loss_turn > 10.0) & (loss_turn <= 50.0))),
        "turn_50_to_100": int(np.sum((~stable) & (loss_turn > 50.0) & (loss_turn <= 100.0))),
        "turn_100_to_200": int(np.sum((~stable) & (loss_turn > 100.0) & (loss_turn <= turns))),
        "survived": int(np.sum(stable)),
    }


def write_tables(root: Path, results: list[dict[str, object]]) -> None:
    summary_fields = [
        "lattice",
        "r_value",
        "strip_rate_percent",
        "qr",
        "qz",
        "turns",
        "phi_step_rad",
        "particle_count",
        "survived_particles",
        "survival_fraction_particles",
        "stable_base_fraction",
        "mean_equivalent_emit_pi_mm_mrad",
        "min_emit_pi_mm_mrad",
        "horizontal_emit_pi_mm_mrad",
        "diagonal_emit_pi_mm_mrad",
        "vertical_emit_pi_mm_mrad",
        "boundary_scan_limit_fraction",
    ]
    with (root / "lattice_da_summary_detailed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for result in results:
            summary = result["summary"]
            particles = result["particles"]
            survived = int(np.sum(particles["stable"].astype(int) == 1))
            writer.writerow(
                {
                    "lattice": summary["lattice"],
                    "r_value": result["r_code"] / 1000.0,
                    "strip_rate_percent": result["strip_rate"],
                    "qr": result["qr"],
                    "qz": result["qz"],
                    "turns": summary["turns"],
                    "phi_step_rad": summary["phi_step_rad"],
                    "particle_count": summary["particle_count"],
                    "survived_particles": survived,
                    "survival_fraction_particles": survived / int(summary["particle_count"]),
                    "stable_base_fraction": summary["stable_base_fraction"],
                    "mean_equivalent_emit_pi_mm_mrad": summary[
                        "da_mean_equivalent_emit_pi_mm_mrad"
                    ],
                    "min_emit_pi_mm_mrad": summary[
                        "da_min_equivalent_emit_pi_mm_mrad"
                    ],
                    "horizontal_emit_pi_mm_mrad": summary[
                        "horizontal_boundary_emit_pi_mm_mrad"
                    ],
                    "diagonal_emit_pi_mm_mrad": summary[
                        "diagonal_boundary_emit_pi_mm_mrad"
                    ],
                    "vertical_emit_pi_mm_mrad": summary[
                        "vertical_boundary_emit_pi_mm_mrad"
                    ],
                    "boundary_scan_limit_fraction": summary[
                        "boundary_scan_limit_fraction"
                    ],
                }
            )

    loss_fields = ["lattice", *loss_bins(results[0]["particles"], 200).keys()]
    with (root / "lattice_da_loss_bins.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=loss_fields)
        writer.writeheader()
        for result in results:
            bins = loss_bins(result["particles"], int(result["summary"]["turns"]))
            writer.writerow({"lattice": result["summary"]["lattice"], **bins})

    theta = results[0]["boundary"]["theta_deg"]
    boundary_fields = ["theta_deg", *[str(result["label"]) for result in results]]
    with (root / "lattice_da_boundary_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(boundary_fields)
        for i, theta_value in enumerate(theta):
            writer.writerow(
                [theta_value]
                + [result["boundary"]["boundary_emit_total_pi_mm_mrad"][i] for result in results]
            )


def survival_curve(particles: np.ndarray, turns: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(0, turns + 1, dtype=float)
    stable = particles["stable"].astype(int) == 1
    loss_turn = particles["turns_survived"].astype(float)
    y = np.array(
        [np.sum(stable | (loss_turn > turn)) for turn in x], dtype=float
    )
    return x, y


def make_plot(root: Path, results: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors

    for i, result in enumerate(results):
        boundary = result["boundary"]
        axes[0, 0].plot(
            boundary["theta_deg"],
            boundary["boundary_emit_total_pi_mm_mrad"],
            "o-",
            color=colors[i],
            label=result["label"],
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("polar direction theta [deg]")
    axes[0, 0].set_ylabel("200-turn boundary emit [pi mm mrad]")
    axes[0, 0].set_title("Origin-connected all-phase DA boundary")
    axes[0, 0].grid(True, which="both", alpha=0.25)
    axes[0, 0].legend()

    for i, result in enumerate(results):
        x, y = survival_curve(result["particles"], int(result["summary"]["turns"]))
        axes[0, 1].step(x, y, where="post", color=colors[i], label=result["label"])
    axes[0, 1].set_xlabel("turn")
    axes[0, 1].set_ylabel("surviving phase particles / 364")
    axes[0, 1].set_title("Cumulative survival")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend()

    labels = [str(result["label"]) for result in results]
    x = np.arange(len(results))
    width = 0.25
    metrics = [
        ("da_min_equivalent_emit_pi_mm_mrad", "minimum"),
        ("diagonal_boundary_emit_pi_mm_mrad", "45 deg"),
        ("vertical_boundary_emit_pi_mm_mrad", "vertical"),
    ]
    for j, (key, label) in enumerate(metrics):
        axes[1, 0].bar(
            x + (j - 1) * width,
            [float(result["summary"][key]) for result in results],
            width,
            label=label,
        )
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("boundary emit [pi mm mrad]")
    axes[1, 0].set_title("Critical 200-turn DA metrics")
    axes[1, 0].grid(True, axis="y", which="both", alpha=0.25)
    axes[1, 0].legend()

    for i, result in enumerate(results):
        strip_rate = float(result["strip_rate"])
        min_emit = float(result["summary"]["da_min_equivalent_emit_pi_mm_mrad"])
        axes[1, 1].scatter(strip_rate, min_emit, s=90, color=colors[i])
        axes[1, 1].annotate(
            str(result["label"]),
            (strip_rate, min_emit),
            xytext=(6, 5),
            textcoords="offset points",
        )
    axes[1, 1].axvline(0.2, color="0.3", ls="--", lw=1.2, label="strip limit 0.2%")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("stripping rate [%]")
    axes[1, 1].set_ylabel("minimum 200-turn DA [pi mm mrad]")
    axes[1, 1].set_title("DA versus stripping-rate constraint")
    axes[1, 1].grid(True, which="both", alpha=0.25)
    axes[1, 1].legend()

    fig.suptitle("300 MeV lattice DA comparison: 200 turns, h_phi=0.001 rad")
    fig.savefig(root / "lattice_da_comparison_detailed.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root")
    args = parser.parse_args()
    root = Path(args.result_root).resolve()
    lattice_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "da_summary.json").is_file()
    )
    results = sorted((load_result(path) for path in lattice_dirs), key=lambda item: item["r_code"])
    if not results:
        raise FileNotFoundError(f"No completed DA results below {root}")
    write_tables(root, results)
    make_plot(root, results)
    print(root / "lattice_da_summary_detailed.csv")
    print(root / "lattice_da_comparison_detailed.png")


if __name__ == "__main__":
    main()
