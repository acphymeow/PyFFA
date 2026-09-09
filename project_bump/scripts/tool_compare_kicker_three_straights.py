import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from main_track import load_bmap_auto
from tool_plot_bmap_with_kickers import evaluate_base_field
from tool_compare_kicker_splits import (
    INSTALLATION_GAP_M,
    LOW_FIELD_THRESHOLD_T,
    MAGNET_LENGTH_M,
    advance_by_s,
    arc_length_between,
    closed_orbit_geometry,
    find_downstream_low_field_window,
    make_group,
    save_track_csv,
)
from tool_optimize_septum_per_kicker_split import candidate_in_straight
from tool_track_extraction_kickers import (
    load_json,
    load_seo,
    make_closed_orbit_particle,
    resolve_config_path,
    run_case,
)


ENERGY_MEV = 600.0
START_PHI_DEG = 140.0
TIME_STEP_NS = 0.02
STRAIGHT_BOUNDARIES_DEG = (146.25, 168.75, 191.25)
GROUP_STARTS_DEG = (146.521143, 169.021176, 191.521176)
FIRST_SEPTUM_STRAIGHT_DEG = 213.75
STRAIGHT_PERIOD_DEG = 22.5
SEPTUM_LENGTH_M = 0.5
SPLITS = ((2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2))
COLORS = ("#2563eb", "#e11d48", "#7c3aed", "#ea580c")


def make_three_straight_kickers(split, base_config, phi_axis_deg, s_axis):
    template = base_config["BumpKicker"]["kickers"][0]
    kickers = []
    next_index = 1
    for count, group_start in zip(split, GROUP_STARTS_DEG):
        group = make_group(
            count, group_start, template, next_index, phi_axis_deg, s_axis
        )
        kickers.extend(group)
        next_index += count
    return kickers


def group_feasibility(kickers, split, windows, phi_axis_deg, s_axis):
    feasibility = []
    cursor = 0
    for count, window in zip(split, windows):
        group = kickers[cursor:cursor + count]
        start = float(group[0]["phi_start_deg"])
        end = float(group[-1]["phi_start_deg"] + group[-1]["phi_span_deg"])
        required = arc_length_between(phi_axis_deg, s_axis, start, end)
        available = arc_length_between(
            phi_axis_deg, s_axis, start, window["end_deg"]
        )
        overhang = max(0.0, required - available)
        feasibility.append(
            {
                "count": count,
                "start_deg": start,
                "end_deg": end,
                "required_span_m": required,
                "available_from_first_kicker_m": available,
                "overhang_m": overhang,
                "fits_low_field_window": overhang <= 1.0e-9,
            }
        )
        cursor += count
    return feasibility


def optimize_septum(track, seo, bmap):
    last_track_deg = float(np.max(track["phi_unwrapped_deg"]))
    candidates = []
    for boundary in np.arange(
        FIRST_SEPTUM_STRAIGHT_DEG, last_track_deg - 2.0, STRAIGHT_PERIOD_DEG
    ):
        candidate = candidate_in_straight(track, seo, bmap, float(boundary))
        if candidate is not None and candidate["min_separation_mm"] > 0.0:
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError("No outward 0.5 m low-field septum candidate was found.")
    return max(candidates, key=lambda item: item["min_separation_mm"]), candidates


def plot_layout(output, results, windows, bmap, seo):
    phi_deg = np.linspace(145.0, 199.8, 4201)
    phi_rad = np.deg2rad(phi_deg)
    r = np.interp(np.mod(phi_rad, 2.0 * np.pi), seo["phi_axis"], seo["r_row"])
    bz, _, _ = evaluate_base_field(
        bmap, 0, r, phi_rad, np.zeros_like(r)
    )

    fig, axes = plt.subplots(
        len(results), 1, figsize=(14.5, 2.5 * len(results) + 0.5),
        sharex=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, result in zip(axes, results):
        ax.plot(phi_deg, bz, color="#475569", lw=1.0, label="base Bz on SEO")
        ax.axhline(LOW_FIELD_THRESHOLD_T, color="#94a3b8", ls=":", lw=0.8)
        ax.axhline(-LOW_FIELD_THRESHOLD_T, color="#94a3b8", ls=":", lw=0.8)
        for window_i, window in enumerate(windows):
            ax.axvspan(
                window["start_deg"], window["end_deg"], color="#bbf7d0",
                alpha=0.36,
                label=f"|Bz| < {LOW_FIELD_THRESHOLD_T:.3f} T"
                if window_i == 0 else None,
            )
        for kicker_i, kicker in enumerate(result["kickers"]):
            start = float(kicker["phi_start_deg"])
            end = start + float(kicker["phi_span_deg"])
            ax.axvspan(
                start, end, color=result["color"], alpha=0.70,
                label="0.20 m kicker" if kicker_i == 0 else None,
            )
            ax.text(
                0.5 * (start + end), 0.88, kicker["name"],
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                rotation=90, fontsize=8, fontweight="bold",
            )
        total_overhang_mm = 1.0e3 * sum(
            item["overhang_m"] for item in result["feasibility"]
        )
        ax.text(
            0.99, 0.08,
            f"{result['label']}: "
            + ("all three groups fit" if total_overhang_mm < 1.0e-6
               else f"overhang = {total_overhang_mm:.0f} mm"),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            color="#166534" if total_overhang_mm < 1.0e-6 else "#991b1b",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        )
        ax.set_ylim(-0.035, 0.035)
        ax.set_ylabel("Bz [T]")
        ax.grid(alpha=0.18)
    axes[0].legend(loc="upper left", fontsize=8, ncol=3)
    axes[-1].set_xlabel("global azimuth [deg]")
    fig.suptitle(
        "Kicker layouts over three adjacent straight sections\n"
        "The installation gap between two kickers in one straight is 0.20 m",
        fontsize=14,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_delta_r(output, results, seo):
    all_delta = []
    trajectories = []
    for result in results:
        track = result["track"]
        phi_rad = np.deg2rad(track["phi_unwrapped_deg"])
        r_seo = np.interp(
            np.mod(phi_rad, 2.0 * np.pi), seo["phi_axis"], seo["r_row"]
        )
        delta = 1.0e3 * (track["r_m"] - r_seo)
        trajectories.append((track["phi_unwrapped_deg"], delta))
        all_delta.append(delta)
    delta_min = float(np.min(np.concatenate(all_delta)))
    delta_max = float(np.max(np.concatenate(all_delta)))
    margin = 0.08 * (delta_max - delta_min)
    y_min, y_max = delta_min - margin, delta_max + margin
    sector_edges = 146.25 + STRAIGHT_PERIOD_DEG * np.arange(17)

    fig, axes = plt.subplots(
        len(results), 1, figsize=(15.0, 2.8 * len(results) + 0.8),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, result, trajectory in zip(axes, results, trajectories):
        for edge_i, edge in enumerate(sector_edges):
            if START_PHI_DEG <= edge <= START_PHI_DEG + 360.0:
                ax.axvline(edge, color="#94a3b8", ls=":", lw=0.65, alpha=0.55)
                ax.text(
                    edge, y_max - 0.02 * (y_max - y_min), f"SS{edge_i + 1}",
                    rotation=90, ha="center", va="top", fontsize=6.2,
                    color="#475569",
                )
        for kicker_i, kicker in enumerate(result["kickers"]):
            start = float(kicker["phi_start_deg"])
            ax.axvspan(
                start, start + float(kicker["phi_span_deg"]),
                color="#f59e0b", alpha=0.23,
                label="kicker" if kicker_i == 0 else None,
            )
        best = result["septum"]
        ax.axvspan(
            best["start_deg"], best["end_deg"], color="#16a34a", alpha=0.30,
            label="optimized 0.5 m septum",
        )
        ax.axhline(
            0.0, color="#111111", ls="--", lw=1.25,
            label=r"closed orbit: $\Delta r=0$",
        )
        ax.plot(
            trajectory[0], trajectory[1], color=result["color"], lw=1.8,
            label=rf"kicked orbit: $\Delta r=r-r_{{SEO}}$",
        )
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(r"$r-r_{SEO}$ [mm]")
        ax.set_title(
            f"{result['label']} — septum {best['start_deg']:.2f}–"
            f"{best['end_deg']:.2f}°; minimum separation = "
            f"{best['min_separation_mm']:.1f} mm",
            fontsize=10.5,
        )
        ax.grid(alpha=0.20)
        ax.legend(loc="lower right", fontsize=7)
    axes[-1].set_xlim(START_PHI_DEG, START_PHI_DEG + 360.0)
    axes[-1].set_xlabel("global azimuth, unwrapped [deg]")
    fig.suptitle(
        "Extraction displacement for kicker layouts over three straight sections",
        fontsize=14,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_candidates(output, results):
    fig, axes = plt.subplots(
        len(results), 1, figsize=(11.5, 2.7 * len(results) + 0.9),
        sharex=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, result in zip(axes, results):
        centers = [item["center_deg"] for item in result["candidates"]]
        minima = [item["min_separation_mm"] for item in result["candidates"]]
        best = result["septum"]
        ax.plot(centers, minima, marker="o", color=result["color"], lw=1.4)
        ax.scatter(
            [best["center_deg"]], [best["min_separation_mm"]], marker="*",
            s=160, color="#16a34a", edgecolor="black", lw=0.5, zorder=4,
        )
        ax.set_ylabel("minimum separation\nover 0.5 m [mm]")
        ax.set_title(
            f"{result['label']}: best {best['start_deg']:.2f}–"
            f"{best['end_deg']:.2f}°, {best['min_separation_mm']:.1f} mm"
        )
        ax.grid(alpha=0.22)
    axes[-1].set_xlabel("candidate septum center azimuth [deg, unwrapped]")
    fig.suptitle(
        f"Per-layout 0.5 m septum optimization, |Bz| < {LOW_FIELD_THRESHOLD_T:.3f} T",
        fontsize=14,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path,
        help="Reuse saved configs/tracks in an existing run directory.",
    )
    args = parser.parse_args()
    config_path = Path("config_track.json").resolve()
    config = load_json(config_path)
    bmap_path = resolve_config_path(config["BmapAndSEO"]["maps"], config_path)
    bmap = load_bmap_auto(
        str(bmap_path), int(config["BmapAndSEO"].get("max_order", 0))
    )
    seo = load_seo(
        bmap_path / config["BmapAndSEO"]["SEO_FileName"], ENERGY_MEV
    )
    particle_p, particle_v, _ = make_closed_orbit_particle(seo, START_PHI_DEG)
    phi_axis_deg, _, s_axis = closed_orbit_geometry(seo)

    windows = [
        find_downstream_low_field_window(bmap, seo, boundary)
        for boundary in STRAIGHT_BOUNDARIES_DEG
    ]
    for window in windows:
        window["length_m"] = arc_length_between(
            phi_axis_deg, s_axis, window["start_deg"], window["end_deg"]
        )

    if args.run_dir is None:
        run_dir = (
            Path("extraction_kicker_three_straight_scan").resolve()
            / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = args.run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    results = []
    for split, color in zip(SPLITS, COLORS):
        label = "+".join(str(value) for value in split)
        case_config = copy.deepcopy(config)
        kickers = make_three_straight_kickers(
            split, config, phi_axis_deg, s_axis
        )
        config_output = run_dir / f"config_split_{'_'.join(map(str, split))}.json"
        track_path = run_dir / f"track_split_{'_'.join(map(str, split))}.csv"
        if args.run_dir is not None and config_output.exists() and track_path.exists():
            print(f"Reusing saved split {label} ...", flush=True)
            case_config = load_json(config_output)
            kickers = case_config["BumpKicker"]["kickers"]
        else:
            case_config["BumpKicker"]["kickers"] = kickers
            config_output.write_text(
                json.dumps(case_config, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Tracking split {label} ...", flush=True)
            case = run_case(
                f"split_{'_'.join(map(str, split))}", bmap, case_config, seo,
                config_path, particle_v, particle_p, run_dir,
                TIME_STEP_NS * 1.0e-9, 1, True,
            )
            save_track_csv(case, track_path)
        track = np.genfromtxt(track_path, delimiter=",", names=True)
        best, candidates = optimize_septum(track, seo, bmap)
        results.append(
            {
                "label": label,
                "split": split,
                "kicker_count": len(kickers),
                "color": color,
                "kickers": kickers,
                "feasibility": group_feasibility(
                    kickers, split, windows, phi_axis_deg, s_axis
                ),
                "maximum_r_m": float(np.max(track["r_m"])),
                "septum": best,
                "candidates": candidates,
                "track": track,
            }
        )

    layout_path = run_dir / "01_three_straight_layout_feasibility.png"
    delta_path = run_dir / "02_three_straight_delta_r_vs_azimuth.png"
    candidate_path = run_dir / "03_three_straight_septum_optimization.png"
    plot_layout(layout_path, results, windows, bmap, seo)
    plot_delta_r(delta_path, results, seo)
    plot_candidates(candidate_path, results)

    serializable_results = []
    for result in results:
        serializable_results.append(
            {
                key: value
                for key, value in result.items()
                if key not in ("track", "color")
            }
        )
    summary = {
        "assumptions": {
            "energy_MeV": ENERGY_MEV,
            "magnet_length_m": MAGNET_LENGTH_M,
            "installation_gap_m": INSTALLATION_GAP_M,
            "kick_per_magnet_mrad": 5.0,
            "polarity": -1,
            "low_field_definition_T": LOW_FIELD_THRESHOLD_T,
            "septum_length_m": SEPTUM_LENGTH_M,
            "official_config_modified": False,
        },
        "low_field_windows": windows,
        "results": serializable_results,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for result in results:
        fit = all(item["fits_low_field_window"] for item in result["feasibility"])
        best = result["septum"]
        print(
            f"{result['label']}: fit={fit}, Rmax={result['maximum_r_m']:.6f} m, "
            f"septum={best['start_deg']:.3f}--{best['end_deg']:.3f} deg, "
            f"min separation={best['min_separation_mm']:.3f} mm"
        )
    print(f"Figure: {layout_path}")
    print(f"Figure: {delta_path}")
    print(f"Figure: {candidate_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
