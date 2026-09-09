import copy
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from main_track import load_bmap_auto
from tool_plot_bmap_with_kickers import evaluate_base_field, symmetric_norm
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
MAGNET_LENGTH_M = 0.20
INSTALLATION_GAP_M = 0.20
LOW_FIELD_THRESHOLD_T = 0.012
FRONT_START_DEG = 146.521143
REAR_START_DEG = 169.021176
SEPTUM_START_DEG = 191.26746533735664
SEPTUM_END_DEG = 193.68920015715602
SPLITS = ((2, 2), (1, 3), (3, 1), (0, 4))


def closed_orbit_geometry(seo, samples=200001):
    phi_deg = np.linspace(0.0, 360.0, samples)
    phi_rad = np.deg2rad(phi_deg)
    r = np.interp(phi_rad, seo["phi_axis"], seo["r_row"])
    x = r * np.cos(phi_rad)
    y = r * np.sin(phi_rad)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    return phi_deg, r, s


def advance_by_s(phi_deg, s_axis, start_deg, distance_m):
    start_s = float(np.interp(start_deg, phi_deg, s_axis))
    return float(np.interp(start_s + distance_m, s_axis, phi_deg))


def arc_length_between(phi_deg, s_axis, start_deg, end_deg):
    return float(np.interp(end_deg, phi_deg, s_axis) - np.interp(start_deg, phi_deg, s_axis))


def make_group(count, first_start_deg, template, first_index, phi_axis_deg, s_axis):
    group = []
    cursor = first_start_deg
    for local_i in range(count):
        start = cursor
        end = advance_by_s(phi_axis_deg, s_axis, start, MAGNET_LENGTH_M)
        item = copy.deepcopy(template)
        item["name"] = f"EK{first_index + local_i}"
        item["L_eff_m"] = MAGNET_LENGTH_M
        item["phi_start_deg"] = start
        item["phi_span_deg"] = end - start
        group.append(item)
        if local_i + 1 < count:
            cursor = advance_by_s(
                phi_axis_deg, s_axis, end, INSTALLATION_GAP_M
            )
    return group


def make_split_kickers(split, base_config, phi_axis_deg, s_axis):
    front_count, rear_count = split
    template = base_config["BumpKicker"]["kickers"][0]
    front = make_group(
        front_count, FRONT_START_DEG, template, 1, phi_axis_deg, s_axis
    )
    rear = make_group(
        rear_count,
        REAR_START_DEG,
        template,
        front_count + 1,
        phi_axis_deg,
        s_axis,
    )
    return front + rear


def find_downstream_low_field_window(
    bmap, seo, straight_start_deg, threshold_t=LOW_FIELD_THRESHOLD_T
):
    phi_deg = np.linspace(straight_start_deg, straight_start_deg + 8.0, 16001)
    phi_rad = np.deg2rad(phi_deg)
    r = np.interp(np.mod(phi_rad, 2.0 * np.pi), seo["phi_axis"], seo["r_row"])
    bz, _, _ = evaluate_base_field(bmap, 0, r, phi_rad, np.zeros_like(r))
    mask = np.abs(bz) < threshold_t
    if not mask[0]:
        raise RuntimeError(
            f"The field at straight start {straight_start_deg} deg is above threshold."
        )
    end_i = 0
    while end_i + 1 < mask.size and mask[end_i + 1]:
        end_i += 1
    return {
        "start_deg": float(phi_deg[0]),
        "end_deg": float(phi_deg[end_i]),
        "max_abs_bz_T": float(np.max(np.abs(bz[:end_i + 1]))),
    }


def save_track_csv(case, path):
    raw = case["trajectory"]
    array = np.column_stack((np.rad2deg(raw[:, 4]), raw[:, 0], raw[:, 2], raw[:, 5], raw[:, -1]))
    np.savetxt(
        path,
        array,
        delimiter=",",
        header="phi_unwrapped_deg,r_m,z_m,Ek_MeV,time_s",
        comments="",
    )


def separation_at_septum(case, seo):
    raw = case["trajectory"]
    phi_deg = np.rad2deg(raw[:, 4])
    mask = (phi_deg >= SEPTUM_START_DEG) & (phi_deg <= SEPTUM_END_DEG)
    r = raw[mask, 0]
    phi = raw[mask, 4]
    r_seo = np.interp(np.mod(phi, 2.0 * np.pi), seo["phi_axis"], seo["r_row"])
    separation_mm = 1.0e3 * (r - r_seo)
    return {
        "min_mm": float(np.min(separation_mm)),
        "max_mm": float(np.max(separation_mm)),
        "mean_mm": float(np.mean(separation_mm)),
    }


def group_feasibility(kickers, count, group_start_index, window, phi_axis_deg, s_axis):
    if count == 0:
        return {
            "count": 0,
            "required_span_m": 0.0,
            "available_from_first_kicker_m": arc_length_between(
                phi_axis_deg, s_axis,
                FRONT_START_DEG if window["start_deg"] < 160.0 else REAR_START_DEG,
                window["end_deg"],
            ),
            "overhang_m": 0.0,
            "fits_low_field_window": True,
        }
    group = kickers[group_start_index:group_start_index + count]
    start = float(group[0]["phi_start_deg"])
    end = float(group[-1]["phi_start_deg"] + group[-1]["phi_span_deg"])
    required = arc_length_between(phi_axis_deg, s_axis, start, end)
    available = arc_length_between(phi_axis_deg, s_axis, start, window["end_deg"])
    overhang = max(0.0, required - available)
    return {
        "count": count,
        "start_deg": start,
        "end_deg": end,
        "required_span_m": required,
        "available_from_first_kicker_m": available,
        "overhang_m": overhang,
        "fits_low_field_window": overhang <= 1.0e-9,
    }


def plot_layout_feasibility(run_dir, results, windows, bmap, seo):
    phi_deg = np.linspace(145.0, 177.5, 2601)
    phi_rad = np.deg2rad(phi_deg)
    r = np.interp(phi_rad, seo["phi_axis"], seo["r_row"])
    bz, _, _ = evaluate_base_field(bmap, 0, r, phi_rad, np.zeros_like(r))

    fig, axes = plt.subplots(4, 1, figsize=(14.0, 9.0), sharex=True, constrained_layout=True)
    for ax, result in zip(axes, results):
        ax.plot(phi_deg, bz, color="#4b5563", linewidth=1.0, label="base Bz on SEO")
        ax.axhline(LOW_FIELD_THRESHOLD_T, color="#9ca3af", ls=":", lw=0.8)
        ax.axhline(-LOW_FIELD_THRESHOLD_T, color="#9ca3af", ls=":", lw=0.8)
        for window_i, window in enumerate(windows):
            ax.axvspan(
                window["start_deg"], window["end_deg"], color="#bbf7d0", alpha=0.38,
                label=f"|Bz| < {LOW_FIELD_THRESHOLD_T:.3f} T" if window_i == 0 else None,
            )
            ax.axvline(window["start_deg"], color="#64748b", ls="--", lw=0.9)
        for kicker_i, kicker in enumerate(result["kickers"]):
            start = float(kicker["phi_start_deg"])
            end = start + float(kicker["phi_span_deg"])
            in_window = any(
                start >= window["start_deg"] - 1.0e-12
                and end <= window["end_deg"] + 1.0e-12
                for window in windows
            )
            color = "#f59e0b" if in_window else "#dc2626"
            ax.axvspan(start, end, color=color, alpha=0.65, zorder=4)
            ax.text(
                0.5 * (start + end), 0.88, kicker["name"],
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                rotation=90, fontsize=8, fontweight="bold",
            )
        overhang_mm = 1.0e3 * (
            result["front_feasibility"]["overhang_m"]
            + result["rear_feasibility"]["overhang_m"]
        )
        status = "fits" if overhang_mm < 1.0e-6 else f"overhang = {overhang_mm:.0f} mm"
        ax.text(
            0.99, 0.08,
            f"{result['label']}: {status}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            color="#166534" if overhang_mm < 1.0e-6 else "#991b1b",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        ax.set_ylim(-0.035, 0.035)
        ax.set_ylabel("Bz [T]")
        ax.grid(alpha=0.18)
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1].set_xlabel("global azimuth [deg]")
    fig.suptitle(
        "Four 0.20 m kickers redistributed between two adjacent straight sections\n"
        "Installation gap is fixed at 0.20 m; red blocks extend beyond the low-field window",
        fontsize=14,
    )
    output = run_dir / "01_kicker_split_layout_feasibility.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_r_azimuth_comparison(run_dir, results, bmap, seo):
    start_deg = START_PHI_DEG
    end_deg = START_PHI_DEG + 360.0
    phi_deg = np.linspace(start_deg, end_deg, 1601)
    phi_rad = np.deg2rad(phi_deg)
    seo_r = np.interp(np.mod(phi_rad, 2.0 * np.pi), seo["phi_axis"], seo["r_row"])

    trajectories = []
    for result in results:
        raw = result["case"]["trajectory"]
        track_phi = np.rad2deg(raw[:, 4])
        mask = (track_phi >= start_deg) & (track_phi <= end_deg)
        trajectories.append((track_phi[mask], raw[mask, 0]))

    all_r = np.concatenate([seo_r, *[item[1] for item in trajectories]])
    r_min = max(bmap.r_min, float(np.min(all_r)) - 0.035)
    r_max = min(bmap.r_max, float(np.max(all_r)) + 0.035)
    r_axis = np.linspace(r_min, r_max, 210)
    r_grid, phi_grid = np.meshgrid(r_axis, phi_rad, indexing="ij")
    bz_grid, _, _ = evaluate_base_field(
        bmap, 0, r_grid, phi_grid, 0.0
    )

    fig = plt.figure(figsize=(15.5, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(4.4, 1.15), hspace=0.04)
    ax = fig.add_subplot(grid[0])
    ax_layout = fig.add_subplot(grid[1], sharex=ax)
    norm = symmetric_norm(bz_grid)
    field = ax.pcolormesh(
        phi_deg, r_axis, bz_grid, cmap="RdBu_r", norm=norm,
        shading="auto", alpha=0.62, rasterized=True,
    )
    sector_period = 360.0 / bmap.Nsectors
    sector_edges = 146.25 + sector_period * np.arange(17)
    for i, edge in enumerate(sector_edges):
        ax.axvline(edge, color="#64748b", ls=":", lw=0.8, alpha=0.65)
        ax_layout.axvline(edge, color="#64748b", ls=":", lw=0.8, alpha=0.65)
        if edge <= end_deg:
            ax.text(
                edge, r_max - 0.006, f"SS{i + 1}", rotation=90,
                ha="center", va="top", fontsize=6.5, color="#374151",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.50, "pad": 0.5},
            )
    ax.plot(phi_deg, seo_r, color="#111111", ls="--", lw=1.6, label="600 MeV SEO")
    colors = ("#2563eb", "#e11d48", "#7c3aed", "#ea580c")
    for result, trajectory, color in zip(results, trajectories, colors):
        separation = result["septum_separation_mm"]
        ax.plot(
            trajectory[0], trajectory[1], color=color, lw=1.8,
            label=(
                f"{result['label']}: septum min/mean = "
                f"{separation['min_mm']:.1f}/{separation['mean_mm']:.1f} mm"
            ),
        )
    ax.axvspan(
        SEPTUM_START_DEG, SEPTUM_END_DEG, color="#16a34a", alpha=0.24,
        label="0.5 m septum",
    )
    ax.set_xlim(start_deg, end_deg)
    ax.set_ylim(r_min, r_max)
    ax.set_ylabel("radius r [m]")
    ax.set_title(
        "One-turn r versus azimuth for four kicker splits\n"
        "Radial range is cropped to the circulating and extraction trajectories"
    )
    ax.grid(alpha=0.16)
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(field, ax=ax, pad=0.012, label="base Bz [T]")

    y_positions = np.arange(len(results))[::-1]
    for row, result, color in zip(y_positions, results, colors):
        ax_layout.axhspan(row - 0.34, row + 0.34, color="#f8fafc", zorder=0)
        for kicker in result["kickers"]:
            start = float(kicker["phi_start_deg"])
            end = start + float(kicker["phi_span_deg"])
            ax_layout.fill_between(
                [start, end], row - 0.27, row + 0.27, color=color, alpha=0.80
            )
    ax_layout.axvspan(SEPTUM_START_DEG, SEPTUM_END_DEG, color="#16a34a", alpha=0.24)
    ax_layout.set_yticks(y_positions, [result["label"] for result in results])
    ax_layout.set_ylim(-0.65, len(results) - 0.35)
    ax_layout.set_xlabel("global azimuth, unwrapped [deg]")
    ax_layout.set_ylabel("layout")
    ax_layout.grid(axis="x", alpha=0.15)
    output = run_dir / "02_kicker_split_one_turn_r_vs_azimuth.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main():
    config_path = Path("config_track.json").resolve()
    config = load_json(config_path)
    bmap_path = resolve_config_path(config["BmapAndSEO"]["maps"], config_path)
    bmap = load_bmap_auto(
        str(bmap_path), int(config["BmapAndSEO"].get("max_order", 0))
    )
    seo_dir = bmap_path / config["BmapAndSEO"]["SEO_FileName"]
    seo = load_seo(seo_dir, ENERGY_MEV)
    particle_p, particle_v, _ = make_closed_orbit_particle(seo, START_PHI_DEG)
    phi_axis_deg, _, s_axis = closed_orbit_geometry(seo)

    windows = [
        find_downstream_low_field_window(bmap, seo, 146.25),
        find_downstream_low_field_window(bmap, seo, 168.75),
    ]
    for window in windows:
        window["length_m"] = arc_length_between(
            phi_axis_deg, s_axis, window["start_deg"], window["end_deg"]
        )

    run_dir = (
        Path("extraction_kicker_split_scan").resolve()
        / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for front_count, rear_count in SPLITS:
        label = f"{front_count}+{rear_count}"
        case_config = copy.deepcopy(config)
        kickers = make_split_kickers(
            (front_count, rear_count), config, phi_axis_deg, s_axis
        )
        case_config["BumpKicker"]["kickers"] = kickers
        case_config_path = run_dir / f"config_split_{front_count}_{rear_count}.json"
        case_config_path.write_text(
            json.dumps(case_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Tracking split {label} ...", flush=True)
        case = run_case(
            f"split_{front_count}_{rear_count}",
            bmap,
            case_config,
            seo,
            config_path,
            particle_v,
            particle_p,
            run_dir,
            TIME_STEP_NS * 1.0e-9,
            1,
            True,
        )
        save_track_csv(case, run_dir / f"track_split_{front_count}_{rear_count}.csv")
        front_feasibility = group_feasibility(
            kickers, front_count, 0, windows[0], phi_axis_deg, s_axis
        )
        rear_feasibility = group_feasibility(
            kickers, rear_count, front_count, windows[1], phi_axis_deg, s_axis
        )
        results.append(
            {
                "label": label,
                "front_count": front_count,
                "rear_count": rear_count,
                "kickers": kickers,
                "front_feasibility": front_feasibility,
                "rear_feasibility": rear_feasibility,
                "septum_separation_mm": separation_at_septum(case, seo),
                "maximum_r_m": float(np.max(case["trajectory"][:, 0])),
                "case": case,
            }
        )

    figures = [
        plot_layout_feasibility(run_dir, results, windows, bmap, seo),
        plot_r_azimuth_comparison(run_dir, results, bmap, seo),
    ]
    serializable_results = []
    for result in results:
        serializable_results.append(
            {key: value for key, value in result.items() if key != "case"}
        )
    summary = {
        "assumptions": {
            "energy_MeV": ENERGY_MEV,
            "magnet_length_m": MAGNET_LENGTH_M,
            "installation_gap_m": INSTALLATION_GAP_M,
            "kick_per_magnet_mrad": 5.0,
            "polarity": -1,
            "low_field_definition_T": LOW_FIELD_THRESHOLD_T,
            "official_config_modified": False,
        },
        "low_field_windows": windows,
        "results": serializable_results,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for figure in figures:
        print(f"Figure: {figure}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
