import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from main_track import load_bmap_auto
from tool_plot_bmap_with_kickers import evaluate_base_field
from tool_track_extraction_kickers import load_json, load_seo, resolve_config_path


SCAN_DIR = Path("extraction_kicker_split_scan/run_20260906_174544")
ENERGY_MEV = 600.0
TRACK_START_DEG = 140.0
SEPTUM_LENGTH_M = 0.5
LOW_FIELD_THRESHOLD_T = 0.012
STRAIGHT_PERIOD_DEG = 22.5
FIRST_DOWNSTREAM_STRAIGHT_DEG = 191.25
SPLITS = ((2, 2), (1, 3), (3, 1), (0, 4))


def candidate_in_straight(track, seo, bmap, boundary_deg):
    phi_deg = np.linspace(boundary_deg, boundary_deg + 7.0, 7001)
    phi_rad = np.deg2rad(phi_deg)
    r_on = np.interp(phi_deg, track["phi_unwrapped_deg"], track["r_m"])
    r_off = np.interp(np.mod(phi_rad, 2.0 * np.pi), seo["phi_axis"], seo["r_row"])
    blade_r = 0.5 * (r_on + r_off)
    blade_x = blade_r * np.cos(phi_rad)
    blade_y = blade_r * np.sin(phi_rad)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(blade_x), np.diff(blade_y)))]
    bz, _, _ = evaluate_base_field(bmap, 0, blade_r, phi_rad, np.zeros_like(blade_r))
    low = np.abs(bz) < LOW_FIELD_THRESHOLD_T
    low_end = 0
    while low_end + 1 < low.size and low[low_end + 1]:
        low_end += 1

    separation_mm = 1.0e3 * (r_on - r_off)
    best = None
    for start_i in range(low_end + 1):
        end_i = int(np.searchsorted(s, s[start_i] + SEPTUM_LENGTH_M, side="left"))
        if end_i > low_end or end_i >= s.size:
            continue
        min_sep = float(np.min(separation_mm[start_i:end_i + 1]))
        candidate = {
            "start_deg": float(phi_deg[start_i]),
            "end_deg": float(phi_deg[end_i]),
            "center_deg": float(0.5 * (phi_deg[start_i] + phi_deg[end_i])),
            "length_m": float(s[end_i] - s[start_i]),
            "min_separation_mm": min_sep,
            "max_separation_mm": float(np.max(separation_mm[start_i:end_i + 1])),
            "mean_separation_mm": float(np.mean(separation_mm[start_i:end_i + 1])),
            "max_abs_bz_T": float(np.max(np.abs(bz[start_i:end_i + 1]))),
            "straight_boundary_deg": float(boundary_deg),
        }
        if best is None or candidate["min_separation_mm"] > best["min_separation_mm"]:
            best = candidate
    return best


def optimize_track(track, seo, bmap):
    last_track_deg = float(np.max(track["phi_unwrapped_deg"]))
    boundaries = np.arange(
        FIRST_DOWNSTREAM_STRAIGHT_DEG,
        last_track_deg - 2.0,
        STRAIGHT_PERIOD_DEG,
    )
    candidates = []
    for boundary in boundaries:
        candidate = candidate_in_straight(track, seo, bmap, float(boundary))
        if candidate is not None and candidate["min_separation_mm"] > 0.0:
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError("No outward, 0.5 m low-field septum candidate was found.")
    return max(candidates, key=lambda item: item["min_separation_mm"]), candidates


def plot_candidates(output, optimized):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    for ax, item in zip(axes.flat, optimized):
        candidates = item["candidates"]
        centers = np.array([candidate["center_deg"] for candidate in candidates])
        minima = np.array([candidate["min_separation_mm"] for candidate in candidates])
        best = item["best"]
        ax.plot(centers, minima, marker="o", color="#2563eb", lw=1.3)
        ax.scatter(
            [best["center_deg"]], [best["min_separation_mm"]],
            marker="*", s=150, color="#15803d", edgecolors="black", linewidths=0.5,
            zorder=5,
        )
        ax.annotate(
            f"best: {best['start_deg']:.2f}–{best['end_deg']:.2f}°\n"
            f"min separation = {best['min_separation_mm']:.1f} mm",
            xy=(best["center_deg"], best["min_separation_mm"]),
            xytext=(0.98, 0.96), textcoords="axes fraction",
            ha="right", va="top", fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "#15803d", "lw": 1.0},
        )
        ax.set_title(f"{item['label']} split")
        ax.set_xlabel("candidate septum center azimuth [deg, unwrapped]")
        ax.set_ylabel("minimum separation over 0.5 m [mm]")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Per-layout septum optimization over downstream low-field straight sections\n"
        f"objective: maximize minimum separation, with |Bz| < {LOW_FIELD_THRESHOLD_T:.3f} T",
        fontsize=14,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_per_layout_r_azimuth(output, optimized, scan_dir, seo, bmap):
    phi_deg = np.linspace(TRACK_START_DEG, TRACK_START_DEG + 360.0, 1601)
    phi_rad = np.deg2rad(phi_deg)
    seo_r = np.interp(np.mod(phi_rad, 2.0 * np.pi), seo["phi_axis"], seo["r_row"])
    tracks = []
    for item, (front, rear) in zip(optimized, SPLITS):
        track = np.genfromtxt(
            scan_dir / f"track_split_{front}_{rear}.csv", delimiter=",", names=True
        )
        tracks.append(track)
    all_r = np.concatenate([seo_r, *[track["r_m"] for track in tracks]])
    r_min = max(bmap.r_min, float(np.min(all_r)) - 0.035)
    r_max = min(bmap.r_max, float(np.max(all_r)) + 0.035)
    r_axis = np.linspace(r_min, r_max, 190)
    r_grid, phi_grid = np.meshgrid(r_axis, phi_rad, indexing="ij")
    bz_grid, _, _ = evaluate_base_field(bmap, 0, r_grid, phi_grid, 0.0)
    vmax = float(np.nanmax(np.abs(bz_grid)))

    fig, axes = plt.subplots(
        2, 2, figsize=(15.0, 9.2), sharex=True, sharey=True,
        constrained_layout=True,
    )
    colors = ("#2563eb", "#e11d48", "#7c3aed", "#ea580c")
    sector_edges = FIRST_DOWNSTREAM_STRAIGHT_DEG - 45.0 + STRAIGHT_PERIOD_DEG * np.arange(17)
    for ax, item, track, color, (front, rear) in zip(
        axes.flat, optimized, tracks, colors, SPLITS
    ):
        field = ax.pcolormesh(
            phi_deg, r_axis, bz_grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            shading="auto", alpha=0.58, rasterized=True,
        )
        for edge_i, edge in enumerate(sector_edges):
            if TRACK_START_DEG <= edge <= TRACK_START_DEG + 360.0:
                ax.axvline(edge, color="#64748b", ls=":", lw=0.75, alpha=0.65)
                ax.text(
                    edge, r_max - 0.004, f"SS{edge_i + 1}", rotation=90,
                    ha="center", va="top", fontsize=6.2, color="#374151",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.5,
                          "pad": 0.4},
                )
        ax.plot(phi_deg, seo_r, color="#111111", ls="--", lw=1.5, label="600 MeV SEO")
        ax.plot(
            track["phi_unwrapped_deg"], track["r_m"], color=color, lw=1.8,
            label=f"kicked trajectory ({front}+{rear})",
        )
        best = item["best"]
        ax.axvspan(
            best["start_deg"], best["end_deg"], color="#16a34a", alpha=0.30,
            label="optimized 0.5 m septum",
        )
        config = load_json(scan_dir / f"config_split_{front}_{rear}.json")
        for kicker_i, kicker in enumerate(config["BumpKicker"]["kickers"]):
            start = float(kicker["phi_start_deg"])
            end = start + float(kicker["phi_span_deg"])
            ax.axvspan(
                start, end, color="#f59e0b", alpha=0.22,
                label="kicker" if kicker_i == 0 else None,
            )
        ax.set_title(
            f"{item['label']} split: septum {best['start_deg']:.2f}–{best['end_deg']:.2f}°\n"
            f"minimum centroid separation over 0.5 m = {best['min_separation_mm']:.1f} mm",
            fontsize=10.5,
        )
        ax.set_xlim(TRACK_START_DEG, TRACK_START_DEG + 360.0)
        ax.set_ylim(r_min, r_max)
        ax.grid(alpha=0.15)
        ax.legend(loc="lower right", fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("global azimuth, unwrapped [deg]")
    for ax in axes[:, 0]:
        ax.set_ylabel("radius r [m]")
    fig.colorbar(field, ax=axes, pad=0.012, label="base Bz [T]")
    fig.suptitle(
        "One-turn r versus azimuth with a separately optimized septum for each layout\n"
        "R range is cropped to the circulating and extraction trajectories",
        fontsize=14,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_per_layout_delta_r_azimuth(output, optimized, scan_dir, seo):
    tracks = []
    delta_r_mm_all = []
    for front, rear in SPLITS:
        track = np.genfromtxt(
            scan_dir / f"track_split_{front}_{rear}.csv", delimiter=",", names=True
        )
        phi_rad = np.deg2rad(track["phi_unwrapped_deg"])
        seo_r = np.interp(
            np.mod(phi_rad, 2.0 * np.pi), seo["phi_axis"], seo["r_row"]
        )
        delta_r_mm = 1.0e3 * (track["r_m"] - seo_r)
        tracks.append((track, delta_r_mm))
        delta_r_mm_all.append(delta_r_mm)

    delta_min = float(np.min(np.concatenate(delta_r_mm_all)))
    delta_max = float(np.max(np.concatenate(delta_r_mm_all)))
    y_margin = 0.08 * (delta_max - delta_min)
    y_min = delta_min - y_margin
    y_max = delta_max + y_margin

    fig, axes = plt.subplots(
        2, 2, figsize=(15.0, 8.8), sharex=True, sharey=True,
        constrained_layout=True,
    )
    colors = ("#2563eb", "#e11d48", "#7c3aed", "#ea580c")
    sector_edges = (
        FIRST_DOWNSTREAM_STRAIGHT_DEG - 45.0
        + STRAIGHT_PERIOD_DEG * np.arange(17)
    )
    for ax, item, track_data, color, (front, rear) in zip(
        axes.flat, optimized, tracks, colors, SPLITS
    ):
        track, delta_r_mm = track_data
        for edge_i, edge in enumerate(sector_edges):
            if TRACK_START_DEG <= edge <= TRACK_START_DEG + 360.0:
                ax.axvline(edge, color="#94a3b8", ls=":", lw=0.65, alpha=0.55)
                ax.text(
                    edge, y_max - 0.02 * (y_max - y_min), f"SS{edge_i + 1}",
                    rotation=90, ha="center", va="top", fontsize=6.2,
                    color="#475569",
                )

        config = load_json(scan_dir / f"config_split_{front}_{rear}.json")
        for kicker_i, kicker in enumerate(config["BumpKicker"]["kickers"]):
            start_deg = float(kicker["phi_start_deg"])
            span_deg = float(kicker["phi_span_deg"])
            ax.axvspan(
                start_deg, start_deg + span_deg, color="#f59e0b", alpha=0.22,
                label="kicker" if kicker_i == 0 else None, zorder=0.5,
            )

        best = item["best"]
        ax.axvspan(
            best["start_deg"], best["end_deg"], color="#16a34a", alpha=0.28,
            label="optimized 0.5 m septum", zorder=0.6,
        )
        ax.axhline(
            0.0, color="#111111", ls="--", lw=1.25,
            label=r"closed-orbit reference: $\Delta r=0$", zorder=2,
        )
        ax.plot(
            track["phi_unwrapped_deg"], delta_r_mm, color=color, lw=1.8,
            label=rf"kicked orbit: $\Delta r=r-r_{{SEO}}$ ({front}+{rear})",
            zorder=3,
        )
        ax.set_xlim(TRACK_START_DEG, TRACK_START_DEG + 360.0)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.20)
        ax.set_title(
            f"{item['label']} split — septum {best['start_deg']:.2f}–"
            f"{best['end_deg']:.2f}°",
            fontsize=10.5,
        )
        ax.legend(loc="lower right", fontsize=7)

    for ax in axes[-1, :]:
        ax.set_xlabel("global azimuth, unwrapped [deg]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"radial displacement $r-r_{SEO}$ [mm]")
    fig.suptitle(
        r"Extraction displacement relative to the 600 MeV closed orbit: "
        r"$\Delta r=r-r_{SEO}$",
        fontsize=14,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    scan_dir = SCAN_DIR.resolve()
    config_path = Path("config_track.json").resolve()
    config = load_json(config_path)
    bmap_path = resolve_config_path(config["BmapAndSEO"]["maps"], config_path)
    bmap = load_bmap_auto(
        str(bmap_path), int(config["BmapAndSEO"].get("max_order", 0))
    )
    seo = load_seo(
        bmap_path / config["BmapAndSEO"]["SEO_FileName"], ENERGY_MEV
    )
    optimized = []
    for front, rear in SPLITS:
        label = f"{front}+{rear}"
        track = np.genfromtxt(
            scan_dir / f"track_split_{front}_{rear}.csv", delimiter=",", names=True
        )
        best, candidates = optimize_track(track, seo, bmap)
        optimized.append({"label": label, "best": best, "candidates": candidates})
        print(
            f"{label}: {best['start_deg']:.3f}--{best['end_deg']:.3f} deg, "
            f"min separation={best['min_separation_mm']:.3f} mm"
        )
    summary = {
        "constraints": {
            "septum_length_m": SEPTUM_LENGTH_M,
            "max_abs_base_bz_T": LOW_FIELD_THRESHOLD_T,
            "objective": "maximize minimum outward centroid separation over the full septum",
        },
        "results": optimized,
    }
    summary_path = scan_dir / "optimized_septum_locations.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    figure_path = scan_dir / "03_optimized_septum_location_scan.png"
    plot_candidates(figure_path, optimized)
    r_azimuth_path = scan_dir / "02_kicker_split_one_turn_r_vs_azimuth.png"
    plot_per_layout_r_azimuth(
        r_azimuth_path, optimized, scan_dir, seo, bmap
    )
    delta_r_path = scan_dir / "05_kicker_split_delta_r_vs_azimuth.png"
    plot_per_layout_delta_r_azimuth(
        delta_r_path, optimized, scan_dir, seo
    )
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure_path}")
    print(f"Figure: {r_azimuth_path}")
    print(f"Figure: {delta_r_path}")


if __name__ == "__main__":
    main()
