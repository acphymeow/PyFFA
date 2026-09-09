import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from main_track import load_bmap_auto
from tool_plot_bmap_with_kickers import (
    add_kicker_regions_cartesian,
    evaluate_base_field,
    symmetric_norm,
)
from tool_track_extraction_kickers import (
    load_json,
    load_seo,
    resolve_config_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create three separate figures for the 500pi multi-particle extraction: "
            "ring layout, one-turn r-versus-azimuth septum placement, and "
            "septum-entrance phase space."
        )
    )
    parser.add_argument("-j", "--config", default="config_track.json")
    parser.add_argument(
        "--run-dir",
        default="extraction_bunch_septum_output/run_20260904_023555",
    )
    parser.add_argument(
        "--central-track-csv",
        default=(
            "extraction_single_particle_output/run_20260903_222702/"
            "single_particle_600MeV_kicker_on.csv"
        ),
    )
    parser.add_argument("--energy-mev", type=float, default=600.0)
    return parser.parse_args()


def load_inputs(args):
    config_path = Path(args.config).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    central_csv = Path(args.central_track_csv).expanduser().resolve()
    config = load_json(config_path)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    kicked = np.load(run_dir / "kicker_on_septum_samples.npz")
    reference = np.load(run_dir / "kicker_off_septum_samples.npz")
    central_track = np.genfromtxt(central_csv, delimiter=",", names=True)

    bmap_path = resolve_config_path(config["BmapAndSEO"]["maps"], config_path)
    if bmap_path.is_file() and bmap_path.name == "config_Bmap.json":
        bmap_path = bmap_path.parent
    bmap = load_bmap_auto(
        str(bmap_path), int(config["BmapAndSEO"].get("max_order", 0))
    )
    seo_dir = bmap_path / config["BmapAndSEO"]["SEO_FileName"]
    seo = load_seo(seo_dir, args.energy_mev)
    return config, summary, kicked, reference, central_track, bmap, seo, run_dir


def station_envelopes(kicked, reference):
    phi = kicked["station_phi_rad"]
    blade_r = kicked["station_blade_r_m"]
    kicked_r = kicked["station_particle_r_m"]
    reference_r = reference["station_particle_r_m"]
    return {
        "phi": phi,
        "s": kicked["station_s_m"],
        "blade_r": blade_r,
        "kicked_min": np.nanmin(kicked_r, axis=1),
        "kicked_max": np.nanmax(kicked_r, axis=1),
        "kicked_mean": np.nanmean(kicked_r, axis=1),
        "reference_min": np.nanmin(reference_r, axis=1),
        "reference_max": np.nanmax(reference_r, axis=1),
        "reference_mean": np.nanmean(reference_r, axis=1),
    }


def plot_ring_layout(run_dir, config, central_track, bmap, seo, envelopes):
    r_axis = np.linspace(bmap.r_min, bmap.r_max, 180)
    phi_axis = np.linspace(0.0, 2.0 * np.pi, 1101)
    r_grid, phi_grid = np.meshgrid(r_axis, phi_axis, indexing="ij")
    bz_grid, _, _ = evaluate_base_field(
        bmap,
        int(config["BmapAndSEO"].get("max_order", 0)),
        r_grid,
        phi_grid,
        0.0,
    )
    x_grid = r_grid * np.cos(phi_grid)
    y_grid = r_grid * np.sin(phi_grid)

    seo_phi = seo["phi_axis"]
    seo_r = seo["r_row"]
    seo_x = seo_r * np.cos(seo_phi)
    seo_y = seo_r * np.sin(seo_phi)

    track_phi = central_track["phi_rad"]
    first_turn = (track_phi >= track_phi[0]) & (track_phi <= track_phi[0] + 2.0 * np.pi)
    blade_x = envelopes["blade_r"] * np.cos(envelopes["phi"])
    blade_y = envelopes["blade_r"] * np.sin(envelopes["phi"])

    fig, ax = plt.subplots(figsize=(9.2, 8.1), constrained_layout=True)
    norm = symmetric_norm(bz_grid)
    field = ax.contourf(
        x_grid,
        y_grid,
        bz_grid,
        levels=np.linspace(norm.vmin, norm.vmax, 81),
        cmap="RdBu_r",
        norm=norm,
        extend="both",
        alpha=0.72,
    )
    ax.plot(
        seo_x,
        seo_y,
        color="#333333",
        linestyle="--",
        linewidth=1.5,
        label="600 MeV closed orbit",
    )
    ax.plot(
        central_track["x_m"][first_turn],
        central_track["y_m"][first_turn],
        color="#d62828",
        linewidth=2.0,
        label="kicked centroid trajectory, first turn",
    )
    add_kicker_regions_cartesian(ax, config["BumpKicker"]["kickers"], labels=True)
    ax.plot(
        blade_x,
        blade_y,
        color="#15803d",
        linewidth=5.0,
        solid_capstyle="butt",
        label="0.5 m septum centerline",
    )
    ax.scatter(
        [blade_x[0]],
        [blade_y[0]],
        s=45,
        color="#15803d",
        edgecolors="black",
        linewidths=0.6,
        zorder=8,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Ring layout: field map, four kickers and 0.5 m septum")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(field, ax=ax, pad=0.02, shrink=0.88, label="base Bz [T]")
    output = run_dir / "01_ring_layout.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def fill_radial_tube(ax, phi, r_lo, r_hi, color, label):
    x_lo, y_lo = r_lo * np.cos(phi), r_lo * np.sin(phi)
    x_hi, y_hi = r_hi * np.cos(phi), r_hi * np.sin(phi)
    ax.fill(
        np.r_[x_lo, x_hi[::-1]],
        np.r_[y_lo, y_hi[::-1]],
        color=color,
        alpha=0.27,
        linewidth=1.0,
        edgecolor=color,
        label=label,
    )


def plot_r_vs_azimuth(
    run_dir, config, summary, central_track, bmap, seo, envelopes
):
    start_deg = float(np.rad2deg(central_track["phi_rad"][0]))
    end_deg = start_deg + 360.0
    phi_plot_deg = np.linspace(start_deg, end_deg, 1441)
    phi_plot_rad = np.deg2rad(phi_plot_deg)
    seo_phi_mod = np.mod(phi_plot_rad, 2.0 * np.pi)
    seo_r_plot = np.interp(seo_phi_mod, seo["phi_axis"], seo["r_row"])
    track_phi_deg = np.rad2deg(central_track["phi_rad"])
    first_turn = (track_phi_deg >= start_deg) & (track_phi_deg <= end_deg)
    plotted_r = np.r_[seo_r_plot, central_track["r_m"][first_turn]]
    view_r_min = max(bmap.r_min, float(np.min(plotted_r)) - 0.04)
    view_r_max = min(bmap.r_max, float(np.max(plotted_r)) + 0.04)
    r_axis = np.linspace(view_r_min, view_r_max, 190)
    r_grid, phi_grid = np.meshgrid(r_axis, phi_plot_rad, indexing="ij")
    bz_grid, _, _ = evaluate_base_field(
        bmap,
        int(config["BmapAndSEO"].get("max_order", 0)),
        r_grid,
        phi_grid,
        0.0,
    )

    fig, ax = plt.subplots(figsize=(15.0, 7.2), constrained_layout=True)
    norm = symmetric_norm(bz_grid)
    field = ax.pcolormesh(
        phi_plot_deg,
        r_axis,
        bz_grid,
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
        alpha=0.72,
        rasterized=True,
    )

    # OPERA cell edges are also the centers of the long straight sections.
    sector_period_deg = 360.0 / float(bmap.Nsectors)
    opera_edge_deg = float(np.rad2deg(bmap.fi_min))
    first_edge_index = int(np.floor((start_deg - opera_edge_deg) / sector_period_deg))
    straight_sections = opera_edge_deg + sector_period_deg * np.arange(
        first_edge_index, first_edge_index + int(bmap.Nsectors) + 2
    )
    straight_sections = straight_sections[
        (straight_sections >= start_deg) & (straight_sections <= end_deg)
    ]
    for index, straight_deg in enumerate(straight_sections):
        ax.axvline(
            straight_deg,
            color="#4b5563",
            linestyle=":",
            linewidth=0.9,
            alpha=0.65,
            label="sector boundary / straight section" if index == 0 else None,
        )
        ax.text(
            straight_deg,
            view_r_max - 0.004,
            f"SS{index + 1}",
            rotation=90,
            ha="center",
            va="top",
            fontsize=7,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55,
                  "pad": 0.8},
            clip_on=True,
            zorder=9,
        )

    ax.plot(
        phi_plot_deg,
        seo_r_plot,
        color="#202020",
        linestyle="--",
        linewidth=1.7,
        label="600 MeV closed orbit",
        zorder=5,
    )
    ax.plot(
        track_phi_deg[first_turn],
        central_track["r_m"][first_turn],
        color="#d62828",
        linewidth=2.1,
        label="kicked centroid trajectory",
        zorder=6,
    )
    ax.axhline(
        float(config["Aperture"]["Rmax"]),
        color="#111827",
        linestyle="-.",
        linewidth=1.0,
        label="outer aperture / map edge",
        zorder=4,
    )

    for kicker_i, kicker in enumerate(config["BumpKicker"]["kickers"]):
        k_start = float(kicker["phi_start_deg"])
        while k_start < start_deg:
            k_start += 360.0
        k_end = k_start + float(kicker["phi_span_deg"])
        ax.axvspan(
            k_start,
            k_end,
            color="#f59e0b",
            alpha=0.28,
            linewidth=0,
            label="kicker azimuth range" if kicker_i == 0 else None,
            zorder=2,
        )
        ax.text(
            0.5 * (k_start + k_end),
            view_r_min + 0.008,
            kicker.get("name", f"K{kicker_i + 1}"),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#7c2d12",
            zorder=7,
        )

    septum_phi_deg = np.rad2deg(envelopes["phi"])
    septum_r = envelopes["blade_r"]
    septum_start = float(septum_phi_deg[0])
    septum_end = float(septum_phi_deg[-1])
    while septum_start < start_deg:
        septum_start += 360.0
        septum_end += 360.0
    ax.axvspan(
        septum_start,
        septum_end,
        color="#16a34a",
        alpha=0.20,
        linewidth=0,
        label="0.5 m septum azimuth range",
        zorder=2,
    )
    septum_phi_shifted = septum_phi_deg + (septum_start - septum_phi_deg[0])
    ax.plot(
        septum_phi_shifted,
        septum_r,
        color="#087b35",
        linewidth=4.0,
        solid_capstyle="butt",
        label="septum centerline",
        zorder=8,
    )

    sep_mm = 1.0e3 * (
        np.interp(septum_phi_shifted, track_phi_deg[first_turn], central_track["r_m"][first_turn])
        - np.interp(np.mod(np.deg2rad(septum_phi_shifted), 2.0 * np.pi), seo["phi_axis"], seo["r_row"])
    )
    max_bz_septum = float(
        np.max(
            np.abs(
                np.interp(
                    septum_phi_shifted,
                    track_phi_deg[first_turn],
                    central_track["Bz_base_T"][first_turn],
                )
            )
        )
    )
    ax.annotate(
        "Selected at the next straight after EK4\n"
        f"centroid separation: {np.min(sep_mm):.1f}–{np.max(sep_mm):.1f} mm\n"
        f"max |base Bz|: {max_bz_septum:.4f} T",
        xy=(septum_start, float(septum_r[0])),
        xytext=(septum_start + 20.0, view_r_min + 0.28 * (view_r_max - view_r_min)),
        arrowprops={"arrowstyle": "->", "color": "#087b35", "linewidth": 1.4},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9,
              "edgecolor": "#087b35"},
        fontsize=9,
        color="#065f2c",
        zorder=10,
    )

    ax.set_xlim(start_deg, end_deg)
    ax.set_ylim(view_r_min, view_r_max)
    ax.set_xlabel("global azimuth, unwrapped [deg]")
    ax.set_ylabel("radius r [m]")
    ax.set_title(
        "One-turn r versus azimuth: basis for the septum location\n"
        "The septum starts at the next low-field straight section downstream of EK4"
    )
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.colorbar(field, ax=ax, pad=0.015, label="base Bz [T]")
    output = run_dir / "02_one_turn_r_vs_azimuth_septum_location.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def entrance_phase_space(samples):
    r = samples["station_particle_r_m"]
    blade_r = samples["station_blade_r_m"]
    s = samples["station_s_m"]
    ds = float(s[1] - s[0])
    x_mm = 1.0e3 * (r[0] - blade_r[0])
    # The first two stations are 5 mm apart.  Subtract the blade-centerline
    # slope so x' is the particle divergence relative to the septum direction.
    xp_mrad = 1.0e3 * (
        (r[1] - r[0]) / ds - (blade_r[1] - blade_r[0]) / ds
    )
    return x_mm, xp_mrad


def plot_entrance_distribution(run_dir, summary, kicked, reference):
    x_on, xp_on = entrance_phase_space(kicked)
    x_off, xp_off = entrance_phase_space(reference)
    waterbag_count = int(summary["particles"]["waterbag"])
    edge = np.arange(x_on.size) >= waterbag_count
    core = ~edge

    fig, ax = plt.subplots(figsize=(9.0, 7.2), constrained_layout=True)
    ax.scatter(
        x_off[core], xp_off[core], s=5, color="#2878b5", alpha=0.25,
        linewidths=0, label="kicker off: waterbag",
    )
    ax.scatter(
        x_on[core], xp_on[core], s=5, color="#d94b41", alpha=0.25,
        linewidths=0, label="kicker on: waterbag",
    )
    ax.scatter(
        x_off[edge], xp_off[edge], s=7, facecolors="none", edgecolors="#145a86",
        alpha=0.55, linewidths=0.45, label="kicker off: 500π edge",
    )
    ax.scatter(
        x_on[edge], xp_on[edge], s=7, facecolors="none", edgecolors="#a32620",
        alpha=0.55, linewidths=0.45, label="kicker on: 500π edge",
    )
    ax.axvline(0.0, color="#15803d", linewidth=2.5, label="septum blade")
    ax.axvspan(ax.get_xlim()[0], 0.0, color="#2878b5", alpha=0.035)
    ax.axvspan(0.0, ax.get_xlim()[1], color="#d94b41", alpha=0.035)
    ax.text(
        0.015,
        0.975,
        "circulating side",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#145a86",
        fontsize=10,
    )
    ax.text(
        0.985,
        0.975,
        "extraction side",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#a32620",
        fontsize=10,
    )
    ax.set_xlabel("radial position relative to septum, x [mm]")
    ax.set_ylabel("radial divergence relative to septum, x' [mrad]")
    ax.set_title(
        "Horizontal phase-space distribution at the septum entrance\n"
        "600 MeV, geometric edge emittance = 500π mm mrad"
    )
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=8, markerscale=1.8)
    output = run_dir / "03_septum_entrance_distribution_500pi.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main():
    args = parse_args()
    config, summary, kicked, reference, central_track, bmap, seo, run_dir = load_inputs(args)
    envelopes = station_envelopes(kicked, reference)
    outputs = [
        plot_ring_layout(run_dir, config, central_track, bmap, seo, envelopes),
        plot_r_vs_azimuth(
            run_dir, config, summary, central_track, bmap, seo, envelopes
        ),
        plot_entrance_distribution(run_dir, summary, kicked, reference),
    ]
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
