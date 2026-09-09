import argparse
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
from tool_track_extraction_kickers import load_json, load_seo, resolve_config_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the first-turn extraction trajectory and a finite-length septum."
    )
    parser.add_argument("-j", "--config", default="config_track.json")
    parser.add_argument(
        "--track-csv",
        default=(
            "extraction_single_particle_output/run_20260903_222702/"
            "single_particle_600MeV_kicker_on.csv"
        ),
    )
    parser.add_argument("--energy-mev", type=float, default=600.0)
    parser.add_argument("--septum-start-deg", type=float, default=191.267)
    parser.add_argument("--septum-length-m", type=float, default=0.5)
    parser.add_argument("--output")
    return parser.parse_args()


def select_first_turn(track):
    phi_unwrapped_deg = np.rad2deg(track["phi_rad"])
    start_deg = float(phi_unwrapped_deg[0])
    mask = (phi_unwrapped_deg >= start_deg) & (phi_unwrapped_deg <= start_deg + 360.0)
    if not np.any(mask):
        raise ValueError("The CSV does not contain a complete first-turn trajectory.")
    return mask, phi_unwrapped_deg


def build_septum_segment(track, phi_unwrapped_deg, start_deg, length_m):
    eligible = np.where(phi_unwrapped_deg >= start_deg)[0]
    if not eligible.size:
        raise ValueError(f"Track does not reach septum start phi={start_deg} deg.")
    start_idx = int(eligible[0])

    phi_tail = track["phi_rad"][start_idx:]
    r_mid_tail = 0.5 * (
        track["r_m"][start_idx:] + track["r_SEO_m"][start_idx:]
    )
    blade_x_tail = r_mid_tail * np.cos(phi_tail)
    blade_y_tail = r_mid_tail * np.sin(phi_tail)
    blade_s_tail = np.r_[
        0.0,
        np.cumsum(np.hypot(np.diff(blade_x_tail), np.diff(blade_y_tail))),
    ]
    end_local = int(np.searchsorted(blade_s_tail, length_m, side="left"))
    if end_local >= blade_s_tail.size:
        raise ValueError("Track ends before the requested septum length is reached.")
    if end_local == 0:
        raise ValueError("The requested septum length is below the trajectory resolution.")

    low_local = end_local - 1
    fraction = (
        (length_m - blade_s_tail[low_local])
        / (blade_s_tail[end_local] - blade_s_tail[low_local])
    )
    base_indices = np.arange(start_idx, start_idx + end_local)

    def with_interpolated_endpoint(values):
        endpoint = (
            values[start_idx + low_local]
            + fraction * (values[start_idx + end_local] - values[start_idx + low_local])
        )
        return np.r_[values[base_indices], endpoint]

    blade_x_endpoint = (
        blade_x_tail[low_local]
        + fraction * (blade_x_tail[end_local] - blade_x_tail[low_local])
    )
    blade_y_endpoint = (
        blade_y_tail[low_local]
        + fraction * (blade_y_tail[end_local] - blade_y_tail[low_local])
    )
    return {
        "phi_rad": with_interpolated_endpoint(track["phi_rad"]),
        "phi_unwrapped_deg": with_interpolated_endpoint(phi_unwrapped_deg),
        "r_kicked_m": with_interpolated_endpoint(track["r_m"]),
        "r_seo_m": with_interpolated_endpoint(track["r_SEO_m"]),
        "bz_base_T": with_interpolated_endpoint(track["Bz_base_T"]),
        "blade_x_m": np.r_[blade_x_tail[:end_local], blade_x_endpoint],
        "blade_y_m": np.r_[blade_y_tail[:end_local], blade_y_endpoint],
        "centerline_s_m": np.r_[blade_s_tail[:end_local], length_m],
    }


def main():
    args = parse_args()
    if args.septum_length_m <= 0.0:
        raise ValueError("--septum-length-m must be positive.")

    config_path = Path(args.config).expanduser().resolve()
    csv_path = Path(args.track_csv).expanduser().resolve()
    config = load_json(config_path)
    track = np.genfromtxt(csv_path, delimiter=",", names=True)

    bmap_path = resolve_config_path(config["BmapAndSEO"]["maps"], config_path)
    if bmap_path.is_file() and bmap_path.name == "config_Bmap.json":
        bmap_path = bmap_path.parent
    bmap = load_bmap_auto(
        str(bmap_path), int(config["BmapAndSEO"].get("max_order", 0))
    )
    seo_dir = bmap_path / config["BmapAndSEO"]["SEO_FileName"]
    seo = load_seo(seo_dir, args.energy_mev)

    first_turn, phi_unwrapped_deg = select_first_turn(track)
    septum = build_septum_segment(
        track, phi_unwrapped_deg, args.septum_start_deg, args.septum_length_m
    )

    r_axis = np.linspace(bmap.r_min, bmap.r_max, 180)
    phi_axis = np.linspace(0.0, 2.0 * np.pi, 1101)
    r_grid, phi_grid = np.meshgrid(r_axis, phi_axis, indexing="ij")
    bz_grid, _, _ = evaluate_base_field(
        bmap, int(config["BmapAndSEO"].get("max_order", 0)), r_grid, phi_grid, 0.0
    )
    x_grid = r_grid * np.cos(phi_grid)
    y_grid = r_grid * np.sin(phi_grid)

    seo_phi = seo["phi_axis"]
    seo_r = seo["r_row"]
    seo_x = seo_r * np.cos(seo_phi)
    seo_y = seo_r * np.sin(seo_phi)

    x_track = track["x_m"]
    y_track = track["y_m"]
    septum_phi = septum["phi_rad"]
    septum_r_kicked = septum["r_kicked_m"]
    septum_r_seo = septum["r_seo_m"]
    septum_blade_x = septum["blade_x_m"]
    septum_blade_y = septum["blade_y_m"]
    septum_seo_x = septum_r_seo * np.cos(septum_phi)
    septum_seo_y = septum_r_seo * np.sin(septum_phi)
    septum_kicked_x = septum_r_kicked * np.cos(septum_phi)
    septum_kicked_y = septum_r_kicked * np.sin(septum_phi)

    separation_mm = (septum_r_kicked - septum_r_seo) * 1.0e3
    bz_local = septum["bz_base_T"]

    fig, (ax_ring, ax_zoom) = plt.subplots(
        1, 2, figsize=(15.2, 7.4), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.08, 1.0)},
    )

    norm = symmetric_norm(bz_grid)
    levels = np.linspace(norm.vmin, norm.vmax, 81)
    field = ax_ring.contourf(
        x_grid, y_grid, bz_grid, levels=levels, cmap="RdBu_r", norm=norm,
        extend="both", alpha=0.72,
    )
    ax_ring.plot(seo_x, seo_y, color="#333333", linestyle="--", linewidth=1.5,
                 label="600 MeV closed orbit")
    ax_ring.plot(x_track[first_turn], y_track[first_turn], color="#d62828", linewidth=2.0,
                 label="kicked trajectory, first turn")
    add_kicker_regions_cartesian(ax_ring, config["BumpKicker"]["kickers"], labels=True)
    ax_ring.plot(septum_blade_x, septum_blade_y, color="#15803d", linewidth=5.0,
                 solid_capstyle="butt", label="0.5 m septum centerline")
    ax_ring.scatter(
        [septum_blade_x[0]], [septum_blade_y[0]], s=42, color="#15803d",
        edgecolors="black", linewidths=0.6, zorder=8,
    )
    ax_ring.set_aspect("equal", adjustable="box")
    ax_ring.set_xlabel("x [m]")
    ax_ring.set_ylabel("y [m]")
    ax_ring.set_title("Ring layout: field map, kickers, trajectory and septum")
    ax_ring.grid(True, alpha=0.2)
    ax_ring.legend(loc="upper right", fontsize=8)
    fig.colorbar(field, ax=ax_ring, pad=0.02, shrink=0.86, label="base Bz [T]")

    ax_zoom.plot(septum_seo_x, septum_seo_y, color="#333333", linestyle="--",
                 linewidth=2.0, label="closed orbit")
    ax_zoom.plot(septum_kicked_x, septum_kicked_y, color="#d62828", linewidth=2.4,
                 label="kicked trajectory")
    ax_zoom.plot(septum_blade_x, septum_blade_y, color="#15803d", linewidth=5.0,
                 solid_capstyle="butt", label="septum centerline (zero thickness)")
    marker_indices = [0, len(septum_phi) // 2, len(septum_phi) - 1]
    for local_idx in marker_indices:
        ax_zoom.plot(
            [septum_seo_x[local_idx], septum_kicked_x[local_idx]],
            [septum_seo_y[local_idx], septum_kicked_y[local_idx]],
            color="#6b7280", linewidth=0.9, linestyle=":",
        )
        ax_zoom.annotate(
            f"{separation_mm[local_idx]:.1f} mm",
            xy=(septum_kicked_x[local_idx], septum_kicked_y[local_idx]),
            xytext=(5, 4), textcoords="offset points", fontsize=8,
        )
    margin = 0.035
    ax_zoom.set_xlim(
        min(np.min(septum_seo_x), np.min(septum_kicked_x)) - margin,
        max(np.max(septum_seo_x), np.max(septum_kicked_x)) + margin,
    )
    ax_zoom.set_ylim(
        min(np.min(septum_seo_y), np.min(septum_kicked_y)) - margin,
        max(np.max(septum_seo_y), np.max(septum_kicked_y)) + margin,
    )
    ax_zoom.set_aspect("equal", adjustable="box")
    ax_zoom.set_xlabel("x [m]")
    ax_zoom.set_ylabel("y [m]")
    ax_zoom.set_title(
        "Septum zoom: blade centerline placed midway between orbit centroids"
    )
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"600 MeV horizontal extraction with four 5 mrad kickers; "
        f"minimum septum separation = {np.min(separation_mm):.2f} mm\n"
        "Blade thickness and beam envelope are not included",
        fontsize=14,
    )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = csv_path.parent / "single_particle_600MeV_septum_layout_0p5m.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)

    print(f"Septum phi range     : {septum['phi_unwrapped_deg'][0]:.6f} to "
          f"{septum['phi_unwrapped_deg'][-1]:.6f} deg")
    print(f"Septum center length : {septum['centerline_s_m'][-1]:.6f} m")
    print(f"Separation min/max   : {np.min(separation_mm):.6f} / "
          f"{np.max(separation_mm):.6f} mm")
    print(f"Max |base Bz|        : {np.max(np.abs(bz_local)):.9f} T")
    print(f"Saved figure         : {output_path}")


if __name__ == "__main__":
    main()
