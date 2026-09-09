import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from main_track import load_bmap_auto
from tool_plot_extraction_with_septum import build_septum_segment
from tool_track_bunch_to_septum import (
    load_twiss_at,
    make_horizontal_waterbag,
    run_case,
)
from tool_track_extraction_kickers import load_json, load_seo, resolve_config_path


SCAN_DIR = Path("extraction_kicker_split_scan/run_20260906_174544")
BASELINE_BUNCH_DIR = Path("extraction_bunch_septum_output/run_20260904_023555")
CENTRAL_TRACK_CSV = Path(
    "extraction_single_particle_output/run_20260903_222702/"
    "single_particle_600MeV_kicker_on.csv"
)
ENERGY_MEV = 600.0
START_PHI_DEG = 140.0
SEPTUM_START_DEG = 191.267
SEPTUM_LENGTH_M = 0.5
WATERBAG_PARTICLES = 4096
BOUNDARY_PARTICLES = 720
EMITTANCE_EDGE_M_RAD = 500.0e-6
TIME_STEP_S = 0.02e-9
MAX_TIME_S = 70.0e-9
SPLITS = ((2, 2), (1, 3), (3, 1), (0, 4))


def load_baseline_samples(path):
    data = np.load(path)
    return {
        "station_s_m": data["station_s_m"],
        "station_phi_rad": data["station_phi_rad"],
        "station_r": data["station_particle_r_m"],
    }


def samples_from_case(case):
    monitor = case["monitor"]
    return {
        "station_s_m": monitor.station_s,
        "station_phi_rad": monitor.station_phi,
        "station_r": monitor.station_r,
    }


def make_septum_descriptor(track, seo, start_deg, end_deg, stations=101):
    phi_deg = np.linspace(start_deg, end_deg, stations)
    phi_rad = np.deg2rad(phi_deg)
    r_on = np.interp(phi_deg, track["phi_unwrapped_deg"], track["r_m"])
    r_off = np.interp(np.mod(phi_rad, 2.0 * np.pi), seo["phi_axis"], seo["r_row"])
    blade_r = 0.5 * (r_on + r_off)
    blade_x = blade_r * np.cos(phi_rad)
    blade_y = blade_r * np.sin(phi_rad)
    centerline_s = np.r_[
        0.0, np.cumsum(np.hypot(np.diff(blade_x), np.diff(blade_y)))
    ]
    return {
        "phi_rad": phi_rad,
        "centerline_s_m": centerline_s,
        "blade_x_m": blade_x,
        "blade_y_m": blade_y,
    }


def entrance_coordinates(kicked, reference):
    r_on = kicked["station_r"]
    r_off = reference["station_r"]
    s = kicked["station_s_m"]
    if not np.allclose(s, reference["station_s_m"], rtol=0.0, atol=1.0e-12):
        raise ValueError("Kicker-on and kicker-off station grids differ.")
    blade_r = 0.5 * (np.nanmean(r_on, axis=1) + np.nanmean(r_off, axis=1))
    ds = float(s[1] - s[0])
    blade_slope = (blade_r[1] - blade_r[0]) / ds

    x_on_mm = 1.0e3 * (r_on[0] - blade_r[0])
    x_off_mm = 1.0e3 * (r_off[0] - blade_r[0])
    xp_on_mrad = 1.0e3 * ((r_on[1] - r_on[0]) / ds - blade_slope)
    xp_off_mrad = 1.0e3 * ((r_off[1] - r_off[0]) / ds - blade_slope)
    return {
        "x_on_mm": x_on_mm,
        "xp_on_mrad": xp_on_mrad,
        "x_off_mm": x_off_mm,
        "xp_off_mrad": xp_off_mrad,
        "blade_r_m": float(blade_r[0]),
        "centroid_separation_mm": float(
            (np.nanmean(r_on[0]) - np.nanmean(r_off[0])) * 1.0e3
        ),
        "radial_free_window_mm": float(
            (np.nanmin(r_on[0]) - np.nanmax(r_off[0])) * 1.0e3
        ),
        "kicked_fraction_outside": float(np.mean(x_on_mm[:WATERBAG_PARTICLES] > 0.0)),
        "reference_fraction_inside": float(np.mean(x_off_mm[:WATERBAG_PARTICLES] < 0.0)),
    }


def plot_phase_spaces(output_path, results):
    colors = {"on": "#d94b41", "off": "#2878b5"}
    all_x = np.concatenate(
        [np.r_[item["phase_space"]["x_on_mm"], item["phase_space"]["x_off_mm"]]
         for item in results]
    )
    all_xp = np.concatenate(
        [np.r_[item["phase_space"]["xp_on_mrad"], item["phase_space"]["xp_off_mrad"]]
         for item in results]
    )
    x_pad = 0.05 * (np.max(all_x) - np.min(all_x))
    xp_pad = 0.06 * (np.max(all_xp) - np.min(all_xp))
    x_limits = (float(np.min(all_x) - x_pad), float(np.max(all_x) + x_pad))
    xp_limits = (float(np.min(all_xp) - xp_pad), float(np.max(all_xp) + xp_pad))

    edge = np.arange(WATERBAG_PARTICLES + BOUNDARY_PARTICLES) >= WATERBAG_PARTICLES
    core = ~edge
    fig, axes = plt.subplots(
        2, 2, figsize=(14.5, 10.5), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for ax, item in zip(axes.flat, results):
        phase = item["phase_space"]
        ax.scatter(
            phase["x_off_mm"][core], phase["xp_off_mrad"][core],
            s=4.0, color=colors["off"], alpha=0.22, linewidths=0,
            label="kicker off: waterbag",
        )
        ax.scatter(
            phase["x_on_mm"][core], phase["xp_on_mrad"][core],
            s=4.0, color=colors["on"], alpha=0.22, linewidths=0,
            label="kicker on: waterbag",
        )
        ax.plot(
            np.r_[phase["x_off_mm"][edge], phase["x_off_mm"][edge][0]],
            np.r_[phase["xp_off_mrad"][edge], phase["xp_off_mrad"][edge][0]],
            color="#145a86", linewidth=0.9, alpha=0.75,
            label="kicker off: 500π edge",
        )
        ax.plot(
            np.r_[phase["x_on_mm"][edge], phase["x_on_mm"][edge][0]],
            np.r_[phase["xp_on_mrad"][edge], phase["xp_on_mrad"][edge][0]],
            color="#a32620", linewidth=0.9, alpha=0.75,
            label="kicker on: 500π edge",
        )
        ax.axvline(0.0, color="#15803d", linewidth=2.2, label="septum blade")
        ax.axvspan(x_limits[0], 0.0, color=colors["off"], alpha=0.025)
        ax.axvspan(0.0, x_limits[1], color=colors["on"], alpha=0.025)
        feasibility = item["feasibility"]
        status = (
            "installation fits"
            if feasibility["overhang_mm"] <= 1.0e-6
            else f"low-field overhang: {feasibility['overhang_mm']:.0f} mm"
        )
        ax.set_title(
            f"{item['label']} split — {status}\n"
            f"septum: {item['septum_start_deg']:.2f}–{item['septum_end_deg']:.2f}°; "
            f"centroid separation = {phase['centroid_separation_mm']:.1f} mm, "
            f"free window = {phase['radial_free_window_mm']:.1f} mm",
            fontsize=10.5,
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*xp_limits)
        ax.grid(alpha=0.20)

    for ax in axes[-1, :]:
        ax.set_xlabel("radial position relative to centered septum, x [mm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("radial divergence relative to septum, x' [mrad]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    fig.suptitle(
        "500π horizontal phase space at the 0.5 m septum entrance\n"
        "Same 4816-particle input distribution; blade centered between on/off centroids in each case",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    scan_dir = SCAN_DIR.resolve()
    baseline_dir = BASELINE_BUNCH_DIR.resolve()
    config_path = Path("config_track.json").resolve()
    config = load_json(config_path)
    scan_summary = json.loads((scan_dir / "summary.json").read_text(encoding="utf-8"))
    optimized_summary = json.loads(
        (scan_dir / "optimized_septum_locations.json").read_text(encoding="utf-8")
    )

    bmap_path = resolve_config_path(config["BmapAndSEO"]["maps"], config_path)
    bmap = load_bmap_auto(
        str(bmap_path), int(config["BmapAndSEO"].get("max_order", 0))
    )
    seo_dir = bmap_path / config["BmapAndSEO"]["SEO_FileName"]
    seo = load_seo(seo_dir, ENERGY_MEV)
    beta_m, alpha = load_twiss_at(seo_dir, ENERGY_MEV, START_PHI_DEG)
    distribution = make_horizontal_waterbag(
        seo,
        START_PHI_DEG,
        beta_m,
        alpha,
        EMITTANCE_EDGE_M_RAD,
        WATERBAG_PARTICLES,
        BOUNDARY_PARTICLES,
    )
    old_initial = np.load(baseline_dir / "initial_horizontal_distribution_500pi.npz")
    if not (
        np.allclose(distribution["x_m"], old_initial["x_m"], rtol=0.0, atol=1.0e-15)
        and np.allclose(distribution["xp_rad"], old_initial["xp_rad"], rtol=0.0, atol=1.0e-15)
    ):
        raise RuntimeError("The regenerated 500pi distribution differs from the baseline run.")

    central_track = np.genfromtxt(CENTRAL_TRACK_CSV.resolve(), delimiter=",", names=True)
    central_phi_deg = np.rad2deg(central_track["phi_rad"])
    septum = build_septum_segment(
        central_track, central_phi_deg, SEPTUM_START_DEG, SEPTUM_LENGTH_M
    )
    reference = load_baseline_samples(
        baseline_dir / "kicker_off_septum_samples.npz"
    )
    baseline_on = load_baseline_samples(
        baseline_dir / "kicker_on_septum_samples.npz"
    )

    phase_dir = scan_dir / "bunch_phase_space"
    phase_dir.mkdir(parents=True, exist_ok=True)
    scan_results = {item["label"]: item for item in scan_summary["results"]}
    optimized_results = {
        item["label"]: item["best"] for item in optimized_summary["results"]
    }
    results = []
    for front, rear in SPLITS:
        label = f"{front}+{rear}"
        optimized = optimized_results[label]
        if label in ("2+2", "1+3"):
            # Both layouts select the same first downstream straight as the
            # existing high-statistics run.  Its first recorded station is
            # 191.267465 deg, 0.017 deg inside the low-field boundary.
            actual_start_deg = float(np.rad2deg(reference["station_phi_rad"][0]))
            actual_end_deg = float(np.rad2deg(reference["station_phi_rad"][-1]))
        else:
            actual_start_deg = float(optimized["start_deg"])
            actual_end_deg = float(optimized["end_deg"])

        if label == "2+2":
            kicked = baseline_on
            reference_this = reference
        elif label == "1+3":
            kicked = load_baseline_samples(
                phase_dir / "split_1_3_septum_samples.npz"
            )
            reference_this = reference
        else:
            case_config = load_json(scan_dir / f"config_split_{front}_{rear}.json")
            single_track = np.genfromtxt(
                scan_dir / f"track_split_{front}_{rear}.csv", delimiter=",", names=True
            )
            septum_this = make_septum_descriptor(
                single_track, seo, actual_start_deg, actual_end_deg
            )
            target_time_s = (
                (actual_end_deg - START_PHI_DEG) / 360.0 * seo["revolution_period_s"]
                + 20.0e-9
            )
            print(
                f"Tracking 500pi bunch for split {label} to optimized septum "
                f"{actual_start_deg:.3f} deg ...", flush=True
            )
            case_on = run_case(
                f"bunch_split_{front}_{rear}",
                bmap,
                case_config,
                seo,
                config_path,
                distribution,
                septum_this,
                TIME_STEP_S,
                target_time_s,
                True,
            )
            case_off = run_case(
                f"reference_for_split_{front}_{rear}",
                bmap,
                case_config,
                seo,
                config_path,
                distribution,
                septum_this,
                TIME_STEP_S,
                target_time_s,
                False,
            )
            kicked = samples_from_case(case_on)
            reference_this = samples_from_case(case_off)
            np.savez_compressed(
                phase_dir / f"split_{front}_{rear}_optimized_septum_samples.npz",
                station_s_m=kicked["station_s_m"],
                station_phi_rad=kicked["station_phi_rad"],
                station_particle_r_m=kicked["station_r"],
                reference_particle_r_m=reference_this["station_r"],
            )
        scan_item = scan_results[label]
        overhang_mm = 1.0e3 * (
            scan_item["front_feasibility"]["overhang_m"]
            + scan_item["rear_feasibility"]["overhang_m"]
        )
        results.append(
            {
                "label": label,
                "septum_start_deg": actual_start_deg,
                "septum_end_deg": actual_end_deg,
                "phase_space": entrance_coordinates(kicked, reference_this),
                "feasibility": {"overhang_mm": float(overhang_mm)},
            }
        )

    output = scan_dir / "04_kicker_split_optimized_septum_entrance_phase_space_500pi.png"
    plot_phase_spaces(output, results)
    scalar_results = []
    for item in results:
        phase = item["phase_space"]
        scalar_results.append(
            {
                "label": item["label"],
                "feasibility": item["feasibility"],
                "septum_start_deg": item["septum_start_deg"],
                "septum_end_deg": item["septum_end_deg"],
                "blade_r_m": phase["blade_r_m"],
                "centroid_separation_mm": phase["centroid_separation_mm"],
                "radial_free_window_mm": phase["radial_free_window_mm"],
                "kicked_waterbag_fraction_outside": phase["kicked_fraction_outside"],
                "reference_waterbag_fraction_inside": phase["reference_fraction_inside"],
            }
        )
    serializable = {
        "assumptions": {
            "energy_MeV": ENERGY_MEV,
            "geometric_edge_emittance_m_rad": EMITTANCE_EDGE_M_RAD,
            "waterbag_particles": WATERBAG_PARTICLES,
            "boundary_particles": BOUNDARY_PARTICLES,
            "blade_placement": "midway between kicker-on and kicker-off centroids at each layout",
            "RF": False,
            "space_charge": False,
        },
        "results": scalar_results,
    }
    summary_path = scan_dir / "bunch_phase_space_summary.json"
    summary_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(serializable, indent=2, ensure_ascii=False))
    print(f"Figure: {output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
