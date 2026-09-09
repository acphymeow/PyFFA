import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from main_track import load_bmap_auto
from tool_compare_kicker_splits_bunch_phase_space import (
    entrance_coordinates,
    make_septum_descriptor,
    samples_from_case,
)
from tool_track_bunch_to_septum import (
    load_twiss_at,
    make_horizontal_waterbag,
    run_case,
)
from tool_track_extraction_kickers import load_json, load_seo, resolve_config_path


SCAN_DIR = Path(
    "extraction_kicker_three_straight_scan/run_20260909_095802"
)
ENERGY_MEV = 600.0
START_PHI_DEG = 140.0
TIME_STEP_S = 0.02e-9
WATERBAG_PARTICLES = 4096
BOUNDARY_PARTICLES = 720
EMITTANCE_EDGE_M_RAD = 500.0e-6
SPLITS = ((2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2))


def plot_phase_spaces(output, results):
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
        1, len(results), figsize=(max(10.5, 5.8 * len(results)), 7.2),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, item in zip(axes, results):
        phase = item["phase_space"]
        ax.scatter(
            phase["x_off_mm"][core], phase["xp_off_mrad"][core],
            s=4.0, color="#2878b5", alpha=0.22, linewidths=0,
            label="kicker off: waterbag",
        )
        ax.scatter(
            phase["x_on_mm"][core], phase["xp_on_mrad"][core],
            s=4.0, color="#d94b41", alpha=0.22, linewidths=0,
            label="kicker on: waterbag",
        )
        ax.plot(
            np.r_[phase["x_off_mm"][edge], phase["x_off_mm"][edge][0]],
            np.r_[phase["xp_off_mrad"][edge], phase["xp_off_mrad"][edge][0]],
            color="#145a86", lw=0.9, alpha=0.78, label="kicker off: 500π edge",
        )
        ax.plot(
            np.r_[phase["x_on_mm"][edge], phase["x_on_mm"][edge][0]],
            np.r_[phase["xp_on_mrad"][edge], phase["xp_on_mrad"][edge][0]],
            color="#a32620", lw=0.9, alpha=0.78, label="kicker on: 500π edge",
        )
        ax.axvline(0.0, color="#15803d", lw=2.2, label="septum blade")
        ax.axvspan(x_limits[0], 0.0, color="#2878b5", alpha=0.025)
        ax.axvspan(0.0, x_limits[1], color="#d94b41", alpha=0.025)
        ax.set_title(
            f"{item['label']}\n"
            f"septum {item['septum_start_deg']:.2f}–"
            f"{item['septum_end_deg']:.2f}°\n"
            f"centroids {phase['centroid_separation_mm']:.1f} mm; "
            f"free window {phase['radial_free_window_mm']:.1f} mm",
            fontsize=10.5,
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*xp_limits)
        ax.set_xlabel("radial position relative to centered septum [mm]")
        ax.grid(alpha=0.20)
    axes[0].set_ylabel("radial divergence relative to septum [mrad]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=3, fontsize=8,
    )
    fig.suptitle(
        "500π horizontal phase space at each independently optimized 0.5 m septum entrance\n"
        "5 mrad kicker layouts over three adjacent straight sections",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.76, wspace=0.08)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def load_saved_samples(path):
    data = np.load(path)
    common = {
        "station_s_m": data["station_s_m"],
        "station_phi_rad": data["station_phi_rad"],
    }
    kicked = dict(common, station_r=data["station_particle_r_m"])
    reference = dict(common, station_r=data["reference_particle_r_m"])
    return kicked, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-dir", type=Path, default=SCAN_DIR,
        help="Single-particle scan directory containing summary.json and tracks.",
    )
    parser.add_argument(
        "--labels", nargs="+", default=["+".join(map(str, item)) for item in SPLITS],
        help="Layouts to run, for example: --labels 2+2+2",
    )
    parser.add_argument(
        "--reuse-samples", action="store_true",
        help="Reuse saved bunch_phase_space NPZ files and only rebuild plots/summaries.",
    )
    args = parser.parse_args()
    split_by_label = {"+".join(map(str, item)): item for item in SPLITS}
    unknown = [label for label in args.labels if label not in split_by_label]
    if unknown:
        raise ValueError(f"Unknown layout label(s): {unknown}")
    selected_splits = [split_by_label[label] for label in args.labels]

    scan_dir = args.scan_dir.resolve()
    config_path = Path("config_track.json").resolve()
    base_config = load_json(config_path)
    scan_summary = json.loads(
        (scan_dir / "summary.json").read_text(encoding="utf-8")
    )
    scan_results = {item["label"]: item for item in scan_summary["results"]}
    bmap_path = resolve_config_path(base_config["BmapAndSEO"]["maps"], config_path)
    bmap = load_bmap_auto(
        str(bmap_path), int(base_config["BmapAndSEO"].get("max_order", 0))
    )
    seo_dir = bmap_path / base_config["BmapAndSEO"]["SEO_FileName"]
    seo = load_seo(seo_dir, ENERGY_MEV)
    beta_m, alpha = load_twiss_at(seo_dir, ENERGY_MEV, START_PHI_DEG)
    distribution = make_horizontal_waterbag(
        seo, START_PHI_DEG, beta_m, alpha, EMITTANCE_EDGE_M_RAD,
        WATERBAG_PARTICLES, BOUNDARY_PARTICLES,
    )
    phase_dir = scan_dir / "bunch_phase_space"
    phase_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for split in selected_splits:
        label = "+".join(map(str, split))
        stem = "_".join(map(str, split))
        scan_item = scan_results[label]
        best = scan_item["septum"]
        case_config = load_json(scan_dir / f"config_split_{stem}.json")
        single_track = np.genfromtxt(
            scan_dir / f"track_split_{stem}.csv", delimiter=",", names=True
        )
        septum = make_septum_descriptor(
            single_track, seo, float(best["start_deg"]), float(best["end_deg"])
        )
        target_time_s = (
            (float(best["end_deg"]) - START_PHI_DEG) / 360.0
            * seo["revolution_period_s"]
            + 20.0e-9
        )
        sample_path = phase_dir / f"split_{stem}_optimized_septum_samples.npz"
        if args.reuse_samples:
            if not sample_path.is_file():
                raise FileNotFoundError(sample_path)
            print(f"Reusing samples for {label} ...", flush=True)
            kicked, reference = load_saved_samples(sample_path)
        else:
            print(
                f"Tracking 500pi bunch for {label} to {best['start_deg']:.3f} deg ...",
                flush=True,
            )
            case_on = run_case(
                f"bunch_split_{stem}", bmap, case_config, seo, config_path,
                distribution, septum, TIME_STEP_S, target_time_s, True,
            )
            case_off = run_case(
                f"reference_split_{stem}", bmap, case_config, seo, config_path,
                distribution, septum, TIME_STEP_S, target_time_s, False,
            )
            kicked = samples_from_case(case_on)
            reference = samples_from_case(case_off)
            np.savez_compressed(
                sample_path,
                station_s_m=kicked["station_s_m"],
                station_phi_rad=kicked["station_phi_rad"],
                station_particle_r_m=kicked["station_r"],
                reference_particle_r_m=reference["station_r"],
            )
        results.append(
            {
                "label": label,
                "septum_start_deg": float(np.rad2deg(kicked["station_phi_rad"][0])),
                "septum_end_deg": float(np.rad2deg(kicked["station_phi_rad"][-1])),
                "phase_space": entrance_coordinates(kicked, reference),
            }
        )

    selection_tag = "__".join(label.replace("+", "_") for label in args.labels)
    output = scan_dir / (
        f"04_three_straight_septum_entrance_phase_space_500pi_{selection_tag}.png"
    )
    plot_phase_spaces(output, results)
    scalar_results = []
    for item in results:
        phase = item["phase_space"]
        scalar_results.append(
            {
                "label": item["label"],
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
            "blade_placement": "midway between on/off centroids for each layout",
            "RF": False,
            "space_charge": False,
        },
        "results": scalar_results,
    }
    summary_path = scan_dir / f"bunch_phase_space_summary_{selection_tag}.json"
    summary_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(serializable, indent=2, ensure_ascii=False))
    print(f"Figure: {output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
