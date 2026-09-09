import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from FFAG_Bunch import FFAG_Bunch
from FFAG_ParasAndConversion import FFAG_ConversionTools
from FFAG_track import FFAG_RungeKutta, FunctionForAccelerationBunch_dt
from main_track import load_bmap_auto
from tool_plot_extraction_with_septum import build_septum_segment
from tool_track_extraction_kickers import (
    SC_GRID_SIZE,
    load_json,
    load_seo,
    make_global_parameters,
    resolve_config_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Track a horizontal Courant-Snyder waterbag from the 600 MeV closed "
            "orbit to the configured extraction septum."
        )
    )
    parser.add_argument("-j", "--config", default="config_track.json")
    parser.add_argument(
        "--central-track-csv",
        default=(
            "extraction_single_particle_output/run_20260903_222702/"
            "single_particle_600MeV_kicker_on.csv"
        ),
    )
    parser.add_argument("--energy-mev", type=float, default=600.0)
    parser.add_argument("--start-phi-deg", type=float, default=140.0)
    parser.add_argument("--septum-start-deg", type=float, default=191.267)
    parser.add_argument("--septum-length-m", type=float, default=0.5)
    parser.add_argument(
        "--emittance-pi-mm-mrad",
        type=float,
        default=500.0,
        help=(
            "Horizontal geometric edge emittance in the project convention: "
            "500 means epsilon_edge=500e-6 m rad (the pi is part of the unit label)."
        ),
    )
    parser.add_argument("--waterbag-particles", type=int, default=4096)
    parser.add_argument("--boundary-particles", type=int, default=720)
    parser.add_argument("--time-step-ns", type=float, default=0.02)
    parser.add_argument("--max-time-ns", type=float, default=70.0)
    parser.add_argument(
        "-o", "--output-dir", default="extraction_bunch_septum_output"
    )
    return parser.parse_args()


def load_twiss_at(seo_dir, energy_mev, phi_deg):
    beta_data = np.loadtxt(seo_dir / "BetaFuncR.txt", skiprows=1)
    alpha_data = np.loadtxt(seo_dir / "AlphaFuncR.txt", skiprows=1)
    energy_axis = beta_data[1:, 0]
    row = int(np.argmin(np.abs(energy_axis - energy_mev))) + 1
    if abs(float(beta_data[row, 0]) - energy_mev) > 1.0e-6:
        raise ValueError("The radial Twiss table has no requested energy row.")
    phi_axis_deg = beta_data[0, 1:]
    beta_m = float(np.interp(phi_deg % 360.0, phi_axis_deg, beta_data[row, 1:]))
    alpha = float(np.interp(phi_deg % 360.0, phi_axis_deg, alpha_data[row, 1:]))
    if beta_m <= 0.0:
        raise ValueError(f"Invalid radial beta function: {beta_m} m.")
    return beta_m, alpha


def make_horizontal_waterbag(
    seo, start_phi_deg, beta_m, alpha, epsilon_edge_m_rad,
    waterbag_particles, boundary_particles,
):
    phi = math.radians(start_phi_deg)
    r_closed = float(np.interp(phi, seo["phi_axis"], seo["r_row"]))
    pr_closed = float(np.interp(phi, seo["phi_axis"], seo["pr_row"]))
    angle_closed = math.asin(pr_closed)

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    idx = np.arange(waterbag_particles, dtype=np.float64)
    theta_waterbag = golden_angle * idx
    rho_waterbag = np.sqrt((idx + 0.5) / waterbag_particles)

    theta_boundary = np.linspace(
        0.0, 2.0 * math.pi, boundary_particles, endpoint=False
    )
    rho_boundary = np.ones(boundary_particles, dtype=np.float64)
    theta = np.r_[theta_waterbag, theta_boundary]
    rho = np.r_[rho_waterbag, rho_boundary]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x = math.sqrt(beta_m * epsilon_edge_m_rad) * rho * cos_theta
    xp = -math.sqrt(epsilon_edge_m_rad / beta_m) * rho * (
        alpha * cos_theta + sin_theta
    )

    total = theta.size
    particle_p = np.zeros((total, 16), dtype=np.float64)
    particle_p[:, 0] = r_closed + x
    particle_p[:, 1] = angle_closed + xp
    particle_p[:, 4] = phi
    particle_p[:, 5] = seo["energy_mev"]
    particle_p[:, 6] = 0.0
    particle_p[:, 13] = 0.0
    particle_p[:, 14] = np.arange(total, dtype=np.float64)
    particle_p[:, 15] = np.arange(total, dtype=np.float64)

    gamma_twiss = (1.0 + alpha * alpha) / beta_m
    invariant = gamma_twiss * x * x + 2.0 * alpha * x * xp + beta_m * xp * xp
    return {
        "particle_p": particle_p,
        "particle_v": FFAG_ConversionTools().ConvertPrzek2Vrzek_boris(particle_p),
        "x_m": x,
        "xp_rad": xp,
        "invariant_m_rad": invariant,
        "is_boundary": np.arange(total) >= waterbag_particles,
        "r_closed_m": r_closed,
        "pr_closed": pr_closed,
        "angle_closed_rad": angle_closed,
    }


class SeptumMonitor:
    def __init__(self, initial_bunch, septum, station_count=101):
        self.save_flag = True
        self.phi_start = float(septum["phi_rad"][0])
        self.phi_end = float(septum["phi_rad"][-1])
        self.station_phi = np.linspace(self.phi_start, self.phi_end, station_count)
        self.station_s = np.interp(
            self.station_phi, septum["phi_rad"], septum["centerline_s_m"]
        )
        self.station_blade_r = np.interp(
            self.station_phi,
            septum["phi_rad"],
            np.hypot(septum["blade_x_m"], septum["blade_y_m"]),
        )
        n_particles = initial_bunch.shape[0]
        self.station_r = np.full((station_count, n_particles), np.nan)
        self.min_gap = np.full(n_particles, np.inf)
        self.max_gap = np.full(n_particles, -np.inf)
        self.entered = np.zeros(n_particles, dtype=bool)
        self.prev_r = initial_bunch[:, 0].copy()
        self.prev_phi = initial_bunch[:, 4].copy()

    def check_and_dump(self, time_pre, time_post, bunch_obj, data_type="Boris"):
        current = bunch_obj.LocalBunch
        current_r = current[:, 0]
        current_phi = current[:, 4]
        alive = bunch_obj.Step_SurviveFlag

        in_septum = alive & (current_phi >= self.phi_start) & (current_phi <= self.phi_end)
        if np.any(in_septum):
            blade_r = np.interp(
                current_phi[in_septum], self.station_phi, self.station_blade_r
            )
            gap = current_r[in_septum] - blade_r
            ids = np.flatnonzero(in_septum)
            self.min_gap[ids] = np.minimum(self.min_gap[ids], gap)
            self.max_gap[ids] = np.maximum(self.max_gap[ids], gap)
            self.entered[ids] = True

        delta_phi = current_phi - self.prev_phi
        moving = alive & np.isfinite(delta_phi) & (delta_phi > 0.0)
        for station_i, target_phi in enumerate(self.station_phi):
            missing = np.isnan(self.station_r[station_i])
            crossed = moving & missing & (self.prev_phi <= target_phi) & (current_phi >= target_phi)
            if np.any(crossed):
                fraction = (target_phi - self.prev_phi[crossed]) / delta_phi[crossed]
                self.station_r[station_i, crossed] = (
                    self.prev_r[crossed]
                    + fraction * (current_r[crossed] - self.prev_r[crossed])
                )

        self.prev_r[:] = current_r
        self.prev_phi[:] = current_phi

    def dump_all_particles(self):
        self.save_flag = True


def run_case(
    name, bmap, config, seo, config_path, distribution, septum,
    time_step_s, max_time_s, kicker_enabled,
):
    monitor = SeptumMonitor(distribution["particle_v"], septum)
    steps = int(math.ceil(max_time_s / time_step_s))
    local_parameters = {
        "stop_condition": {
            "max_stepsN": steps,
            "max_turn": 10.0,
            "max_time": max_time_s + time_step_s,
        },
        "time_step": time_step_s,
        "step_dumps": monitor,
        "enable_SC": False,
        "SC_type": 2,
        "SC_step": 20,
    }
    bunch = FFAG_Bunch(
        distribution["particle_v"].copy(), marcosize=1.0, sc_grid_size=SC_GRID_SIZE
    )
    bunch.SEO_fi_axis = seo["phi_axis"]
    bunch.SEO_Ek_axis = seo["energy_axis"]
    bunch.SEO_r_matrix = seo["r_matrix"]
    bunch.SEO_pr_matrix = seo["pr_matrix"]
    bunch.SEO_perimeter = seo["seo_info"][:, 8]

    global_parameters = make_global_parameters(
        bmap, config, seo, config_path, kicker_enabled
    )
    FFAG_RungeKutta().rk4_solve_dt_bunch3_withSC_boris(
        FunctionForAccelerationBunch_dt,
        0.0,
        bunch,
        global_parameters,
        local_parameters,
    )
    return {
        "name": name,
        "monitor": monitor,
        "bunch": bunch,
        "survived": bunch.LocalBunch[:, 8] != 0.0,
    }


def finite_stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1.0)),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def summarize_case(case, expected_side, waterbag_count):
    monitor = case["monitor"]
    entered = monitor.entered
    if expected_side == "outside":
        safe = entered & (monitor.min_gap > 0.0)
        critical_gap = monitor.min_gap
    else:
        safe = entered & (monitor.max_gap < 0.0)
        critical_gap = -monitor.max_gap
    waterbag = np.arange(entered.size) < waterbag_count
    return {
        "entered_count": int(np.count_nonzero(entered)),
        "survived_count": int(np.count_nonzero(case["survived"])),
        "safe_all_count": int(np.count_nonzero(safe)),
        "safe_all_fraction": float(np.count_nonzero(safe) / entered.size),
        "safe_waterbag_count": int(np.count_nonzero(safe & waterbag)),
        "safe_waterbag_fraction": float(
            np.count_nonzero(safe & waterbag) / waterbag_count
        ),
        "critical_clearance_m": finite_stats(critical_gap[entered]),
    }


def plot_results(run_dir, septum, kicked, reference, summary):
    phi = kicked["monitor"].station_phi
    s = kicked["monitor"].station_s
    blade_r = kicked["monitor"].station_blade_r
    kicked_r = kicked["monitor"].station_r
    reference_r = reference["monitor"].station_r

    def envelope(matrix):
        return np.nanmin(matrix, axis=1), np.nanmax(matrix, axis=1)

    kicked_min, kicked_max = envelope(kicked_r)
    ref_min, ref_max = envelope(reference_r)
    kicked_centroid = np.nanmean(kicked_r, axis=1)
    ref_centroid = np.nanmean(reference_r, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)
    ax = axes[0]
    blade_x = blade_r * np.cos(phi)
    blade_y = blade_r * np.sin(phi)

    def fill_tube(r_lo, r_hi, color, alpha, label):
        x_lo, y_lo = r_lo * np.cos(phi), r_lo * np.sin(phi)
        x_hi, y_hi = r_hi * np.cos(phi), r_hi * np.sin(phi)
        ax.fill(
            np.r_[x_lo, x_hi[::-1]],
            np.r_[y_lo, y_hi[::-1]],
            color=color,
            alpha=alpha,
            linewidth=0.8,
            edgecolor=color,
            label=label,
        )

    fill_tube(ref_min, ref_max, "#2878b5", 0.28, "kicker off: full 500π envelope")
    fill_tube(kicked_min, kicked_max, "#d94b41", 0.28, "kicker on: full 500π envelope")
    ax.plot(ref_centroid * np.cos(phi), ref_centroid * np.sin(phi), color="#145a86", lw=1.3)
    ax.plot(kicked_centroid * np.cos(phi), kicked_centroid * np.sin(phi), color="#a32620", lw=1.3)
    ax.plot(blade_x, blade_y, color="#1b8e3e", lw=3.0, label="zero-thickness septum centerline")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("0.5 m septum: multi-particle envelopes")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    ax.fill_between(
        s, 1.0e3 * (ref_min - blade_r), 1.0e3 * (ref_max - blade_r),
        color="#2878b5", alpha=0.28, label="kicker off envelope",
    )
    ax.fill_between(
        s, 1.0e3 * (kicked_min - blade_r), 1.0e3 * (kicked_max - blade_r),
        color="#d94b41", alpha=0.28, label="kicker on envelope",
    )
    ax.plot(s, 1.0e3 * (ref_centroid - blade_r), color="#145a86", lw=1.4)
    ax.plot(s, 1.0e3 * (kicked_centroid - blade_r), color="#a32620", lw=1.4)
    ax.axhline(0.0, color="#1b8e3e", lw=2.2, label="septum")
    ax.set_xlabel("distance along septum centerline [m]")
    ax.set_ylabel("radial position relative to septum [mm]")
    ax.set_title(
        "Positive = extraction side\n"
        f"minimum free blade window = {summary['minimum_free_blade_window_mm']:.2f} mm"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    output = run_dir / "bunch_600MeV_500pi_septum_0p5m.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main():
    args = parse_args()
    if args.emittance_pi_mm_mrad <= 0.0:
        raise ValueError("Emittance must be positive.")
    if args.waterbag_particles <= 0 or args.boundary_particles <= 0:
        raise ValueError("Particle counts must be positive.")

    config_path = Path(args.config).expanduser().resolve()
    central_csv = Path(args.central_track_csv).expanduser().resolve()
    config = load_json(config_path)
    central_track = np.genfromtxt(central_csv, delimiter=",", names=True)
    central_phi_deg = np.rad2deg(central_track["phi_rad"])
    septum = build_septum_segment(
        central_track, central_phi_deg, args.septum_start_deg, args.septum_length_m
    )

    bmap_path = resolve_config_path(config["BmapAndSEO"]["maps"], config_path)
    if bmap_path.is_file() and bmap_path.name == "config_Bmap.json":
        bmap_path = bmap_path.parent
    bmap = load_bmap_auto(
        str(bmap_path), int(config["BmapAndSEO"].get("max_order", 0))
    )
    seo_dir = bmap_path / config["BmapAndSEO"]["SEO_FileName"]
    seo = load_seo(seo_dir, args.energy_mev)
    beta_m, alpha = load_twiss_at(
        seo_dir, args.energy_mev, args.start_phi_deg
    )
    epsilon_edge = args.emittance_pi_mm_mrad * 1.0e-6
    distribution = make_horizontal_waterbag(
        seo,
        args.start_phi_deg,
        beta_m,
        alpha,
        epsilon_edge,
        args.waterbag_particles,
        args.boundary_particles,
    )

    run_dir = (
        Path(args.output_dir).expanduser().resolve()
        / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        run_dir / "initial_horizontal_distribution_500pi.npz",
        x_m=distribution["x_m"],
        xp_rad=distribution["xp_rad"],
        invariant_m_rad=distribution["invariant_m_rad"],
        is_boundary=distribution["is_boundary"],
    )

    time_step_s = args.time_step_ns * 1.0e-9
    max_time_s = args.max_time_ns * 1.0e-9
    print(
        f"Tracking {distribution['particle_p'].shape[0]} particles: "
        f"{args.waterbag_particles} waterbag + {args.boundary_particles} edge"
    )
    kicked = run_case(
        "kicker_on", bmap, config, seo, config_path, distribution, septum,
        time_step_s, max_time_s, True,
    )
    reference = run_case(
        "kicker_off", bmap, config, seo, config_path, distribution, septum,
        time_step_s, max_time_s, False,
    )

    kicked_min = np.nanmin(kicked["monitor"].station_r, axis=1)
    reference_max = np.nanmax(reference["monitor"].station_r, axis=1)
    free_window = kicked_min - reference_max
    min_window_i = int(np.nanargmin(free_window))
    summary = {
        "assumptions": {
            "energy_MeV": args.energy_mev,
            "horizontal_geometric_edge_emittance_m_rad": epsilon_edge,
            "emittance_label_pi_mm_mrad": args.emittance_pi_mm_mrad,
            "vertical_coordinates": "z=pz=0",
            "space_charge": False,
            "RF": False,
            "septum_thickness_m": 0.0,
            "distribution": "deterministic horizontal Courant-Snyder waterbag plus exact edge",
        },
        "initial_twiss": {
            "phi_deg": args.start_phi_deg,
            "beta_x_m": beta_m,
            "alpha_x": alpha,
            "maximum_abs_x_mm": float(np.max(np.abs(distribution["x_m"])) * 1.0e3),
            "maximum_invariant_m_rad": float(np.max(distribution["invariant_m_rad"])),
        },
        "particles": {
            "waterbag": args.waterbag_particles,
            "edge": args.boundary_particles,
            "total": int(distribution["particle_p"].shape[0]),
        },
        "septum": {
            "length_m": args.septum_length_m,
            "start_phi_deg": float(septum["phi_unwrapped_deg"][0]),
            "end_phi_deg": float(septum["phi_unwrapped_deg"][-1]),
        },
        "kicker_on": summarize_case(kicked, "outside", args.waterbag_particles),
        "kicker_off": summarize_case(reference, "inside", args.waterbag_particles),
        "minimum_free_blade_window_mm": float(np.min(free_window) * 1.0e3),
        "minimum_free_blade_window_at_s_m": float(
            kicked["monitor"].station_s[min_window_i]
        ),
        "envelopes_overlap": bool(np.any(free_window <= 0.0)),
        "time_step_ns": args.time_step_ns,
        "max_time_ns": args.max_time_ns,
    }

    for case in (kicked, reference):
        np.savez_compressed(
            run_dir / f"{case['name']}_septum_samples.npz",
            station_s_m=case["monitor"].station_s,
            station_phi_rad=case["monitor"].station_phi,
            station_blade_r_m=case["monitor"].station_blade_r,
            station_particle_r_m=case["monitor"].station_r,
            minimum_gap_m=case["monitor"].min_gap,
            maximum_gap_m=case["monitor"].max_gap,
            entered=case["monitor"].entered,
            survived=case["survived"],
        )

    figure = plot_results(run_dir, septum, kicked, reference, summary)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Figure: {figure}")
    print(f"Summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
