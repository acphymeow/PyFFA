import os
from datetime import datetime
from itertools import groupby
from operator import itemgetter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from FFAG_Bunch import FFAG_Bunch
from FFAG_Distribution import LoadSEOParams
from FFAG_Field import FFAG_BField_Error, FFAG_Bfield_analytical, FFAG_EField_new
from FFAG_ParasAndConversion import FFAG_ConversionTools, FFAG_GlobalParameters
from FFAG_dump import StepDump
from FFAG_track import FFAG_RungeKutta, FunctionForAccelerationBunch_dt


TARGET_EK_MEV = 300.0
TARGET_GAP_INDEX = 0
GAP_THRESHOLD_T = 0.08
MIN_GAP_WIDTH_POINTS = 2
MIN_STRAIGHT_GAP_WIDTH_DEG = 2.0

EMAP_PATH = "./Emaps_test/FrequencyCurve_noRF.txt"
OUTPUT_DIR = "./analytical_k_reverse_injection_compare_output"
TRACK_TIME_NS = 120.0
TIME_STEP_NS = 0.02
DUMP_INTERVAL_NS = 0.10
APERTURE_R_MAX_M = 12.0
APERTURE_Z_ABS_M = 0.10
SC_GRID_SIZE = (8, 16, 16, 10.0)
FIELD_COMPARE_R_M = 11.5
FIELD_COMPARE_PHI_POINTS = 1000
OPEN_OUTPUT_PLOTS = os.environ.get("PYFFA_OPEN_PLOTS", os.environ.get("PYFFA_SHOW_PLOTS", "1")).lower() not in ("0", "false", "no")
PLOTS_TO_OPEN = []

BMAP_SPECS = [
    {
        "key": "k7p63_B1p65_R465",
        "label": "Bmap_FD16_k7p63_B1p65_R465",
        "path": "./Bmaps/Bmap_FD16_k7p63_B1p65_R465",
        "color": "#eb5757",
        "max_order": 2,
    },
    {
        "key": "k7p63_B1p65_R365",
        "label": "Bmap_FD16_k7p63_B1p65_R365",
        "path": "./Bmaps/Bmap_FD16_k7p63_B1p65_R365",
        "color": "#2f80ed",
        "max_order": 2,
    },
    {
        "key": "k7p63_B1p55_R365",
        "label": "Bmap_FD16_k7p63_B1p55_R365",
        "path": "./Bmaps/Bmap_FD16_k7p63_B1p55_R365",
        "color": "#27ae60",
        "max_order": 2,
    },
]


def load_seo_table(path):
    try:
        data = np.loadtxt(path)
    except ValueError:
        data = np.loadtxt(path, skiprows=1)
    phi_axis = data[0, 1:]
    ek_axis = data[1:, 0]
    values = data[1:, 1:]
    return ek_axis, phi_axis, values


def save_and_maybe_open(fig, output_path, dpi=180, open_after_run=False):
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    if OPEN_OUTPUT_PLOTS and open_after_run:
        PLOTS_TO_OPEN.append(os.path.abspath(output_path))


def open_output_plots():
    if not OPEN_OUTPUT_PLOTS:
        return
    for path in PLOTS_TO_OPEN:
        if os.name == "nt":
            os.startfile(path)
        else:
            print(f"Open plot manually: {path}")


def select_nearest_energy(ek_axis, target_ek):
    idx = int(np.argmin(np.abs(np.asarray(ek_axis) - target_ek)))
    return idx, float(ek_axis[idx])


def load_seo_inputs(seo_path):
    ek_r, phi_full, seo_r = load_seo_table(os.path.join(seo_path, "SEO_r.txt"))
    ek_pr, phi_pr, seo_pr = load_seo_table(os.path.join(seo_path, "SEO_pr.txt"))
    ek_bz, phi_two_cell, seo_bz_two_cell = load_seo_table(os.path.join(seo_path, "Bz.txt"))
    if not np.allclose(ek_r, ek_pr) or not np.allclose(phi_full, phi_pr):
        raise ValueError(f"SEO_r.txt and SEO_pr.txt axes differ in {seo_path}")
    return {
        "ek_axis": ek_r,
        "phi_full": phi_full,
        "seo_r": seo_r,
        "seo_pr": seo_pr,
        "bz_ek_axis": ek_bz,
        "phi_two_cell": phi_two_cell,
        "seo_bz_two_cell": seo_bz_two_cell,
    }


def find_periodic_gaps(theta_rad, bz_array, threshold=GAP_THRESHOLD_T, min_gap_width=MIN_GAP_WIDTH_POINTS):
    theta_rad = np.asarray(theta_rad, dtype=np.float64)
    bz_array = np.asarray(bz_array, dtype=np.float64)
    gap_mask = np.abs(bz_array) < threshold
    gap_indices = np.where(gap_mask)[0]
    if gap_indices.size == 0:
        return []

    groups = []
    for _, group in groupby(enumerate(gap_indices), lambda ix: ix[0] - ix[1]):
        groups.append(list(map(itemgetter(1), group)))
    if len(groups) >= 2 and gap_mask[0] and gap_mask[-1]:
        groups = [groups[-1] + groups[0]] + groups[1:-1]
    groups = [group for group in groups if len(group) >= min_gap_width]
    if len(groups) == 0:
        return []

    dtheta = float(np.median(np.diff(theta_rad)))
    period_rad = float(theta_rad[-1] - theta_rad[0] + dtheta)
    period_deg = float(np.rad2deg(period_rad))
    theta_deg = np.rad2deg(theta_rad)
    gaps = []
    for group in groups:
        wraps = bool(np.any(np.diff(group) < 0))
        if wraps:
            deg_unwrapped = []
            previous = group[0]
            wrapped_part = False
            for idx in group:
                value = float(theta_deg[idx])
                if idx < previous:
                    wrapped_part = True
                if wrapped_part:
                    value += period_deg
                deg_unwrapped.append(value)
                previous = idx
            start_deg = float(theta_deg[group[0]])
            end_deg = float(theta_deg[group[-1]])
            width_deg = float((period_deg - start_deg) + end_deg)
            center_deg = float(deg_unwrapped[len(deg_unwrapped) // 2] % period_deg)
        else:
            start_deg = float(theta_deg[group[0]])
            end_deg = float(theta_deg[group[-1]])
            width_deg = float(end_deg - start_deg)
            center_deg = float(theta_deg[group[len(group) // 2]])

        gaps.append({
            "id": len(gaps) + 1,
            "indices": group,
            "start_deg": start_deg,
            "end_deg": end_deg,
            "center_deg": center_deg,
            "center_rad": float(np.deg2rad(center_deg)),
            "width_deg": width_deg,
            "npts": len(group),
            "wraps": wraps,
        })

    gaps.sort(key=lambda item: item["center_deg"])
    for idx, gap in enumerate(gaps):
        gap["id"] = idx + 1
    return gaps


def get_bfields_for_points(bmap, r, phi_global, z, max_order):
    r = np.asarray(r, dtype=np.float64)
    phi_global = np.asarray(phi_global, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    sector_period = 2.0 * np.pi / bmap.Nsectors
    phi_cell = np.mod(phi_global, sector_period)
    inj = np.ones_like(r, dtype=np.bool_)
    survive = np.ones_like(r, dtype=np.bool_)
    bz = np.zeros_like(r, dtype=np.float64)
    br = np.zeros_like(r, dtype=np.float64)
    bphi = np.zeros_like(r, dtype=np.float64)
    bmap.GetBfields(r, phi_cell, z, inj, survive, bz, br, bphi, SearchMod=False)
    if max_order >= 1:
        bmap.AddHigherOrderBmaps(r, phi_cell, z, bz, br, bphi, max_order)
    bx = br * np.cos(phi_global) - bphi * np.sin(phi_global)
    by = br * np.sin(phi_global) + bphi * np.cos(phi_global)
    b_abs = np.sqrt(br**2 + bphi**2 + bz**2)
    return bz, br, bphi, bx, by, b_abs


def select_foil_point(bmap, seo, max_order):
    seo_idx, ek_selected = select_nearest_energy(seo["ek_axis"], TARGET_EK_MEV)
    bz_idx, bz_ek_selected = select_nearest_energy(seo["bz_ek_axis"], ek_selected)
    if abs(bz_ek_selected - ek_selected) > 1e-6:
        raise ValueError(f"No matching Bz row for Ek={ek_selected:.6f} MeV.")

    phi_bz_raw = seo["phi_two_cell"]
    bz_line = seo["seo_bz_two_cell"][bz_idx]
    if phi_bz_raw[-1] > 2.0 * np.pi:
        mask = phi_bz_raw < 2.0 * np.pi
        phi_bz = phi_bz_raw[mask]
        bz_line = bz_line[mask]
    else:
        phi_bz = phi_bz_raw

    gaps = find_periodic_gaps(phi_bz, bz_line)
    straight_gaps = [gap for gap in gaps if gap["width_deg"] >= MIN_STRAIGHT_GAP_WIDTH_DEG]
    if len(straight_gaps) == 0:
        straight_gaps = gaps
    if len(straight_gaps) == 0:
        raise RuntimeError("No usable low-field straight-section gap was found.")

    selected_gap = straight_gaps[TARGET_GAP_INDEX % len(straight_gaps)]
    phi = float(selected_gap["center_rad"] % (2.0 * np.pi))
    r = float(np.interp(phi, seo["phi_full"], seo["seo_r"][seo_idx]))
    pr = float(np.interp(phi, seo["phi_full"], seo["seo_pr"][seo_idx]))
    bz = float(get_bfields_for_points(bmap, np.array([r]), np.array([phi]), np.array([0.0]), max_order)[0][0])
    return {
        "seo_idx": seo_idx,
        "ek_selected": ek_selected,
        "phi_bz": phi_bz,
        "bz_line": bz_line,
        "gaps": gaps,
        "straight_gaps": straight_gaps,
        "selected_gap": selected_gap,
        "selected_point": {
            "Ek_MeV": ek_selected,
            "phi_rad": phi,
            "phi_deg": float(np.rad2deg(phi)),
            "r_m": r,
            "pr": pr,
            "z_m": 0.0,
            "pz": 0.0,
            "x_m": float(r * np.cos(phi)),
            "y_m": float(r * np.sin(phi)),
            "bz_T": bz,
            "gap_width_deg": selected_gap["width_deg"],
        },
    }


def make_reverse_particle(point):
    particle_p = np.zeros((1, 16), dtype=np.float64)
    particle_p[0, 0] = point["r_m"]
    particle_p[0, 1] = point["pr"]
    particle_p[0, 2] = point["z_m"]
    particle_p[0, 3] = point["pz"]
    particle_p[0, 4] = point["phi_rad"]
    particle_p[0, 5] = point["Ek_MeV"]

    conv = FFAG_ConversionTools()
    particle_v = conv.ConvertPrzek2Vrzek_boris(particle_p)
    particle_v[:, 1] *= -1.0
    particle_v[:, 3] *= -1.0
    particle_v[:, 5] *= -1.0
    particle_p_reversed = conv.ConvertVrzek2Przek_boris(particle_v)
    return particle_p, particle_v, particle_p_reversed


def make_global_parameters(bmap, bmap_path, ek_mev):
    seo_path = os.path.join(bmap_path, "resultsSEO")
    _, _, seo_info = LoadSEOParams(os.path.join(seo_path, "SEO_ini.txt"), ek_mev)
    global_parameters = FFAG_GlobalParameters()
    global_parameters.AddBMap(bmap)
    global_parameters.AddEMap(FFAG_EField_new(EMAP_PATH, False))
    global_parameters.AddSEOInfo(seo_info)
    global_parameters.AddBHarmonics(FFAG_BField_Error([], False))
    aperture_r_max = min(float(bmap.r_max), APERTURE_R_MAX_M)
    global_parameters.Aperture_m = np.array([bmap.r_min, aperture_r_max, -APERTURE_Z_ABS_M, APERTURE_Z_ABS_M], dtype=np.float64)
    global_parameters.Aperture_enable = True
    return global_parameters, seo_info


def run_reverse_tracking(spec, bmap, seo, particle_v):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_dir = os.path.join(OUTPUT_DIR, f"dump_{spec['key']}_300MeV_gap{TARGET_GAP_INDEX}_{run_id}")
    os.makedirs(dump_dir, exist_ok=True)
    global_parameters, seo_info = make_global_parameters(bmap, spec["path"], TARGET_EK_MEV)
    step_dump = StepDump(
        start_time=-1.0e-15,
        end_time=TRACK_TIME_NS * 1e-9,
        time_interval=DUMP_INTERVAL_NS * 1e-9,
        tracked_particle_id_global=[0],
        save_folder=dump_dir,
    )
    local_parameters = {
        "stop_condition": {
            "max_stepsN": int(TRACK_TIME_NS / TIME_STEP_NS) + 2000,
            "max_turn": 2.0,
            "max_time": TRACK_TIME_NS * 1e-9,
        },
        "time_step": TIME_STEP_NS * 1e-9,
        "step_dumps": step_dump,
        "enable_SC": False,
        "SC_type": 2,
        "SC_step": 20,
    }

    bunch = FFAG_Bunch(particle_v, marcosize=1.0, sc_grid_size=SC_GRID_SIZE)
    bunch.SEO_fi_axis = seo["phi_full"]
    bunch.SEO_Ek_axis = seo["ek_axis"]
    bunch.SEO_r_matrix = seo["seo_r"]
    bunch.SEO_pr_matrix = seo["seo_pr"]
    bunch.SEO_perimeter = seo_info[:, 8]
    FFAG_RungeKutta().rk4_solve_dt_bunch3_withSC_boris(
        FunctionForAccelerationBunch_dt,
        0.0,
        bunch,
        global_parameters,
        local_parameters,
    )
    if not step_dump.save_flag:
        step_dump.dump_all_particles()
        step_dump.save_flag = True

    particle_path = os.path.join(dump_dir, "particle_0_rank_0.npz")
    if not os.path.exists(particle_path):
        raise RuntimeError(f"StepDump did not produce {particle_path}.")
    return dump_dir, particle_path


def load_and_augment_trajectory(particle_path, initial_particle_p_reversed, bmap, max_order):
    dumped = np.load(particle_path)["particles"]
    initial_row = np.hstack((initial_particle_p_reversed[0], [0.0]))
    trajectory = np.vstack((initial_row, dumped))

    r = trajectory[:, 0]
    pr = trajectory[:, 1]
    z = trajectory[:, 2]
    pz = trajectory[:, 3]
    phi = trajectory[:, 4]
    ek = trajectory[:, 5]
    time_s = trajectory[:, -1]
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    bz, br, bphi, bx, by, b_abs = get_bfields_for_points(bmap, r, phi, z, max_order)
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    return {
        "raw": trajectory,
        "r": r,
        "pr": pr,
        "z": z,
        "pz": pz,
        "phi": phi,
        "ek": ek,
        "time_s": time_s,
        "x": x,
        "y": y,
        "bz": bz,
        "br": br,
        "bphi": bphi,
        "bx": bx,
        "by": by,
        "b_abs": b_abs,
        "s": s,
    }


def first_valid_segment(traj, bmap):
    aperture_r_max = min(float(bmap.r_max), APERTURE_R_MAX_M)
    valid = (
        (traj["r"] >= bmap.r_min)
        & (traj["r"] <= aperture_r_max)
        & (np.abs(traj["z"]) <= APERTURE_Z_ABS_M)
    )
    if not valid[0]:
        raise RuntimeError("The reverse trajectory starts outside the analytical field table.")
    invalid_after_start = np.where(~valid)[0]
    end_idx = int(invalid_after_start[0] - 1) if invalid_after_start.size else traj["r"].size - 1
    sliced = {}
    for key, value in traj.items():
        if isinstance(value, np.ndarray) and value.shape[0] == traj["r"].shape[0]:
            sliced[key] = value[: end_idx + 1]
        else:
            sliced[key] = value
    return sliced


def save_trajectory_csv(traj, output_path):
    data = np.column_stack((
        traj["time_s"],
        traj["s"],
        traj["r"],
        traj["pr"],
        traj["z"],
        traj["pz"],
        traj["phi"],
        np.rad2deg(np.mod(traj["phi"], 2.0 * np.pi)),
        traj["ek"],
        traj["x"],
        traj["y"],
        traj["br"],
        traj["bphi"],
        traj["bz"],
        traj["bx"],
        traj["by"],
        traj["b_abs"],
    ))
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="time_s,s_m,r_m,pr,z_m,pz,phi_rad,phi_mod_deg,Ek_MeV,x_m,y_m,Br_T,Bphi_T,Bz_T,Bx_T,By_T,Babs_T",
        comments="",
        fmt="%.12e",
    )


def save_r_fi_bz_csv(traj, output_path):
    data = np.column_stack((traj["r"], traj["phi"], traj["bz"]))
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="r,fi,bz",
        comments="",
        fmt="%.12e",
    )


def build_bz_background(bmap, max_order, r_stride=2, phi_points=1440):
    r_axis = bmap.r_axis[::r_stride]
    phi_axis = np.linspace(0.0, 2.0 * np.pi, phi_points, endpoint=False)
    rr, pp = np.meshgrid(r_axis, phi_axis, indexing="ij")
    zz = np.zeros_like(rr)
    bz = get_bfields_for_points(bmap, rr.ravel(), pp.ravel(), zz.ravel(), max_order)[0].reshape(rr.shape)

    r_step = float(np.mean(np.diff(r_axis)))
    r_edges = np.concatenate((
        [r_axis[0] - 0.5 * r_step],
        0.5 * (r_axis[:-1] + r_axis[1:]),
        [r_axis[-1] + 0.5 * r_step],
    ))
    r_edges = np.clip(r_edges, bmap.r_min, bmap.r_max)
    phi_edges = np.linspace(0.0, 2.0 * np.pi, phi_points + 1, endpoint=True)
    rr_edges, pp_edges = np.meshgrid(r_edges, phi_edges, indexing="ij")
    return rr_edges * np.cos(pp_edges), rr_edges * np.sin(pp_edges), bz


def sample_fixed_radius_field_vs_fi(bmap, max_order, r_value):
    sector_period = 2.0 * np.pi / bmap.Nsectors
    phi_cell = np.linspace(0.0, sector_period, FIELD_COMPARE_PHI_POINTS + 1, endpoint=True)
    r = np.full_like(phi_cell, float(r_value), dtype=np.float64)
    z = np.zeros_like(phi_cell, dtype=np.float64)
    bz, br, bphi, bx, by, b_abs = get_bfields_for_points(bmap, r, phi_cell, z, max_order)
    return {
        "phi_cell": phi_cell,
        "r": r,
        "z": z,
        "Bz": bz,
        "Br": br,
        "Bphi": bphi,
        "Bx": bx,
        "By": by,
        "Babs": b_abs,
    }


def save_fixed_radius_field_csv(results, output_path):
    phi_cell = results[0]["fixed_radius_field"]["phi_cell"]
    columns = [phi_cell, np.rad2deg(phi_cell)]
    header = ["fi_rad", "fi_deg"]
    for item in results:
        field = item["fixed_radius_field"]
        columns.append(field["Bz"])
        header.append(f"Bz_{item['key']}_T")
    np.savetxt(output_path, np.column_stack(columns), delimiter=",", header=",".join(header), comments="", fmt="%.12e")


def plot_fixed_radius_field_vs_fi(results, output_path):
    phi_deg = np.rad2deg(results[0]["fixed_radius_field"]["phi_cell"])
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for item in results:
        ax.plot(
            phi_deg,
            item["fixed_radius_field"]["Bz"],
            color=item["color"],
            lw=1.7,
            label=item["label"],
        )
    ax.set_xlabel("cell azimuth fi [deg]")
    ax.set_ylabel("Bz [T]")
    ax.set_title(f"Analytical Bz comparison at R = {FIELD_COMPARE_R_M:.4f} m, z = 0")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    save_and_maybe_open(fig, output_path, dpi=180, open_after_run=True)


def plot_tracks_xy(results, output_path):
    fig, ax = plt.subplots(figsize=(9.0, 8.2), constrained_layout=True)
    for item in results:
        orbit = item["orbit"]
        traj = item["traj_valid"]
        foil = item["foil"]
        color = item["color"]
        ax.plot(orbit["x"], orbit["y"], color=color, lw=1.0, alpha=0.42, ls="--", label=f"{item['label']} 300 MeV SEO")
        ax.plot(traj["x"], traj["y"], color=color, lw=1.9, label=f"{item['label']} reverse track")
        ax.scatter([foil["x_m"]], [foil["y_m"]], color=color, marker="o", s=46, edgecolors="black", linewidths=0.5)
        ax.scatter([traj["x"][-1]], [traj["y"][-1]], color=color, marker="s", s=34, edgecolors="black", linewidths=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Analytical reverse injection tracks: k comparison")
    ax.grid(True, alpha=0.24)
    ax.legend(loc="best")
    save_and_maybe_open(fig, output_path, dpi=200)


def plot_r_and_bz_vs_s(results, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True, constrained_layout=True)
    for item in results:
        traj = item["traj_valid"]
        axes[0].plot(traj["s"], traj["r"], color=item["color"], lw=1.6, label=item["label"])
        axes[1].plot(traj["s"], traj["bz"], color=item["color"], lw=1.6, label=item["label"])
    axes[0].set_ylabel("R [m]")
    axes[0].set_title("Reverse trajectory R and Bz along s")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].set_xlabel("s along reverse trajectory [m]")
    axes[1].set_ylabel("Bz [T]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    save_and_maybe_open(fig, output_path, dpi=180)


def plot_gap_bz(results, output_path):
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    for item in results:
        phi_deg = np.rad2deg(item["phi_bz"])
        ax.plot(phi_deg, item["bz_line"], color=item["color"], lw=1.5, label=item["label"])
        gap = item["selected_gap"]
        ax.axvline(gap["center_deg"], color=item["color"], lw=1.1, ls="--")
    ax.axhline(GAP_THRESHOLD_T, color="#555555", lw=0.9, ls=":")
    ax.axhline(-GAP_THRESHOLD_T, color="#555555", lw=0.9, ls=":")
    ax.set_xlabel("phi in one-turn SEO Bz [deg]")
    ax.set_ylabel("Bz [T]")
    ax.set_title("300 MeV SEO low-field gap selection")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    save_and_maybe_open(fig, output_path, dpi=180)


def plot_track_on_own_bz(item, output_path):
    x_bg, y_bg, bz_bg = build_bz_background(item["bmap"], item["max_order"])
    vabs = float(np.nanmax(np.abs(bz_bg)))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    traj = item["traj_valid"]
    orbit = item["orbit"]
    foil = item["foil"]

    fig, ax = plt.subplots(figsize=(9.0, 8.0), constrained_layout=True)
    mesh = ax.pcolormesh(x_bg, y_bg, bz_bg, shading="auto", cmap="RdBu_r", norm=norm)
    ax.plot(orbit["x"], orbit["y"], color="white", lw=1.0, alpha=0.82, label="300 MeV SEO")
    ax.plot(traj["x"], traj["y"], color="#111111", lw=1.8, label="reverse track")
    ax.scatter([foil["x_m"]], [foil["y_m"]], c="#f2c94c", s=72, edgecolors="black", linewidths=0.8, label="foil point")
    ax.scatter([traj["x"][-1]], [traj["y"][-1]], c="#eb5757", s=42, edgecolors="black", linewidths=0.55, label="valid end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{item['label']} reverse injection trajectory on Bz")
    ax.text(
        0.015,
        0.015,
        f"foil: phi={foil['phi_deg']:.3f} deg, R={foil['r_m']:.8f} m\n"
        f"end: R={traj['r'][-1]:.8f} m, phi={np.rad2deg(np.mod(traj['phi'][-1], 2*np.pi)):.3f} deg\n"
        f"time={traj['time_s'][-1]*1e9:.2f} ns, s={traj['s'][-1]:.3f} m",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 4},
    )
    ax.legend(loc="upper right")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("Bz [T]")
    save_and_maybe_open(fig, output_path, dpi=200)


def save_summary_csv(results, output_path):
    rows = []
    for item in results:
        foil = item["foil"]
        traj = item["traj_valid"]
        rows.append([
            item["key"],
            item["label"],
            foil["Ek_MeV"],
            foil["phi_rad"],
            foil["phi_deg"],
            foil["r_m"],
            foil["pr"],
            foil["bz_T"],
            foil["gap_width_deg"],
            traj["time_s"][-1] * 1e9,
            traj["s"][-1],
            np.nanmin(traj["r"]),
            np.nanmax(traj["r"]),
            traj["r"][-1],
            np.rad2deg(np.mod(traj["phi"][-1], 2.0 * np.pi)),
            np.nanmin(traj["bz"]),
            np.nanmax(traj["bz"]),
        ])
    header = (
        "key,label,Ek_MeV,foil_phi_rad,foil_phi_deg,foil_R_m,foil_pr,foil_Bz_T,"
        "gap_width_deg,track_time_ns,path_length_m,R_min_m,R_max_m,end_R_m,end_phi_deg,Bz_min_T,Bz_max_T"
    )
    np.savetxt(output_path, np.asarray(rows, dtype=object), delimiter=",", header=header, comments="", fmt="%s")


def process_one_spec(spec):
    bmap = FFAG_Bfield_analytical(os.path.join(spec["path"], "config_Bmap.json"), spec["max_order"])
    seo = load_seo_inputs(os.path.join(spec["path"], "resultsSEO"))
    foil_pack = select_foil_point(bmap, seo, spec["max_order"])
    particle_p, particle_v, particle_p_reversed = make_reverse_particle(foil_pack["selected_point"])
    dump_dir, particle_path = run_reverse_tracking(spec, bmap, seo, particle_v)
    traj = load_and_augment_trajectory(particle_path, particle_p_reversed, bmap, spec["max_order"])
    traj_valid = first_valid_segment(traj, bmap)

    seo_idx = foil_pack["seo_idx"]
    phi_full = seo["phi_full"]
    r_row = seo["seo_r"][seo_idx]
    orbit = {
        "phi": phi_full,
        "r": r_row,
        "x": r_row * np.cos(phi_full),
        "y": r_row * np.sin(phi_full),
    }

    item = dict(spec)
    item.update({
        "bmap": bmap,
        "seo": seo,
        "dump_dir": dump_dir,
        "particle_path": particle_path,
        "foil": foil_pack["selected_point"],
        "selected_gap": foil_pack["selected_gap"],
        "phi_bz": foil_pack["phi_bz"],
        "bz_line": foil_pack["bz_line"],
        "orbit": orbit,
        "traj": traj,
        "traj_valid": traj_valid,
    })
    return item


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    for spec in BMAP_SPECS:
        print(f"Processing {spec['label']} ...")
        item = process_one_spec(spec)
        item["fixed_radius_field"] = sample_fixed_radius_field_vs_fi(
            item["bmap"],
            item["max_order"],
            FIELD_COMPARE_R_M,
        )
        results.append(item)
        traj_csv = os.path.join(OUTPUT_DIR, f"{spec['key']}_300MeV_reverse_injection_Rle12_track.csv")
        r_fi_bz_csv = os.path.join(OUTPUT_DIR, f"{spec['key']}_300MeV_reverse_injection_Rle12_r_fi_bz.csv")
        save_trajectory_csv(item["traj_valid"], traj_csv)
        save_r_fi_bz_csv(item["traj_valid"], r_fi_bz_csv)
        item["traj_csv"] = traj_csv
        item["r_fi_bz_csv"] = r_fi_bz_csv
        plot_track_on_own_bz(item, os.path.join(OUTPUT_DIR, f"{spec['key']}_reverse_injection_on_Bz.png"))

    summary_csv = os.path.join(OUTPUT_DIR, "compare_k_reverse_injection_summary.csv")
    tracks_xy_png = os.path.join(OUTPUT_DIR, "compare_k_reverse_tracks_xy.png")
    r_bz_s_png = os.path.join(OUTPUT_DIR, "compare_k_reverse_R_and_Bz_vs_s.png")
    gap_bz_png = os.path.join(OUTPUT_DIR, "compare_k_gap_Bz_selection.png")
    fixed_radius_field_csv = os.path.join(OUTPUT_DIR, f"compare_k_Bz_vs_fi_R{FIELD_COMPARE_R_M:.2f}m.csv")
    fixed_radius_field_png = os.path.join(OUTPUT_DIR, f"compare_k_Bz_vs_fi_R{FIELD_COMPARE_R_M:.2f}m.png")
    save_summary_csv(results, summary_csv)
    save_fixed_radius_field_csv(results, fixed_radius_field_csv)
    plot_tracks_xy(results, tracks_xy_png)
    plot_r_and_bz_vs_s(results, r_bz_s_png)
    plot_gap_bz(results, gap_bz_png)
    plot_fixed_radius_field_vs_fi(results, fixed_radius_field_png)

    print(f"Output dir          : {os.path.abspath(OUTPUT_DIR)}")
    print(f"Target Ek           : {TARGET_EK_MEV:.6f} MeV")
    print(f"Target gap index    : {TARGET_GAP_INDEX}")
    print(f"Aperture Rmax       : {APERTURE_R_MAX_M:.6f} m")
    print(f"Field compare R     : {FIELD_COMPARE_R_M:.6f} m")
    for item in results:
        foil = item["foil"]
        traj = item["traj_valid"]
        fixed_field = item["fixed_radius_field"]
        print(f"{item['label']}:")
        print(f"  foil phi/R/pr/Bz  : {foil['phi_deg']:.6f} deg, {foil['r_m']:.10f} m, {foil['pr']:.10e}, {foil['bz_T']:.6e} T")
        print(f"  gap width         : {foil['gap_width_deg']:.6f} deg")
        print(f"  valid samples     : {traj['r'].size}")
        print(f"  valid time/path   : {traj['time_s'][-1] * 1e9:.6f} ns, {traj['s'][-1]:.6f} m")
        print(f"  R range/end       : {np.nanmin(traj['r']):.6f} to {np.nanmax(traj['r']):.6f} m, end {traj['r'][-1]:.10f} m")
        print(f"  fixed-R Bz range  : {np.nanmin(fixed_field['Bz']):.6e} to {np.nanmax(fixed_field['Bz']):.6e} T")
        print(f"  track CSV         : {os.path.abspath(item['traj_csv'])}")
        print(f"  r-fi-bz CSV       : {os.path.abspath(item['r_fi_bz_csv'])}")
    print(f"Saved summary CSV   : {os.path.abspath(summary_csv)}")
    print(f"Saved fixed-R CSV   : {os.path.abspath(fixed_radius_field_csv)}")
    print(f"Saved xy plot       : {os.path.abspath(tracks_xy_png)}")
    print(f"Saved R/Bz vs s     : {os.path.abspath(r_bz_s_png)}")
    print(f"Saved gap Bz plot   : {os.path.abspath(gap_bz_png)}")
    print(f"Saved fixed-R Bz    : {os.path.abspath(fixed_radius_field_png)}")
    open_output_plots()


if __name__ == "__main__":
    main()
