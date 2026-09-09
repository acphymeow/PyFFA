import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from FFAG_Bunch import FFAG_Bunch
from FFAG_Distribution import LoadSEOParams
from FFAG_Field import FFAG_BField_Error, FFAG_Bfield_3D_opera, FFAG_EField_new
from FFAG_ParasAndConversion import FFAG_ConversionTools, FFAG_GlobalParameters
from FFAG_dump import StepDump
from FFAG_track import FFAG_RungeKutta, FunctionForAccelerationBunch_dt
from tool_inj_gap_opera3d import (
    BMAP_PATH,
    GAP_THRESHOLD_T,
    MIN_STRAIGHT_GAP_WIDTH_DEG,
    OUTPUT_DIR as GAP_OUTPUT_DIR,
    SEO_FOLDER,
    TARGET_EK_MEV,
    TARGET_GAP_INDEX,
    build_bz_background,
    find_periodic_gaps,
    gap_points_from_full_ring,
    load_seo_inputs,
    make_bz_norm,
    select_nearest_energy,
)


EMAP_PATH = "./Emaps_test/FrequencyCurve_noRF.txt"
OUTPUT_DIR = "./opera3d_reverse_track_output"
TRACK_TIME_NS = 120.0
TIME_STEP_NS = 0.02
DUMP_INTERVAL_NS = 0.10
SC_GRID_SIZE = (8, 16, 16, 10.0)
APERTURE_R_MAX_M = 12.0


def select_foil_point(bmap, seo_path):
    seo = load_seo_inputs(seo_path)
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

    gaps = find_periodic_gaps(phi_bz, bz_line, threshold=GAP_THRESHOLD_T, min_gap_width=2)
    straight_gaps = [gap for gap in gaps if gap["width_deg"] >= MIN_STRAIGHT_GAP_WIDTH_DEG]
    if len(straight_gaps) == 0:
        straight_gaps = gaps
    if len(straight_gaps) == 0:
        raise RuntimeError("No usable low-field straight-section gap was found.")

    selected_gap = straight_gaps[TARGET_GAP_INDEX % len(straight_gaps)]
    points = gap_points_from_full_ring(
        straight_gaps,
        seo["phi_full"],
        seo["seo_r"][seo_idx],
        seo["seo_pr"][seo_idx],
        ek_selected,
        bmap,
    )
    selected_point = [point for point in points if point["gap_id_in_template"] == selected_gap["id"]][0]

    return {
        "seo": seo,
        "seo_idx": seo_idx,
        "ek_selected": ek_selected,
        "phi_full": seo["phi_full"],
        "r_row": seo["seo_r"][seo_idx],
        "pr_row": seo["seo_pr"][seo_idx],
        "straight_gaps": straight_gaps,
        "gap_points": points,
        "selected_gap": selected_gap,
        "selected_point": selected_point,
    }


def make_reverse_particle(point):
    particle_p = np.zeros((1, 16), dtype=np.float64)
    particle_p[0, 0] = point["r_m"]
    particle_p[0, 1] = point["pr"]
    particle_p[0, 2] = 0.0
    particle_p[0, 3] = 0.0
    particle_p[0, 4] = point["phi_rad"]
    particle_p[0, 5] = point["Ek_MeV"]
    particle_p[0, 6] = 0.0
    particle_p[0, 7] = 0.0
    particle_p[0, 8] = 0.0
    particle_p[0, 9] = 0.0
    particle_p[0, 13] = 0.0
    particle_p[0, 14] = 0.0
    particle_p[0, 15] = 0.0

    conv = FFAG_ConversionTools()
    particle_v = conv.ConvertPrzek2Vrzek_boris(particle_p)
    particle_v[:, 1] *= -1.0
    particle_v[:, 3] *= -1.0
    particle_v[:, 5] *= -1.0
    particle_p_reversed = conv.ConvertVrzek2Przek_boris(particle_v)
    return particle_p, particle_v, particle_p_reversed


def make_global_parameters(bmap, seo_path, ek_mev):
    _, _, seo_info = LoadSEOParams(os.path.join(seo_path, "SEO_ini.txt"), ek_mev)
    global_parameters = FFAG_GlobalParameters()
    global_parameters.AddBMap(bmap)
    global_parameters.AddEMap(FFAG_EField_new(EMAP_PATH, False))
    global_parameters.AddSEOInfo(seo_info)
    global_parameters.AddBHarmonics(FFAG_BField_Error([], False))
    aperture_r_max = min(float(bmap.r_max), APERTURE_R_MAX_M)
    global_parameters.Aperture_m = np.array([bmap.r_min, aperture_r_max, -0.10, 0.10], dtype=np.float64)
    global_parameters.Aperture_enable = True
    return global_parameters, seo_info


def run_reverse_tracking(bmap, seo_pack, particle_v):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_dir = os.path.join(OUTPUT_DIR, f"dump_300MeV_gap{TARGET_GAP_INDEX}_{run_id}")

    seo_path = os.path.join(BMAP_PATH, SEO_FOLDER)
    global_parameters, seo_info = make_global_parameters(bmap, seo_path, seo_pack["ek_selected"])
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
    bunch.SEO_fi_axis = seo_pack["seo"]["phi_full"]
    bunch.SEO_Ek_axis = seo_pack["seo"]["ek_axis"]
    bunch.SEO_r_matrix = seo_pack["seo"]["seo_r"]
    bunch.SEO_pr_matrix = seo_pack["seo"]["seo_pr"]
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
    return bunch, dump_dir, particle_path


def load_and_augment_trajectory(particle_path, initial_particle_p_reversed, bmap):
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
    inj = np.ones_like(r, dtype=np.bool_)
    survive = np.ones_like(r, dtype=np.bool_)
    bz = np.zeros_like(r, dtype=np.float64)
    br = np.zeros_like(r, dtype=np.float64)
    bphi = np.zeros_like(r, dtype=np.float64)
    bmap.GetBfields(r, phi, z, inj, survive, bz, br, bphi, SearchMod=False)
    bx = br * np.cos(phi) - bphi * np.sin(phi)
    by = br * np.sin(phi) + bphi * np.cos(phi)
    b_abs = np.sqrt(br**2 + bphi**2 + bz**2)
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


def save_field_sample_csv(traj, output_path):
    data = np.column_stack((
        traj["r"],
        traj["phi"],
        traj["z"],
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
        header="r,fi,z,x,y,Br,Bphi,Bz,Bx,By,Babs",
        comments="",
        fmt="%.12e",
    )


def save_1p05_like_csv(traj, output_path):
    data = np.column_stack((traj["r"], traj["phi"], traj["bz"]))
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="r,fi,bz",
        comments="",
        fmt="%.12e",
    )


def first_valid_opera_segment(traj, bmap):
    aperture_r_max = min(float(bmap.r_max), APERTURE_R_MAX_M)
    valid = (
        (traj["r"] >= bmap.r_min)
        & (traj["r"] <= aperture_r_max)
        & (np.abs(traj["z"]) <= bmap.y_max)
    )
    if not valid[0]:
        raise RuntimeError("The reverse trajectory starts outside the OPERA 3D field table.")

    invalid_after_start = np.where(~valid)[0]
    end_idx = int(invalid_after_start[0] - 1) if invalid_after_start.size else traj["r"].size - 1
    end_idx = max(0, end_idx)
    sliced = {}
    for key, value in traj.items():
        if isinstance(value, np.ndarray) and value.shape[0] == traj["r"].shape[0]:
            sliced[key] = value[: end_idx + 1]
        else:
            sliced[key] = value
    return sliced, valid, end_idx


def plot_reverse_track_on_bz(bmap, seo_pack, traj_valid, output_path, mirror_x=False):
    x_bg, y_bg, bz_bg = build_bz_background(bmap)
    x_sign = -1.0 if mirror_x else 1.0
    x_bg = x_sign * x_bg
    norm = make_bz_norm(bz_bg)
    phi_full = seo_pack["phi_full"]
    r_row = seo_pack["r_row"]
    foil = seo_pack["selected_point"]

    fig, ax = plt.subplots(figsize=(9.0, 8.0), constrained_layout=True)
    mesh = ax.pcolormesh(x_bg, y_bg, bz_bg, shading="auto", cmap="RdBu_r", norm=norm)
    ax.plot(x_sign * r_row * np.cos(phi_full), r_row * np.sin(phi_full), color="white", lw=1.0, alpha=0.85, label="300 MeV SEO")
    ax.plot(x_sign * traj_valid["x"], traj_valid["y"], color="#111111", lw=1.8, label="reverse track, R<=12 m")
    ax.scatter([x_sign * foil["x_m"]], [foil["y_m"]], c="#f2c94c", s=78, edgecolors="black", linewidths=0.8, label="foil point")
    ax.scatter([x_sign * traj_valid["x"][-1]], [traj_valid["y"][-1]], c="#eb5757", s=48, edgecolors="black", linewidths=0.6, label="valid end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title_suffix = " mirrored about x=0" if mirror_x else ""
    ax.set_title(f"OPERA 3D reverse injection trajectory on mid-plane Bz{title_suffix}")
    ax.text(
        0.015,
        0.015,
        f"foil: phi={foil['phi_deg']:.3f} deg, r={foil['r_m']:.8f} m\n"
        f"R<=12 end: r={traj_valid['r'][-1]:.8f} m, phi={np.rad2deg(np.mod(traj_valid['phi'][-1], 2*np.pi)):.3f} deg\n"
        f"time={traj_valid['time_s'][-1]*1e9:.2f} ns, s={traj_valid['s'][-1]:.3f} m",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 4},
    )
    ax.legend(loc="upper right")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("Bz [T]")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_bz_along_track(traj, output_path, flip_s=False):
    if flip_s:
        s_plot = traj["s"][-1] - traj["s"]
        foil_s = s_plot[0]
        end_s = s_plot[-1]
        title_suffix = " flipped along horizontal axis"
    else:
        s_plot = traj["s"]
        foil_s = traj["s"][0]
        end_s = traj["s"][-1]
        title_suffix = ""

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.plot(s_plot, traj["bz"], lw=1.5, color="black")
    ax.scatter([foil_s], [traj["bz"][0]], c="#f2c94c", edgecolors="black", label="foil")
    ax.scatter([end_s], [traj["bz"][-1]], c="#eb5757", edgecolors="black", label="reverse end")
    ax.set_xlabel("s along reverse trajectory [m]")
    ax.set_ylabel("Bz [T]")
    ax.set_title(f"Bz sampled along OPERA 3D reverse injection trajectory{title_suffix}")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seo_path = os.path.join(BMAP_PATH, SEO_FOLDER)
    bmap = FFAG_Bfield_3D_opera(BMAP_PATH)
    seo_pack = select_foil_point(bmap, seo_path)
    particle_p, particle_v, particle_p_reversed = make_reverse_particle(seo_pack["selected_point"])
    _, dump_dir, particle_path = run_reverse_tracking(bmap, seo_pack, particle_v)

    traj = load_and_augment_trajectory(particle_path, particle_p_reversed, bmap)
    traj_valid, valid_mask, valid_end_idx = first_valid_opera_segment(traj, bmap)
    csv_path = os.path.join(OUTPUT_DIR, "opera3d_300MeV_reverse_injection_Rle12_track.csv")
    field_csv_path = os.path.join(OUTPUT_DIR, "opera3d_300MeV_reverse_injection_Rle12_coords_fields.csv")
    simple_field_csv_path = os.path.join(OUTPUT_DIR, "opera3d_300MeV_reverse_injection_Rle12_r_fi_bz.csv")
    overlay_png = os.path.join(OUTPUT_DIR, "opera3d_300MeV_reverse_injection_on_Bz.png")
    overlay_mirror_png = os.path.join(OUTPUT_DIR, "opera3d_300MeV_reverse_injection_on_Bz_mirror_x0.png")
    bz_png = os.path.join(OUTPUT_DIR, "opera3d_300MeV_reverse_injection_Bz_along_s.png")
    bz_flip_png = os.path.join(OUTPUT_DIR, "opera3d_300MeV_reverse_injection_Bz_along_s_flip_x.png")
    save_trajectory_csv(traj_valid, csv_path)
    save_field_sample_csv(traj_valid, field_csv_path)
    save_1p05_like_csv(traj_valid, simple_field_csv_path)
    plot_reverse_track_on_bz(bmap, seo_pack, traj_valid, overlay_png)
    plot_reverse_track_on_bz(bmap, seo_pack, traj_valid, overlay_mirror_png, mirror_x=True)
    plot_bz_along_track(traj_valid, bz_png)
    plot_bz_along_track(traj_valid, bz_flip_png, flip_s=True)

    foil = seo_pack["selected_point"]
    valid_r_min = float(np.nanmin(traj_valid["r"]))
    valid_r_max = float(np.nanmax(traj_valid["r"]))
    print(f"Gap output dir      : {os.path.abspath(GAP_OUTPUT_DIR)}")
    print(f"Reverse dump dir    : {os.path.abspath(dump_dir)}")
    print(f"Aperture Rmax       : {APERTURE_R_MAX_M:.6f} m")
    print(f"Foil point          : phi={foil['phi_deg']:.6f} deg, r={foil['r_m']:.10f} m, pr={foil['pr']:.10f}")
    print(f"Reverse initial pr  : {particle_p_reversed[0, 1]:.10f}")
    print(f"R<=12 samples       : {traj_valid['r'].size}")
    print(f"R<=12 track time    : {traj_valid['time_s'][-1] * 1e9:.6f} ns")
    print(f"R<=12 path length   : {traj_valid['s'][-1]:.6f} m")
    print(f"R<=12 r range       : {valid_r_min:.6f} m to {valid_r_max:.6f} m")
    print(f"R<=12 end point     : r={traj_valid['r'][-1]:.10f} m, phi={np.rad2deg(np.mod(traj_valid['phi'][-1], 2*np.pi)):.6f} deg")
    print(f"Saved R<=12 track   : {os.path.abspath(csv_path)}")
    print(f"Saved coord fields  : {os.path.abspath(field_csv_path)}")
    print(f"Saved r-fi-bz       : {os.path.abspath(simple_field_csv_path)}")
    print(f"Saved overlay plot  : {os.path.abspath(overlay_png)}")
    print(f"Saved mirror plot   : {os.path.abspath(overlay_mirror_png)}")
    print(f"Saved Bz plot       : {os.path.abspath(bz_png)}")
    print(f"Saved flipped Bz    : {os.path.abspath(bz_flip_png)}")


if __name__ == "__main__":
    main()
