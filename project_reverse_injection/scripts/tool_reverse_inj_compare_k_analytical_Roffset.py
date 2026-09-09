import os

import numpy as np

import tool_reverse_inj_compare_k_analytical as base


TARGET_EK_MEV = 300.0
RADIAL_OFFSET_M = -0.030
OUTPUT_DIR = "./analytical_k_reverse_injection_Roffset_compare_output"


def select_offset_foil_point(bmap, seo, max_order):
    foil_pack = base.select_foil_point(bmap, seo, max_order)
    closed_point = foil_pack["selected_point"]

    point = dict(closed_point)
    point["closed_r_m"] = closed_point["r_m"]
    point["closed_pr"] = closed_point["pr"]
    point["closed_bz_T"] = closed_point["bz_T"]
    point["radial_offset_m"] = RADIAL_OFFSET_M
    point["r_m"] = closed_point["r_m"] + RADIAL_OFFSET_M
    point["x_m"] = float(point["r_m"] * np.cos(point["phi_rad"]))
    point["y_m"] = float(point["r_m"] * np.sin(point["phi_rad"]))

    bz = base.get_bfields_for_points(
        bmap,
        np.array([point["r_m"]]),
        np.array([point["phi_rad"]]),
        np.array([0.0]),
        max_order,
    )[0]
    point["bz_T"] = float(bz[0])

    foil_pack["selected_point_closed_orbit"] = closed_point
    foil_pack["selected_point"] = point
    return foil_pack


def process_one_spec(spec):
    bmap = base.FFAG_Bfield_analytical(os.path.join(spec["path"], "config_Bmap.json"), spec["max_order"])
    seo = base.load_seo_inputs(os.path.join(spec["path"], "resultsSEO"))
    foil_pack = select_offset_foil_point(bmap, seo, spec["max_order"])
    particle_p, particle_v, particle_p_reversed = base.make_reverse_particle(foil_pack["selected_point"])
    dump_dir, particle_path = base.run_reverse_tracking(spec, bmap, seo, particle_v)
    traj = base.load_and_augment_trajectory(particle_path, particle_p_reversed, bmap, spec["max_order"])
    traj_valid = base.first_valid_segment(traj, bmap)

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
        "closed_foil": foil_pack["selected_point_closed_orbit"],
        "selected_gap": foil_pack["selected_gap"],
        "phi_bz": foil_pack["phi_bz"],
        "bz_line": foil_pack["bz_line"],
        "orbit": orbit,
        "traj": traj,
        "traj_valid": traj_valid,
    })
    return item


def save_offset_summary_csv(results, output_path):
    rows = []
    for item in results:
        foil = item["foil"]
        closed = item["closed_foil"]
        traj = item["traj_valid"]
        rows.append([
            item["key"],
            item["label"],
            foil["Ek_MeV"],
            foil["phi_rad"],
            foil["phi_deg"],
            closed["r_m"],
            closed["pr"],
            closed["bz_T"],
            foil["radial_offset_m"],
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
        "key,label,Ek_MeV,phi_rad,phi_deg,"
        "closed_R_m,closed_pr,closed_Bz_T,radial_offset_m,"
        "start_R_m,start_pr,start_Bz_T,gap_width_deg,"
        "track_time_ns,path_length_m,R_min_m,R_max_m,end_R_m,end_phi_deg,Bz_min_T,Bz_max_T"
    )
    np.savetxt(output_path, np.asarray(rows, dtype=object), delimiter=",", header=header, comments="", fmt="%s")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base.OUTPUT_DIR = OUTPUT_DIR

    results = []
    for spec in base.BMAP_SPECS:
        print(f"Processing {spec['label']} with radial offset {RADIAL_OFFSET_M:+.3f} m ...")
        item = process_one_spec(spec)
        item["fixed_radius_field"] = base.sample_fixed_radius_field_vs_fi(
            item["bmap"],
            item["max_order"],
            base.FIELD_COMPARE_R_M,
        )
        results.append(item)

        traj_csv = os.path.join(OUTPUT_DIR, f"{spec['key']}_300MeV_reverse_injection_Roffset_Rle12_track.csv")
        r_fi_bz_csv = os.path.join(OUTPUT_DIR, f"{spec['key']}_300MeV_reverse_injection_Roffset_Rle12_r_fi_bz.csv")
        base.save_trajectory_csv(item["traj_valid"], traj_csv)
        base.save_r_fi_bz_csv(item["traj_valid"], r_fi_bz_csv)
        item["traj_csv"] = traj_csv
        item["r_fi_bz_csv"] = r_fi_bz_csv
        base.plot_track_on_own_bz(item, os.path.join(OUTPUT_DIR, f"{spec['key']}_Roffset_reverse_injection_on_Bz.png"))

    summary_csv = os.path.join(OUTPUT_DIR, "compare_k_reverse_injection_Roffset_summary.csv")
    tracks_xy_png = os.path.join(OUTPUT_DIR, "compare_k_Roffset_reverse_tracks_xy.png")
    r_bz_s_png = os.path.join(OUTPUT_DIR, "compare_k_Roffset_reverse_R_and_Bz_vs_s.png")
    gap_bz_png = os.path.join(OUTPUT_DIR, "compare_k_Roffset_gap_Bz_selection.png")
    fixed_radius_bz_csv = os.path.join(OUTPUT_DIR, f"compare_k_Bz_vs_fi_R{base.FIELD_COMPARE_R_M:.2f}m.csv")
    fixed_radius_bz_png = os.path.join(OUTPUT_DIR, f"compare_k_Bz_vs_fi_R{base.FIELD_COMPARE_R_M:.2f}m.png")

    save_offset_summary_csv(results, summary_csv)
    base.save_fixed_radius_field_csv(results, fixed_radius_bz_csv)
    base.plot_tracks_xy(results, tracks_xy_png)
    base.plot_r_and_bz_vs_s(results, r_bz_s_png)
    base.plot_gap_bz(results, gap_bz_png)
    base.plot_fixed_radius_field_vs_fi(results, fixed_radius_bz_png)

    print(f"Output dir          : {os.path.abspath(OUTPUT_DIR)}")
    print(f"Target Ek           : {TARGET_EK_MEV:.6f} MeV")
    print(f"Radial offset       : {RADIAL_OFFSET_M:+.6f} m")
    print(f"Aperture Rmax       : {base.APERTURE_R_MAX_M:.6f} m")
    for item in results:
        foil = item["foil"]
        closed = item["closed_foil"]
        traj = item["traj_valid"]
        print(f"{item['label']}:")
        print(f"  closed phi/R/pr   : {closed['phi_deg']:.6f} deg, {closed['r_m']:.10f} m, {closed['pr']:.10e}")
        print(f"  start R/pr/Bz     : {foil['r_m']:.10f} m, {foil['pr']:.10e}, {foil['bz_T']:.6e} T")
        print(f"  valid samples     : {traj['r'].size}")
        print(f"  valid time/path   : {traj['time_s'][-1] * 1e9:.6f} ns, {traj['s'][-1]:.6f} m")
        print(f"  R range/end       : {np.nanmin(traj['r']):.6f} to {np.nanmax(traj['r']):.6f} m, end {traj['r'][-1]:.10f} m")
        print(f"  track CSV         : {os.path.abspath(item['traj_csv'])}")
        print(f"  r-fi-bz CSV       : {os.path.abspath(item['r_fi_bz_csv'])}")
    print(f"Saved summary CSV   : {os.path.abspath(summary_csv)}")
    print(f"Saved xy plot       : {os.path.abspath(tracks_xy_png)}")
    print(f"Saved R/Bz vs s     : {os.path.abspath(r_bz_s_png)}")
    print(f"Saved gap Bz plot   : {os.path.abspath(gap_bz_png)}")
    print(f"Saved fixed-R Bz    : {os.path.abspath(fixed_radius_bz_png)}")
    base.open_output_plots()


if __name__ == "__main__":
    main()
