import os
from pathlib import Path

import numpy as np

import tool_reverse_inj_compare_k_analytical as base


# ===== input here =====
BMAP_DIR = "./Bmaps/Bmap_FD16_k7p63_B1p45_R165_Fp0p5"
OUTPUT_DIR = "./reverse_one_bmap_output"
TARGET_EK_MEV = 300.0
RADIAL_OFFSETS_M = [-0.030]  # 0 means closed orbit point; -0.030 means R_closed - 30 mm
MAX_ORDER = 2
TRACK_TIME_NS = 120.0
TIME_STEP_NS = 0.02
DUMP_INTERVAL_NS = 0.10
APERTURE_R_MAX_M = 12.0


def off_key(v):
    mm = int(round(v * 1000.0))
    return "off0mm" if mm == 0 else f"off{'p' if mm > 0 else 'm'}{abs(mm)}mm"


def make_offset_point(p0, bmap, offset_m):
    p = dict(p0)
    p["closed_r_m"], p["closed_pr"], p["radial_offset_m"] = p0["r_m"], p0["pr"], float(offset_m)
    p["r_m"] = p0["r_m"] + float(offset_m)
    p["x_m"], p["y_m"] = float(p["r_m"] * np.cos(p["phi_rad"])), float(p["r_m"] * np.sin(p["phi_rad"]))
    p["bz_T"] = float(base.get_bfields_for_points(bmap, [p["r_m"]], [p["phi_rad"]], [0.0], MAX_ORDER)[0][0])
    return p


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    base.OUTPUT_DIR, base.TARGET_EK_MEV = OUTPUT_DIR, TARGET_EK_MEV
    base.TRACK_TIME_NS, base.TIME_STEP_NS, base.DUMP_INTERVAL_NS = TRACK_TIME_NS, TIME_STEP_NS, DUMP_INTERVAL_NS
    base.APERTURE_R_MAX_M = APERTURE_R_MAX_M
    base.OPEN_OUTPUT_PLOTS = False

    bmap_dir = Path(BMAP_DIR)
    tag = bmap_dir.name.replace("Bmap_FD16_", "")
    bmap = base.FFAG_Bfield_analytical(str(bmap_dir / "config_Bmap.json"), MAX_ORDER)
    seo = base.load_seo_inputs(str(bmap_dir / "resultsSEO"))
    foil_pack = base.select_foil_point(bmap, seo, MAX_ORDER)
    closed = foil_pack["selected_point"]
    rows = []

    for offset_m in RADIAL_OFFSETS_M:
        key = f"{tag}_{off_key(offset_m)}"
        point = make_offset_point(closed, bmap, offset_m)
        spec = {"key": key, "label": key, "path": str(bmap_dir).replace("\\", "/"), "max_order": MAX_ORDER}
        _, pv, pp_rev = base.make_reverse_particle(point)
        _, particle_path = base.run_reverse_tracking(spec, bmap, seo, pv)
        traj = base.first_valid_segment(base.load_and_augment_trajectory(particle_path, pp_rev, bmap, MAX_ORDER), bmap)

        track_csv = Path(OUTPUT_DIR) / f"{key}_track.csv"
        r_fi_bz_csv = Path(OUTPUT_DIR) / f"{key}_r_fi_bz.csv"
        base.save_trajectory_csv(traj, track_csv)
        base.save_r_fi_bz_csv(traj, r_fi_bz_csv)
        rows.append([key, offset_m, point["phi_deg"], point["closed_r_m"], point["r_m"], point["pr"],
                     point["bz_T"], traj["s"][-1], np.nanmax(np.abs(traj["bz"])), track_csv, r_fi_bz_csv])
        print(f"{key}: max|Bz|={np.nanmax(np.abs(traj['bz'])):.6f} T, s={traj['s'][-1]:.6f} m")
        print(f"  {track_csv}")
        print(f"  {r_fi_bz_csv}")

    summary = Path(OUTPUT_DIR) / f"{tag}_reverse_summary.csv"
    header = "key,offset_m,foil_phi_deg,closed_R_m,start_R_m,start_pr,start_Bz_T,path_length_m,max_abs_Bz_T,track_csv,r_fi_bz_csv"
    np.savetxt(summary, np.asarray(rows, dtype=object), delimiter=",", header=header, comments="", fmt="%s")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
