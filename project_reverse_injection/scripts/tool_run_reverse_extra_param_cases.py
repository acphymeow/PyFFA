import csv
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

import main_SEO
import tool_reverse_inj_compare_k_analytical as base
from FFAG_Field import FFAG_Bfield_analytical
from FFAG_HighOrder import BmapParams_to_ConvexParams, Generate_Bmaps_MultiProcess, calculate_s
from FFAG_ISO import CheckISOBmapConfigNew


BMAP_ROOT = Path("./Bmaps")
OUTPUT_DIR = Path("./reverse_extra_param_scan_output")
EXPAND_ORDER = 0
MAX_ORDER = 0
SEO_EK_MEV = np.array([300.0, 350.0])
RADIAL_OFFSET_M = -0.030
TRACK_TIME_NS = 120.0
TIME_STEP_NS = 0.02
DUMP_INTERVAL_NS = 0.10
APERTURE_R_MAX_M = 12.0
BHARMONICS = {"enable": False, "harmonics": []}

BASE = {"k": 7.630003, "B": 1.65, "R": 0.465074, "F": 0.0, "s": 49.973388}
CASES = [
    ("DF ratio", dict(BASE, R=0.265074)),
    ("F iron angle", dict(BASE, F=0.75)),
]


def tag_float(v, nd=2):
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "n")


def tag_r(v):
    return f"{int(round(v * 1000.0 - 0.074)):03d}"


def tag_f(v):
    if abs(v) < 1e-12:
        return "0"
    return ("p" if v > 0 else "n") + tag_float(abs(v), 3)


def folder_name(c):
    return f"Bmap_FD16_k{tag_float(c['k'], 2)}_B{tag_float(c['B'], 2)}_R{tag_r(c['R'])}_F{tag_f(c['F'])}_s{tag_float(c['s'], 2)}"


def make_config(folder, c):
    f = float(c["F"])
    bmap = {
        "Type": "SCALE",
        "NSector": 16,
        "theta_step_rad": np.deg2rad(0.005),
        "rmin_max_step_m": (10.8, 12.2, 0.001),
        "interval_1": (5.625 / 22.5, (5.9083 + f) / 22.5, 2.38755 / 22.5, (2.95415 - f) / 22.5, 5.625 / 22.5),
        "positive_or_negative_1": (0, 1.0, 0, -float(c["R"]), 0),
        "fringe_width_1": (0, 1.17, 0, 1.17, 0),
        "SpiralAngle_deg": float(c["s"]),
        "orbital_freq_MHz": 4.5,
        "k_value": float(c["k"]),
        "B0_max_T": float(c["B"]),
        "R0_m": 12.0,
    }
    bmap["B0_T"] = bmap["B0_max_T"] / (bmap["R0_m"] ** bmap["k_value"])
    return {
        "config": {"date": 0.0, "folder": str(folder).replace("\\", "/"), "expand_order": EXPAND_ORDER},
        "machine": {"energy_inj": 300, "energy_ext": 600},
        "Bmap": bmap,
        "BzInfo": {"IsoCoef": None, "IsoError": None},
    }


def bmap_complete(folder):
    return all((folder / name).exists() for name in ("config_Bmap.json", "Br_order_0.npz", "Bz_order_0.npz", "Bfi_order_0.npz"))


def generate_bmap(folder, c):
    if bmap_complete(folder):
        print(f"Bmap exists: {folder.name}")
        return
    folder.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    cfg = make_config(folder, c)
    cfg["Bmap"]["coefficients_tupel"] = CheckISOBmapConfigNew(cfg)
    cfg["Bmap"]["Bmean"], cfg["Bmap"]["r_axis"] = None, None
    seed = OUTPUT_DIR / f"{folder.name}_config_Bmap_seed.json"
    seed.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    bmap = cfg["Bmap"]
    rmin, rmax, rstep = bmap["rmin_max_step_m"]
    r_axis = np.linspace(rmin, rmax, int(round((rmax - rmin) / rstep)) + 1)
    fi_axis = np.linspace(0.0, 2.0 * np.pi / bmap["NSector"], int(round(2.0 * np.pi / bmap["NSector"] / bmap["theta_step_rad"])) + 1)
    s_fit = Polynomial.fit(r_axis, calculate_s(r_axis, 0.0, bmap["SpiralAngle_deg"]), 5)
    convex = BmapParams_to_ConvexParams(bmap)

    FFAG_Bfield_analytical(str(seed), 0, AddNumricalMap=False).UpdataConfig()
    br, bz, bfi = Generate_Bmaps_MultiProcess(EXPAND_ORDER, r_axis, fi_axis, convex, s_fit, bmap["coefficients_tupel"])
    for order in range(len(br)):
        np.savez(folder / f"Br_order_{order}.npz", Br_Taylor_coeff=br[order])
        np.savez(folder / f"Bz_order_{order}.npz", Bz_Taylor_coeff=bz[order])
        np.savez(folder / f"Bfi_order_{order}.npz", Bfi_Taylor_coeff=bfi[order])
    print(f"Generated Bmap: {folder.name}")


def seo_complete(folder):
    p = folder / "resultsSEO"
    if not all((p / name).exists() for name in ("SEO_ini.txt", "SEO_r.txt", "SEO_pr.txt", "Bz.txt")):
        return False
    try:
        data = np.loadtxt(p / "SEO_r.txt")
        return data.ndim == 2 and data.shape[0] >= 3
    except Exception:
        return False


def run_seo(folder):
    if seo_complete(folder):
        print(f"SEO exists: {folder.name}")
        return
    print(f"Running SEO: {folder.name}")
    seo_path = Path(main_SEO.main(SEO_EK_MEV, str(folder).replace("\\", "/"), "resultsSEO", BHARMONICS))
    cfg = {
        "config": {
            "start_Ek": float(SEO_EK_MEV[0]),
            "end_Ek": float(SEO_EK_MEV[-1]),
            "delta_Ek": 50,
            "extra_Ek": [],
            "Bmap_path": str(folder).replace("\\", "/"),
            "SEO_folder": "resultsSEO",
        },
        "BHarmonics": BHARMONICS,
    }
    (seo_path / "config_SEO.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def off_key(v):
    mm = int(round(v * 1000.0))
    return "off0mm" if mm == 0 else f"off{'p' if mm > 0 else 'm'}{abs(mm)}mm"


def make_offset_point(p0, bmap):
    p = dict(p0)
    p["closed_r_m"], p["closed_pr"], p["radial_offset_m"] = p0["r_m"], p0["pr"], RADIAL_OFFSET_M
    p["r_m"] = p0["r_m"] + RADIAL_OFFSET_M
    p["x_m"], p["y_m"] = float(p["r_m"] * np.cos(p["phi_rad"])), float(p["r_m"] * np.sin(p["phi_rad"]))
    p["bz_T"] = float(base.get_bfields_for_points(bmap, [p["r_m"]], [p["phi_rad"]], [0.0], MAX_ORDER)[0][0])
    return p


def run_reverse(group, folder):
    print(f"Running reverse injection: {folder.name}")
    base.OUTPUT_DIR = str(OUTPUT_DIR)
    base.TARGET_EK_MEV = float(SEO_EK_MEV[0])
    base.TRACK_TIME_NS, base.TIME_STEP_NS, base.DUMP_INTERVAL_NS = TRACK_TIME_NS, TIME_STEP_NS, DUMP_INTERVAL_NS
    base.APERTURE_R_MAX_M = APERTURE_R_MAX_M
    base.OPEN_OUTPUT_PLOTS = False
    bmap = base.FFAG_Bfield_analytical(str(folder / "config_Bmap.json"), MAX_ORDER)
    seo = base.load_seo_inputs(str(folder / "resultsSEO"))
    point = make_offset_point(base.select_foil_point(bmap, seo, MAX_ORDER)["selected_point"], bmap)
    key = f"{folder.name.replace('Bmap_FD16_', '')}_{off_key(RADIAL_OFFSET_M)}"
    spec = {"key": key, "label": key, "path": str(folder).replace("\\", "/"), "max_order": MAX_ORDER}
    _, particle_v, particle_p_reversed = base.make_reverse_particle(point)
    _, particle_path = base.run_reverse_tracking(spec, bmap, seo, particle_v)
    traj = base.first_valid_segment(base.load_and_augment_trajectory(particle_path, particle_p_reversed, bmap, MAX_ORDER), bmap)
    track_csv = OUTPUT_DIR / f"{key}_track.csv"
    simple_csv = OUTPUT_DIR / f"{key}_r_fi_bz.csv"
    base.save_trajectory_csv(traj, track_csv)
    base.save_r_fi_bz_csv(traj, simple_csv)
    return {
        "group": group,
        "folder": folder.name,
        "key": key,
        "offset_m": RADIAL_OFFSET_M,
        "foil_phi_deg": point["phi_deg"],
        "closed_R_m": point["closed_r_m"],
        "start_R_m": point["r_m"],
        "start_Bz_T": point["bz_T"],
        "samples": int(traj["r"].size),
        "path_length_m": float(traj["s"][-1]),
        "R_min_m": float(np.nanmin(traj["r"])),
        "R_max_m": float(np.nanmax(traj["r"])),
        "Bz_min_T": float(np.nanmin(traj["bz"])),
        "Bz_max_T": float(np.nanmax(traj["bz"])),
        "max_abs_Bz_T": float(np.nanmax(np.abs(traj["bz"]))),
        "track_csv": str(track_csv),
        "r_fi_bz_csv": str(simple_csv),
    }


def save_summary(rows):
    header = ["group", "folder", "key", "offset_m", "foil_phi_deg", "closed_R_m", "start_R_m", "start_Bz_T",
              "samples", "path_length_m", "R_min_m", "R_max_m", "Bz_min_T", "Bz_max_T", "max_abs_Bz_T",
              "track_csv", "r_fi_bz_csv"]
    with open(OUTPUT_DIR / "extra_param_reverse_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []
    failures = []
    for group, c in CASES:
        name = folder_name(c)
        folder = BMAP_ROOT / name
        try:
            generate_bmap(folder, c)
            run_seo(folder)
            row = run_reverse(group, folder)
            rows.append(row)
            print(f"  {name}: max|Bz|={row['max_abs_Bz_T']:.6f} T, s={row['path_length_m']:.6f} m")
        except Exception as exc:
            failures.append((name, repr(exc)))
            print(f"FAILED: {name}: {exc}")
    save_summary(rows)
    if failures:
        (OUTPUT_DIR / "extra_param_failed_cases.txt").write_text("\n".join(f"{name}: {err}" for name, err in failures), encoding="utf-8")
    print(f"finished: {len(rows)} ok, {len(failures)} failed")
    print(f"summary: {OUTPUT_DIR / 'extra_param_reverse_summary.csv'}")


if __name__ == "__main__":
    main()
