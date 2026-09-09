import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

import main_SEO
import tool_reverse_inj_compare_k_analytical as base
from FFAG_Field import FFAG_Bfield_analytical
from FFAG_HighOrder import BmapParams_to_ConvexParams, Generate_Bmaps_MultiProcess, calculate_s
from FFAG_ISO import CheckISOBmapConfigNew


BMAP_ROOT = Path("./Bmaps")
OUTPUT_DIR = Path("./reverse_R_F_offset_grid_output")
EXPAND_ORDER = 0
MAX_ORDER = 0
SEO_EK_MEV = np.array([300.0, 350.0])
TRACK_TIME_NS = 120.0
TIME_STEP_NS = 0.02
DUMP_INTERVAL_NS = 0.10
APERTURE_R_MAX_M = 12.0
BHARMONICS = {"enable": False, "harmonics": []}

BASE = {"k": 7.630003, "B": 1.65, "s": 49.973388}
R_VALUES = [0.265074, 0.365074, 0.465074, 0.565074]
F_VALUES_DEG = [0.0, 0.5, 0.75]
RADIAL_OFFSETS_M = [0.0, -0.015, -0.030, -0.045, -0.060]


def tag_float(v, nd=2):
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "n")


def tag_r(v):
    return f"{int(round(v * 1000.0 - 0.074)):03d}"


def tag_f(v):
    if abs(v) < 1e-12:
        return "0"
    return ("p" if v > 0 else "n") + tag_float(abs(v), 3)


def off_key(v):
    mm = int(round(v * 1000.0))
    return "off0mm" if mm == 0 else f"off{'p' if mm > 0 else 'm'}{abs(mm)}mm"


def folder_name(r_ratio, f_deg):
    return f"Bmap_FD16_k{tag_float(BASE['k'], 2)}_B{tag_float(BASE['B'], 2)}_R{tag_r(r_ratio)}_F{tag_f(f_deg)}_s{tag_float(BASE['s'], 2)}"


def make_config(folder, r_ratio, f_deg):
    bmap = {
        "Type": "SCALE",
        "NSector": 16,
        "theta_step_rad": np.deg2rad(0.005),
        "rmin_max_step_m": (10.8, 12.2, 0.001),
        "interval_1": (5.625 / 22.5, (5.9083 + f_deg) / 22.5, 2.38755 / 22.5, (2.95415 - f_deg) / 22.5, 5.625 / 22.5),
        "positive_or_negative_1": (0, 1.0, 0, -float(r_ratio), 0),
        "fringe_width_1": (0, 1.17, 0, 1.17, 0),
        "SpiralAngle_deg": BASE["s"],
        "orbital_freq_MHz": 4.5,
        "k_value": BASE["k"],
        "B0_max_T": BASE["B"],
        "R0_m": 12.0,
    }
    if bmap["interval_1"][3] <= 0:
        raise ValueError(f"D iron width must stay positive, F={f_deg} deg is too large.")
    bmap["B0_T"] = bmap["B0_max_T"] / (bmap["R0_m"] ** bmap["k_value"])
    return {
        "config": {"date": 0.0, "folder": str(folder).replace("\\", "/"), "expand_order": EXPAND_ORDER},
        "machine": {"energy_inj": 300, "energy_ext": 600},
        "Bmap": bmap,
        "BzInfo": {"IsoCoef": None, "IsoError": None},
    }


def bmap_complete(folder):
    return all((folder / name).exists() for name in ("config_Bmap.json", "Br_order_0.npz", "Bz_order_0.npz", "Bfi_order_0.npz"))


def generate_bmap(folder, r_ratio, f_deg):
    if bmap_complete(folder):
        print(f"Bmap exists: {folder.name}")
        return
    folder.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    cfg = make_config(folder, r_ratio, f_deg)
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
    return all((p / name).exists() and (p / name).stat().st_size > 0
               for name in ("SEO_ini.txt", "SEO_r.txt", "SEO_pr.txt", "Bz.txt"))


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


def make_offset_point(p0, bmap, offset_m):
    p = dict(p0)
    p["closed_r_m"], p["closed_pr"], p["radial_offset_m"] = p0["r_m"], p0["pr"], float(offset_m)
    p["r_m"] = p0["r_m"] + float(offset_m)
    p["x_m"], p["y_m"] = float(p["r_m"] * np.cos(p["phi_rad"])), float(p["r_m"] * np.sin(p["phi_rad"]))
    p["bz_T"] = float(base.get_bfields_for_points(bmap, [p["r_m"]], [p["phi_rad"]], [0.0], MAX_ORDER)[0][0])
    return p


def existing_row(folder, r_ratio, f_deg, offset_m, csv_path):
    if not csv_path.exists():
        return None
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    r = np.asarray(d["r"], dtype=float)
    fi = np.asarray(d["fi"], dtype=float)
    bz = np.asarray(d["bz"], dtype=float)
    ok = np.isfinite(r) & np.isfinite(fi) & np.isfinite(bz)
    r, fi, bz = r[ok], fi[ok], bz[ok]
    x, y = r * np.cos(fi), r * np.sin(fi)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    return {
        "folder": folder.name,
        "R_ratio": r_ratio,
        "F_deg": f_deg,
        "offset_m": offset_m,
        "offset_mm": int(round(offset_m * 1000.0)),
        "samples": int(r.size),
        "path_length_m": float(s[-1]),
        "R_min_m": float(np.nanmin(r)),
        "R_max_m": float(np.nanmax(r)),
        "Bz_min_T": float(np.nanmin(bz)),
        "Bz_max_T": float(np.nanmax(bz)),
        "max_abs_Bz_T": float(np.nanmax(np.abs(bz))),
        "track_csv": "",
        "r_fi_bz_csv": str(csv_path),
        "reused_existing": True,
    }


def run_reverse_for_offset(folder, r_ratio, f_deg, bmap, seo, closed_point, offset_m):
    key = f"{folder.name.replace('Bmap_FD16_', '')}_{off_key(offset_m)}"
    track_csv = OUTPUT_DIR / f"{key}_track.csv"
    simple_csv = OUTPUT_DIR / f"{key}_r_fi_bz.csv"
    reused = existing_row(folder, r_ratio, f_deg, offset_m, simple_csv)
    if reused is not None:
        print(f"  reuse offset {offset_m * 1000.0:+.0f} mm: max|Bz|={reused['max_abs_Bz_T']:.6f} T")
        return reused

    point = make_offset_point(closed_point, bmap, offset_m)
    spec = {"key": key, "label": key, "path": str(folder).replace("\\", "/"), "max_order": MAX_ORDER}
    _, particle_v, particle_p_reversed = base.make_reverse_particle(point)
    _, particle_path = base.run_reverse_tracking(spec, bmap, seo, particle_v)
    traj = base.first_valid_segment(base.load_and_augment_trajectory(particle_path, particle_p_reversed, bmap, MAX_ORDER), bmap)
    base.save_trajectory_csv(traj, track_csv)
    base.save_r_fi_bz_csv(traj, simple_csv)
    row = {
        "folder": folder.name,
        "R_ratio": r_ratio,
        "F_deg": f_deg,
        "offset_m": offset_m,
        "offset_mm": int(round(offset_m * 1000.0)),
        "closed_R_m": point["closed_r_m"],
        "start_R_m": point["r_m"],
        "start_Bz_T": point["bz_T"],
        "foil_phi_deg": point["phi_deg"],
        "samples": int(traj["r"].size),
        "path_length_m": float(traj["s"][-1]),
        "R_min_m": float(np.nanmin(traj["r"])),
        "R_max_m": float(np.nanmax(traj["r"])),
        "Bz_min_T": float(np.nanmin(traj["bz"])),
        "Bz_max_T": float(np.nanmax(traj["bz"])),
        "max_abs_Bz_T": float(np.nanmax(np.abs(traj["bz"]))),
        "track_csv": str(track_csv),
        "r_fi_bz_csv": str(simple_csv),
        "reused_existing": False,
    }
    print(f"  offset {offset_m * 1000.0:+.0f} mm: max|Bz|={row['max_abs_Bz_T']:.6f} T")
    return row


def save_summary(rows):
    header = ["folder", "R_ratio", "F_deg", "offset_m", "offset_mm", "closed_R_m", "start_R_m", "start_Bz_T",
              "foil_phi_deg", "samples", "path_length_m", "R_min_m", "R_max_m", "Bz_min_T", "Bz_max_T",
              "max_abs_Bz_T", "track_csv", "r_fi_bz_csv", "reused_existing"]
    with open(OUTPUT_DIR / "R_F_offset_grid_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmaps(rows):
    for offset_m in RADIAL_OFFSETS_M:
        grid = np.full((len(F_VALUES_DEG), len(R_VALUES)), np.nan)
        for row in rows:
            if abs(float(row["offset_m"]) - offset_m) > 1e-12:
                continue
            i = F_VALUES_DEG.index(float(row["F_deg"]))
            j = R_VALUES.index(float(row["R_ratio"]))
            grid[i, j] = float(row["max_abs_Bz_T"])
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(R_VALUES)), [f"{v:.3f}" for v in R_VALUES])
        ax.set_yticks(range(len(F_VALUES_DEG)), [f"{v:.2f}" for v in F_VALUES_DEG])
        ax.set_xlabel("DF ratio R")
        ax.set_ylabel("F angle increase [deg]")
        ax.set_title(f"max |Bz| on reverse track, offset {offset_m * 1000.0:+.0f} mm")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="white" if grid[i, j] > np.nanmean(grid) else "black", fontsize=9)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("max |Bz| [T]")
        fig.savefig(OUTPUT_DIR / f"heatmap_max_abs_Bz_offset_{off_key(offset_m)}.png", dpi=180)
        plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    base.OUTPUT_DIR = str(OUTPUT_DIR)
    base.TARGET_EK_MEV = float(SEO_EK_MEV[0])
    base.TRACK_TIME_NS, base.TIME_STEP_NS, base.DUMP_INTERVAL_NS = TRACK_TIME_NS, TIME_STEP_NS, DUMP_INTERVAL_NS
    base.APERTURE_R_MAX_M = APERTURE_R_MAX_M
    base.OPEN_OUTPUT_PLOTS = False

    rows, failures = [], []
    for r_ratio in R_VALUES:
        for f_deg in F_VALUES_DEG:
            folder = BMAP_ROOT / folder_name(r_ratio, f_deg)
            try:
                print(f"\nCase R={r_ratio:.6f}, F={f_deg:.3f} deg -> {folder.name}")
                generate_bmap(folder, r_ratio, f_deg)
                run_seo(folder)
                plt.close("all")
                bmap = base.FFAG_Bfield_analytical(str(folder / "config_Bmap.json"), MAX_ORDER)
                seo = base.load_seo_inputs(str(folder / "resultsSEO"))
                closed_point = base.select_foil_point(bmap, seo, MAX_ORDER)["selected_point"]
                for offset_m in RADIAL_OFFSETS_M:
                    try:
                        rows.append(run_reverse_for_offset(folder, r_ratio, f_deg, bmap, seo, closed_point, offset_m))
                        save_summary(rows)
                    except Exception as exc:
                        failures.append((f"{folder.name}, offset {offset_m * 1000.0:+.0f} mm", repr(exc)))
                        print(f"FAILED: {folder.name}, offset {offset_m * 1000.0:+.0f} mm: {exc}")
            except Exception as exc:
                failures.append((folder.name, repr(exc)))
                print(f"FAILED: {folder.name}: {exc}")
    save_summary(rows)
    plot_heatmaps(rows)
    if failures:
        (OUTPUT_DIR / "R_F_offset_grid_failed_cases.txt").write_text("\n".join(f"{name}: {err}" for name, err in failures), encoding="utf-8")
    print(f"\nfinished: {len(rows)} rows, {len(failures)} failed cases")
    print(f"summary: {OUTPUT_DIR / 'R_F_offset_grid_summary.csv'}")


if __name__ == "__main__":
    main()
